#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapanything_config import load_service_config
from noesis.ds8_runtime import _CalibrationProvider, _load_camera_labels
from noesis.virtual_twin.artifacts import write_json, write_points_glb, write_points_npz, write_textured_mesh_glb
from noesis.virtual_twin.builder import calibration_fingerprint, revision_id_from_clock
from noesis.virtual_twin.geometry import (
    backproject_depth,
    transform_points,
)
from noesis.virtual_twin.store import VirtualTwinStore


DEFAULT_CAMERAS = ("living-room", "kitchen", "family-room")
CACHED_DEPTH_COLOR_SOURCE = "mapanything_cached_zarr_depth_confidence_texture"
ZARR_RGB_COLOR_SOURCE = "mapanything_persisted_zarr_rgb_texture"
LIVE_RGB_COLOR_SOURCE = "pipeline_camera_uri_live_capture_dewarped_to_depth_frame"


@dataclass(frozen=True)
class DepthSnapshot:
    path: Path
    timestamp_us: int
    depth: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray
    rgb: np.ndarray | None = None
    source_paths: tuple[Path, ...] = ()
    source_timestamps_us: tuple[int, ...] = ()
    fusion_meta: dict[str, Any] | None = None
    snapshot_role: str | None = None
    fusion_level: str | None = None


@dataclass(frozen=True)
class RgbFrame:
    source_index: int
    image_bgr: np.ndarray
    source_uri: str
    dewarper_config: str | None
    transformed: bool


@dataclass(frozen=True)
class CameraCalibration:
    source_id: int
    camera_id: str
    snapshot: Any
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    floor_y: float


@dataclass(frozen=True)
class FrameCloud:
    frame_id: str
    snapshot: DepthSnapshot
    depth_clip_m: float
    points_camera: np.ndarray
    points_world: np.ndarray
    pixels: np.ndarray
    confidence: np.ndarray
    image_bgr: np.ndarray
    image_ref: str


def _read_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected mapping at {path}")
    return payload


def _resolve_repo_path(path: Path | str | None, fallback: Path | str) -> Path:
    raw = Path(path) if path is not None else Path(fallback)
    raw = raw.expanduser()
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


@dataclass(frozen=True)
class DewarperFrameTransform:
    config_path: Path
    source_intrinsics: np.ndarray
    distortion: np.ndarray
    rectified_intrinsics: np.ndarray
    output_size: tuple[int, int]


def _camera_source(pipeline_cfg: dict[str, Any], camera_labels: dict[int, str], camera_id: str) -> tuple[int, str, dict[str, Any]]:
    sources = pipeline_cfg.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError("pipeline config has no sources list")
    wanted = str(camera_id).strip()
    for source_id, label in camera_labels.items():
        if str(label).strip() != wanted:
            continue
        source = sources[int(source_id)] if int(source_id) < len(sources) else None
        if not isinstance(source, dict) or not isinstance(source.get("uri"), str):
            raise RuntimeError(f"camera {wanted!r} has no configured URI in pipeline sources")
        return int(source_id), str(source["uri"]), source
    raise RuntimeError(f"camera {wanted!r} not found in camera labels: {sorted(camera_labels.values())}")


def _parse_float_pair(value: str, *, field: str, path: Path) -> tuple[float, float]:
    parts = [part.strip() for part in str(value or "").split(";") if part.strip()]
    if len(parts) != 2:
        raise RuntimeError(f"dewarper config {path} field {field} must contain two semicolon-separated numbers")
    return float(parts[0]), float(parts[1])


def _parse_float_list(value: str, *, field: str, path: Path) -> list[float]:
    parts = [part.strip() for part in str(value or "").split(";") if part.strip()]
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise RuntimeError(f"dewarper config {path} field {field} contains non-numeric values") from exc


def _load_dewarper_frame_transform(path: Path) -> DewarperFrameTransform:
    cfg_path = _resolve_repo_path(path, path)
    if not cfg_path.exists():
        raise RuntimeError(f"configured dewarper file does not exist: {cfg_path}")
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.optionxform = str
    parser.read(cfg_path, encoding="utf-8")
    if "property" not in parser or "surface0" not in parser:
        raise RuntimeError(f"dewarper config {cfg_path} must contain [property] and [surface0]")
    props = parser["property"]
    surface = parser["surface0"]
    output_w = int(float(props.get("output-width", surface.get("width", "0"))))
    output_h = int(float(props.get("output-height", surface.get("height", "0"))))
    src_fx, src_fy = _parse_float_pair(surface.get("focal-length", ""), field="focal-length", path=cfg_path)
    dst_fx, dst_fy = _parse_float_pair(surface.get("dst-focal-length", ""), field="dst-focal-length", path=cfg_path)
    src_cx = float(surface.get("src-x0", "nan"))
    src_cy = float(surface.get("src-y0", "nan"))
    dst_cx, dst_cy = _parse_float_pair(surface.get("dst-principal-point", ""), field="dst-principal-point", path=cfg_path)
    if output_w <= 0 or output_h <= 0:
        raise RuntimeError(f"dewarper config {cfg_path} has invalid output size {output_w}x{output_h}")
    distortion = np.asarray(_parse_float_list(surface.get("distortion", ""), field="distortion", path=cfg_path), dtype=np.float64)
    if distortion.size < 4:
        raise RuntimeError(f"dewarper config {cfg_path} needs at least four fisheye distortion coefficients")
    source_intrinsics = np.asarray(
        [[src_fx, 0.0, src_cx], [0.0, src_fy, src_cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rectified_intrinsics = np.asarray(
        [[dst_fx, 0.0, dst_cx], [0.0, dst_fy, dst_cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(source_intrinsics)) or not np.all(np.isfinite(rectified_intrinsics)):
        raise RuntimeError(f"dewarper config {cfg_path} contains non-finite intrinsics")
    return DewarperFrameTransform(
        config_path=cfg_path,
        source_intrinsics=source_intrinsics,
        distortion=distortion.reshape((-1, 1)),
        rectified_intrinsics=rectified_intrinsics,
        output_size=(output_w, output_h),
    )


def _dewarper_transform_for_source(source_cfg: dict[str, Any]) -> DewarperFrameTransform | None:
    dewarper = source_cfg.get("dewarper")
    if not isinstance(dewarper, dict) or not bool(dewarper.get("enable", False)):
        return None
    config_file = str(dewarper.get("config-file") or "").strip()
    if not config_file:
        raise RuntimeError("source dewarper is enabled but has no config-file")
    return _load_dewarper_frame_transform(Path(config_file))


def _apply_dewarper_transform(frame_bgr: np.ndarray, transform: DewarperFrameTransform) -> np.ndarray:
    image_h, image_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    source_k = np.asarray(transform.source_intrinsics, dtype=np.float64).copy()
    authored_w, authored_h = int(transform.output_size[0]), int(transform.output_size[1])
    if image_w > 0 and image_h > 0 and (image_w, image_h) != (authored_w, authored_h):
        sx = float(image_w) / float(authored_w)
        sy = float(image_h) / float(authored_h)
        source_k[0, 0] *= sx
        source_k[0, 2] *= sx
        source_k[1, 1] *= sy
        source_k[1, 2] *= sy
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        source_k,
        np.asarray(transform.distortion, dtype=np.float64).reshape((-1, 1)),
        np.eye(3, dtype=np.float64),
        np.asarray(transform.rectified_intrinsics, dtype=np.float64).reshape(3, 3),
        tuple(int(v) for v in transform.output_size),
        cv2.CV_32FC1,
    )
    return cv2.remap(frame_bgr, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _capture_rgb_frames(
    *,
    pipeline_cfg: dict[str, Any],
    camera_labels: dict[int, str],
    camera_id: str,
    count: int,
    frame_stride: int,
    max_frames_read: int,
) -> list[RgbFrame]:
    _source_id, uri, source_cfg = _camera_source(pipeline_cfg, camera_labels, camera_id)
    transform = _dewarper_transform_for_source(source_cfg)
    cap = cv2.VideoCapture(uri[7:] if uri.startswith("file://") else uri)
    if not cap.isOpened():
        raise RuntimeError(f"unable to open camera source URI for RGB depth cloud: {uri}")
    frames: list[RgbFrame] = []
    try:
        idx = 0
        while idx < int(max_frames_read) and len(frames) < int(count):
            ok, raw = cap.read()
            if not ok or raw is None:
                break
            if idx % max(1, int(frame_stride)) == 0:
                image = _apply_dewarper_transform(raw, transform) if transform is not None else raw
                frames.append(
                    RgbFrame(
                        source_index=int(idx),
                        image_bgr=np.asarray(image, dtype=np.uint8),
                        source_uri=uri,
                        dewarper_config=str(transform.config_path) if transform is not None else None,
                        transformed=transform is not None,
                    )
                )
            idx += 1
    finally:
        cap.release()
    if len(frames) < int(count):
        raise RuntimeError(f"captured {len(frames)} RGB frames for {camera_id}, expected {count} from {uri}")
    return frames


def _depth_texture_bgr(snapshot: DepthSnapshot, *, min_confidence: float, depth_clip_m: float) -> np.ndarray:
    depth = np.asarray(snapshot.depth, dtype=np.float32)
    valid = _valid_depth_mask(snapshot, min_confidence, depth_clip_m)
    image = np.zeros((int(depth.shape[0]), int(depth.shape[1]), 3), dtype=np.uint8)
    values = depth[valid]
    if values.size == 0:
        return image
    low = float(np.percentile(values, 2.0))
    high = float(np.percentile(values, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return image
    normalized = np.clip((depth - low) / max(1e-6, high - low), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    confidence = np.asarray(snapshot.confidence, dtype=np.float32)
    conf_valid = confidence[valid]
    conf_high = float(np.percentile(conf_valid, 95.0)) if conf_valid.size else 1.0
    if not np.isfinite(conf_high) or conf_high <= 0.0:
        conf_high = 1.0
    confidence_weight = np.clip(confidence / conf_high, 0.0, 1.0)
    shaded = colored.astype(np.float32) * (0.35 + 0.65 * confidence_weight[:, :, None])
    image[valid] = np.clip(shaded[valid], 0.0, 255.0).astype(np.uint8)
    image[~valid] = np.asarray([18, 18, 18], dtype=np.uint8)
    return image


def _cached_depth_texture_frames(
    snapshots: Sequence[DepthSnapshot],
    *,
    min_confidence: float,
    depth_clip_percentile: float,
    max_depth_m: float,
) -> list[RgbFrame]:
    frames: list[RgbFrame] = []
    for index, snapshot in enumerate(snapshots):
        depth_clip_m = _depth_clip(snapshot, min_confidence, depth_clip_percentile, max_depth_m)
        frames.append(
            RgbFrame(
                source_index=int(index),
                image_bgr=_depth_texture_bgr(snapshot, min_confidence=min_confidence, depth_clip_m=depth_clip_m),
                source_uri=f"mapanything_zarr:{_relative_path(snapshot.path)}",
                dewarper_config=None,
                transformed=False,
            )
        )
    return frames


def _zarr_rgb_frames(snapshots: Sequence[DepthSnapshot]) -> list[RgbFrame]:
    frames: list[RgbFrame] = []
    for index, snapshot in enumerate(snapshots):
        if snapshot.rgb is None:
            sources = [_relative_path(path) for path in _snapshot_source_paths(snapshot)]
            raise RuntimeError(f"snapshot has no persisted RGB layer: {sources}")
        rgb = np.asarray(snapshot.rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError(f"snapshot RGB layer has invalid shape: {snapshot.path} rgb={rgb.shape}")
        frames.append(
            RgbFrame(
                source_index=int(index),
                image_bgr=np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1], dtype=np.uint8),
                source_uri=f"mapanything_zarr_rgb:{_relative_path(snapshot.path)}",
                dewarper_config=None,
                transformed=False,
            )
        )
    return frames


def _snapshot_entries(depth_base: Path, camera_id: str) -> list[tuple[int, Path]]:
    root = Path(depth_base) / str(camera_id)
    if not root.exists():
        return []
    rows: list[tuple[int, Path]] = []
    for path in root.rglob("*.zarr"):
        try:
            rows.append((int(path.stem), path))
        except ValueError:
            continue
    rows.sort(key=lambda item: item[0])
    return rows


def _load_depth_snapshot(path: Path) -> DepthSnapshot:
    group = zarr.open_group(str(path), mode="r")
    depth = np.asarray(np.array(group["depth_z"]), dtype=np.float32)
    confidence = np.asarray(np.array(group["conf"]), dtype=np.float32)
    mask = np.asarray(np.array(group["mask"]), dtype=bool)
    if depth.ndim != 2 or confidence.shape != depth.shape or mask.shape != depth.shape:
        raise RuntimeError(f"bad MapAnything zarr shape at {path}: depth={depth.shape} conf={confidence.shape} mask={mask.shape}")
    rgb = None
    if "rgb" in group:
        rgb_arr = np.asarray(np.array(group["rgb"]), dtype=np.uint8)
        if rgb_arr.ndim == 3 and rgb_arr.shape[2] >= 3:
            rgb = np.ascontiguousarray(rgb_arr[:, :, :3], dtype=np.uint8)
        else:
            raise RuntimeError(f"bad MapAnything zarr RGB shape at {path}: rgb={rgb_arr.shape}")
    timestamp_us = int(group.attrs.get("timestamp_us", path.stem))
    attrs = dict(group.attrs.asdict() if hasattr(group.attrs, "asdict") else dict(group.attrs))
    source_paths_raw = attrs.get("source_snapshot_paths")
    source_timestamps_raw = attrs.get("source_timestamps_us")
    fusion_meta_raw = attrs.get("fusion_meta")
    try:
        if isinstance(source_paths_raw, str):
            source_paths = tuple(Path(item) for item in json.loads(source_paths_raw))
        elif isinstance(source_paths_raw, (list, tuple)):
            source_paths = tuple(Path(str(item)) for item in source_paths_raw)
        else:
            source_paths = (Path(path),)
    except Exception:
        source_paths = (Path(path),)
    try:
        if isinstance(source_timestamps_raw, str):
            source_timestamps = tuple(int(item) for item in json.loads(source_timestamps_raw))
        elif isinstance(source_timestamps_raw, (list, tuple)):
            source_timestamps = tuple(int(item) for item in source_timestamps_raw)
        else:
            source_timestamps = (int(timestamp_us),)
    except Exception:
        source_timestamps = (int(timestamp_us),)
    fusion_meta = None
    try:
        if isinstance(fusion_meta_raw, str) and fusion_meta_raw.strip():
            parsed = json.loads(fusion_meta_raw)
            if isinstance(parsed, dict):
                fusion_meta = parsed
        elif isinstance(fusion_meta_raw, dict):
            fusion_meta = dict(fusion_meta_raw)
    except Exception:
        fusion_meta = None
    return DepthSnapshot(
        path=Path(path),
        timestamp_us=timestamp_us,
        depth=depth,
        confidence=confidence,
        mask=mask,
        rgb=rgb,
        source_paths=source_paths,
        source_timestamps_us=source_timestamps,
        fusion_meta=fusion_meta,
        snapshot_role=str(attrs.get("snapshot_role") or "").strip() or None,
        fusion_level=str(attrs.get("fusion_level") or "").strip() or None,
    )


def _snapshot_source_paths(snapshot: DepthSnapshot) -> tuple[Path, ...]:
    return tuple(snapshot.source_paths or (snapshot.path,))


def _snapshot_source_timestamps(snapshot: DepthSnapshot) -> tuple[int, ...]:
    return tuple(snapshot.source_timestamps_us or (int(snapshot.timestamp_us),))


def _valid_fraction(snapshot: DepthSnapshot, min_confidence: float) -> float:
    valid = (
        np.asarray(snapshot.mask, dtype=bool)
        & np.isfinite(snapshot.depth)
        & (snapshot.depth > 0.0)
        & np.isfinite(snapshot.confidence)
        & (snapshot.confidence >= float(min_confidence))
    )
    return float(np.count_nonzero(valid) / max(1, valid.size))


def _snapshot_depth_percentile(snapshot: DepthSnapshot, min_confidence: float, depth_clip_m: float, percentile: float) -> float | None:
    valid = _valid_depth_mask(snapshot, min_confidence, depth_clip_m)
    values = np.asarray(snapshot.depth, dtype=np.float32)[valid]
    if values.size <= 0:
        return None
    return float(np.percentile(values, float(percentile)))


def _latest_snapshots(depth_base: Path, camera_id: str, count: int, min_confidence: float) -> list[DepthSnapshot]:
    snapshots: list[DepthSnapshot] = []
    for _ts, path in reversed(_snapshot_entries(depth_base, camera_id)):
        try:
            snapshot = _load_depth_snapshot(path)
        except Exception:
            continue
        if _valid_fraction(snapshot, min_confidence) <= 0.0:
            continue
        snapshots.append(snapshot)
        if len(snapshots) >= int(count):
            break
    snapshots.reverse()
    if len(snapshots) < int(count):
        raise RuntimeError(f"found {len(snapshots)} usable snapshots for {camera_id}, expected {count} under {depth_base}")
    return snapshots


def _is_derived_snapshot(snapshot: DepthSnapshot) -> bool:
    role = str(snapshot.snapshot_role or "").strip().lower()
    level = str(snapshot.fusion_level or "").strip().lower()
    return role in {"capture_event_fused", "reconstruction_fused"} or level in {"intra_capture", "inter_capture"}


def _is_capture_event_snapshot(snapshot: DepthSnapshot) -> bool:
    role = str(snapshot.snapshot_role or "").strip().lower()
    level = str(snapshot.fusion_level or "").strip().lower()
    return role == "capture_event_fused" or level == "intra_capture"


def _group_raw_snapshot_entries(entries: Sequence[tuple[int, Path]], *, event_gap_s: float) -> list[list[tuple[int, Path]]]:
    gap_us = max(1, int(float(event_gap_s) * 1_000_000))
    groups: list[list[tuple[int, Path]]] = []
    current: list[tuple[int, Path]] = []
    for ts, path in sorted(entries, key=lambda item: item[0]):
        if current and int(ts) - int(current[-1][0]) > gap_us:
            groups.append(current)
            current = []
        current.append((int(ts), path))
    if current:
        groups.append(current)
    return groups


def _event_snapshot_candidates(
    snapshots: Sequence[DepthSnapshot],
    *,
    max_raw_snapshots: int,
    min_confidence: float,
) -> list[DepthSnapshot]:
    if len(snapshots) <= max(1, int(max_raw_snapshots)):
        return list(snapshots)
    ranked = sorted(
        snapshots,
        key=lambda snapshot: (_valid_fraction(snapshot, min_confidence), int(snapshot.timestamp_us)),
        reverse=True,
    )
    selected = ranked[: max(1, int(max_raw_snapshots))]
    selected.sort(key=lambda snapshot: int(snapshot.timestamp_us))
    return selected


def _fuse_depth_snapshots(
    snapshots: Sequence[DepthSnapshot],
    *,
    camera_id: str,
    min_confidence: float,
    min_observations: int,
    depth_agreement_m: float,
    fusion_level: str = "inter_capture",
    fusion_mode: str = "per_pixel_weighted_depth_consensus",
) -> DepthSnapshot:
    if len(snapshots) <= 1:
        return snapshots[0]
    shape = tuple(int(dim) for dim in snapshots[0].depth.shape)
    for snapshot in snapshots:
        if tuple(int(dim) for dim in snapshot.depth.shape) != shape:
            raise RuntimeError(f"{camera_id}: cannot fuse snapshots with mismatched depth shapes")

    depth_stack = np.stack([np.asarray(snapshot.depth, dtype=np.float32) for snapshot in snapshots], axis=0)
    conf_stack = np.stack([np.asarray(snapshot.confidence, dtype=np.float32) for snapshot in snapshots], axis=0)
    mask_stack = np.stack([np.asarray(snapshot.mask, dtype=bool) for snapshot in snapshots], axis=0)
    valid = mask_stack & np.isfinite(depth_stack) & (depth_stack > 0.0) & np.isfinite(conf_stack) & (conf_stack >= float(min_confidence))
    masked_depth = np.ma.array(depth_stack, mask=~valid)
    median_depth = np.ma.median(masked_depth, axis=0).filled(np.nan).astype(np.float32)
    tolerance = float(depth_agreement_m) + np.nan_to_num(median_depth, nan=0.0, posinf=0.0, neginf=0.0) * 0.025
    agreeing = valid & np.isfinite(median_depth)[None, :, :] & (np.abs(depth_stack - median_depth[None, :, :]) <= tolerance[None, :, :])
    support = np.count_nonzero(agreeing, axis=0)
    required = max(1, min(int(min_observations), len(snapshots)))
    fused_mask = support >= required

    weights = np.where(agreeing, np.clip(conf_stack, 0.0, None), 0.0).astype(np.float32, copy=False)
    weight_sum = np.sum(weights, axis=0)
    weighted_depth_sum = np.sum(np.where(agreeing, depth_stack, 0.0) * weights, axis=0)
    fused_depth = np.zeros(shape, dtype=np.float32)
    np.divide(weighted_depth_sum, weight_sum, out=fused_depth, where=weight_sum > 0.0)
    fallback = fused_mask & ~(weight_sum > 0.0) & np.isfinite(median_depth)
    fused_depth[fallback] = median_depth[fallback]
    fused_depth[~fused_mask] = 0.0

    conf_sum = np.sum(np.where(agreeing, conf_stack, 0.0), axis=0)
    fused_conf = np.zeros(shape, dtype=np.float32)
    np.divide(conf_sum, support, out=fused_conf, where=support > 0)
    fused_conf[~fused_mask] = 0.0

    rgb_inputs: list[np.ndarray] = []
    for snapshot in snapshots:
        if snapshot.rgb is None:
            continue
        rgb = np.asarray(snapshot.rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            continue
        if rgb.shape[:2] != shape:
            rgb = cv2.resize(rgb[:, :, :3], (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        rgb_inputs.append(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8))
    fused_rgb = None
    if rgb_inputs:
        if len(rgb_inputs) == 1:
            fused_rgb = rgb_inputs[0]
        else:
            fused_rgb = np.median(np.stack(rgb_inputs, axis=0).astype(np.float32), axis=0).astype(np.uint8)

    source_paths = tuple(path for snapshot in snapshots for path in _snapshot_source_paths(snapshot))
    source_timestamps = tuple(ts for snapshot in snapshots for ts in _snapshot_source_timestamps(snapshot))
    fusion_meta = {
        "mode": fusion_mode,
        "fusion_level": str(fusion_level),
        "source_count": int(len(snapshots)),
        "source_zarrs": [_relative_path(path) for path in source_paths],
        "source_timestamps_us": [int(ts) for ts in source_timestamps],
        "min_observations": int(required),
        "depth_agreement_m": float(depth_agreement_m),
        "support_valid_fraction": float(np.count_nonzero(fused_mask) / max(1, fused_mask.size)),
        "median_support": float(np.median(support[fused_mask])) if np.any(fused_mask) else 0.0,
        "rgb_source_count": int(len(rgb_inputs)),
    }
    child_fusions = [snapshot.fusion_meta for snapshot in snapshots if snapshot.fusion_meta is not None]
    if child_fusions:
        fusion_meta["child_fusions"] = child_fusions
    return DepthSnapshot(
        path=snapshots[-1].path,
        timestamp_us=int(snapshots[-1].timestamp_us),
        depth=fused_depth,
        confidence=fused_conf,
        mask=fused_mask,
        rgb=fused_rgb,
        source_paths=source_paths,
        source_timestamps_us=source_timestamps,
        fusion_meta=fusion_meta,
        snapshot_role="reconstruction_fused" if fusion_level == "inter_capture" else "capture_event_fused",
        fusion_level=str(fusion_level),
    )


def _latest_capture_snapshots(
    depth_base: Path,
    camera_id: str,
    count: int,
    *,
    min_confidence: float,
    event_gap_s: float,
    max_raw_snapshots_per_capture: int,
    min_observations: int,
    depth_agreement_m: float,
    prefer_persisted: bool,
) -> list[DepthSnapshot]:
    entries = _snapshot_entries(depth_base, camera_id)
    persisted: list[DepthSnapshot] = []
    raw_entries: list[tuple[int, Path]] = []
    persisted_source_paths: set[str] = set()
    for ts, path in entries:
        try:
            snapshot = _load_depth_snapshot(path)
        except Exception:
            continue
        if _valid_fraction(snapshot, min_confidence) <= 0.0:
            continue
        if _is_capture_event_snapshot(snapshot):
            persisted.append(snapshot)
            persisted_source_paths.update(str(path) for path in _snapshot_source_paths(snapshot))
        elif not _is_derived_snapshot(snapshot):
            raw_entries.append((int(ts), path))

    capture_events: list[DepthSnapshot] = []
    if prefer_persisted:
        capture_events.extend(persisted)

    needed = max(0, int(count) - len(capture_events))
    if needed > 0:
        groups = _group_raw_snapshot_entries(raw_entries, event_gap_s=float(event_gap_s))
        for group in reversed(groups):
            if all(str(path) in persisted_source_paths for _ts, path in group):
                continue
            loaded: list[DepthSnapshot] = []
            for _ts, path in group:
                try:
                    snapshot = _load_depth_snapshot(path)
                except Exception:
                    continue
                if _valid_fraction(snapshot, min_confidence) <= 0.0:
                    continue
                loaded.append(snapshot)
            loaded = _event_snapshot_candidates(
                loaded,
                max_raw_snapshots=max_raw_snapshots_per_capture,
                min_confidence=min_confidence,
            )
            if not loaded:
                continue
            if len(loaded) == 1:
                capture = loaded[0]
            else:
                capture = _fuse_depth_snapshots(
                    loaded,
                    camera_id=camera_id,
                    min_confidence=float(min_confidence),
                    min_observations=int(min_observations),
                    depth_agreement_m=float(depth_agreement_m),
                    fusion_level="intra_capture",
                    fusion_mode="inferred_capture_event_from_raw_burst",
                )
            capture_events.append(capture)
            if len(capture_events) >= int(count):
                break

    capture_events.sort(key=lambda snapshot: int(snapshot.timestamp_us))
    if len(capture_events) < int(count):
        raise RuntimeError(f"found {len(capture_events)} usable capture events for {camera_id}, expected {count} under {depth_base}")
    return capture_events[-int(count):]


def _camera_calibrations(pipeline_config: Path, cameras_config: Path) -> dict[str, CameraCalibration]:
    pipeline = _read_yaml(pipeline_config)
    labels = _load_camera_labels(cameras_config)
    provider = _CalibrationProvider(cameras_config, pipeline)
    provider.set_camera_labels(labels)
    out: dict[str, CameraCalibration] = {}
    for source_id, camera_id in labels.items():
        snapshot = provider.snapshot(int(source_id), str(camera_id))
        if snapshot is None:
            continue
        world_to_camera = np.asarray(snapshot.extrinsics_col_major, dtype=np.float64).reshape((4, 4), order="F")
        camera_to_world = np.linalg.inv(world_to_camera)
        out[str(camera_id)] = CameraCalibration(
            source_id=int(source_id),
            camera_id=str(camera_id),
            snapshot=snapshot,
            world_to_camera=world_to_camera,
            camera_to_world=camera_to_world,
            intrinsics=np.asarray(snapshot.intrinsics, dtype=np.float64).reshape(3, 3),
            floor_y=float(snapshot.floor_y),
        )
    return out


def _load_scene_similarity(pipeline_config: Path, cameras_config: Path, alignment_config: Path) -> dict[str, Any]:
    try:
        provider = _CalibrationProvider(cameras_config, _read_yaml(pipeline_config))
        bundle = provider.calibration_bundle()
        sim = ((bundle.get("align") or {}).get("scene_similarity") or {})
        if isinstance(sim, dict) and isinstance(sim.get("world_to_scene_col_major"), list):
            return dict(sim)
    except Exception:
        pass
    try:
        payload = json.loads(Path(alignment_config).read_text(encoding="utf-8"))
    except Exception:
        return {}
    sim = payload.get("scene_similarity")
    if sim is None and isinstance(payload.get("align"), dict):
        sim = payload["align"].get("scene_similarity")
    return dict(sim) if isinstance(sim, dict) else {}


def _relative_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _valid_depth_mask(snapshot: DepthSnapshot, min_confidence: float, depth_clip_m: float | None = None) -> np.ndarray:
    valid = (
        np.asarray(snapshot.mask, dtype=bool)
        & np.isfinite(snapshot.depth)
        & (snapshot.depth > 0.0)
        & np.isfinite(snapshot.confidence)
        & (snapshot.confidence >= float(min_confidence))
    )
    if depth_clip_m is not None and np.isfinite(depth_clip_m):
        valid &= snapshot.depth <= float(depth_clip_m)
    return valid


def _depth_clip(snapshot: DepthSnapshot, min_confidence: float, percentile: float, max_depth_m: float) -> float:
    valid = _valid_depth_mask(snapshot, min_confidence)
    values = snapshot.depth[valid]
    if values.size == 0:
        return float(max_depth_m)
    pct = float(np.percentile(values, float(percentile)))
    return float(min(float(max_depth_m), max(0.5, pct + 0.25)))


def _build_frame_clouds(
    *,
    camera: CameraCalibration,
    snapshots: Sequence[DepthSnapshot],
    rgb_frames: Sequence[RgbFrame],
    min_confidence: float,
    pixel_step: int,
    depth_clip_percentile: float,
    max_depth_m: float,
) -> list[FrameCloud]:
    frames: list[FrameCloud] = []
    for index, snapshot in enumerate(snapshots, start=1):
        if index > len(rgb_frames):
            raise RuntimeError(f"{camera.camera_id}: missing RGB frame for depth snapshot {index}")
        rgb_frame = rgb_frames[index - 1]
        image_bgr = np.asarray(rgb_frame.image_bgr, dtype=np.uint8)
        if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
            raise RuntimeError(f"{camera.camera_id}: RGB frame {index} is not a color image: {image_bgr.shape}")
        if image_bgr.shape[:2] != snapshot.depth.shape:
            image_bgr = cv2.resize(
                image_bgr[:, :, :3],
                (int(snapshot.depth.shape[1]), int(snapshot.depth.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )
        frame_id = f"{camera.camera_id}_{index:04d}"
        depth_clip_m = _depth_clip(snapshot, min_confidence, depth_clip_percentile, max_depth_m)
        valid = _valid_depth_mask(snapshot, min_confidence, depth_clip_m)
        points_camera, pixels = backproject_depth(
            snapshot.depth,
            camera.intrinsics,
            mask=valid,
            confidence=None,
            pixel_step=max(1, int(pixel_step)),
        )
        points_world = transform_points(points_camera, camera.camera_to_world)
        confidence = snapshot.confidence[pixels[:, 1], pixels[:, 0]] if pixels.size else np.zeros((0,), dtype=np.float32)
        frames.append(
            FrameCloud(
                frame_id=frame_id,
                snapshot=snapshot,
                depth_clip_m=depth_clip_m,
                points_camera=points_camera,
                points_world=points_world,
                pixels=pixels,
                confidence=np.asarray(confidence, dtype=np.float32),
                image_bgr=image_bgr[:, :, :3],
                image_ref=str(rgb_frame.source_uri),
            )
        )
    return frames


def _frame_rgb_colors(frame: FrameCloud) -> np.ndarray:
    pix = np.asarray(frame.pixels, dtype=np.int32).reshape((-1, 2))
    if pix.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    image = np.asarray(frame.image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"{frame.frame_id}: missing RGB image for point color sampling")
    h, w = image.shape[:2]
    x = np.clip(pix[:, 0], 0, w - 1)
    y = np.clip(pix[:, 1], 0, h - 1)
    bgr = image[y, x, :3]
    return bgr[:, ::-1].astype(np.uint8, copy=False)


def _bounded_stride(point_count: int, budget: int) -> int:
    return int(math.ceil(point_count / float(max(1, budget)))) if point_count > budget else 1


def _estimate_observed_floor_y(frames: Sequence[FrameCloud], calibrated_floor_y: float) -> tuple[float, dict[str, Any]]:
    samples = []
    for frame in frames:
        points = np.asarray(frame.points_world, dtype=np.float32).reshape((-1, 3))
        if points.shape[0] == 0:
            continue
        samples.append(points[::_bounded_stride(points.shape[0], 60000), 1].astype(np.float64))
    if not samples:
        return float(calibrated_floor_y), {"source": "calibrated_floor_y", "reason": "no_points"}
    y = np.concatenate(samples, axis=0)
    y = y[np.isfinite(y)]
    if y.size < 128:
        return float(calibrated_floor_y), {"source": "calibrated_floor_y", "reason": "too_few_points"}

    near_calibrated = float(np.count_nonzero(np.abs(y - float(calibrated_floor_y)) <= 0.35) / max(1, y.size))
    low_p01 = float(np.percentile(y, 1.0))
    low_p03 = float(np.percentile(y, 3.0))
    low_p08 = float(np.percentile(y, 8.0))
    observed = float(np.median([low_p01, low_p03, low_p08]))
    use_observed = near_calibrated < 0.018 and abs(observed - float(calibrated_floor_y)) > 0.38
    selected = observed if use_observed else float(calibrated_floor_y)
    return selected, {
        "source": "observed_depth_low_height_quantile" if use_observed else "calibrated_floor_y",
        "calibrated_floor_y": float(calibrated_floor_y),
        "selected_floor_y": float(selected),
        "near_calibrated_fraction": near_calibrated,
        "observed_low_p01": low_p01,
        "observed_low_p03": low_p03,
        "observed_low_p08": low_p08,
    }


def _clip_frame_clouds_to_ceiling(
    frames: Sequence[FrameCloud],
    *,
    floor_y: float,
    enabled: bool,
    percentile: float,
    min_height_m: float,
) -> tuple[list[FrameCloud], dict[str, Any]]:
    rows = []
    total_source = 0
    total_kept = 0
    if not enabled:
        return list(frames), {"status": "disabled"}

    y_samples: list[np.ndarray] = []
    for frame in frames:
        points = np.asarray(frame.points_world, dtype=np.float32).reshape((-1, 3))
        total_source += int(points.shape[0])
        if points.shape[0] <= 0:
            continue
        y = points[:, 1]
        y = y[np.isfinite(y)]
        if y.size:
            y_samples.append(y.astype(np.float64, copy=False))
    if not y_samples:
        return list(frames), {
            "status": "skipped",
            "reason": "no_points",
            "source_point_count": int(total_source),
        }

    y_all = np.concatenate(y_samples, axis=0)
    pct = float(np.clip(float(percentile), 50.0, 99.95))
    percentile_y = float(np.percentile(y_all, pct))
    min_threshold_y = float(floor_y) + max(0.0, float(min_height_m))
    threshold_y = max(percentile_y, min_threshold_y)

    clipped: list[FrameCloud] = []
    for frame in frames:
        points_world = np.asarray(frame.points_world, dtype=np.float32).reshape((-1, 3))
        if points_world.shape[0] <= 0:
            clipped.append(frame)
            rows.append({"frame_id": frame.frame_id, "source_point_count": 0, "kept_point_count": 0, "removed_point_count": 0})
            continue
        keep = np.isfinite(points_world[:, 1]) & (points_world[:, 1] <= threshold_y)
        kept = int(np.count_nonzero(keep))
        total_kept += kept
        rows.append(
            {
                "frame_id": frame.frame_id,
                "source_point_count": int(points_world.shape[0]),
                "kept_point_count": kept,
                "removed_point_count": int(points_world.shape[0] - kept),
            }
        )
        clipped.append(
            FrameCloud(
                frame_id=frame.frame_id,
                snapshot=frame.snapshot,
                depth_clip_m=frame.depth_clip_m,
                points_camera=frame.points_camera[keep],
                points_world=points_world[keep],
                pixels=frame.pixels[keep],
                confidence=frame.confidence[keep],
                image_bgr=frame.image_bgr,
                image_ref=frame.image_ref,
            )
        )

    removed = int(total_source - total_kept)
    return clipped, {
        "status": "applied",
        "source": "top_world_y_percentile_with_min_height_guard",
        "axis": "backend_world_m_y",
        "threshold_y": float(threshold_y),
        "percentile": pct,
        "percentile_y": percentile_y,
        "floor_y": float(floor_y),
        "min_height_m": float(min_height_m),
        "min_threshold_y": float(min_threshold_y),
        "source_point_count": int(total_source),
        "kept_point_count": int(total_kept),
        "removed_point_count": removed,
        "removed_fraction": float(removed / max(1, total_source)),
        "per_frame": rows,
    }


def _triangle_is_continuous(points_camera: np.ndarray, depths: np.ndarray, *, max_edge_m: float, max_depth_delta_m: float) -> bool:
    tri = np.asarray(points_camera, dtype=np.float32).reshape((3, 3))
    z = np.asarray(depths, dtype=np.float32).reshape((3,))
    if not np.all(np.isfinite(tri)) or not np.all(np.isfinite(z)) or np.any(z <= 0.0):
        return False
    median_depth = float(np.median(z))
    edge_limit = max(float(max_edge_m), 0.18 + median_depth * 0.08)
    depth_limit = max(float(max_depth_delta_m), 0.10 + median_depth * 0.05)
    if float(np.max(z) - np.min(z)) > depth_limit:
        return False
    edge01 = float(np.linalg.norm(tri[0] - tri[1]))
    edge12 = float(np.linalg.norm(tri[1] - tri[2]))
    edge20 = float(np.linalg.norm(tri[2] - tri[0]))
    return max(edge01, edge12, edge20) <= edge_limit


def _frame_mesh_arrays(
    frame: FrameCloud,
    camera: CameraCalibration,
    *,
    min_confidence: float,
    mesh_pixel_step: int,
    max_edge_m: float,
    max_depth_delta_m: float,
    ceiling_clip_y: float | None,
    atlas_cols: int,
    atlas_rows: int,
    atlas_col: int,
    atlas_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    step = max(1, int(mesh_pixel_step))
    valid = _valid_depth_mask(frame.snapshot, min_confidence, frame.depth_clip_m)
    points_camera, pixels = backproject_depth(
        frame.snapshot.depth,
        camera.intrinsics,
        mask=valid,
        confidence=None,
        pixel_step=step,
    )
    if points_camera.shape[0] < 3:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.uint32),
            {"frame_id": frame.frame_id, "vertex_count": 0, "triangle_count": 0, "rejected_triangle_count": 0},
        )
    points_world = transform_points(points_camera, camera.camera_to_world)
    removed_by_ceiling = 0
    if ceiling_clip_y is not None and np.isfinite(float(ceiling_clip_y)):
        keep = np.isfinite(points_world[:, 1]) & (points_world[:, 1] <= float(ceiling_clip_y))
        removed_by_ceiling = int(points_world.shape[0] - np.count_nonzero(keep))
        points_camera = points_camera[keep]
        points_world = points_world[keep]
        pixels = pixels[keep]
    if points_camera.shape[0] < 3:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.uint32),
            {
                "frame_id": frame.frame_id,
                "vertex_count": 0,
                "triangle_count": 0,
                "rejected_triangle_count": 0,
                "ceiling_clip_removed_vertex_count": int(removed_by_ceiling),
            },
        )
    height, width = frame.snapshot.depth.shape
    atlas_cols = max(1, int(atlas_cols))
    atlas_rows = max(1, int(atlas_rows))
    texcoords = np.stack(
        [
            (float(atlas_col) + (pixels[:, 0].astype(np.float32) + 0.5) / max(1.0, float(width))) / float(atlas_cols),
            (float(atlas_row) + (pixels[:, 1].astype(np.float32) + 0.5) / max(1.0, float(height))) / float(atlas_rows),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    index_grid = np.full((height, width), -1, dtype=np.int32)
    index_grid[pixels[:, 1], pixels[:, 0]] = np.arange(pixels.shape[0], dtype=np.int32)

    triangles: list[tuple[int, int, int]] = []
    rejected = 0
    for y in range(0, max(0, height - step), step):
        y1 = y + step
        if y1 >= height:
            continue
        for x in range(0, max(0, width - step), step):
            x1 = x + step
            if x1 >= width:
                continue
            a = int(index_grid[y, x])
            b = int(index_grid[y, x1])
            c = int(index_grid[y1, x])
            d = int(index_grid[y1, x1])
            if a >= 0 and c >= 0 and b >= 0:
                idx = np.asarray([a, c, b], dtype=np.int32)
                if _triangle_is_continuous(points_camera[idx], points_camera[idx, 2], max_edge_m=max_edge_m, max_depth_delta_m=max_depth_delta_m):
                    triangles.append((a, c, b))
                else:
                    rejected += 1
            if b >= 0 and c >= 0 and d >= 0:
                idx = np.asarray([b, c, d], dtype=np.int32)
                if _triangle_is_continuous(points_camera[idx], points_camera[idx, 2], max_edge_m=max_edge_m, max_depth_delta_m=max_depth_delta_m):
                    triangles.append((b, c, d))
                else:
                    rejected += 1

    indices = np.asarray(triangles, dtype=np.uint32).reshape((-1,)) if triangles else np.zeros((0,), dtype=np.uint32)
    return points_world, texcoords, indices, {
        "frame_id": frame.frame_id,
        "vertex_count": int(points_world.shape[0]),
        "triangle_count": int(indices.shape[0] // 3),
        "rejected_triangle_count": int(rejected),
        "ceiling_clip_removed_vertex_count": int(removed_by_ceiling),
        "mesh_pixel_step": int(step),
        "texture_tile": {
            "col": int(atlas_col),
            "row": int(atlas_row),
            "atlas_cols": int(atlas_cols),
            "atlas_rows": int(atlas_rows),
        },
    }


def _build_texture_atlas(frames: Sequence[FrameCloud]) -> tuple[np.ndarray, dict[str, Any]]:
    if not frames:
        return np.zeros((1, 1, 3), dtype=np.uint8), {"frame_count": 0, "cols": 1, "rows": 1, "width": 1, "height": 1}
    tile_h, tile_w = frames[0].image_bgr.shape[:2]
    if tile_w <= 0 or tile_h <= 0:
        raise RuntimeError("cannot build mesh texture atlas from empty RGB keyframe")
    count = len(frames)
    cols = int(math.ceil(math.sqrt(float(count))))
    rows = int(math.ceil(count / float(max(1, cols))))
    atlas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    tiles: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        row = index // cols
        col = index % cols
        image_bgr = np.asarray(frame.image_bgr, dtype=np.uint8)
        if image_bgr.shape[:2] != (tile_h, tile_w):
            image_bgr = cv2.resize(image_bgr[:, :, :3], (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        image_rgb = image_bgr[:, :, :3][:, :, ::-1]
        y0 = row * tile_h
        x0 = col * tile_w
        atlas[y0 : y0 + tile_h, x0 : x0 + tile_w] = image_rgb
        tiles.append({"frame_id": frame.frame_id, "col": int(col), "row": int(row), "image_size": [int(tile_w), int(tile_h)]})
    return atlas, {
        "frame_count": int(count),
        "cols": int(cols),
        "rows": int(rows),
        "tile_width": int(tile_w),
        "tile_height": int(tile_h),
        "width": int(atlas.shape[1]),
        "height": int(atlas.shape[0]),
        "tiles": tiles,
    }


def _build_room_mesh(
    frames: Sequence[FrameCloud],
    camera: CameraCalibration,
    *,
    min_confidence: float,
    mesh_pixel_step: int,
    max_edge_m: float,
    max_depth_delta_m: float,
    ceiling_clip_y: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    vertices_rows: list[np.ndarray] = []
    texcoord_rows: list[np.ndarray] = []
    index_rows: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    offset = 0
    texture_atlas, texture_meta = _build_texture_atlas(frames)
    atlas_cols = int(texture_meta.get("cols") or 1)
    atlas_rows = int(texture_meta.get("rows") or 1)
    for frame_index, frame in enumerate(frames):
        vertices, texcoords, indices, metrics = _frame_mesh_arrays(
            frame,
            camera,
            min_confidence=min_confidence,
            mesh_pixel_step=mesh_pixel_step,
            max_edge_m=max_edge_m,
            max_depth_delta_m=max_depth_delta_m,
            ceiling_clip_y=ceiling_clip_y,
            atlas_cols=atlas_cols,
            atlas_rows=atlas_rows,
            atlas_col=frame_index % atlas_cols,
            atlas_row=frame_index // atlas_cols,
        )
        per_frame.append(metrics)
        if vertices.shape[0] <= 0 or indices.shape[0] <= 0:
            continue
        vertices_rows.append(vertices.astype(np.float32, copy=False))
        texcoord_rows.append(texcoords.astype(np.float32, copy=False))
        index_rows.append((indices + np.uint32(offset)).astype(np.uint32, copy=False))
        offset += int(vertices.shape[0])
    vertices_all = np.concatenate(vertices_rows, axis=0) if vertices_rows else np.zeros((0, 3), dtype=np.float32)
    texcoords_all = np.concatenate(texcoord_rows, axis=0) if texcoord_rows else np.zeros((0, 2), dtype=np.float32)
    indices_all = np.concatenate(index_rows, axis=0) if index_rows else np.zeros((0,), dtype=np.uint32)
    return vertices_all, texcoords_all, indices_all, texture_atlas, {
        "schema": "noesis.room_reconstruction.stream_rgb_textured_mesh.v1",
        "source": "mapanything_depth_grid_triangulated_with_rgb_texture_atlas",
        "mesh_pixel_step": int(max(1, mesh_pixel_step)),
        "max_edge_m": float(max_edge_m),
        "max_depth_delta_m": float(max_depth_delta_m),
        "ceiling_clip_y": float(ceiling_clip_y) if ceiling_clip_y is not None and np.isfinite(float(ceiling_clip_y)) else None,
        "vertex_count": int(vertices_all.shape[0]),
        "triangle_count": int(indices_all.shape[0] // 3),
        "texture": texture_meta,
        "per_frame": per_frame,
    }


def _frame_color_source(frames: Sequence[FrameCloud]) -> str:
    refs = [str(frame.image_ref or "") for frame in frames]
    if refs and all(ref.startswith("mapanything_zarr_rgb:") for ref in refs):
        return ZARR_RGB_COLOR_SOURCE
    if refs and all(ref.startswith("mapanything_zarr:") for ref in refs):
        return CACHED_DEPTH_COLOR_SOURCE
    return "dewarped_live_camera_rgb_texture_atlas"


def _mesh_source_for_color_source(color_source: str) -> str:
    if color_source == ZARR_RGB_COLOR_SOURCE:
        return "mapanything_depth_grid_triangulated_with_persisted_zarr_rgb_texture_atlas"
    if color_source == CACHED_DEPTH_COLOR_SOURCE:
        return "mapanything_depth_grid_triangulated_with_cached_depth_texture_atlas"
    return "mapanything_depth_grid_triangulated_with_live_rgb_texture_atlas"


def _keyframe_source_for_color_source(color_source: str) -> str:
    if color_source == ZARR_RGB_COLOR_SOURCE:
        return ZARR_RGB_COLOR_SOURCE
    if color_source == CACHED_DEPTH_COLOR_SOURCE:
        return CACHED_DEPTH_COLOR_SOURCE
    return LIVE_RGB_COLOR_SOURCE


def _write_revision(
    *,
    store: VirtualTwinStore,
    revision_id: str,
    camera: CameraCalibration,
    frames: Sequence[FrameCloud],
    points_budget: int,
    min_confidence: float,
    mesh_pixel_step: int,
    mesh_max_edge_m: float,
    mesh_max_depth_delta_m: float,
    ceiling_clip: dict[str, Any],
    observed_floor_y: float,
    floor_estimate: dict[str, Any],
    scene_similarity: dict[str, Any],
    update_latest: bool,
) -> dict[str, Any]:
    revision_dir = store.revision_dir(revision_id)
    revision_dir.mkdir(parents=True, exist_ok=True)
    all_points = np.concatenate([frame.points_world for frame in frames], axis=0) if frames else np.zeros((0, 3), dtype=np.float32)
    all_conf = np.concatenate([frame.confidence for frame in frames], axis=0) if frames else np.zeros((0,), dtype=np.float32)
    all_colors = np.concatenate([_frame_rgb_colors(frame) for frame in frames], axis=0) if frames else np.zeros((0, 3), dtype=np.uint8)
    if all_points.shape[0] == 0:
        raise RuntimeError(f"{camera.camera_id}: no points generated")
    if all_colors.shape[0] != all_points.shape[0]:
        raise RuntimeError(f"{camera.camera_id}: RGB color count does not match point count")
    stride = _bounded_stride(all_points.shape[0], points_budget)
    served_points = all_points[::stride]
    served_colors = all_colors[::stride]
    write_points_glb(revision_dir / "room_points.glb", served_points, served_colors)
    write_points_npz(
        revision_dir / "room_points.npz",
        all_points,
        all_colors,
        confidence=all_conf.astype(np.float32),
    )
    ceiling_clip_y_raw = ceiling_clip.get("threshold_y") if isinstance(ceiling_clip, dict) else None
    ceiling_clip_y = float(ceiling_clip_y_raw) if ceiling_clip_y_raw is not None and np.isfinite(float(ceiling_clip_y_raw)) else None
    mesh_vertices, mesh_texcoords, mesh_indices, mesh_texture, mesh_meta = _build_room_mesh(
        frames,
        camera,
        min_confidence=float(min_confidence),
        mesh_pixel_step=int(mesh_pixel_step),
        max_edge_m=float(mesh_max_edge_m),
        max_depth_delta_m=float(mesh_max_depth_delta_m),
        ceiling_clip_y=ceiling_clip_y,
    )
    if mesh_vertices.shape[0] > 0 and mesh_indices.shape[0] > 0:
        write_textured_mesh_glb(revision_dir / "room_mesh.glb", mesh_vertices, mesh_indices, mesh_texcoords, mesh_texture)
    mesh_meta = {
        **mesh_meta,
        "revision_id": revision_id,
        "camera": camera.camera_id,
        "coordinate_frame": "backend_world_m_stream_mesh",
        "generated_ts_us": int(time.time() * 1_000_000),
        "color_source": _frame_color_source(frames),
        "color_space": "sRGB",
        "ceiling_clip": ceiling_clip,
    }
    color_source = str(mesh_meta.get("color_source") or _frame_color_source(frames))
    mesh_source = _mesh_source_for_color_source(color_source)
    keyframe_source = _keyframe_source_for_color_source(color_source)
    mesh_meta["source"] = mesh_source
    write_json(revision_dir / "room_mesh_meta.json", mesh_meta)
    keyframes_dir = revision_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    keyframe_refs: dict[str, str] = {}
    for frame in frames:
        rel = Path("keyframes") / f"{frame.frame_id}.png"
        if not cv2.imwrite(str(revision_dir / rel), np.asarray(frame.image_bgr, dtype=np.uint8)):
            raise RuntimeError(f"{camera.camera_id}: failed to write RGB keyframe {rel}")
        keyframe_refs[frame.frame_id] = str(rel)

    bounds_min = np.min(all_points, axis=0)
    bounds_max = np.max(all_points, axis=0)
    source_snapshots = []
    for frame in frames:
        p50 = _snapshot_depth_percentile(frame.snapshot, 0.0, frame.depth_clip_m, 50.0)
        p95 = _snapshot_depth_percentile(frame.snapshot, 0.0, frame.depth_clip_m, 95.0)
        source_paths = _snapshot_source_paths(frame.snapshot)
        source_timestamps = _snapshot_source_timestamps(frame.snapshot)
        row = {
            "zarr": _relative_path(frame.snapshot.path),
            "timestamp_us": int(frame.snapshot.timestamp_us),
            "source_zarrs": [_relative_path(path) for path in source_paths],
            "source_timestamps_us": [int(ts) for ts in source_timestamps],
            "valid_fraction": _valid_fraction(frame.snapshot, 0.0),
            "depth_clip_m": float(frame.depth_clip_m),
            "depth_p50_m": p50,
            "depth_p95_m": p95,
            "point_count": int(frame.points_world.shape[0]),
        }
        if frame.snapshot.snapshot_role:
            row["snapshot_role"] = frame.snapshot.snapshot_role
        if frame.snapshot.fusion_level:
            row["fusion_level"] = frame.snapshot.fusion_level
        if frame.snapshot.fusion_meta is not None:
            row["fusion"] = frame.snapshot.fusion_meta
        source_snapshots.append(row)
    room_meta = {
        "schema": "noesis.room_reconstruction.stream_points.v4",
        "revision_id": revision_id,
        "camera": camera.camera_id,
        "source": "mapanything_zarr_current_calibration_backprojection_texture_colored",
        "coordinate_frame": "backend_world_m_stream_points",
        "generated_ts_us": int(time.time() * 1_000_000),
        "source_frame_count": len(frames),
        "source_snapshots": source_snapshots,
        "rgb_keyframes": keyframe_refs,
        "intrinsics": camera.intrinsics.tolist(),
        "extrinsics_col_major": [float(x) for x in camera.snapshot.extrinsics_col_major],
        "floor_y": float(observed_floor_y),
        "calibrated_floor_y": float(camera.floor_y),
        "floor_estimate": floor_estimate,
        "point_count": int(served_points.shape[0]),
        "source_point_count": int(all_points.shape[0]),
        "source_sample_stride": int(stride),
        "ceiling_clip": ceiling_clip,
        "bounds": {
            "min": [float(x) for x in bounds_min],
            "max": [float(x) for x in bounds_max],
        },
        "color_source": color_source,
        "color_space": "sRGB",
        "calibration_source": "config/cameras.yaml + config/camera_calibration.json via _CalibrationProvider",
    }
    write_json(revision_dir / "room_points_meta.json", room_meta)

    frame_rows = [
        {
            "frame_id": frame.frame_id,
            "camera_id": camera.camera_id,
            "source_ref": _relative_path(frame.snapshot.path),
            "source_refs": [_relative_path(path) for path in _snapshot_source_paths(frame.snapshot)],
            "revision_artifacts": {
                "calibration": {
                    "camera_id": camera.camera_id,
                    "intrinsics": camera.intrinsics.tolist(),
                    "extrinsics_col_major": [float(x) for x in camera.snapshot.extrinsics_col_major],
                    "floor_y": float(camera.floor_y),
                    "image_size": [int(camera.snapshot.image_size[0]), int(camera.snapshot.image_size[1])],
                    "unit_scale": float(camera.snapshot.unit_scale),
                },
                "mapanything_zarr": _relative_path(frame.snapshot.path),
                "mapanything_zarrs": [_relative_path(path) for path in _snapshot_source_paths(frame.snapshot)],
                "rgb_keyframe": keyframe_refs.get(frame.frame_id),
                "texture_source_ref": frame.image_ref,
            },
            "image_size": [int(frame.snapshot.depth.shape[1]), int(frame.snapshot.depth.shape[0])],
            "mapanything_valid_depth_coverage": _valid_fraction(frame.snapshot, 0.0),
            "surfel_count": int(frame.points_world.shape[0]),
        }
        for frame in frames
    ]
    artifacts = {
        "manifest": "manifest.json",
        "metrics": "metrics.json",
        "room_mesh_glb": "room_mesh.glb",
        "room_mesh_meta": "room_mesh_meta.json",
        "room_points_glb": "room_points.glb",
        "room_points_meta": "room_points_meta.json",
        "room_points_npz": "room_points.npz",
        "tracking_alignment": "tracking_alignment.json",
    }
    if mesh_vertices.shape[0] <= 0 or mesh_indices.shape[0] <= 0:
        artifacts.pop("room_mesh_glb", None)
    manifest = {
        "schema": "noesis.virtual_twin.revision.v1",
        "revision_id": revision_id,
        "camera": camera.camera_id,
        "created_ts_us": int(time.time() * 1_000_000),
        "label": revision_id,
        "source_frame_ids": [frame.frame_id for frame in frames],
        "calibration_fingerprints": {camera.camera_id: calibration_fingerprint(camera.snapshot)},
        "model_fingerprints": {
            "mapanything": {
                "source": "ds8_persisted_zarr_latest",
                "snapshot_paths": sorted(
                    {
                        _relative_path(path)
                        for frame in frames
                        for path in _snapshot_source_paths(frame.snapshot)
                    }
                ),
                "min_confidence": None,
            },
            "rgb_keyframes": {
                "source": keyframe_source,
                "keyframe_paths": list(keyframe_refs.values()),
            },
            "rgb_mesh": {
                "source": mesh_source,
                "mesh_pixel_step": int(mesh_pixel_step),
                "max_edge_m": float(mesh_max_edge_m),
                "max_depth_delta_m": float(mesh_max_depth_delta_m),
                "ceiling_clip": ceiling_clip,
                "texture": mesh_meta.get("texture"),
                "vertex_count": int(mesh_meta.get("vertex_count") or 0),
                "triangle_count": int(mesh_meta.get("triangle_count") or 0),
            },
            "scene_prior": {
                "source": "calibration_scene_similarity",
                "scene_similarity": scene_similarity,
            },
        },
        "source_file_refs": {
            "frames": frame_rows,
        },
        "coordinate_frames": {
            "room_mesh_glb": "backend_world_m_stream_mesh",
            "room_mesh_meta": "backend_world_m_stream_mesh",
            "room_points_glb": "backend_world_m_stream_points",
            "room_points_meta": "backend_world_m_stream_points",
            "room_points_npz": "backend_world_m_stream_points",
            "tracking_alignment": "menon_scene_units_from_camera_scene_prior",
        },
        "artifacts": artifacts,
    }
    tracking_alignment = {
        "revision_id": revision_id,
        "camera": camera.camera_id,
        "pose_correction": {
            "status": "reference_only",
            "method": "calibration_scene_similarity_for_menon_render",
            "world_to_menon_scene_col_major": scene_similarity.get("world_to_scene_col_major"),
            "scene_per_m": scene_similarity.get("scene_per_m") or scene_similarity.get("scale"),
        },
        "floor_plane": {
            "frame": "backend_world_m",
            "normal": [0.0, 1.0, 0.0],
            "offset": -float(observed_floor_y),
            "floor_y": float(observed_floor_y),
            "calibrated_floor_y": float(camera.floor_y),
            "floor_estimate": floor_estimate,
        },
    }
    coverage = [row["mapanything_valid_depth_coverage"] for row in frame_rows]
    metrics = {
        "revision_id": revision_id,
        "camera": camera.camera_id,
        "frame_count": len(frames),
        "accepted_plane_count": 0,
        "floor_plane_count": 0,
        "wall_plane_count": 0,
        "mapanything_valid_depth_coverage_min": float(np.min(coverage)) if coverage else 0.0,
        "mapanything_valid_depth_coverage_median": float(np.median(coverage)) if coverage else 0.0,
        "browser_render_budget": {
            "artifact_type": "stream_depth_textured_room_mesh_and_point_cloud",
            "point_budget": int(points_budget),
            "source_point_count": int(all_points.shape[0]),
            "served_glb_point_count": int(served_points.shape[0]),
            "source_sample_stride": int(stride),
            "mesh_vertex_count": int(mesh_meta.get("vertex_count") or 0),
            "mesh_triangle_count": int(mesh_meta.get("triangle_count") or 0),
            "mesh_pixel_step": int(mesh_pixel_step),
        },
        "room_points": room_meta,
        "room_mesh": mesh_meta,
        "ceiling_clip": ceiling_clip,
        "floor_estimate": floor_estimate,
        "plane_extraction": {
            "status": "removed",
            "reason": "rgb_depth_cloud_view_no_2d_planes",
        },
        "gates": {
            "stream_points_present": bool(served_points.shape[0] > 0),
            "stream_mesh_present": bool(mesh_vertices.shape[0] > 0 and mesh_indices.shape[0] > 0),
            "ceiling_clip_conservative": bool(float(ceiling_clip.get("removed_fraction") or 0.0) <= 0.025),
            "texture_color_source_present": bool(all_colors.shape[0] == all_points.shape[0]),
            "model_geometry_used_for_rendering": False,
        },
    }
    write_json(revision_dir / "manifest.json", manifest)
    write_json(revision_dir / "metrics.json", metrics)
    write_json(revision_dir / "tracking_alignment.json", tracking_alignment)
    if update_latest:
        store.write_latest_revision_id(revision_id)
    return {
        "revision_id": revision_id,
        "revision_dir": str(revision_dir),
        "camera": camera.camera_id,
        "point_count": int(served_points.shape[0]),
        "source_point_count": int(all_points.shape[0]),
        "mesh_vertex_count": int(mesh_meta.get("vertex_count") or 0),
        "mesh_triangle_count": int(mesh_meta.get("triangle_count") or 0),
        "ceiling_clip": ceiling_clip,
        "plane_count": 0,
        "floor_plane_count": 0,
        "wall_plane_count": 0,
        "floor_estimate": floor_estimate,
        "color_source": mesh_meta.get("color_source") or CACHED_DEPTH_COLOR_SOURCE,
        "mesh_color_source": mesh_meta.get("color_source") or CACHED_DEPTH_COLOR_SOURCE,
        "point_color_source": room_meta.get("color_source") or CACHED_DEPTH_COLOR_SOURCE,
        "source_frame_count": int(len(frames)),
        "snapshots": sorted(
            {
                _relative_path(path)
                for frame in frames
                for path in _snapshot_source_paths(frame.snapshot)
            }
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    service_config = load_service_config()
    depth_base = _resolve_repo_path(args.mapanything_depth_base, service_config.storage.depth_base)
    cameras_config = _resolve_repo_path(args.cameras_config, REPO_ROOT / "config" / "cameras.yaml")
    pipeline_config = _resolve_repo_path(args.pipeline_config, REPO_ROOT / "config" / "infer.yaml")
    alignment_config = _resolve_repo_path(args.alignment_config, REPO_ROOT / "config" / "ply_alignment.json")
    pipeline = _read_yaml(pipeline_config)
    camera_labels = _load_camera_labels(cameras_config)
    calibrations = _camera_calibrations(pipeline_config, cameras_config)
    scene_similarity = _load_scene_similarity(pipeline_config, cameras_config, alignment_config)
    store = VirtualTwinStore(Path(args.output_root) if args.output_root else None)
    store.ensure_root()
    cameras = list(args.camera or [])
    if not cameras or "all" in cameras:
        cameras = list(DEFAULT_CAMERAS)
    results = []
    capture_count = max(1, int(args.capture_count if args.capture_count is not None else args.snapshots))
    inter_capture_fusion_enabled = not (bool(args.no_inter_capture_fusion) or bool(args.no_fuse_snapshots))
    intra_capture_fused_any = False
    inter_capture_fused_any = False
    for camera_id in cameras:
        if camera_id not in calibrations:
            raise RuntimeError(f"camera {camera_id!r} has no calibration snapshot")
        camera = calibrations[camera_id]
        snapshots = _latest_capture_snapshots(
            depth_base,
            camera_id,
            capture_count,
            min_confidence=float(args.min_confidence),
            event_gap_s=float(args.capture_event_gap_s),
            max_raw_snapshots_per_capture=int(args.raw_snapshots_per_capture),
            min_observations=int(args.intra_capture_min_observations),
            depth_agreement_m=float(args.intra_capture_depth_agreement_m),
            prefer_persisted=not bool(args.ignore_persisted_capture_events),
        )
        if any(_is_capture_event_snapshot(snapshot) for snapshot in snapshots):
            intra_capture_fused_any = True
        if len(snapshots) > 1 and inter_capture_fusion_enabled:
            inter_capture_fused_any = True
            snapshots = [
                _fuse_depth_snapshots(
                    snapshots,
                    camera_id=camera_id,
                    min_confidence=float(args.min_confidence),
                    min_observations=int(args.fusion_min_observations),
                    depth_agreement_m=float(args.fusion_depth_agreement_m),
                    fusion_level="inter_capture",
                    fusion_mode="latest_capture_events_reconstruction_consensus",
                )
            ]
        rgb_source = str(args.rgb_source or "zarr-rgb").strip().lower()
        if rgb_source == "live":
            rgb_frames = _capture_rgb_frames(
                pipeline_cfg=pipeline,
                camera_labels=camera_labels,
                camera_id=camera_id,
                count=len(snapshots),
                frame_stride=int(args.rgb_frame_stride),
                max_frames_read=int(args.rgb_max_frames_read),
            )
        elif rgb_source == "cached-depth":
            rgb_frames = _cached_depth_texture_frames(
                snapshots,
                min_confidence=float(args.min_confidence),
                depth_clip_percentile=float(args.depth_clip_percentile),
                max_depth_m=float(args.max_depth_m),
            )
        elif rgb_source == "zarr-rgb":
            rgb_frames = _zarr_rgb_frames(snapshots)
        else:
            raise RuntimeError(f"unsupported RGB source mode: {args.rgb_source!r}")
        frames = _build_frame_clouds(
            camera=camera,
            snapshots=snapshots,
            rgb_frames=rgb_frames,
            min_confidence=float(args.min_confidence),
            pixel_step=int(args.pixel_step),
            depth_clip_percentile=float(args.depth_clip_percentile),
            max_depth_m=float(args.max_depth_m),
        )
        observed_floor_y, floor_estimate = _estimate_observed_floor_y(frames, camera.floor_y)
        frames, ceiling_clip = _clip_frame_clouds_to_ceiling(
            frames,
            floor_y=observed_floor_y,
            enabled=not bool(args.disable_ceiling_clip),
            percentile=float(args.ceiling_clip_percentile),
            min_height_m=float(args.ceiling_clip_min_height_m),
        )
        revision_id = args.revision_id
        if len(cameras) > 1 or not revision_id:
            mode_name = "stream_rgbmesh" if rgb_source in {"live", "zarr-rgb"} else "stream_depthmesh"
            prefix = f"vt_{camera_id.replace('-', '_')}_{mode_name}"
            revision_id = revision_id_from_clock(prefix)
        results.append(
            _write_revision(
                store=store,
                revision_id=revision_id,
                camera=camera,
                frames=frames,
                points_budget=int(args.points_budget),
                min_confidence=float(args.min_confidence),
                mesh_pixel_step=int(args.mesh_pixel_step),
                mesh_max_edge_m=float(args.mesh_max_edge_m),
                mesh_max_depth_delta_m=float(args.mesh_max_depth_delta_m),
                ceiling_clip=ceiling_clip,
                observed_floor_y=observed_floor_y,
                floor_estimate=floor_estimate,
                scene_similarity=scene_similarity,
                update_latest=bool(args.update_latest and camera_id == cameras[-1]),
            )
        )
    return {
        "schema": "noesis.stream_room_reconstruction.build_result.v1",
        "depth_base": str(depth_base),
        "rgb_source": str(args.rgb_source),
        "snapshot_count_per_camera": int(capture_count),
        "capture_count_per_camera": int(capture_count),
        "capture_event_gap_s": float(args.capture_event_gap_s),
        "raw_snapshots_per_capture": int(args.raw_snapshots_per_capture),
        "intra_capture_fusion_enabled": True,
        "intra_capture_fused": bool(intra_capture_fused_any),
        "inter_capture_fusion_enabled": bool(inter_capture_fusion_enabled),
        "inter_capture_fused": bool(inter_capture_fused_any),
        "fused_snapshots": bool(inter_capture_fused_any),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Menon room reconstruction artifacts directly from latest MapAnything depth snapshots.")
    parser.add_argument("--camera", action="append", choices=(*DEFAULT_CAMERAS, "all"), default=None)
    parser.add_argument("--pipeline-config", type=Path, default=REPO_ROOT / "config" / "infer.yaml")
    parser.add_argument("--cameras-config", type=Path, default=REPO_ROOT / "config" / "cameras.yaml")
    parser.add_argument("--alignment-config", type=Path, default=REPO_ROOT / "config" / "ply_alignment.json")
    parser.add_argument("--mapanything-depth-base", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--revision-id", default=None, help="Only valid for a single --camera build.")
    parser.add_argument("--snapshots", type=int, default=4, help="Backward-compatible alias for --capture-count.")
    parser.add_argument("--capture-count", type=int, default=None, help="Number of capture events to use per camera.")
    parser.add_argument("--capture-event-gap-s", type=float, default=10.0, help="Group raw snapshots into one capture event when gaps stay below this many seconds.")
    parser.add_argument("--raw-snapshots-per-capture", type=int, default=12, help="Maximum raw snapshots to fuse inside one inferred capture event.")
    parser.add_argument("--ignore-persisted-capture-events", action="store_true", help="Infer capture events from raw snapshots even when fused capture-event Zarrs exist.")
    parser.add_argument(
        "--rgb-source",
        choices=("zarr-rgb", "cached-depth", "live"),
        default="zarr-rgb",
        help="zarr-rgb reads the persisted RGB layer beside depth; cached-depth uses depth/confidence false color; live explicitly opens the configured camera URI.",
    )
    parser.add_argument("--no-fuse-snapshots", action="store_true", help="Backward-compatible alias for --no-inter-capture-fusion.")
    parser.add_argument("--no-inter-capture-fusion", action="store_true", help="Keep selected capture events as separate source frames.")
    parser.add_argument("--intra-capture-min-observations", type=int, default=2)
    parser.add_argument("--intra-capture-depth-agreement-m", type=float, default=0.18)
    parser.add_argument("--fusion-min-observations", type=int, default=2)
    parser.add_argument("--fusion-depth-agreement-m", type=float, default=0.18)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--pixel-step", type=int, default=5)
    parser.add_argument("--points-budget", type=int, default=360_000)
    parser.add_argument("--depth-clip-percentile", type=float, default=99.4)
    parser.add_argument("--max-depth-m", type=float, default=12.0)
    parser.add_argument("--mesh-pixel-step", type=int, default=5)
    parser.add_argument("--mesh-max-edge-m", type=float, default=0.34)
    parser.add_argument("--mesh-max-depth-delta-m", type=float, default=0.22)
    parser.add_argument("--ceiling-clip-percentile", type=float, default=98.8)
    parser.add_argument("--ceiling-clip-min-height-m", type=float, default=2.15)
    parser.add_argument("--disable-ceiling-clip", action="store_true")
    parser.add_argument("--rgb-frame-stride", type=int, default=12)
    parser.add_argument("--rgb-max-frames-read", type=int, default=240)
    parser.add_argument("--update-latest", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        result = build(parse_args())
    except Exception as exc:
        print(f"stream room reconstruction build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
