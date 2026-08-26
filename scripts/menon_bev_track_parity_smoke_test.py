#!/usr/bin/env python3
"""Validate camera-local BEV telemetry against exact tracking frames.

The inline BEV is intentionally in the active floorplan's camera-local ground
frame.  Canonical tracking remains in ``backend_world_m``.  This gate therefore
never subtracts those coordinates directly: it independently applies the
reviewed world-to-camera calibration, image-ray/floor intersection, or
registered-depth unprojection appropriate to each declared ``displaySource``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DS9.scripts import ds9_floorplan_live_gate as floorplan_gate  # noqa: E402

from scripts.internal_auth_client import (  # noqa: E402
    RequiredInternalAuth,
    add_auth_token_file_argument,
    configure_required_auth_environment,
    connect_required_websocket,
    load_required_internal_auth,
)

BEV_FRAME = "camera_local_ground_m"
BEV_FRAME_MODE = "camera_local"
BEV_UNITS = "meters"
TRACK_WORLD_FRAME = "backend_world_m"
ALLOWED_DISPLAY_SOURCES = frozenset(
    {
        "world_to_camera_local",
        "registered_depth_anchor",
        "floor_contact_ray",
        "image_depth_anchor",
        "image_anchor",
    }
)
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_CAPTURED_MESSAGES = 100_000


@dataclass(frozen=True)
class _FrameContract:
    frame: str
    frame_mode: str
    units: str


@dataclass(frozen=True)
class _CameraCalibration:
    source_id: int
    camera_id: str
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    floor_y: float
    image_size: Tuple[int, int]


def _read_mapping(path: Path, *, label: str, yaml_input: bool) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    raw = resolved.read_bytes()
    if not raw or len(raw) > MAX_CONFIG_BYTES:
        raise ValueError(f"{label} is empty or exceeds {MAX_CONFIG_BYTES} bytes")
    try:
        payload = yaml.safe_load(raw) if yaml_input else json.loads(raw)
    except Exception as exc:
        raise ValueError(f"unable to decode {label}: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain an object")
    return payload


def _load_frame_contract(pipeline_config: Path) -> _FrameContract:
    payload = _read_mapping(pipeline_config, label="pipeline config", yaml_input=True)
    bev = payload.get("bev")
    if not isinstance(bev, Mapping):
        raise ValueError("pipeline config is missing the reviewed BEV contract")
    frame = str(bev.get("frame") or "").strip()
    if frame != BEV_FRAME:
        raise ValueError(
            f"pipeline BEV frame must be {BEV_FRAME}; configured={frame or '<missing>'}"
        )
    return _FrameContract(frame=BEV_FRAME, frame_mode=BEV_FRAME_MODE, units=BEV_UNITS)


def _positive_dimension(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _finite(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _usable_registered_depth_m(track: Mapping[str, object]) -> Optional[float]:
    """Independently enforce the canonical registered-depth wire contract."""

    if str(track.get("depth_status") or "").strip().lower() != "ok":
        return None
    if str(track.get("depth_registration_status") or "").strip().lower() != "ok":
        return None
    registered = _finite(track.get("depth_registered_m"))
    if registered is None or registered <= 0.0:
        return None
    used_raw = track.get("depth_used_m")
    if used_raw is not None:
        used = _finite(used_raw)
        if (
            used is None
            or used <= 0.0
            or not math.isclose(
                used,
                registered,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            return None
    return float(registered)


def _load_calibrations(
    *,
    pipeline_config: Path,
    cameras_config: Path,
    calibration_config: Path,
    alignment_config: Path,
) -> Dict[str, _CameraCalibration]:
    pipeline = _read_mapping(pipeline_config, label="pipeline config", yaml_input=True)
    camera_doc = _read_mapping(cameras_config, label="cameras config", yaml_input=True)
    extrinsics_doc = _read_mapping(
        calibration_config, label="camera calibration", yaml_input=False
    )
    alignment_doc = _read_mapping(
        alignment_config, label="scene alignment", yaml_input=False
    )
    streammux = pipeline.get("streammux")
    if not isinstance(streammux, Mapping):
        raise ValueError("pipeline streammux contract is missing")
    target_w = _positive_dimension(streammux.get("width"), "streammux.width")
    target_h = _positive_dimension(streammux.get("height"), "streammux.height")
    camera_rows = camera_doc.get("cameras")
    models = camera_doc.get("intrinsics_models")
    extrinsics = extrinsics_doc.get("cameras")
    if not isinstance(camera_rows, Mapping) or not isinstance(models, Mapping):
        raise ValueError("cameras config lacks camera/model mappings")
    if not isinstance(extrinsics, Mapping):
        raise ValueError("camera calibration lacks a cameras mapping")
    floor_y = _finite(alignment_doc.get("floor_y"))
    if floor_y is None:
        raise ValueError("scene alignment floor_y must be finite")

    result: Dict[str, _CameraCalibration] = {}
    for raw_source_id, raw_camera in camera_rows.items():
        if not isinstance(raw_camera, Mapping):
            raise ValueError(f"camera row {raw_source_id!r} must be an object")
        source_id = int(raw_source_id)
        camera_id = str(raw_camera.get("name") or "").strip()
        model_id = str(raw_camera.get("model") or "").strip()
        model = models.get(model_id)
        if not camera_id or not isinstance(model, Mapping):
            raise ValueError(f"camera row {raw_source_id!r} lacks reviewed intrinsics")
        intrinsics = model.get("intrinsics")
        if not isinstance(intrinsics, Mapping):
            raise ValueError(f"camera model {model_id} lacks intrinsics")
        fx = _finite(intrinsics.get("fx"))
        fy = _finite(intrinsics.get("fy"))
        cx = _finite(intrinsics.get("cx"))
        cy = _finite(intrinsics.get("cy"))
        if any(value is None for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:  # type: ignore[operator]
            raise ValueError(f"camera model {model_id} intrinsics are invalid")
        resolution = model.get("resolution")
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            source_w = _positive_dimension(resolution[0], f"{model_id}.resolution[0]")
            source_h = _positive_dimension(resolution[1], f"{model_id}.resolution[1]")
        else:
            source_w, source_h = target_w, target_h
        scale_x = float(target_w) / float(source_w)
        scale_y = float(target_h) / float(source_h)
        k = np.array(
            [
                [float(fx) * scale_x, 0.0, float(cx) * scale_x],
                [0.0, float(fy) * scale_y, float(cy) * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        extrinsic_row = extrinsics.get(camera_id)
        raw_e = extrinsic_row.get("E") if isinstance(extrinsic_row, Mapping) else None
        if not isinstance(raw_e, list) or len(raw_e) != 16:
            raise ValueError(f"camera {camera_id} lacks a 16-value world-to-camera E")
        e_values = [_finite(value) for value in raw_e]
        if any(value is None for value in e_values):
            raise ValueError(f"camera {camera_id} E contains non-finite values")
        e = np.asarray(e_values, dtype=np.float64).reshape((4, 4), order="F")
        if not np.all(np.isfinite(e)) or abs(float(np.linalg.det(e))) <= 1e-9:
            raise ValueError(f"camera {camera_id} E is singular")
        if camera_id in result:
            raise ValueError(f"duplicate camera id: {camera_id}")
        result[camera_id] = _CameraCalibration(
            source_id=source_id,
            camera_id=camera_id,
            intrinsics=k,
            world_to_camera=e,
            floor_y=float(floor_y),
            image_size=(target_w, target_h),
        )
    if not result:
        raise ValueError("no reviewed camera calibrations were loaded")
    return result


def _spawn_runtime(
    args: argparse.Namespace, auth: RequiredInternalAuth
) -> subprocess.Popen:
    ws_url = urlparse(str(args.ws))
    ws_port = ws_url.port or 6040
    cmd = [
        sys.executable,
        "DS9/noesis/ds9_runtime.py",
        "--pipeline-config",
        str(args.pipeline_config),
        "--cameras-config",
        str(args.cameras_config),
        "--ws-port",
        str(ws_port),
        "--depth-enable-seconds",
        "0",
        "--disable-rest",
    ]
    env = os.environ.copy()
    configure_required_auth_environment(env, auth)
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    env.setdefault("NOESIS_MAPANYTHING_POSTPROCESS_ENABLED", "0")
    env.setdefault("NOESIS_CALIBRATION_POSE_ONLY", "1")
    return subprocess.Popen(cmd, env=env)


async def _recv_json(ws, *, timeout_s: float) -> Optional[Dict[str, object]]:
    try:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return None
    if isinstance(msg, (bytes, bytearray)):
        return None
    try:
        payload = json.loads(msg)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _strict_int(value: object, *, minimum: int = 0) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _bounds(payload: Mapping[str, object]) -> Optional[Tuple[float, float, float, float]]:
    values = tuple(
        _finite(payload.get(key)) for key in ("xMin", "xMax", "zMin", "zMax")
    )
    if any(value is None for value in values):
        return None
    x_min, x_max, z_min, z_max = (float(value) for value in values)  # type: ignore[arg-type]
    if x_max <= x_min or z_max <= z_min:
        return None
    return x_min, x_max, z_min, z_max


def _point_in_bounds(
    point: Tuple[float, float], bounds: Tuple[float, float, float, float]
) -> bool:
    x, z = point
    x_min, x_max, z_min, z_max = bounds
    return x_min - 1e-6 <= x <= x_max + 1e-6 and z_min - 1e-6 <= z <= z_max + 1e-6


def _track_image_size(
    track: Mapping[str, object],
    tracking: Mapping[str, object],
    calibration: _CameraCalibration,
) -> Tuple[int, int]:
    for value in (
        track.get("image_size"),
        tracking.get("image_size"),
        tracking.get("frame_size"),
    ):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                width = _positive_dimension(value[0], "track image width")
                height = _positive_dimension(value[1], "track image height")
                return width, height
            except ValueError:
                continue
    return calibration.image_size


def _raw_anchor_uv(
    track: Mapping[str, object], method: str
) -> Optional[Tuple[float, float]]:
    method_key = str(method or "").strip().lower()
    if method_key in {"image_foot", "image_base"}:
        value = track.get(method_key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            u, v = _finite(value[0]), _finite(value[1])
            if u is not None and v is not None:
                return float(u), float(v)
    bbox = track.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        parsed = tuple(_finite(value) for value in bbox[:4])
        if all(value is not None for value in parsed):
            left, top, width, height = (float(value) for value in parsed)  # type: ignore[arg-type]
            if width > 0.0 and height > 0.0:
                return left + width * 0.5, top + height
    return None


def _scaled_anchor_uv(
    track: Mapping[str, object],
    tracking: Mapping[str, object],
    calibration: _CameraCalibration,
    method: str,
) -> Optional[Tuple[float, float]]:
    raw = _raw_anchor_uv(track, method)
    if raw is None:
        return None
    source_w, source_h = _track_image_size(track, tracking, calibration)
    target_w, target_h = calibration.image_size
    u = min(max(raw[0], 0.0), float(source_w)) * float(target_w) / float(source_w)
    v = min(max(raw[1], 0.0), float(source_h)) * float(target_h) / float(source_h)
    return float(u), float(v)


def _world_to_camera_local(
    track: Mapping[str, object], calibration: _CameraCalibration
) -> Optional[Tuple[float, float]]:
    world = track.get("world")
    if track.get("world_valid") is not True or not isinstance(world, (list, tuple)) or len(world) != 3:
        return None
    world_x, world_z = _finite(world[0]), _finite(world[2])
    if world_x is None or world_z is None:
        return None
    point = np.array(
        [float(world_x), float(calibration.floor_y), float(world_z), 1.0],
        dtype=np.float64,
    )
    local = calibration.world_to_camera @ point
    if abs(float(local[3])) <= 1e-9:
        return None
    x = float(local[0] / local[3])
    z = float(local[2] / local[3])
    return (x, z) if math.isfinite(x) and math.isfinite(z) else None


def _depth_to_camera_local(
    track: Mapping[str, object],
    tracking: Mapping[str, object],
    calibration: _CameraCalibration,
    method: str,
) -> Optional[Tuple[float, float]]:
    depth = _usable_registered_depth_m(track)
    anchor = _scaled_anchor_uv(track, tracking, calibration, method)
    if depth is None or not 0.05 < depth < 50.0 or anchor is None:
        return None
    fx = float(calibration.intrinsics[0, 0])
    cx = float(calibration.intrinsics[0, 2])
    return (float((anchor[0] - cx) * depth / fx), float(depth))


def _ray_floor_to_camera_local(
    *,
    u: float,
    v: float,
    calibration: _CameraCalibration,
) -> Optional[Tuple[float, float]]:
    try:
        camera_to_world = np.linalg.inv(calibration.world_to_camera)
        origin = camera_to_world[:3, 3]
        direction_camera = np.linalg.inv(calibration.intrinsics) @ np.array(
            [float(u), float(v), 1.0], dtype=np.float64
        )
        direction_world = camera_to_world[:3, :3] @ direction_camera
        denominator = float(direction_world[1])
        if abs(denominator) <= 1e-9:
            return None
        distance = (float(calibration.floor_y) - float(origin[1])) / denominator
        if not math.isfinite(distance) or distance < 0.0:
            return None
        hit = origin + distance * direction_world
        local = calibration.world_to_camera @ np.array(
            [float(hit[0]), float(hit[1]), float(hit[2]), 1.0], dtype=np.float64
        )
        if abs(float(local[3])) <= 1e-9:
            return None
        point = (float(local[0] / local[3]), float(local[2] / local[3]))
        return point if all(math.isfinite(value) for value in point) else None
    except (ValueError, np.linalg.LinAlgError):
        return None


def _floor_contact_candidates(
    track: Mapping[str, object],
    tracking: Mapping[str, object],
    calibration: _CameraCalibration,
    method: str,
) -> List[Tuple[float, float]]:
    candidates: List[Tuple[int, int, float, float]] = []
    seen: set[Tuple[int, int]] = set()

    def add(priority: int, raw: Optional[Tuple[float, float]]) -> None:
        if raw is None:
            return
        source_w, source_h = _track_image_size(track, tracking, calibration)
        target_w, target_h = calibration.image_size
        u = min(max(raw[0], 0.0), float(source_w)) * target_w / source_w
        v = min(max(raw[1], 0.0), float(source_h)) * target_h / source_h
        key = (int(round(u * 10.0)), int(round(v * 10.0)))
        if key in seen:
            return
        seen.add(key)
        candidates.append((priority, len(candidates), float(u), float(v)))

    add(0, _raw_anchor_uv(track, method))
    add(0, _raw_anchor_uv(track, "image_foot"))
    add(1, _raw_anchor_uv(track, "image_base"))
    bbox = track.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        parsed = tuple(_finite(value) for value in bbox[:4])
        if all(value is not None for value in parsed):
            left, top, width, height = (float(value) for value in parsed)  # type: ignore[arg-type]
            if width > 0.0 and height > 0.0:
                add(20, (left + width * 0.5, top + height))
                add(21, (left + width * 0.30, top + height))
                add(22, (left + width * 0.70, top + height))
                add(23, (left + width * 0.5, top + height * 0.95))
    candidates.sort(key=lambda row: (row[0], row[1]))
    return [(row[2], row[3]) for row in candidates]


def _floorplan_alignment_matrix(
    authority: Mapping[str, object],
) -> Optional[np.ndarray]:
    alignment = authority.get("ray_to_floorplan_alignment")
    if not isinstance(alignment, Mapping) or alignment.get("quality") != "ok":
        return None
    raw = alignment.get("matrix_2x3")
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in raw)
    ):
        return None
    try:
        matrix = np.asarray(raw, dtype=np.float64).reshape(2, 3)
    except (TypeError, ValueError):
        return None
    return matrix if np.all(np.isfinite(matrix)) else None


def _apply_floorplan_alignment(
    point: Tuple[float, float],
    authority: Mapping[str, object],
) -> Optional[Tuple[float, float]]:
    matrix = _floorplan_alignment_matrix(authority)
    if matrix is None:
        return None
    result = matrix @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    parsed = (float(result[0]), float(result[1]))
    return parsed if all(math.isfinite(value) for value in parsed) else None


def _expected_local_point(
    *,
    display_source: str,
    point_method: str,
    track: Mapping[str, object],
    tracking: Mapping[str, object],
    bev: Mapping[str, object],
    calibration: _CameraCalibration,
    bounds: Tuple[float, float, float, float],
    authority: Mapping[str, object],
) -> Optional[Tuple[float, float]]:
    if display_source == "world_to_camera_local":
        return _world_to_camera_local(track, calibration)
    if display_source == "registered_depth_anchor":
        return _depth_to_camera_local(
            track,
            tracking,
            calibration,
            point_method,
        )
    if display_source == "image_depth_anchor":
        return _depth_to_camera_local(
            track,
            tracking,
            calibration,
            point_method,
        )
    if display_source in {"floor_contact_ray", "image_anchor"}:
        alignment = bev.get("floorplanAlignment")
        alignment_applied = (
            isinstance(alignment, Mapping) and alignment.get("applied") is True
        )
        for u, v in _floor_contact_candidates(
            track, tracking, calibration, point_method
        ):
            candidate = _ray_floor_to_camera_local(
                u=u,
                v=v,
                calibration=calibration,
            )
            if (
                candidate is not None
                and display_source == "floor_contact_ray"
                and alignment_applied
            ):
                candidate = _apply_floorplan_alignment(candidate, authority)
            if candidate is not None and _point_in_bounds(candidate, bounds):
                return candidate
        return None
    return None


def _authority_by_camera(
    report: Mapping[str, object],
    *,
    calibrations: Mapping[str, _CameraCalibration],
) -> Dict[str, Mapping[str, object]]:
    rows = report.get("cameras")
    if not isinstance(rows, list):
        raise ValueError("sealed floorplan authority camera rows are missing")
    result: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("sealed floorplan authority camera row is invalid")
        camera_id = row.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id or camera_id in result:
            raise ValueError("sealed floorplan authority camera inventory is invalid")
        result[camera_id] = row
    if set(result) != set(calibrations):
        raise ValueError("sealed floorplan authority does not cover reviewed cameras N/N")
    return result


def _bev_matches_floorplan_authority(
    bev: Mapping[str, object],
    authority: Mapping[str, object],
) -> bool:
    expected_bounds = authority.get("bounds")
    actual_bounds = bev.get("floorplanBounds")
    if not isinstance(expected_bounds, Mapping) or not isinstance(actual_bounds, Mapping):
        return False
    expected_bounds_values = tuple(
        _finite(expected_bounds.get(key))
        for key in ("min_x", "max_x", "min_z", "max_z")
    )
    actual_bounds_values = tuple(
        _finite(actual_bounds.get(key))
        for key in ("min_x", "max_x", "min_z", "max_z")
    )
    if (
        any(value is None for value in expected_bounds_values)
        or actual_bounds_values != expected_bounds_values
    ):
        return False
    expected_alignment = _floorplan_alignment_matrix(authority)
    alignment_summary = bev.get("floorplanAlignment")
    alignment_applied = (
        isinstance(alignment_summary, Mapping)
        and alignment_summary.get("applied") is True
    )
    if alignment_applied != (expected_alignment is not None):
        return False
    return bool(
        bev.get("boundsSource") == "active_floorplan"
        and bev.get("floorplanCoordinateSpace") == "floorplan_normalized_v1"
        and bev.get("floorplanSnapshotTsUs") == authority.get("snapshot_ts_us")
        and bev.get("floorplanTsUs") == authority.get("floorplan_ts_us")
        and bev.get("floorplanSnapshotId") == authority.get("snapshot_id")
        and bev.get("floorplanSnapshotContentSha256")
        == authority.get("snapshot_content_sha256")
        and bev.get("floorplanCalibrationFingerprint")
        == authority.get("calibration_fingerprint")
        and bev.get("floorplanGridShape") == authority.get("grid_shape")
        and _finite(bev.get("floorplanGridResM"))
        == _finite(authority.get("grid_res_m"))
        and tuple(_finite(bev.get(key)) for key in ("xMin", "xMax", "zMin", "zMax"))
        == expected_bounds_values
    )


def _evaluate_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    contract: _FrameContract,
    calibrations: Mapping[str, _CameraCalibration],
    floorplan_authority: Mapping[str, Mapping[str, object]],
    acquisition_started_at_us: int,
    acquisition_finished_at_us: int,
) -> Dict[str, object]:
    if (
        acquisition_started_at_us <= 0
        or acquisition_finished_at_us < acquisition_started_at_us
    ):
        raise ValueError("BEV acquisition window is invalid")
    if set(floorplan_authority) != set(calibrations):
        raise ValueError("BEV floorplan authority does not cover configured cameras N/N")

    tracking_frames: Dict[Tuple[str, int, int], Mapping[str, object]] = {}
    bev_frames: List[Mapping[str, object]] = []
    track_total = 0
    track_world_valid = 0
    track_world_frame_ok = 0
    duplicate_tracking_frames = 0
    tracking_identity_violations = 0
    freshness_violations = 0
    for payload in messages:
        if payload.get("type") == "tracking":
            camera_id = payload.get("camera_id")
            source_id = _strict_int(payload.get("source_id"))
            frame_id = _strict_int(payload.get("frame_id"))
            observed_at_us = _strict_int(payload.get("observed_at_us"), minimum=1)
            tracks = payload.get("tracks")
            camera = (
                calibrations.get(camera_id) if isinstance(camera_id, str) else None
            )
            if (
                camera is None
                or source_id is None
                or source_id != camera.source_id
                or frame_id is None
                or observed_at_us is None
                or not isinstance(tracks, list)
            ):
                tracking_identity_violations += 1
                continue
            if not (
                acquisition_started_at_us
                <= observed_at_us
                <= acquisition_finished_at_us
            ):
                freshness_violations += 1
                continue
            key = (camera_id, source_id, frame_id)
            if key in tracking_frames:
                duplicate_tracking_frames += 1
                continue
            tracking_frames[key] = payload
            for track in tracks:
                if not isinstance(track, Mapping):
                    tracking_identity_violations += 1
                    continue
                track_total += 1
                if track.get("world_valid") is True:
                    world = track.get("world")
                    if isinstance(world, (list, tuple)) and len(world) == 3:
                        track_world_valid += 1
                        if track.get("world_frame") == TRACK_WORLD_FRAME:
                            track_world_frame_ok += 1
        elif payload.get("type") == "bev-frame":
            bev_frames.append(payload)

    comparisons: List[float] = []
    display_source_counts: Dict[str, int] = {}
    local_frame_violations = 0
    identity_violations = 0
    association_violations = 0
    stale_association_violations = 0
    duplicate_bev_frames = 0
    reused_tracking_frame_violations = 0
    floorplan_authority_violations = 0
    display_source_violations = 0
    bounds_violations = 0
    trail_violations = 0
    geometry_unavailable = 0
    duplicate_tracker_ids = 0
    exact_associations = 0
    paired_bev_frames = 0
    footpoint_total = 0
    trail_total = 0
    paired_cameras: set[str] = set()
    consumed_tracking_frames: set[Tuple[str, int, int]] = set()
    seen_bev_frames: set[Tuple[str, int, int]] = set()

    for bev in bev_frames:
        if (
            bev.get("frame") != contract.frame
            or bev.get("world_frame") != contract.frame
            or bev.get("frame_mode") != contract.frame_mode
            or bev.get("units") != contract.units
        ):
            local_frame_violations += 1
        camera_id = bev.get("cameraId")
        source_id = _strict_int(bev.get("sourceId"))
        frame_id = _strict_int(bev.get("frameId"))
        observed_at_us = _strict_int(bev.get("observedAtUs"), minimum=1)
        camera = calibrations.get(camera_id) if isinstance(camera_id, str) else None
        authority = (
            floorplan_authority.get(camera_id)
            if isinstance(camera_id, str)
            else None
        )
        frame_bounds = _bounds(bev)
        if (
            camera is None
            or authority is None
            or source_id is None
            or frame_id is None
            or observed_at_us is None
            or source_id != camera.source_id
            or frame_bounds is None
        ):
            identity_violations += 1
            continue
        if not (
            acquisition_started_at_us
            <= observed_at_us
            <= acquisition_finished_at_us
        ):
            freshness_violations += 1
            continue
        frame_key = (camera.camera_id, source_id, frame_id)
        if frame_key in seen_bev_frames:
            duplicate_bev_frames += 1
            continue
        seen_bev_frames.add(frame_key)
        authority_matches = _bev_matches_floorplan_authority(bev, authority)
        if not authority_matches:
            floorplan_authority_violations += 1

        tracking = tracking_frames.get(frame_key)
        tracking_observed_at = (
            _strict_int(tracking.get("observed_at_us"), minimum=1)
            if isinstance(tracking, Mapping)
            else None
        )
        pair_valid = True
        if tracking is None:
            association_violations += 1
            pair_valid = False
        elif tracking_observed_at != observed_at_us:
            stale_association_violations += 1
            pair_valid = False
        elif frame_key in consumed_tracking_frames:
            reused_tracking_frame_violations += 1
            pair_valid = False
        elif not authority_matches:
            pair_valid = False
        else:
            consumed_tracking_frames.add(frame_key)
            paired_bev_frames += 1
            paired_cameras.add(camera.camera_id)

        tracks = tracking.get("tracks") if isinstance(tracking, Mapping) else None
        tracks_by_id: Dict[int, Mapping[str, object]] = {}
        for track in tracks or []:
            tracker_id = (
                _strict_int(track.get("tracker_id"))
                if isinstance(track, Mapping)
                else None
            )
            if tracker_id is None:
                tracking_identity_violations += 1
                continue
            if tracker_id in tracks_by_id:
                duplicate_tracker_ids += 1
                continue
            tracks_by_id[tracker_id] = track

        raw_footpoints = bev.get("footpoints")
        if not isinstance(raw_footpoints, list):
            identity_violations += 1
            raw_footpoints = []
        for point in raw_footpoints:
            footpoint_total += 1
            if not isinstance(point, Mapping):
                identity_violations += 1
                continue
            display_source = str(point.get("displaySource") or "").strip()
            display_source_counts[display_source or "<missing>"] = (
                display_source_counts.get(display_source or "<missing>", 0) + 1
            )
            if display_source not in ALLOWED_DISPLAY_SOURCES:
                display_source_violations += 1
                continue
            tracker_id = _strict_int(point.get("trackerId"))
            point_frame_id = _strict_int(point.get("frameId"))
            x, z = _finite(point.get("x")), _finite(point.get("y"))
            if (
                tracker_id is None
                or point_frame_id != frame_id
                or x is None
                or z is None
            ):
                identity_violations += 1
                continue
            actual = (float(x), float(z))
            if not _point_in_bounds(actual, frame_bounds):
                bounds_violations += 1
            track = tracks_by_id.get(tracker_id)
            if not pair_valid or track is None:
                association_violations += 1
                continue
            if _strict_int(track.get("frame_id")) != frame_id:
                stale_association_violations += 1
                continue
            stable_id = point.get("stableId")
            track_stable_id = track.get("stable_id")
            if stable_id not in (None, "", -1) and track_stable_id != stable_id:
                identity_violations += 1
                continue
            exact_associations += 1
            expected = _expected_local_point(
                display_source=display_source,
                point_method=str(point.get("method") or ""),
                track=track,
                tracking=tracking,
                bev=bev,
                calibration=camera,
                bounds=frame_bounds,
                authority=authority,
            )
            if expected is None:
                geometry_unavailable += 1
                continue
            comparisons.append(
                math.hypot(actual[0] - expected[0], actual[1] - expected[1])
            )

        raw_trails = bev.get("trails")
        if not isinstance(raw_trails, list):
            trail_violations += 1
            raw_trails = []
        for trail in raw_trails:
            trail_total += 1
            if (
                not isinstance(trail, Mapping)
                or _strict_int(trail.get("trackerId")) is None
            ):
                trail_violations += 1
                continue
            points = trail.get("points")
            if not isinstance(points, list) or not points:
                trail_violations += 1
                continue
            previous_t: Optional[int] = None
            for point in points:
                if not isinstance(point, Mapping):
                    trail_violations += 1
                    continue
                trail_point = (_finite(point.get("x")), _finite(point.get("y")))
                point_t = _strict_int(point.get("t"), minimum=1)
                if (
                    trail_point[0] is None
                    or trail_point[1] is None
                    or point_t is None
                    or (previous_t is not None and point_t < previous_t)
                    or not _point_in_bounds(
                        (float(trail_point[0]), float(trail_point[1])),
                        frame_bounds,
                    )
                ):
                    trail_violations += 1
                previous_t = point_t if point_t is not None else previous_t

    missing_paired_cameras = sorted(set(calibrations).difference(paired_cameras))
    values = sorted(comparisons)
    result: Dict[str, object] = {
        "configured_bev_frame": contract.frame,
        "configured_bev_frame_mode": contract.frame_mode,
        "configured_bev_units": contract.units,
        "acquisition_started_at_us": acquisition_started_at_us,
        "acquisition_finished_at_us": acquisition_finished_at_us,
        "configured_camera_count": len(calibrations),
        "paired_camera_count": len(paired_cameras),
        "paired_cameras": sorted(paired_cameras),
        "missing_paired_cameras": missing_paired_cameras,
        "track_total": track_total,
        "track_world_valid": track_world_valid,
        "track_world_frame_backend_world_m": track_world_frame_ok,
        "tracking_frame_count": len(tracking_frames),
        "duplicate_tracking_frames": duplicate_tracking_frames,
        "tracking_identity_violations": tracking_identity_violations,
        "bev_total": len(bev_frames),
        "paired_bev_frame_count": paired_bev_frames,
        "duplicate_bev_frames": duplicate_bev_frames,
        "reused_tracking_frame_violations": reused_tracking_frame_violations,
        "bev_camera_local_frame_ok": len(bev_frames) - local_frame_violations,
        "footpoint_total": footpoint_total,
        "trail_total": trail_total,
        "exact_frame_associations": exact_associations,
        "comparisons": len(comparisons),
        "display_source_counts": dict(sorted(display_source_counts.items())),
        "local_frame_violations": local_frame_violations,
        "identity_violations": identity_violations,
        "association_violations": association_violations,
        "stale_association_violations": stale_association_violations,
        "freshness_violations": freshness_violations,
        "floorplan_authority_violations": floorplan_authority_violations,
        "display_source_violations": display_source_violations,
        "bounds_violations": bounds_violations,
        "trail_violations": trail_violations,
        "duplicate_tracker_ids": duplicate_tracker_ids,
        "geometry_unavailable": geometry_unavailable,
        "mean_err_m": float(sum(values) / len(values)) if values else None,
        "p95_err_m": (
            float(values[int(0.95 * (len(values) - 1))]) if values else None
        ),
    }
    return result


async def _run(
    uri: str,
    duration_s: float,
    auth: RequiredInternalAuth,
    *,
    contract: _FrameContract,
    calibrations: Mapping[str, _CameraCalibration],
    floorplan_authority: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    messages: List[Mapping[str, object]] = []
    acquisition_started_at_us = time.time_ns() // 1_000
    async with connect_required_websocket(uri, auth, max_size=None) as ws:
        end_at = time.time() + max(5.0, float(duration_s))
        while time.time() < end_at:
            payload = await _recv_json(ws, timeout_s=2.0)
            if not payload or payload.get("type") not in {"tracking", "bev-frame"}:
                continue
            messages.append(payload)
            if len(messages) > MAX_CAPTURED_MESSAGES:
                raise RuntimeError("BEV parity observation bound exceeded")
    acquisition_finished_at_us = time.time_ns() // 1_000
    return _evaluate_messages(
        messages,
        contract=contract,
        calibrations=calibrations,
        floorplan_authority=floorplan_authority,
        acquisition_started_at_us=acquisition_started_at_us,
        acquisition_finished_at_us=acquisition_finished_at_us,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exact-frame camera-local BEV/tracking geometry gate."
    )
    parser.add_argument("--ws", default="ws://127.0.0.1:6040", help="WebSocket URL")
    parser.add_argument(
        "--pipeline-config", type=Path, default=Path("DS9/config/infer.yaml")
    )
    parser.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("config/camera_calibration.json"),
    )
    parser.add_argument(
        "--alignment-config",
        type=Path,
        default=Path("config/ply_alignment.json"),
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="Collection window seconds"
    )
    parser.add_argument(
        "--p95-threshold-m",
        type=float,
        default=1.5,
        help="Maximum camera-local p95 geometry error in meters",
    )
    parser.add_argument(
        "--no-spawn", action="store_true", help="Do not spawn the DS9 runtime"
    )
    parser.add_argument("--floorplan-authority", type=Path, required=True)
    parser.add_argument("--floorplan-source", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--runtime-lane", required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--runtime-run-id", required=True)
    add_auth_token_file_argument(parser)
    args = parser.parse_args()

    try:
        contract = _load_frame_contract(args.pipeline_config)
        calibrations = _load_calibrations(
            pipeline_config=args.pipeline_config,
            cameras_config=args.cameras_config,
            calibration_config=args.calibration_config,
            alignment_config=args.alignment_config,
        )
        threshold = float(args.p95_threshold_m)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("p95 threshold must be finite and positive")
        auth = load_required_internal_auth(args.auth_token_file)
        sealed_report = floorplan_gate.load_and_validate_sealed_authority(
            args.floorplan_authority,
            args.floorplan_source,
            session_id=args.session_id,
            runtime_lane=args.runtime_lane,
            runtime_instance_id=args.runtime_instance_id,
            runtime_run_id=args.runtime_run_id,
        )
        floorplan_authority = _authority_by_camera(
            sealed_report,
            calibrations=calibrations,
        )
    except Exception as exc:
        print(f"[FAIL] BEV gate configuration unavailable: {exc}")
        return 1

    proc: Optional[subprocess.Popen] = None
    if not args.no_spawn:
        proc = _spawn_runtime(args, auth)
        time.sleep(8.0)

    try:
        summary = asyncio.run(
            _run(
                str(args.ws),
                float(args.duration),
                auth,
                contract=contract,
                calibrations=calibrations,
                floorplan_authority=floorplan_authority,
            )
        )
        print(json.dumps(summary, indent=2))

        if int(summary.get("track_world_valid", 0)) <= 0:
            print("[FAIL] no world-valid tracking samples observed")
            return 1
        if int(summary.get("track_world_frame_backend_world_m", 0)) != int(
            summary.get("track_world_valid", 0)
        ):
            print("[FAIL] world-valid tracking samples are not all backend_world_m")
            return 1
        if int(summary.get("bev_total", 0)) <= 0:
            print("[FAIL] no BEV samples observed")
            return 1
        if int(summary.get("paired_camera_count", 0)) != len(calibrations):
            print("[FAIL] exact paired BEV frames do not cover configured cameras N/N")
            return 1
        if int(summary.get("footpoint_total", 0)) <= 0:
            print("[FAIL] no occupied BEV footpoints observed")
            return 1
        if int(summary.get("exact_frame_associations", 0)) <= 0:
            print("[FAIL] no exact source/frame/time BEV-to-track associations")
            return 1
        if int(summary.get("comparisons", 0)) <= 0:
            print("[FAIL] no independently calibrated BEV geometry comparisons")
            return 1
        if int(summary.get("comparisons", 0)) != int(
            summary.get("footpoint_total", 0)
        ):
            print("[FAIL] not every emitted BEV footpoint has independent geometry")
            return 1

        violation_fields = (
            "duplicate_tracking_frames",
            "tracking_identity_violations",
            "duplicate_bev_frames",
            "reused_tracking_frame_violations",
            "local_frame_violations",
            "identity_violations",
            "association_violations",
            "stale_association_violations",
            "freshness_violations",
            "floorplan_authority_violations",
            "display_source_violations",
            "bounds_violations",
            "trail_violations",
            "duplicate_tracker_ids",
            "geometry_unavailable",
        )
        violations = {
            field: int(summary.get(field, 0) or 0)
            for field in violation_fields
            if int(summary.get(field, 0) or 0) != 0
        }
        if violations:
            print(f"[FAIL] BEV contract violations: {json.dumps(violations, sort_keys=True)}")
            return 1

        p95 = summary.get("p95_err_m")
        if not isinstance(p95, (int, float)):
            print("[FAIL] invalid p95 error metric")
            return 1
        if float(p95) > threshold:
            print(
                f"[FAIL] camera-local BEV geometry p95 error too high: "
                f"{p95:.6f} m > {threshold:.6f} m"
            )
            return 1

        print(
            "[PASS] exact-frame camera-local BEV contract validated "
            f"(p95_err_m={float(p95):.6f}, threshold_m={threshold:.6f})"
        )
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
