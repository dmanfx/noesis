#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np
import yaml
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapanything_config import load_service_config
from noesis.calibration.scene_registration import solve_scene_similarity
from noesis.ds8_runtime import _CalibrationProvider, _load_camera_labels
from noesis.virtual_twin.builder import (
    VirtualTwinBuildError,
    VirtualTwinFrameInput,
    build_virtual_twin_revision,
    revision_id_from_clock,
    sha256_file,
)
from noesis.virtual_twin.store import VirtualTwinStore
from noesis.virtual_twin.zeroplane_adapter import (
    PrecomputedZeroPlaneAdapter,
    ZeroPlaneAdapterError,
    ZeroPlaneCommandAdapter,
)


LOGGER = logging.getLogger("virtual_twin_builder")


def _default_mapanything_refresh_url() -> str:
    explicit = (
        os.environ.get("NOESIS_MAPANYTHING_REFRESH_URL")
        or os.environ.get("NOESIS_DEPTH_REFRESH_URL")
    )
    if explicit:
        return explicit

    host = os.environ.get("NOESIS_DEPTH_REST_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = (
        os.environ.get("NOESIS_DEPTH_REST_PORT")
        or os.environ.get("NOESIS_DS8_REST_PORT")
        or "8082"
    )
    return f"http://{host}:{port}/api/v1/depth/refresh"


@dataclass(frozen=True)
class MapAnythingSnapshot:
    path: Path
    timestamp_us: int
    depth: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class DewarperFrameTransform:
    config_path: Path
    source_intrinsics: np.ndarray
    distortion: np.ndarray
    rectified_intrinsics: np.ndarray
    output_size: tuple[int, int]


def _video_capture_from_uri(uri: str) -> cv2.VideoCapture:
    if uri.startswith("file://"):
        return cv2.VideoCapture(uri[7:])
    return cv2.VideoCapture(uri)


def _iter_source_frames(*, uri: str, frame_stride: int, max_frames_read: int) -> Iterable[tuple[int, np.ndarray]]:
    cap = _video_capture_from_uri(uri)
    if not cap.isOpened():
        raise VirtualTwinBuildError(f"unable to open camera source URI for virtual-twin keyframes: {uri}")
    try:
        idx = 0
        yielded = 0
        while idx < int(max_frames_read):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % max(1, int(frame_stride)) == 0:
                yielded += 1
                yield yielded, frame
            idx += 1
    finally:
        cap.release()


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise VirtualTwinBuildError(f"expected mapping at {path}")
    return payload


def _camera_source(pipeline_cfg: dict[str, Any], camera_labels: dict[int, str], camera: str) -> tuple[int, str, dict[str, Any]]:
    sources = pipeline_cfg.get("sources")
    if not isinstance(sources, list):
        raise VirtualTwinBuildError("pipeline config has no sources list")
    wanted = str(camera).strip()
    for source_id, label in camera_labels.items():
        if label == wanted:
            source = sources[int(source_id)] if int(source_id) < len(sources) else None
            if not isinstance(source, dict) or not isinstance(source.get("uri"), str):
                raise VirtualTwinBuildError(f"camera {wanted} has no configured URI in pipeline sources")
            return int(source_id), str(source["uri"]), source
    raise VirtualTwinBuildError(f"camera {wanted!r} not found in camera labels: {sorted(camera_labels.values())}")


def _scaled_snapshot(snapshot, frame_shape: tuple[int, int, int]):
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    snap_w, snap_h = int(snapshot.image_size[0]), int(snapshot.image_size[1])
    if (snap_w, snap_h) == (frame_w, frame_h):
        return snapshot
    k = np.asarray(snapshot.intrinsics, dtype=np.float64).copy()
    if snap_w > 0 and snap_h > 0:
        sx = float(frame_w) / float(snap_w)
        sy = float(frame_h) / float(snap_h)
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
    return replace(snapshot, intrinsics=k, image_size=(frame_w, frame_h))


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _parse_float_pair(value: str, *, field: str, path: Path) -> tuple[float, float]:
    parts = [part.strip() for part in str(value or "").split(";") if part.strip()]
    if len(parts) != 2:
        raise VirtualTwinBuildError(f"dewarper config {path} field {field} must contain two semicolon-separated numbers")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise VirtualTwinBuildError(f"dewarper config {path} field {field} contains non-numeric values") from exc


def _parse_float_list(value: str, *, field: str, path: Path) -> list[float]:
    parts = [part.strip() for part in str(value or "").split(";") if part.strip()]
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise VirtualTwinBuildError(f"dewarper config {path} field {field} contains non-numeric values") from exc


def _load_dewarper_frame_transform(path: Path) -> DewarperFrameTransform:
    cfg_path = _resolve_repo_path(path)
    if not cfg_path.exists():
        raise VirtualTwinBuildError(f"configured dewarper file does not exist: {cfg_path}")
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.optionxform = str
    try:
        parser.read(cfg_path, encoding="utf-8")
    except Exception as exc:
        raise VirtualTwinBuildError(f"failed to read dewarper config {cfg_path}: {exc}") from exc
    if "property" not in parser or "surface0" not in parser:
        raise VirtualTwinBuildError(f"dewarper config {cfg_path} must contain [property] and [surface0]")
    props = parser["property"]
    surface = parser["surface0"]
    try:
        output_w = int(float(props.get("output-width", surface.get("width", "0"))))
        output_h = int(float(props.get("output-height", surface.get("height", "0"))))
        src_fx, src_fy = _parse_float_pair(surface.get("focal-length", ""), field="focal-length", path=cfg_path)
        dst_fx, dst_fy = _parse_float_pair(surface.get("dst-focal-length", ""), field="dst-focal-length", path=cfg_path)
        src_cx = float(surface.get("src-x0", "nan"))
        src_cy = float(surface.get("src-y0", "nan"))
        dst_cx, dst_cy = _parse_float_pair(surface.get("dst-principal-point", ""), field="dst-principal-point", path=cfg_path)
    except ValueError as exc:
        raise VirtualTwinBuildError(f"dewarper config {cfg_path} contains invalid numeric values") from exc
    if output_w <= 0 or output_h <= 0:
        raise VirtualTwinBuildError(f"dewarper config {cfg_path} has invalid output size {output_w}x{output_h}")
    distortion = np.asarray(_parse_float_list(surface.get("distortion", ""), field="distortion", path=cfg_path), dtype=np.float64)
    if distortion.size < 4:
        raise VirtualTwinBuildError(f"dewarper config {cfg_path} needs at least four fisheye distortion coefficients")
    source_intrinsics = np.asarray(
        [[src_fx, 0.0, src_cx], [0.0, src_fy, src_cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rectified_intrinsics = np.asarray(
        [[dst_fx, 0.0, dst_cx], [0.0, dst_fy, dst_cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(source_intrinsics)) or not np.all(np.isfinite(rectified_intrinsics)):
        raise VirtualTwinBuildError(f"dewarper config {cfg_path} contains non-finite intrinsics")
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
        raise VirtualTwinBuildError("source dewarper is enabled but has no config-file")
    return _load_dewarper_frame_transform(Path(config_file))


def _apply_dewarper_transform(frame_bgr: np.ndarray, transform: DewarperFrameTransform) -> np.ndarray:
    image_h, image_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    source_k = np.asarray(transform.source_intrinsics, dtype=np.float64).copy()
    # The checked-in dewarper configs are authored for the camera stream
    # resolution. If OpenCV gives us a different source size, scale only the
    # source camera matrix; the DS8 dewarper output size/intrinsics stay fixed.
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


def _frame_snapshot_for_transform(snapshot: Any, transform: DewarperFrameTransform | None, frame_shape: tuple[int, int, int]) -> Any:
    if transform is None:
        return _scaled_snapshot(snapshot, frame_shape)
    return replace(
        snapshot,
        intrinsics=np.asarray(transform.rectified_intrinsics, dtype=np.float64).reshape(3, 3),
        image_size=tuple(int(v) for v in transform.output_size),
    )


def _resolve_depth_base(path: Path | None, configured_depth_base: str) -> Path:
    raw = Path(path) if path is not None else Path(configured_depth_base)
    raw = raw.expanduser()
    if raw.is_absolute():
        return raw
    return (REPO_ROOT / raw).resolve()


def _snapshot_entries(depth_base: Path, camera_id: str) -> list[tuple[int, Path]]:
    camera_root = Path(depth_base) / str(camera_id)
    if not camera_root.exists():
        return []
    entries: list[tuple[int, Path]] = []
    for path in camera_root.rglob("*.zarr"):
        try:
            ts_us = int(path.stem)
        except ValueError:
            continue
        entries.append((ts_us, path))
    entries.sort(key=lambda item: item[0])
    return entries


def _load_mapanything_snapshot(path: Path) -> MapAnythingSnapshot:
    try:
        group = zarr.open_group(str(path), mode="r")
        depth = np.asarray(np.array(group["depth_z"]), dtype=np.float32)
        confidence = np.asarray(np.array(group["conf"]), dtype=np.float32)
        mask = np.asarray(np.array(group["mask"]), dtype=bool)
    except Exception as exc:
        raise VirtualTwinBuildError(f"failed to load MapAnything snapshot {path}: {exc}") from exc
    if depth.ndim != 2 or confidence.shape != depth.shape or mask.shape != depth.shape:
        raise VirtualTwinBuildError(
            f"MapAnything snapshot {path} has incompatible array shapes: "
            f"depth={depth.shape} conf={confidence.shape} mask={mask.shape}"
        )
    try:
        ts_us = int(group.attrs.get("timestamp_us", path.stem))
    except Exception:
        ts_us = int(path.stem)
    return MapAnythingSnapshot(
        path=Path(path),
        timestamp_us=ts_us,
        depth=depth,
        confidence=confidence,
        mask=mask,
    )


def _valid_depth_coverage(snapshot: MapAnythingSnapshot, *, min_confidence: float) -> float:
    valid = (
        np.asarray(snapshot.mask, dtype=bool)
        & np.isfinite(snapshot.depth)
        & (np.asarray(snapshot.depth, dtype=np.float32) > 0.0)
        & np.isfinite(snapshot.confidence)
    )
    if np.isfinite(float(min_confidence)):
        valid &= np.asarray(snapshot.confidence, dtype=np.float32) >= float(min_confidence)
    total = int(valid.size)
    return float(np.count_nonzero(valid)) / float(total) if total else 0.0


def _latest_mapanything_snapshot(
    *,
    depth_base: Path,
    camera_id: str,
    min_ts_us: int = 0,
    min_confidence: float,
) -> MapAnythingSnapshot | None:
    for ts_us, path in reversed(_snapshot_entries(depth_base, camera_id)):
        if int(ts_us) <= int(min_ts_us):
            continue
        try:
            snapshot = _load_mapanything_snapshot(path)
        except VirtualTwinBuildError as exc:
            LOGGER.debug("skipping incomplete MapAnything snapshot %s: %s", path, exc)
            continue
        if _valid_depth_coverage(snapshot, min_confidence=min_confidence) > 0.0:
            return snapshot
    return None


def _trigger_mapanything_refresh(refresh_url: str, *, seconds: int, timeout_s: float) -> dict[str, Any]:
    url = str(refresh_url).strip()
    if not url:
        raise VirtualTwinBuildError("MapAnything refresh URL is empty")
    separator = "&" if "?" in url else "?"
    request_url = f"{url}{separator}{urlencode({'seconds': int(seconds)})}"
    try:
        request = Request(request_url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
            status = int(getattr(response, "status", 0) or 0)
            body = response.read().decode("utf-8")
    except Exception as exc:
        raise VirtualTwinBuildError(
            f"failed to trigger DS8 MapAnything refresh at {request_url}: {exc}"
        ) from exc
    if status < 200 or status >= 300:
        raise VirtualTwinBuildError(f"DS8 MapAnything refresh returned HTTP {status}: {body[:240]}")
    try:
        payload = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError as exc:
        raise VirtualTwinBuildError(f"DS8 MapAnything refresh returned non-JSON payload: {body[:240]}") from exc
    return payload if isinstance(payload, dict) else {"payload": payload}


def _wait_for_mapanything_snapshot(
    *,
    depth_base: Path,
    camera_id: str,
    min_ts_us: int,
    timeout_s: float,
    min_confidence: float,
) -> MapAnythingSnapshot:
    deadline = time.time() + max(0.1, float(timeout_s))
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            snapshot = _latest_mapanything_snapshot(
                depth_base=depth_base,
                camera_id=camera_id,
                min_ts_us=int(min_ts_us),
                min_confidence=float(min_confidence),
            )
            if snapshot is not None:
                return snapshot
        except VirtualTwinBuildError as exc:
            last_error = exc
        time.sleep(0.25)
    detail = f" last load error: {last_error}" if last_error else ""
    raise VirtualTwinBuildError(
        f"timed out waiting for fresh DS8 MapAnything snapshot for {camera_id} "
        f"under {depth_base} newer than {min_ts_us}.{detail}"
    )


def _resize_depth_snapshot(snapshot: MapAnythingSnapshot, frame_shape: tuple[int, int, int]) -> MapAnythingSnapshot:
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    snap_h, snap_w = [int(v) for v in snapshot.depth.shape[:2]]
    if (snap_w, snap_h) == (frame_w, frame_h):
        return snapshot
    size = (frame_w, frame_h)
    depth = cv2.resize(snapshot.depth, size, interpolation=cv2.INTER_LINEAR).astype(np.float32, copy=False)
    confidence = cv2.resize(snapshot.confidence, size, interpolation=cv2.INTER_LINEAR).astype(np.float32, copy=False)
    mask = cv2.resize(snapshot.mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return MapAnythingSnapshot(
        path=snapshot.path,
        timestamp_us=snapshot.timestamp_us,
        depth=depth,
        confidence=confidence,
        mask=mask,
    )


def _model_refs(args: argparse.Namespace) -> dict[str, Any]:
    refs: dict[str, Any] = {
        "repo": str(Path(args.zeroplane_repo)),
        "config": str(Path(args.zeroplane_config)),
        "checkpoint": str(Path(args.zeroplane_checkpoint)),
    }
    if Path(args.zeroplane_checkpoint).exists():
        refs["checkpoint_sha256"] = sha256_file(Path(args.zeroplane_checkpoint))
    return refs


def _initial_world_to_scene_from_bundle(bundle: dict[str, Any]) -> list[float] | None:
    align = bundle.get("align")
    if not isinstance(align, dict):
        return None
    scene_similarity = align.get("scene_similarity")
    if not isinstance(scene_similarity, dict):
        return None
    values = scene_similarity.get("world_to_scene_col_major")
    if not isinstance(values, list) or len(values) != 16:
        return None
    try:
        return [float(x) for x in values]
    except Exception:
        return None


def _initial_world_to_scene_from_alignment(path: Path) -> list[float] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    scene_similarity = payload.get("scene_similarity")
    if not isinstance(scene_similarity, dict):
        scene_similarity = (payload.get("align") or {}).get("scene_similarity") if isinstance(payload.get("align"), dict) else None
    if not isinstance(scene_similarity, dict):
        return None
    values = scene_similarity.get("world_to_scene_col_major")
    if not isinstance(values, list) or len(values) != 16:
        return None
    try:
        return [float(x) for x in values]
    except Exception:
        return None


def _camera_center_from_snapshot(snapshot: Any) -> list[float]:
    matrix = np.asarray(snapshot.extrinsics_col_major, dtype=np.float64).reshape((4, 4), order="F")
    center = np.linalg.inv(matrix)[:3, 3]
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise VirtualTwinBuildError(f"calibration snapshot for {snapshot.camera_id} has a non-finite camera center")
    return [float(x) for x in center]


def _normalize_camera_key(value: str) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _load_menon_device_camera_positions(path: Path, camera_ids: Iterable[str]) -> dict[str, list[float]]:
    device_path = Path(path).expanduser()
    if not device_path.is_absolute():
        device_path = (REPO_ROOT / device_path).resolve()
    try:
        payload = json.loads(device_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VirtualTwinBuildError(f"Menon device config not found for scene prior: {device_path}") from exc
    except Exception as exc:
        raise VirtualTwinBuildError(f"failed to read Menon device config for scene prior {device_path}: {exc}") from exc
    if not isinstance(payload, list):
        raise VirtualTwinBuildError(f"Menon device config must contain a device list: {device_path}")

    camera_key_to_id = {_normalize_camera_key(camera_id): str(camera_id) for camera_id in camera_ids}
    camera_phrase_to_id = {
        _normalize_camera_key(str(camera_id).replace("-", " ") + " camera"): str(camera_id)
        for camera_id in camera_ids
    }
    out: dict[str, list[float]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        category = str(item.get("category") or item.get("type") or "")
        normalized_name = _normalize_camera_key(name)
        if "camera" not in normalized_name and _normalize_camera_key(category) != "camera":
            continue
        camera_id = camera_phrase_to_id.get(normalized_name)
        if camera_id is None:
            for key, candidate in camera_key_to_id.items():
                if key and key in normalized_name:
                    camera_id = candidate
                    break
        if camera_id is None:
            continue
        pos = item.get("position")
        if not isinstance(pos, dict):
            continue
        try:
            point = [float(pos["x"]), float(pos["y"]), float(pos["z"])]
        except Exception:
            continue
        if not all(np.isfinite(point)):
            continue
        out[str(camera_id)] = point
    return out


def _scene_prior_from_menon_devices(
    *,
    calibration_provider: _CalibrationProvider,
    camera_labels: dict[int, str],
    devices_config: Path,
) -> dict[str, Any]:
    camera_ids = sorted({str(v) for v in camera_labels.values() if str(v).strip()})
    scene_positions = _load_menon_device_camera_positions(devices_config, camera_ids)
    correspondences: list[dict[str, Any]] = []
    for source_id, camera_id in sorted(camera_labels.items()):
        camera_id = str(camera_id)
        scene_position = scene_positions.get(camera_id)
        if scene_position is None:
            continue
        snapshot = calibration_provider.snapshot(int(source_id), camera_id)
        if snapshot is None:
            continue
        correspondences.append(
            {
                "camera_id": camera_id,
                "world_position_m": _camera_center_from_snapshot(snapshot),
                "scene_position": scene_position,
            }
        )
    if len(correspondences) < 2:
        raise VirtualTwinBuildError(
            "Menon device scene prior needs at least two camera device correspondences; "
            f"got {len(correspondences)} from {devices_config}"
        )
    prior = solve_scene_similarity(
        correspondences,
        source="menon_virtual_device_camera_similarity",
        residual_units="scene_units",
    )
    prior["devices_config"] = str(Path(devices_config).expanduser())
    return prior


def build(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_path = Path(args.pipeline_config)
    cameras_path = Path(args.cameras_config)
    pipeline_cfg = _load_yaml(pipeline_path)
    camera_labels = _load_camera_labels(cameras_path)
    source_id, uri, source_cfg = _camera_source(pipeline_cfg, camera_labels, args.camera)
    dewarper_transform = _dewarper_transform_for_source(source_cfg)
    model_obj_raw = str(args.model_obj or os.environ.get("NOESIS_MENON_STRUCTURAL_OBJ", "")).strip()
    if not model_obj_raw:
        raise VirtualTwinBuildError("Menon structural OBJ is required via --model-obj or NOESIS_MENON_STRUCTURAL_OBJ")
    model_obj = Path(model_obj_raw).expanduser()
    if not model_obj.exists():
        raise VirtualTwinBuildError(f"Menon structural OBJ does not exist: {model_obj}")

    revision_id = args.revision_id or revision_id_from_clock("vt_living_room")
    store = VirtualTwinStore(Path(args.output_root) if args.output_root else None)
    staging_dir = Path(store.root) / "staging" / revision_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = staging_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    calibration_provider = _CalibrationProvider(cameras_path, pipeline_cfg)
    calibration_provider.set_camera_labels(camera_labels)
    snapshot = calibration_provider.snapshot(source_id, args.camera)
    if snapshot is None:
        raise VirtualTwinBuildError(f"calibration snapshot unavailable for {args.camera}")
    calib_bundle = calibration_provider.calibration_bundle()
    scene_prior_refs: dict[str, Any]
    if str(args.scene_prior_source) == "menon-devices":
        scene_prior_refs = _scene_prior_from_menon_devices(
            calibration_provider=calibration_provider,
            camera_labels=camera_labels,
            devices_config=Path(args.menon_devices_config),
        )
        initial_world_to_scene = [float(x) for x in scene_prior_refs["world_to_scene_col_major"]]
    elif str(args.scene_prior_source) == "alignment":
        scene_prior_refs = {
            "source": "alignment_config_scene_similarity",
            "alignment_config": str(Path(args.alignment_config)),
        }
        initial_world_to_scene = _initial_world_to_scene_from_bundle(calib_bundle)
        if initial_world_to_scene is None:
            initial_world_to_scene = _initial_world_to_scene_from_alignment(Path(args.alignment_config))
        if initial_world_to_scene is None:
            raise VirtualTwinBuildError(
                "calibration lacks scene_similarity.world_to_scene_col_major; "
                "virtual-twin registration needs an explicit Menon scene prior"
            )
        scene_prior_refs["world_to_scene_col_major"] = [float(x) for x in initial_world_to_scene]
    else:
        raise VirtualTwinBuildError(f"unsupported scene prior source: {args.scene_prior_source}")

    camera_scene_anchor = None
    for row in scene_prior_refs.get("correspondences", []) if isinstance(scene_prior_refs, dict) else []:
        if isinstance(row, dict) and str(row.get("camera_id")) == str(args.camera):
            camera_scene_anchor = {
                "camera_id": str(args.camera),
                "source": scene_prior_refs.get("source", str(args.scene_prior_source)),
                "world_position_m": row.get("world_position_m"),
                "scene_position": row.get("scene_position"),
            }
            break
    if str(args.scene_prior_source) == "menon-devices" and camera_scene_anchor is None:
        raise VirtualTwinBuildError(f"Menon device scene prior did not include active camera {args.camera!r}")

    map_config = load_service_config()
    depth_base = _resolve_depth_base(args.mapanything_depth_base, map_config.storage.depth_base)
    min_confidence = float(map_config.performance.min_conf)
    baseline_snapshot = _latest_mapanything_snapshot(
        depth_base=depth_base,
        camera_id=args.camera,
        min_ts_us=0,
        min_confidence=min_confidence,
    )
    baseline_ts_us = int(baseline_snapshot.timestamp_us) if baseline_snapshot is not None else 0
    if args.no_mapanything_refresh:
        if baseline_snapshot is None:
            raise VirtualTwinBuildError(
                f"--no-mapanything-refresh requested, but no usable persisted MapAnything snapshot exists "
                f"for {args.camera} under {depth_base}"
            )
        LOGGER.info("using existing DS8 MapAnything snapshot %s", baseline_snapshot.path)

    if args.zeroplane_precomputed_dir:
        zeroplane = PrecomputedZeroPlaneAdapter(Path(args.zeroplane_precomputed_dir))
    else:
        zeroplane = ZeroPlaneCommandAdapter(
            repo_dir=Path(args.zeroplane_repo),
            checkpoint_path=Path(args.zeroplane_checkpoint),
            config_path=Path(args.zeroplane_config),
            output_dir=staging_dir / "zeroplane",
            runner_script=Path(args.zeroplane_runner),
            device=args.zeroplane_device,
        )

    frame_inputs: list[VirtualTwinFrameInput] = []
    snapshot_paths: list[str] = []
    refresh_payloads: list[dict[str, Any]] = []
    next_snapshot_min_ts_us = int(baseline_ts_us)
    for local_index, raw_frame_bgr in _iter_source_frames(
        uri=uri,
        frame_stride=args.frame_stride,
        max_frames_read=args.max_frames_read,
    ):
        if len(frame_inputs) >= int(args.keyframes):
            break
        frame_bgr = (
            _apply_dewarper_transform(raw_frame_bgr, dewarper_transform)
            if dewarper_transform is not None
            else raw_frame_bgr
        )
        frame_id = f"{args.camera}_{local_index:04d}"
        frame_path = frames_dir / f"{frame_id}.png"
        if not cv2.imwrite(str(frame_path), frame_bgr):
            raise VirtualTwinBuildError(f"failed to write captured keyframe {frame_path}")
        frame_snapshot = _frame_snapshot_for_transform(snapshot, dewarper_transform, frame_bgr.shape)
        if args.no_mapanything_refresh:
            assert baseline_snapshot is not None
            map_snapshot = baseline_snapshot
        else:
            refresh_payload = _trigger_mapanything_refresh(
                args.mapanything_refresh_url,
                seconds=int(args.mapanything_refresh_seconds),
                timeout_s=min(10.0, float(args.mapanything_refresh_timeout)),
            )
            refresh_payloads.append(refresh_payload)
            LOGGER.info(
                "triggered DS8 MapAnything refresh for keyframe %s (%ss via %s)",
                frame_id,
                int(args.mapanything_refresh_seconds),
                args.mapanything_refresh_url,
            )
            map_snapshot = _wait_for_mapanything_snapshot(
                depth_base=depth_base,
                camera_id=args.camera,
                min_ts_us=next_snapshot_min_ts_us,
                timeout_s=float(args.mapanything_refresh_timeout),
                min_confidence=min_confidence,
            )
            next_snapshot_min_ts_us = max(next_snapshot_min_ts_us, int(map_snapshot.timestamp_us))
        map_snapshot = _resize_depth_snapshot(map_snapshot, frame_bgr.shape)
        coverage = _valid_depth_coverage(map_snapshot, min_confidence=min_confidence)
        if coverage <= 0.0:
            raise VirtualTwinBuildError(
                f"persisted MapAnything snapshot {map_snapshot.path} has no usable depth for {frame_id}"
            )
        zp_frame = zeroplane.infer(
            frame_id=frame_id,
            image_path=frame_path,
            intrinsics=frame_snapshot.intrinsics,
            image_size=(int(frame_bgr.shape[1]), int(frame_bgr.shape[0])),
        )
        snapshot_paths.append(str(map_snapshot.path))
        frame_inputs.append(
            VirtualTwinFrameInput(
                frame_id=frame_id,
                camera_id=args.camera,
                image_bgr=frame_bgr,
                map_depth=np.asarray(map_snapshot.depth, dtype=np.float32),
                map_confidence=np.asarray(map_snapshot.confidence, dtype=np.float32),
                map_mask=np.asarray(map_snapshot.mask, dtype=bool),
                calibration=frame_snapshot,
                plane_candidates=zp_frame.planes,
                source_ref=(
                    f"rgb={frame_path}; "
                    f"mapanything_zarr={map_snapshot.path}; "
                    f"mapanything_ts_us={map_snapshot.timestamp_us}; "
                    f"frame_preprocess="
                    f"{'ds8_nvdewarper:'+str(dewarper_transform.config_path) if dewarper_transform is not None else 'none'}"
                ),
            )
        )
        LOGGER.info(
            "captured keyframe %s with %d ZeroPlane plane candidates and DS8 MapAnything coverage %.3f from %s",
            frame_id,
            len(zp_frame.planes),
            coverage,
            map_snapshot.path,
        )

    if len(frame_inputs) < int(args.keyframes):
        raise VirtualTwinBuildError(f"captured {len(frame_inputs)} keyframes, expected {int(args.keyframes)}")

    return build_virtual_twin_revision(
        frames=frame_inputs,
        menon_obj_path=model_obj,
        store=store,
        revision_id=revision_id,
        camera_label=args.camera,
        mapanything_refs={
            "source": "ds8_pipeline_depth_snapshots",
            "depth_base": str(depth_base),
            "refresh_url": None if args.no_mapanything_refresh else str(args.mapanything_refresh_url),
            "refresh_seconds": None if args.no_mapanything_refresh else int(args.mapanything_refresh_seconds),
            "refresh_payloads": refresh_payloads,
            "min_confidence": min_confidence,
            "snapshot_paths": snapshot_paths,
            "frame_preprocess": (
                {
                    "method": "ds8_nvdewarper_mirror",
                    "config_file": str(dewarper_transform.config_path),
                    "output_size": [int(dewarper_transform.output_size[0]), int(dewarper_transform.output_size[1])],
                    "rectified_intrinsics": np.asarray(dewarper_transform.rectified_intrinsics, dtype=np.float64).tolist(),
                }
                if dewarper_transform is not None
                else {"method": "none"}
            ),
        },
        zeroplane_refs=_model_refs(args),
        scene_prior_refs=scene_prior_refs,
        camera_scene_anchor=camera_scene_anchor,
        browser_point_budget=int(args.browser_point_budget),
        texture_tile_px=int(args.texture_tile_px),
        min_texture_coverage=float(args.min_texture_coverage),
        texture_exposure=float(args.texture_exposure),
        texture_gamma=float(args.texture_gamma),
        texture_contrast=float(args.texture_contrast),
        min_texture_colorfulness=0.0 if args.allow_grayscale_texture else float(args.min_texture_colorfulness),
        min_mapanything_coverage=float(args.min_mapanything_coverage),
        min_zero_planes=int(args.min_zero_planes),
        min_registration_correspondences=int(args.min_registration_correspondences),
        max_room_model_leakage=float(args.max_room_model_leakage),
        initial_world_to_menon_scene_col_major=initial_world_to_scene,
        update_latest=not args.no_update_latest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline MapAnything+ZeroPlane virtual-twin reconstruction")
    parser.add_argument("--camera", default="living-room")
    parser.add_argument("--pipeline-config", type=Path, default=REPO_ROOT / "config" / "infer.yaml")
    parser.add_argument("--cameras-config", type=Path, default=REPO_ROOT / "config" / "cameras.yaml")
    parser.add_argument("--alignment-config", type=Path, default=REPO_ROOT / "config" / "ply_alignment.json")
    parser.add_argument(
        "--scene-prior-source",
        choices=("menon-devices", "alignment"),
        default="menon-devices",
        help="Initial backend-world to Menon-scene prior. Use Menon device placement by default.",
    )
    parser.add_argument(
        "--menon-devices-config",
        type=Path,
        default=REPO_ROOT.parent / "Menon" / "public" / "config" / "virtual-devices.json",
        help="Menon virtual-devices JSON used when --scene-prior-source=menon-devices.",
    )
    parser.add_argument("--model-obj", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--revision-id", default=None)
    parser.add_argument("--keyframes", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-frames-read", type=int, default=900)
    parser.add_argument("--mapanything-depth-base", type=Path, default=None)
    parser.add_argument("--mapanything-refresh-url", default=_default_mapanything_refresh_url())
    parser.add_argument("--mapanything-refresh-seconds", type=int, default=10)
    parser.add_argument("--mapanything-refresh-timeout", type=float, default=45.0)
    parser.add_argument(
        "--no-mapanything-refresh",
        action="store_true",
        help="Use the latest existing DS8 MapAnything Zarr snapshot instead of triggering /api/v1/depth/refresh.",
    )
    parser.add_argument("--zeroplane-repo", type=Path, default=REPO_ROOT / "external" / "ZeroPlane")
    parser.add_argument("--zeroplane-config", type=Path, default=REPO_ROOT / "external" / "ZeroPlane" / "configs" / "ZeroPlaneNYUV2" / "dust3r_large_dpt_bs16_50ep.yaml")
    parser.add_argument("--zeroplane-checkpoint", type=Path, default=REPO_ROOT / "external" / "ZeroPlane" / "checkpoints" / "dust3r_encoder_released.pth")
    parser.add_argument("--zeroplane-runner", type=Path, default=REPO_ROOT / "scripts" / "run_zeroplane_inference.py")
    parser.add_argument("--zeroplane-device", default="cuda")
    parser.add_argument("--zeroplane-precomputed-dir", type=Path, default=None)
    parser.add_argument("--browser-point-budget", type=int, default=150_000)
    parser.add_argument("--texture-tile-px", type=int, default=96)
    parser.add_argument("--min-texture-coverage", type=float, default=0.02)
    parser.add_argument("--texture-exposure", type=float, default=2.2)
    parser.add_argument("--texture-gamma", type=float, default=0.65)
    parser.add_argument("--texture-contrast", type=float, default=1.08)
    parser.add_argument("--min-texture-colorfulness", type=float, default=2.0)
    parser.add_argument(
        "--allow-grayscale-texture",
        action="store_true",
        help="Allow a grayscale diagnostic texture artifact when the captured keyframes have no RGB chroma.",
    )
    parser.add_argument("--min-mapanything-coverage", type=float, default=0.02)
    parser.add_argument("--min-zero-planes", type=int, default=2)
    parser.add_argument("--min-registration-correspondences", type=int, default=3)
    parser.add_argument("--max-room-model-leakage", type=float, default=0.03)
    parser.add_argument("--no-update-latest", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(name)s: %(message)s")
    try:
        result = build(args)
    except (VirtualTwinBuildError, ZeroPlaneAdapterError) as exc:
        LOGGER.error("%s", exc)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
