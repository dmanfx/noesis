#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.depth_source import MapAnythingDepthSource
from mapanything_config import load_service_config
from noesis.calibration.depth_registration import (
    calibration_fingerprint_from_snapshot,
    write_depth_registration_bundle,
)
from noesis.calibration.depth_registration_builder import (
    DepthRegistrationBuildError,
    build_registration_entry,
)
from noesis.depth_tracking_materialization import (
    DA2_CHECKPOINT_PATH,
    DA2_REPO_DIR,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL_NAME,
)
from noesis.ds8_runtime import (
    _CalibrationProvider,
    _build_depth_registration_profile_fingerprints,
    _load_camera_labels,
)


LOGGER = logging.getLogger("depth_registration_builder")


@dataclass(frozen=True, slots=True)
class _DenseFramePair:
    frame_bgr: np.ndarray
    da2_depth: np.ndarray
    ma_depth: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class _MapAnythingSnapshot:
    path: Path
    ts_us: int
    depth: np.ndarray
    conf: np.ndarray
    mask: np.ndarray


class _Da2Runner:
    def __init__(self, *, device: str = "cuda") -> None:
        import torch
        import torch.nn as nn

        if str(DA2_REPO_DIR) not in sys.path:
            sys.path.insert(0, str(DA2_REPO_DIR))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
            from depth_anything_v2.dinov2_layers import attention as da2_attention  # type: ignore
            from depth_anything_v2.dinov2_layers import block as da2_block  # type: ignore
        finally:
            if str(DA2_REPO_DIR) in sys.path:
                sys.path.remove(str(DA2_REPO_DIR))

        da2_attention.XFORMERS_AVAILABLE = False
        da2_block.XFORMERS_AVAILABLE = False

        class _Wrapper(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
                self.register_buffer("mean", mean)
                self.register_buffer("std", std)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
                x = x.float().div(255.0)
                x = (x - self.mean) / self.std
                return self.inner(x)

        model = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            max_depth=20.0,
        )
        state = torch.load(DA2_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self._torch = torch
        self._device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self._model = _Wrapper(model.eval()).eval().to(self._device)
        self._input_size = (int(DEFAULT_INPUT_SIZE[0]), int(DEFAULT_INPUT_SIZE[1]))
        self.profile = {
            "model_name": DEFAULT_MODEL_NAME,
            "input_size": list(self._input_size),
            "runtime_device": str(self._device),
        }

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        torch = self._torch
        width, height = self._input_size
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            depth = self._model(tensor)
        if hasattr(depth, "detach"):
            depth_np = depth.detach().float().cpu().numpy()[0, 0]
        else:
            depth_np = np.asarray(depth, dtype=np.float32)[0, 0]
        return cv2.resize(depth_np.astype(np.float32, copy=False), (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def _video_capture_from_uri(uri: str) -> cv2.VideoCapture:
    if uri.startswith("file://"):
        return cv2.VideoCapture(uri[7:])
    return cv2.VideoCapture(uri)


def _iter_source_frames(
    *,
    uri: str,
    frame_stride: int,
    max_frames_read: int,
) -> Iterable[tuple[int, np.ndarray]]:
    cap = _video_capture_from_uri(uri)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source URI: {uri}")
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


def _resolve_pipeline_cfg_path(pipeline_config_path: Path, value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return Path("")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    if raw.startswith(("config/", "models/", "pipelines/")):
        return (REPO_ROOT / candidate).resolve()
    return (Path(pipeline_config_path).parent / candidate).resolve()


def _semicolon_floats(raw: str, *, expected: int, key: str) -> list[float]:
    values = [part.strip() for part in str(raw or "").split(";") if part.strip()]
    if len(values) < expected:
        raise RuntimeError(f"dewarper {key} requires {expected} values; got {len(values)}")
    return [float(value) for value in values[:expected]]


def _dewarper_frame_rectifier(
    *,
    source_cfg: Mapping[str, Any],
    pipeline_config_path: Path,
) -> Callable[[np.ndarray], np.ndarray] | None:
    dewarp_cfg = source_cfg.get("dewarper")
    if not (isinstance(dewarp_cfg, dict) and bool(dewarp_cfg.get("enable", False))):
        return None

    config_raw = str(dewarp_cfg.get("config-file") or "").strip()
    if not config_raw:
        raise RuntimeError("dewarper enabled but source has no config-file")
    config_path = _resolve_pipeline_cfg_path(pipeline_config_path, config_raw)
    if not config_path.exists():
        raise RuntimeError(f"dewarper config not found: {config_path}")

    parser = configparser.ConfigParser(inline_comment_prefixes=("#",), strict=False)
    parser.optionxform = str
    parser.read(config_path, encoding="utf-8")
    if "property" not in parser or "surface0" not in parser:
        raise RuntimeError(f"dewarper config missing [property] or [surface0]: {config_path}")

    props = parser["property"]
    surface = parser["surface0"]
    projection_type = int(float(surface.get("projection-type", "0")))
    if projection_type != 4:
        LOGGER.warning(
            "Depth-registration builder can only mirror FISH_PERSPECTIVE dewarper configs; "
            "skipping CPU rectification for projection-type=%s config=%s",
            projection_type,
            config_path,
        )
        return None

    output_w = int(float(props.get("output-width", surface.get("width", "0"))))
    output_h = int(float(props.get("output-height", surface.get("height", "0"))))
    calib_w = int(float(surface.get("width", str(output_w))))
    calib_h = int(float(surface.get("height", str(output_h))))
    if output_w <= 0 or output_h <= 0 or calib_w <= 0 or calib_h <= 0:
        raise RuntimeError(f"dewarper config has invalid dimensions: {config_path}")

    src_focal = _semicolon_floats(surface.get("focal-length", ""), expected=2, key="focal-length")
    distortion = _semicolon_floats(surface.get("distortion", ""), expected=4, key="distortion")
    dst_focal = _semicolon_floats(
        surface.get("dst-focal-length", surface.get("focal-length", "")),
        expected=2,
        key="dst-focal-length",
    )
    dst_pp = _semicolon_floats(
        surface.get(
            "dst-principal-point",
            f"{(output_w - 1) * 0.5};{(output_h - 1) * 0.5}",
        ),
        expected=2,
        key="dst-principal-point",
    )

    src_k = np.array(
        [
            [float(src_focal[0]), 0.0, float(surface.get("src-x0"))],
            [0.0, float(src_focal[1]), float(surface.get("src-y0"))],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dst_k = np.array(
        [
            [float(dst_focal[0]), 0.0, float(dst_pp[0])],
            [0.0, float(dst_focal[1]), float(dst_pp[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    d = np.asarray(distortion, dtype=np.float64).reshape(4, 1)
    maps_by_shape: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    warned_shape_mismatch = False

    LOGGER.info(
        "Depth-registration builder will mirror dewarper config=%s output=%dx%d",
        config_path,
        output_w,
        output_h,
    )

    def _rectify(frame_bgr: np.ndarray) -> np.ndarray:
        nonlocal warned_shape_mismatch
        frame_h, frame_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        key = (frame_w, frame_h)
        maps = maps_by_shape.get(key)
        if maps is None:
            k = src_k.copy()
            if (frame_w, frame_h) != (calib_w, calib_h):
                if not warned_shape_mismatch:
                    LOGGER.warning(
                        "Dewarper config %s calibrated for %dx%d but source frames are %dx%d; scaling source K",
                        config_path,
                        calib_w,
                        calib_h,
                        frame_w,
                        frame_h,
                    )
                    warned_shape_mismatch = True
                sx = float(frame_w) / float(calib_w)
                sy = float(frame_h) / float(calib_h)
                k[0, 0] *= sx
                k[0, 2] *= sx
                k[1, 1] *= sy
                k[1, 2] *= sy
            maps = cv2.fisheye.initUndistortRectifyMap(
                k,
                d,
                np.eye(3, dtype=np.float64),
                dst_k,
                (output_w, output_h),
                cv2.CV_16SC2,
            )
            maps_by_shape[key] = maps
        map1, map2 = maps
        return cv2.remap(
            frame_bgr,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    return _rectify


def _dense_pair_from_frame(
    *,
    camera_id: str,
    frame_bgr: np.ndarray,
    da2_runner: _Da2Runner,
    map_source: MapAnythingDepthSource,
    calib_bundle: dict[str, Any],
    sample_pixel_step: int,
    row_start_frac: float,
    frame_index: int,
) -> _DenseFramePair:
    timestamp_s = time.time() + (float(frame_index) * 2.0)
    ma_result = map_source.maybe_infer_mono(camera_id, frame_bgr, calib_bundle, timestamp_s=timestamp_s)
    if ma_result is None:
        raise DepthRegistrationBuildError(f"MapAnything inference did not produce a frame for camera {camera_id}")
    da2_depth = da2_runner.infer(frame_bgr)
    ma_depth = np.asarray(ma_result.depth, dtype=np.float32)
    ma_conf = np.asarray(ma_result.conf, dtype=np.float32)
    ma_mask = np.asarray(ma_result.mask, dtype=bool)
    if da2_depth.shape != ma_depth.shape:
        raise DepthRegistrationBuildError(
            f"Shape mismatch between DAv2 {da2_depth.shape} and MapAnything {ma_depth.shape} for camera {camera_id}"
        )
    row_start = int(max(0, min(ma_depth.shape[0] - 1, round(float(row_start_frac) * float(ma_depth.shape[0])))))
    valid = np.isfinite(da2_depth) & np.isfinite(ma_depth) & (da2_depth > 0.0) & (ma_depth > 0.1)
    valid &= ma_mask
    valid &= ma_conf >= float(map_source.min_conf)
    if row_start > 0:
        rows = np.arange(ma_depth.shape[0], dtype=np.int32)[:, None]
        valid &= rows >= row_start
    if sample_pixel_step > 1:
        grid = np.zeros_like(valid, dtype=bool)
        grid[::sample_pixel_step, ::sample_pixel_step] = True
        valid &= grid
    if not np.any(valid):
        raise DepthRegistrationBuildError(f"No valid dense overlap samples for camera {camera_id}")
    return _DenseFramePair(
        frame_bgr=np.asarray(frame_bgr, dtype=np.uint8),
        da2_depth=np.asarray(da2_depth, dtype=np.float32),
        ma_depth=np.asarray(ma_depth, dtype=np.float32),
        valid_mask=np.asarray(valid, dtype=bool),
    )


def _snapshot_entries(depth_base: Path, camera_id: str) -> list[tuple[int, Path]]:
    camera_dir = Path(depth_base) / str(camera_id)
    entries: list[tuple[int, Path]] = []
    if not camera_dir.exists():
        return entries
    for path in camera_dir.rglob("*.zarr"):
        try:
            ts_us = int(path.stem)
        except ValueError:
            continue
        entries.append((ts_us, path))
    entries.sort(key=lambda item: item[0])
    return entries


def _load_mapanything_snapshot(path: Path) -> _MapAnythingSnapshot:
    try:
        group = zarr.open_group(str(path), mode="r")
        depth = np.asarray(np.array(group["depth_z"]), dtype=np.float32)
        conf = np.asarray(np.array(group["conf"]), dtype=np.float32)
        mask = np.asarray(np.array(group["mask"]), dtype=bool)
    except Exception as exc:
        raise DepthRegistrationBuildError(f"failed to load MapAnything snapshot {path}: {exc}") from exc
    if depth.ndim != 2 or conf.shape != depth.shape or mask.shape != depth.shape:
        raise DepthRegistrationBuildError(
            f"MapAnything snapshot {path} has incompatible shapes: depth={depth.shape} conf={conf.shape} mask={mask.shape}"
        )
    try:
        ts_us = int(group.attrs.get("timestamp_us", path.stem))
    except Exception:
        ts_us = int(path.stem)
    return _MapAnythingSnapshot(
        path=Path(path),
        ts_us=ts_us,
        depth=depth,
        conf=conf,
        mask=mask,
    )


def _valid_snapshot_coverage(snapshot: _MapAnythingSnapshot, *, min_conf: float) -> float:
    valid = (
        np.asarray(snapshot.mask, dtype=bool)
        & np.isfinite(snapshot.depth)
        & (np.asarray(snapshot.depth, dtype=np.float32) > 0.0)
        & np.isfinite(snapshot.conf)
    )
    if np.isfinite(float(min_conf)):
        valid &= np.asarray(snapshot.conf, dtype=np.float32) >= float(min_conf)
    return float(np.count_nonzero(valid)) / float(valid.size) if valid.size else 0.0


def _latest_mapanything_snapshots(
    *,
    depth_base: Path,
    camera_id: str,
    count: int,
    min_conf: float,
) -> list[_MapAnythingSnapshot]:
    snapshots: list[_MapAnythingSnapshot] = []
    for _ts_us, path in reversed(_snapshot_entries(depth_base, camera_id)):
        try:
            snapshot = _load_mapanything_snapshot(path)
        except DepthRegistrationBuildError as exc:
            LOGGER.debug("Skipping incomplete MapAnything snapshot %s: %s", path, exc)
            continue
        if _valid_snapshot_coverage(snapshot, min_conf=min_conf) <= 0.0:
            continue
        snapshots.append(snapshot)
        if len(snapshots) >= max(1, int(count)):
            break
    snapshots.reverse()
    return snapshots


def _resize_snapshot_to_frame(
    snapshot: _MapAnythingSnapshot,
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_h, frame_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    if snapshot.depth.shape == (frame_h, frame_w):
        return snapshot.depth, snapshot.conf, snapshot.mask
    depth = cv2.resize(snapshot.depth, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    conf = cv2.resize(snapshot.conf, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(
        snapshot.mask.astype(np.uint8, copy=False),
        (frame_w, frame_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    return depth.astype(np.float32, copy=False), conf.astype(np.float32, copy=False), mask


def _dense_pair_from_snapshot(
    *,
    camera_id: str,
    frame_bgr: np.ndarray,
    da2_runner: _Da2Runner,
    snapshot: _MapAnythingSnapshot,
    sample_pixel_step: int,
    row_start_frac: float,
    min_conf: float,
) -> _DenseFramePair:
    da2_depth = da2_runner.infer(frame_bgr)
    ma_depth, ma_conf, ma_mask = _resize_snapshot_to_frame(snapshot, frame_bgr)
    if da2_depth.shape != ma_depth.shape:
        raise DepthRegistrationBuildError(
            f"Shape mismatch between DAv2 {da2_depth.shape} and persisted MapAnything {ma_depth.shape} for camera {camera_id}"
        )
    row_start = int(max(0, min(ma_depth.shape[0] - 1, round(float(row_start_frac) * float(ma_depth.shape[0])))))
    valid = np.isfinite(da2_depth) & np.isfinite(ma_depth) & (da2_depth > 0.0) & (ma_depth > 0.1)
    valid &= np.asarray(ma_mask, dtype=bool)
    valid &= np.asarray(ma_conf, dtype=np.float32) >= float(min_conf)
    if row_start > 0:
        rows = np.arange(ma_depth.shape[0], dtype=np.int32)[:, None]
        valid &= rows >= row_start
    if sample_pixel_step > 1:
        grid = np.zeros_like(valid, dtype=bool)
        grid[::sample_pixel_step, ::sample_pixel_step] = True
        valid &= grid
    if not np.any(valid):
        raise DepthRegistrationBuildError(f"No valid dense overlap samples for camera {camera_id}")
    return _DenseFramePair(
        frame_bgr=np.asarray(frame_bgr, dtype=np.uint8),
        da2_depth=np.asarray(da2_depth, dtype=np.float32),
        ma_depth=np.asarray(ma_depth, dtype=np.float32),
        valid_mask=np.asarray(valid, dtype=bool),
    )


def _stable_pixel_masks(
    frame_pairs: Sequence[_DenseFramePair],
    *,
    max_luma_mad: float = 12.0,
    max_depth_delta_m: float = 0.75,
) -> list[np.ndarray]:
    if not frame_pairs:
        return []
    if len(frame_pairs) == 1:
        return [np.ones_like(frame_pairs[0].valid_mask, dtype=bool)]

    gray_stack = np.stack(
        [cv2.cvtColor(pair.frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) for pair in frame_pairs],
        axis=0,
    )
    gray_median = np.median(gray_stack, axis=0)
    gray_mad = np.median(np.abs(gray_stack - gray_median[None, ...]), axis=0)

    da_stack = np.stack([pair.da2_depth.astype(np.float32, copy=False) for pair in frame_pairs], axis=0)
    ma_stack = np.stack([pair.ma_depth.astype(np.float32, copy=False) for pair in frame_pairs], axis=0)
    da_median = np.median(da_stack, axis=0)
    ma_median = np.median(ma_stack, axis=0)

    masks: list[np.ndarray] = []
    for idx, pair in enumerate(frame_pairs):
        gray_stable = gray_mad <= float(max_luma_mad)
        da_stable = np.abs(pair.da2_depth - da_median) <= float(max_depth_delta_m)
        ma_stable = np.abs(pair.ma_depth - ma_median) <= float(max_depth_delta_m)
        stable = pair.valid_mask & gray_stable & da_stable & ma_stable
        # A small close-range dynamic region should not poison the whole frame; keep
        # only if there is still meaningful support after stability filtering.
        if int(np.count_nonzero(stable)) < 4096:
            stable = pair.valid_mask
        masks.append(stable)
        LOGGER.info(
            "Temporal stability mask frame=%d retained=%d base_valid=%d",
            idx + 1,
            int(np.count_nonzero(stable)),
            int(np.count_nonzero(pair.valid_mask)),
        )
    return masks


def _flatten_stable_pairs(
    frame_pairs: Sequence[_DenseFramePair],
    stable_masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    raw_samples: list[np.ndarray] = []
    ref_samples: list[np.ndarray] = []
    for pair, stable in zip(frame_pairs, stable_masks):
        stable_mask = np.asarray(stable, dtype=bool)
        valid = pair.valid_mask & stable_mask
        if not np.any(valid):
            continue
        raw_samples.append(pair.da2_depth[valid].reshape(-1))
        ref_samples.append(pair.ma_depth[valid].reshape(-1))
    if not raw_samples or not ref_samples:
        raise DepthRegistrationBuildError("No stable dense overlap samples remain after temporal filtering")
    return np.concatenate(raw_samples, axis=0), np.concatenate(ref_samples, axis=0)


def _selected_cameras(
    pipeline_cfg: dict[str, Any],
    camera_labels: dict[int, str],
    selected: list[str] | None,
) -> list[tuple[int, str, str, Mapping[str, Any]]]:
    sources = pipeline_cfg.get("sources") or []
    if not isinstance(sources, list):
        raise RuntimeError("pipeline config missing sources list")
    selected_set = {str(v).strip() for v in selected or [] if str(v).strip()}
    result: list[tuple[int, str, str, Mapping[str, Any]]] = []
    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        camera_id = str(camera_labels.get(int(idx), f"camera_{idx}")).strip()
        if selected_set and camera_id not in selected_set and str(idx) not in selected_set:
            continue
        uri = str(source.get("uri") or "").strip()
        if not uri:
            continue
        result.append((int(idx), camera_id, uri, source))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DS8 DAv2-to-MapAnything depth-registration artifacts.")
    parser.add_argument("--pipeline-config", type=Path, default=REPO_ROOT / "config" / "infer.yaml")
    parser.add_argument("--cameras-config", type=Path, default=REPO_ROOT / "config" / "cameras.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "config" / "depth_registration.json")
    parser.add_argument("--camera", action="append", default=None, help="Camera id/name to process (repeatable).")
    parser.add_argument("--frames-per-camera", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-frames-read", type=int, default=240)
    parser.add_argument("--sample-pixel-step", type=int, default=6)
    parser.add_argument("--row-start-frac", type=float, default=0.35)
    parser.add_argument("--max-luma-mad", type=float, default=12.0, help="Temporal luma MAD threshold for stable support masking.")
    parser.add_argument("--max-depth-delta-m", type=float, default=0.75, help="Per-pixel temporal depth delta threshold for stable support masking.")
    parser.add_argument(
        "--mapanything-source",
        choices=("snapshots", "service"),
        default="snapshots",
        help="Use persisted DS8 MapAnything Zarr snapshots by default; service mode is explicit one-off maintenance.",
    )
    parser.add_argument(
        "--mapanything-depth-base",
        type=Path,
        default=None,
        help="Depth snapshot root produced by the DS8 runtime MapAnything branch.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    pipeline_cfg = json.loads(Path(args.pipeline_config).read_text(encoding="utf-8")) if args.pipeline_config.suffix == ".json" else None
    if pipeline_cfg is None:
        import yaml

        pipeline_cfg = yaml.safe_load(Path(args.pipeline_config).read_text(encoding="utf-8")) or {}
    if not isinstance(pipeline_cfg, dict):
        raise RuntimeError(f"Pipeline config must decode to a mapping: {args.pipeline_config}")

    camera_labels = _load_camera_labels(Path(args.cameras_config))
    calibration_provider = _CalibrationProvider(Path(args.cameras_config), pipeline_cfg, tracking_mode="baseline")
    calibration_provider.set_camera_labels(camera_labels)
    calib_bundle = calibration_provider.calibration_bundle()
    service_cfg = load_service_config()
    map_source = MapAnythingDepthSource(service_cfg) if args.mapanything_source == "service" else None
    snapshot_depth_base = Path(args.mapanything_depth_base or service_cfg.storage.depth_base)
    min_conf = float(service_cfg.performance.min_conf)
    da2_runner = _Da2Runner()
    depth_profile, mapanything_profile = _build_depth_registration_profile_fingerprints(
        pipeline_cfg,
        pipeline_path=Path(args.pipeline_config).resolve(),
    )
    entries: dict[str, Any] = {}
    try:
        for source_id, camera_id, uri, source_cfg in _selected_cameras(pipeline_cfg, camera_labels, args.camera):
            LOGGER.info("Building depth registration for camera=%s uri=%s", camera_id, uri)
            snapshot = calibration_provider.snapshot(int(source_id), camera_id)
            if snapshot is None:
                raise RuntimeError(f"Calibration unavailable for camera {camera_id}")
            rectifier = _dewarper_frame_rectifier(
                source_cfg=source_cfg,
                pipeline_config_path=Path(args.pipeline_config),
            )
            frame_pairs: list[_DenseFramePair] = []
            if args.mapanything_source == "snapshots":
                snapshots = _latest_mapanything_snapshots(
                    depth_base=snapshot_depth_base,
                    camera_id=camera_id,
                    count=int(args.frames_per_camera),
                    min_conf=min_conf,
                )
                if not snapshots:
                    raise RuntimeError(
                        f"No persisted DS8 MapAnything snapshots found for camera {camera_id} under {snapshot_depth_base}"
                    )
                LOGGER.info(
                    "Using %d persisted DS8 MapAnything snapshots for %s from %s",
                    len(snapshots),
                    camera_id,
                    snapshot_depth_base,
                )
                snapshot_index = 0
                for frame_index, frame_bgr in _iter_source_frames(
                    uri=uri,
                    frame_stride=int(args.frame_stride),
                    max_frames_read=int(args.max_frames_read),
                ):
                    if rectifier is not None:
                        frame_bgr = rectifier(frame_bgr)
                    ma_snapshot = snapshots[snapshot_index]
                    try:
                        pair = _dense_pair_from_snapshot(
                            camera_id=camera_id,
                            frame_bgr=frame_bgr,
                            da2_runner=da2_runner,
                            snapshot=ma_snapshot,
                            sample_pixel_step=max(1, int(args.sample_pixel_step)),
                            row_start_frac=float(args.row_start_frac),
                            min_conf=min_conf,
                        )
                    except DepthRegistrationBuildError as exc:
                        LOGGER.warning("Skipping frame %s for %s: %s", frame_index, camera_id, exc)
                        continue
                    frame_pairs.append(pair)
                    LOGGER.info(
                        "Collected registration frame %d/%d for %s from snapshot=%s (%d samples)",
                        len(frame_pairs),
                        len(snapshots),
                        camera_id,
                        ma_snapshot.path,
                        int(np.count_nonzero(pair.valid_mask)),
                    )
                    snapshot_index += 1
                    if snapshot_index >= len(snapshots):
                        break
            else:
                if map_source is None:
                    raise RuntimeError("MapAnything service source selected but source client is unavailable")
                for frame_index, frame_bgr in _iter_source_frames(
                    uri=uri,
                    frame_stride=int(args.frame_stride),
                    max_frames_read=int(args.max_frames_read),
                ):
                    if rectifier is not None:
                        frame_bgr = rectifier(frame_bgr)
                    try:
                        pair = _dense_pair_from_frame(
                            camera_id=camera_id,
                            frame_bgr=frame_bgr,
                            da2_runner=da2_runner,
                            map_source=map_source,
                            calib_bundle=calib_bundle,
                            sample_pixel_step=max(1, int(args.sample_pixel_step)),
                            row_start_frac=float(args.row_start_frac),
                            frame_index=frame_index,
                        )
                    except DepthRegistrationBuildError as exc:
                        LOGGER.warning("Skipping frame %s for %s: %s", frame_index, camera_id, exc)
                        continue
                    frame_pairs.append(pair)
                    LOGGER.info(
                        "Collected registration frame %d/%d for %s (%d samples)",
                        len(frame_pairs),
                        int(args.frames_per_camera),
                        camera_id,
                        int(np.count_nonzero(pair.valid_mask)),
                    )
                    if len(frame_pairs) >= int(args.frames_per_camera):
                        break
            if not frame_pairs:
                raise RuntimeError(f"No usable registration frames were collected for camera {camera_id}")
            stable_masks = _stable_pixel_masks(
                frame_pairs,
                max_luma_mad=float(args.max_luma_mad),
                max_depth_delta_m=float(args.max_depth_delta_m),
            )
            raw_depth_m, registered_depth_m = _flatten_stable_pairs(frame_pairs, stable_masks)
            entry = build_registration_entry(
                camera_id=camera_id,
                calibration_fingerprint=calibration_fingerprint_from_snapshot(snapshot),
                dav2_profile=depth_profile | {"builder_runtime_device": da2_runner.profile["runtime_device"]},
                mapanything_profile=mapanything_profile,
                provenance={
                    "source_uri": uri,
                    "mapanything_source": str(args.mapanything_source),
                    "mapanything_depth_base": str(snapshot_depth_base) if args.mapanything_source == "snapshots" else None,
                    "frames_per_camera": int(len(frame_pairs)),
                    "sample_pixel_step": int(args.sample_pixel_step),
                    "row_start_frac": float(args.row_start_frac),
                    "max_luma_mad": float(args.max_luma_mad),
                    "max_depth_delta_m": float(args.max_depth_delta_m),
                },
                raw_depth_m=raw_depth_m,
                registered_depth_m=registered_depth_m,
            )
            entries[camera_id] = entry
        if not entries:
            raise RuntimeError("No depth registration entries were built")
        write_depth_registration_bundle(path=Path(args.output), entries=entries)
        LOGGER.info("Wrote depth registration bundle: %s", Path(args.output).resolve())
        return 0
    finally:
        try:
            map_source.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
