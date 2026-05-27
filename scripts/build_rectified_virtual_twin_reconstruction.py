#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapanything_config import load_service_config
from noesis.ds8_runtime import _CalibrationProvider, _load_camera_labels
from noesis.virtual_twin.builder import (
    VirtualTwinBuildError,
    VirtualTwinFrameInput,
    build_virtual_twin_revision,
    revision_id_from_clock,
)
from noesis.virtual_twin.store import VirtualTwinStore
from noesis.virtual_twin.zeroplane_adapter import ZeroPlaneAdapterError, ZeroPlaneCommandAdapter
from scripts.build_virtual_twin_reconstruction import (
    _camera_source,
    _initial_world_to_scene_from_alignment,
    _initial_world_to_scene_from_bundle,
    _iter_source_frames,
    _load_yaml,
    _model_refs,
    _scaled_snapshot,
)

try:
    from mapanything.models import MapAnything  # type: ignore
    from mapanything.utils.image import preprocess_inputs  # type: ignore
except ImportError as exc:  # pragma: no cover
    MapAnything = None  # type: ignore
    preprocess_inputs = None  # type: ignore
    MAPANYTHING_IMPORT_ERROR = exc
else:
    MAPANYTHING_IMPORT_ERROR = None


LOGGER = logging.getLogger("rectified_virtual_twin_builder")


@dataclass
class PreparedFrame:
    frame_id: str
    frame_path: Path
    source_ref: str
    image_size: tuple[int, int]
    image_bgr: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray
    calibration: Any
    intrinsics: np.ndarray
    manifest_row: dict[str, Any]


def _load_camera_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise VirtualTwinBuildError(f"expected camera config mapping at {path}")
    return payload


def _camera_model_name(cameras_cfg: dict[str, Any], camera_labels: dict[int, str], camera: str) -> str:
    cameras = cameras_cfg.get("cameras") or {}
    for source_id, label in camera_labels.items():
        if label != camera:
            continue
        row = cameras.get(int(source_id)) or cameras.get(str(source_id))
        if isinstance(row, dict) and str(row.get("model") or "").strip():
            return str(row["model"]).strip()
    raise VirtualTwinBuildError(f"camera {camera!r} has no intrinsics model in cameras config")


def _fisheye_distortion_for_camera(cameras_cfg: dict[str, Any], camera_labels: dict[int, str], camera: str) -> np.ndarray:
    model_name = _camera_model_name(cameras_cfg, camera_labels, camera)
    model = (cameras_cfg.get("intrinsics_models") or {}).get(model_name)
    if not isinstance(model, dict):
        raise VirtualTwinBuildError(f"intrinsics model {model_name!r} is missing")
    calibration = model.get("calibration") or {}
    distortion_model = str(calibration.get("distortion_model") or calibration.get("model") or "").lower()
    coeffs = calibration.get("distortion_coeffs")
    if "fisheye" not in distortion_model or not isinstance(coeffs, list) or len(coeffs) != 4:
        raise VirtualTwinBuildError(
            f"camera {camera} model {model_name} does not expose a 4-coefficient OpenCV fisheye calibration"
        )
    values = np.asarray([float(x) for x in coeffs], dtype=np.float64).reshape(4, 1)
    if not np.all(np.isfinite(values)):
        raise VirtualTwinBuildError(f"camera {camera} fisheye coefficients contain non-finite values")
    return values


def _iter_revision_keyframes(source_revision: Path, *, keyframes: int) -> Iterable[tuple[int, np.ndarray, Path]]:
    paths = sorted((Path(source_revision) / "keyframes").glob("*.png"))
    if not paths:
        raise VirtualTwinBuildError(f"source revision has no keyframes under {Path(source_revision) / 'keyframes'}")
    for idx, path in enumerate(paths[: int(keyframes)], start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise VirtualTwinBuildError(f"failed to read source keyframe {path}")
        yield idx, image, path


def _iter_capture_keyframes(*, uri: str, frame_stride: int, max_frames_read: int, keyframes: int) -> Iterable[tuple[int, np.ndarray, None]]:
    count = 0
    for local_index, frame in _iter_source_frames(uri=uri, frame_stride=frame_stride, max_frames_read=max_frames_read):
        count += 1
        if count > int(keyframes):
            break
        yield int(local_index), frame, None


def _rectification_maps(
    *,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    balance: float,
    fov_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = (int(image_size[0]), int(image_size[1]))
    k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    d = np.asarray(distortion, dtype=np.float64).reshape(4, 1)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k,
        d,
        size,
        np.eye(3, dtype=np.float64),
        balance=float(balance),
        new_size=size,
        fov_scale=float(fov_scale),
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        k,
        d,
        np.eye(3, dtype=np.float64),
        new_k,
        size,
        cv2.CV_16SC2,
    )
    return new_k.astype(np.float64), map1, map2


def _squeeze_to_array(value: Any, *, channel_last: bool) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        array = array[..., 0] if channel_last else array[0]
    return array


def _resize_mapanything_output(
    depth: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    frame_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    if depth.shape == (frame_h, frame_w) and confidence.shape == (frame_h, frame_w) and mask.shape == (frame_h, frame_w):
        return depth.astype(np.float32), confidence.astype(np.float32), mask.astype(bool)
    size = (frame_w, frame_h)
    return (
        cv2.resize(depth.astype(np.float32), size, interpolation=cv2.INTER_LINEAR).astype(np.float32),
        cv2.resize(confidence.astype(np.float32), size, interpolation=cv2.INTER_LINEAR).astype(np.float32),
        cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool),
    )


class DirectMapAnythingRunner:
    def __init__(self, *, model_id: str, device: str, amp_dtype: str) -> None:
        if MapAnything is None or preprocess_inputs is None:
            raise VirtualTwinBuildError(f"MapAnything is unavailable: {MAPANYTHING_IMPORT_ERROR}")
        self._amp_dtype = str(amp_dtype)
        self._device = torch.device(device)
        LOGGER.info("loading MapAnything model %s on %s", model_id, self._device)
        self._model = MapAnything.from_pretrained(model_id, cache_dir=os.environ.get("MAPANYTHING_WEIGHTS"))  # type: ignore[operator]
        self._model.to(self._device)
        self._model.eval()

    def infer(
        self,
        *,
        image_bgr: np.ndarray,
        intrinsics: np.ndarray,
        memory_efficient: bool,
        apply_mask: bool,
        mask_edges: bool,
        confidence_percentile: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        prepared = preprocess_inputs(  # type: ignore[operator]
            [{"img": rgb, "intrinsics": np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)}],
            resize_mode="fixed_mapping",
        )
        with torch.inference_mode():
            predictions = self._model.infer(  # type: ignore[attr-defined]
                prepared,
                memory_efficient_inference=bool(memory_efficient),
                use_amp=True,
                amp_dtype=self._amp_dtype,
                apply_mask=bool(apply_mask),
                mask_edges=bool(mask_edges),
                confidence_percentile=float(confidence_percentile),
            )
        if not predictions:
            raise VirtualTwinBuildError("MapAnything returned no predictions")
        pred = predictions[0]
        if pred.get("depth_z") is None or pred.get("conf") is None:
            raise VirtualTwinBuildError("MapAnything prediction missing depth_z or conf")
        depth = _squeeze_to_array(pred["depth_z"], channel_last=True).astype(np.float32)
        conf = _squeeze_to_array(pred["conf"], channel_last=False).astype(np.float32)
        mask = (
            _squeeze_to_array(pred["mask"], channel_last=True).astype(bool)
            if pred.get("mask") is not None
            else np.ones_like(depth, dtype=bool)
        )
        return depth, conf, mask


def _rectified_snapshot(snapshot: Any, intrinsics: np.ndarray, image_size: tuple[int, int]) -> Any:
    return replace(snapshot, intrinsics=np.asarray(intrinsics, dtype=np.float64).reshape(3, 3), image_size=image_size)


def build(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_cfg = _load_yaml(Path(args.pipeline_config))
    cameras_cfg = _load_camera_yaml(Path(args.cameras_config))
    camera_labels = _load_camera_labels(Path(args.cameras_config))
    source_id, uri = _camera_source(pipeline_cfg, camera_labels, args.camera)

    model_obj_raw = str(args.model_obj or os.environ.get("NOESIS_MENON_STRUCTURAL_OBJ", "")).strip()
    if not model_obj_raw:
        raise VirtualTwinBuildError("Menon structural OBJ is required via --model-obj or NOESIS_MENON_STRUCTURAL_OBJ")
    model_obj = Path(model_obj_raw).expanduser()
    if not model_obj.exists():
        raise VirtualTwinBuildError(f"Menon structural OBJ does not exist: {model_obj}")

    calibration_provider = _CalibrationProvider(Path(args.cameras_config), pipeline_cfg)
    calibration_provider.set_camera_labels(camera_labels)
    snapshot = calibration_provider.snapshot(source_id, args.camera)
    if snapshot is None:
        raise VirtualTwinBuildError(f"calibration snapshot unavailable for {args.camera}")
    calib_bundle = calibration_provider.calibration_bundle()
    initial_world_to_scene = _initial_world_to_scene_from_bundle(calib_bundle)
    if initial_world_to_scene is None:
        initial_world_to_scene = _initial_world_to_scene_from_alignment(Path(args.alignment_config))
    if initial_world_to_scene is None:
        raise VirtualTwinBuildError(
            "calibration lacks scene_similarity.world_to_scene_col_major; "
            "virtual-twin registration needs the existing Menon scene similarity as its explicit prior"
        )

    revision_id = args.revision_id or revision_id_from_clock("vt_living_room_rectified")
    store = VirtualTwinStore(Path(args.output_root) if args.output_root else None)
    staging_dir = Path(store.root) / "staging" / revision_id
    frames_dir = staging_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    distortion = _fisheye_distortion_for_camera(cameras_cfg, camera_labels, args.camera)
    if args.source_revision:
        source_revision = Path(args.source_revision)
        if not source_revision.is_absolute():
            source_revision = (VirtualTwinStore().root / "revisions" / str(args.source_revision)).resolve()
        source_frames = _iter_revision_keyframes(source_revision, keyframes=int(args.keyframes))
        source_description = str(source_revision)
    else:
        source_frames = _iter_capture_keyframes(
            uri=uri,
            frame_stride=int(args.frame_stride),
            max_frames_read=int(args.max_frames_read),
            keyframes=int(args.keyframes),
        )
        source_description = uri

    map_config = load_service_config()
    mapanything = DirectMapAnythingRunner(
        model_id=str(map_config.inference.model_id),
        device=str(args.mapanything_device or map_config.inference.device),
        amp_dtype=str(map_config.inference.amp_dtype_name),
    )

    prepared_frames: list[PreparedFrame] = []
    rectification_rows: list[dict[str, Any]] = []
    for local_index, frame_bgr, source_path in source_frames:
        if len(prepared_frames) >= int(args.keyframes):
            break
        frame_snapshot = _scaled_snapshot(snapshot, frame_bgr.shape)
        image_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        rectified_k, map1, map2 = _rectification_maps(
            intrinsics=frame_snapshot.intrinsics,
            distortion=distortion,
            image_size=image_size,
            balance=float(args.rectify_balance),
            fov_scale=float(args.rectify_fov_scale),
        )
        rectified_bgr = cv2.remap(frame_bgr, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        valid_rectified = cv2.remap(
            np.ones(frame_bgr.shape[:2], dtype=np.uint8),
            map1,
            map2,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        frame_id = f"{args.camera}_{local_index:04d}_rectified"
        frame_path = frames_dir / f"{frame_id}.png"
        if not cv2.imwrite(str(frame_path), rectified_bgr):
            raise VirtualTwinBuildError(f"failed to write rectified keyframe {frame_path}")

        depth, confidence, mask = mapanything.infer(
            image_bgr=rectified_bgr,
            intrinsics=rectified_k,
            memory_efficient=bool(args.mapanything_memory_efficient),
            apply_mask=bool(map_config.inference.apply_mask),
            mask_edges=bool(map_config.inference.mask_edges),
            confidence_percentile=float(map_config.inference.confidence_percentile),
        )
        raw_depth_shape = [int(depth.shape[1]), int(depth.shape[0])]
        depth, confidence, mask = _resize_mapanything_output(depth, confidence, mask, rectified_bgr.shape)
        mask = mask & valid_rectified
        usable = mask & np.isfinite(depth) & (depth > 0.0) & np.isfinite(confidence)
        coverage = float(np.count_nonzero(usable) / max(1, usable.size))
        if coverage <= 0.0:
            raise VirtualTwinBuildError(f"MapAnything produced no usable rectified depth for {frame_id}")

        row = {
            "frame_id": frame_id,
            "source_ref": str(source_path or source_description),
            "source_intrinsics": np.asarray(frame_snapshot.intrinsics, dtype=np.float64).reshape(3, 3).tolist(),
            "rectified_intrinsics": rectified_k.tolist(),
            "valid_rectified_pixel_ratio": float(np.count_nonzero(valid_rectified) / max(1, valid_rectified.size)),
            "mapanything_raw_depth_shape": raw_depth_shape,
            "mapanything_depth_shape": [int(depth.shape[1]), int(depth.shape[0])],
            "mapanything_valid_depth_coverage": coverage,
            "zeroplane_plane_count": None,
        }
        rectification_rows.append(row)
        prepared_frames.append(
            PreparedFrame(
                frame_id=frame_id,
                frame_path=frame_path,
                source_ref=row["source_ref"],
                image_size=image_size,
                image_bgr=rectified_bgr,
                depth=depth,
                confidence=confidence,
                mask=mask,
                calibration=_rectified_snapshot(frame_snapshot, rectified_k, image_size),
                intrinsics=rectified_k,
                manifest_row=row,
            )
        )
        LOGGER.info("prepared %s: rectified valid %.3f, MapAnything coverage %.3f", frame_id, row["valid_rectified_pixel_ratio"], coverage)

    if len(prepared_frames) < int(args.keyframes):
        raise VirtualTwinBuildError(f"prepared {len(prepared_frames)} keyframes, expected {int(args.keyframes)}")

    del mapanything
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    LOGGER.info("released MapAnything before ZeroPlane planar inference")

    zeroplane = ZeroPlaneCommandAdapter(
        repo_dir=Path(args.zeroplane_repo),
        checkpoint_path=Path(args.zeroplane_checkpoint),
        config_path=Path(args.zeroplane_config),
        output_dir=staging_dir / "zeroplane",
        runner_script=Path(args.zeroplane_runner),
        device=args.zeroplane_device,
    )
    frame_inputs: list[VirtualTwinFrameInput] = []
    for prepared in prepared_frames:
        zp_frame = zeroplane.infer(
            frame_id=prepared.frame_id,
            image_path=prepared.frame_path,
            intrinsics=prepared.intrinsics,
            image_size=prepared.image_size,
        )
        prepared.manifest_row["zeroplane_plane_count"] = int(len(zp_frame.planes))
        frame_inputs.append(
            VirtualTwinFrameInput(
                frame_id=prepared.frame_id,
                camera_id=args.camera,
                image_bgr=prepared.image_bgr,
                map_depth=prepared.depth,
                map_confidence=prepared.confidence,
                map_mask=prepared.mask,
                calibration=prepared.calibration,
                plane_candidates=zp_frame.planes,
                source_ref=(
                    f"rectified_rgb={prepared.frame_path}; "
                    f"source_rgb={prepared.source_ref}; "
                    f"rectify_balance={float(args.rectify_balance)}"
                ),
            )
        )
        LOGGER.info("completed %s with %d ZeroPlane planes", prepared.frame_id, len(zp_frame.planes))

    mapanything_refs = {
        "source": "direct_mapanything_on_rectified_rgb",
        "model_id": str(map_config.inference.model_id),
        "device": str(args.mapanything_device or map_config.inference.device),
        "preprocess": "opencv_fisheye_rectify_then_fixed_mapping",
        "rectify_balance": float(args.rectify_balance),
        "rectify_fov_scale": float(args.rectify_fov_scale),
        "fisheye_distortion_coeffs": [float(x) for x in distortion.reshape(-1)],
        "frames": rectification_rows,
        "min_confidence": float(map_config.performance.min_conf),
    }
    return build_virtual_twin_revision(
        frames=frame_inputs,
        menon_obj_path=model_obj,
        store=store,
        revision_id=revision_id,
        camera_label=args.camera,
        mapanything_refs=mapanything_refs,
        zeroplane_refs=_model_refs(args),
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
    parser = argparse.ArgumentParser(
        description="Build an offline virtual twin from fisheye-rectified RGB, fresh MapAnything, and fresh ZeroPlane."
    )
    parser.add_argument("--camera", default="living-room")
    parser.add_argument("--pipeline-config", type=Path, default=REPO_ROOT / "config" / "infer.yaml")
    parser.add_argument("--cameras-config", type=Path, default=REPO_ROOT / "config" / "cameras.yaml")
    parser.add_argument("--alignment-config", type=Path, default=REPO_ROOT / "config" / "ply_alignment.json")
    parser.add_argument("--model-obj", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--revision-id", default=None)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--keyframes", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-frames-read", type=int, default=900)
    parser.add_argument("--rectify-balance", type=float, default=0.0)
    parser.add_argument("--rectify-fov-scale", type=float, default=1.0)
    parser.add_argument("--mapanything-device", default=None)
    parser.add_argument("--mapanything-memory-efficient", action="store_true")
    parser.add_argument("--zeroplane-repo", type=Path, default=REPO_ROOT / "external" / "ZeroPlane")
    parser.add_argument(
        "--zeroplane-config",
        type=Path,
        default=REPO_ROOT / "external" / "ZeroPlane" / "configs" / "ZeroPlaneNYUV2" / "dust3r_large_dpt_bs16_50ep.yaml",
    )
    parser.add_argument(
        "--zeroplane-checkpoint",
        type=Path,
        default=REPO_ROOT / "external" / "ZeroPlane" / "checkpoints" / "dust3r_encoder_released.pth",
    )
    parser.add_argument("--zeroplane-runner", type=Path, default=REPO_ROOT / "scripts" / "run_zeroplane_inference.py")
    parser.add_argument("--zeroplane-device", default="cuda")
    parser.add_argument("--browser-point-budget", type=int, default=150_000)
    parser.add_argument("--texture-tile-px", type=int, default=96)
    parser.add_argument("--min-texture-coverage", type=float, default=0.02)
    parser.add_argument("--texture-exposure", type=float, default=2.2)
    parser.add_argument("--texture-gamma", type=float, default=0.65)
    parser.add_argument("--texture-contrast", type=float, default=1.08)
    parser.add_argument("--min-texture-colorfulness", type=float, default=2.0)
    parser.add_argument("--allow-grayscale-texture", action="store_true")
    parser.add_argument("--min-mapanything-coverage", type=float, default=0.02)
    parser.add_argument("--min-zero-planes", type=int, default=2)
    parser.add_argument("--min-registration-correspondences", type=int, default=3)
    parser.add_argument("--max-room-model-leakage", type=float, default=0.03)
    parser.add_argument("--no-update-latest", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = build(args)
    except (VirtualTwinBuildError, ZeroPlaneAdapterError) as exc:
        LOGGER.error("%s", exc)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
