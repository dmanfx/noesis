#!/usr/bin/env python3
"""Generate SV3DT/MV3DT camInfo YAMLs from Noesis calibration data.

Usage:
  python3 scripts/generate_v3dt_caminfo.py

This writes camInfo_<camera-name>.yml files under config/v3dt/ by default.

Units:
  - `config/camera_calibration.json` extrinsics are stored in METERS.
  - SV3DT camInfo is generated in METERS by default (`NOESIS_V3DT_CAMINFO_WORLD_SCALE=1`).
    You may set `NOESIS_V3DT_CAMINFO_WORLD_SCALE=100` if you want SV3DT world units
    in centimeters, but then you must retune SV3DT world-space noise/thresholds.
  - Baseline defaults (working across all cameras):
    - `NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p`
    - `NOESIS_V3DT_CAMINFO_INVERT_E=1`
    - `NOESIS_V3DT_CAMINFO_Y_FLIP=1`
    - `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy` (swap Y/Z; SV3DT Z-up)
  - `NOESIS_V3DT_CAMINFO_WORLD_AXES` can remap world axes (e.g., `xzy` swaps Y/Z)
    to align calibration conventions with DeepStream's SV3DT world frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.calibration.pose_v1 import normalize_pose_v1, pose_to_E_col_major


def _load_yaml(path: Path) -> Dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def _load_streammux_size(path: Path) -> Tuple[int, int]:
    data = _load_yaml(path)
    streammux = data.get("streammux") or {}
    width = int(streammux.get("width", 1920) or 1920)
    height = int(streammux.get("height", 1080) or 1080)
    return width, height


def _intrinsics_from_model(models: Dict, model_name: str) -> Dict:
    model = models.get(model_name)
    if not model:
        raise KeyError(f"Intrinsics model '{model_name}' not found in cameras.yaml")
    intrinsics = model.get("intrinsics") or {}
    required = ("fx", "fy", "cx", "cy")
    if not all(key in intrinsics for key in required):
        raise KeyError(f"Intrinsics model '{model_name}' missing one of {required}")
    return intrinsics


def _resolution_from_model(model: Optional[Dict]) -> Optional[Tuple[int, int]]:
    if not isinstance(model, dict):
        return None
    res = model.get("resolution")
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        try:
            w = int(res[0])
            h = int(res[1])
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    intr = model.get("intrinsics") or {}
    res = intr.get("resolution")
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        try:
            w = int(res[0])
            h = int(res[1])
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    return None


def _scale_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    target_w: int,
    target_h: int,
    base_res: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float, float, float]:
    # Intrinsics in config/cameras.yaml are specified at the camera's native calibrated
    # pixel resolution. If available, use the explicit base resolution;
    # otherwise infer from the principal point:
    #   base_w ≈ 2*cx, base_h ≈ 2*cy
    # then scale to the streammux output size (target_w/target_h).
    #
    # This keeps scaling consistent when a source is upscaled (e.g., 1280×720 → 1920×1080).
    if base_res:
        base_w = float(base_res[0])
        base_h = float(base_res[1])
    else:
        base_w = 2.0 * float(cx) if cx else 0.0
        base_h = 2.0 * float(cy) if cy else 0.0
    scale_x = float(target_w) / base_w if base_w else 1.0
    scale_y = float(target_h) / base_h if base_h else 1.0
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y


def _projection_matrix(k: np.ndarray, e_col_major: list, target_h: int, invert_e: bool = False) -> np.ndarray:
    e_mat = np.array(e_col_major, dtype=np.float64).reshape((4, 4), order="F")
    if invert_e:
        r = e_mat[:3, :3]
        t = e_mat[:3, 3]
        e_inv = np.eye(4, dtype=np.float64)
        e_inv[:3, :3] = r.T
        e_inv[:3, 3] = -r.T @ t
        e_mat = e_inv
    axis_map = _axis_mapping_from_env()
    if axis_map is not None:
        axis_map_4 = np.eye(4, dtype=np.float64)
        axis_map_4[:3, :3] = axis_map
        e_mat = e_mat @ axis_map_4
    # World scale: 1.0 = meters (Noesis canonical), 100.0 = centimeters (NVIDIA SV3DT samples).
    scale_env = str(os.environ.get("NOESIS_V3DT_CAMINFO_WORLD_SCALE", "1") or "").strip()
    try:
        world_scale = float(scale_env)
    except Exception:
        world_scale = 1.0
    if not np.isfinite(world_scale) or world_scale <= 0.0:
        world_scale = 1.0
    e_mat = e_mat.copy()
    e_mat[:3, 3] *= float(world_scale)
    rt = e_mat[:3, :]
    p = k @ rt
    
    # Optional image-space Y-flip (default: off). Keep this as an escape hatch for
    # coordinate-system mismatches, but do not enable it by default.
    #
    # Formula: y_flipped = h - y_original, achieved via: P_flip = [[1,0,0],[0,-1,h],[0,0,1]] @ P
    y_flip_env = str(os.environ.get("NOESIS_V3DT_CAMINFO_Y_FLIP", "1") or "").strip().lower()
    y_flip = y_flip_env in ("1", "true", "yes", "y", "on")
    if y_flip:
        # Apply image-space Y-flip: new_row1 = -row1 + h*row2
        p_orig = p.copy()
        p[1, :] = -p_orig[1, :] + target_h * p_orig[2, :]
    
    return p


def _axis_mapping_from_env() -> Optional[np.ndarray]:
    spec = str(os.environ.get("NOESIS_V3DT_CAMINFO_WORLD_AXES", "xzy") or "").strip().lower()
    if not spec or spec == "xyz":
        return None

    tokens = [tok for tok in spec.replace(",", " ").split() if tok]
    if len(tokens) == 1 and len(tokens[0]) == 3 and all(ch in "xyz" for ch in tokens[0]):
        tokens = list(tokens[0])

    if len(tokens) != 3:
        print(f"Warning: invalid NOESIS_V3DT_CAMINFO_WORLD_AXES='{spec}', expected 3 axes")
        return None

    basis = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    used = set()
    cols = []
    for token in tokens:
        sign = -1.0 if token.startswith("-") else 1.0
        axis = token.lstrip("+-")
        if axis not in basis:
            print(f"Warning: invalid axis '{token}' in NOESIS_V3DT_CAMINFO_WORLD_AXES='{spec}'")
            return None
        if axis in used:
            print(f"Warning: duplicate axis '{axis}' in NOESIS_V3DT_CAMINFO_WORLD_AXES='{spec}'")
            return None
        used.add(axis)
        cols.append(sign * basis[axis])
    return np.stack(cols, axis=1)


def _write_caminfo(output_dir: Path, camera_name: str, proj: np.ndarray, model_height: float, model_radius: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # World scale: 1.0 = meters (Noesis canonical), 100.0 = centimeters (NVIDIA SV3DT samples).
    scale_env = str(os.environ.get("NOESIS_V3DT_CAMINFO_WORLD_SCALE", "1") or "").strip()
    try:
        world_scale = float(scale_env)
    except Exception:
        world_scale = 1.0
    if not np.isfinite(world_scale) or world_scale <= 0.0:
        world_scale = 1.0

    # model_height and model_radius are provided in meters; scale if needed
    height = float(model_height)
    radius = float(model_radius)
    if np.isfinite(height) and 0.0 < height <= 10.0:
        height *= float(world_scale)
    if np.isfinite(radius) and 0.0 < radius <= 10.0:
        radius *= float(world_scale)
    matrix_type_env = str(os.environ.get("NOESIS_V3DT_CAMINFO_MATRIX_TYPE", "w2p") or "").strip().lower()
    key = "projectionMatrix_3x4_w2p" if matrix_type_env in ("w2p", "3x4_w2p", "projectionmatrix_3x4_w2p") else "projectionMatrix_3x4"
    caminfo = {
        key: proj.flatten(order="C").tolist(),
        "modelInfo": {"height": float(height), "radius": float(radius)},
    }
    output_path = output_dir / f"camInfo_{camera_name}.yml"
    with output_path.open("w") as handle:
        yaml.safe_dump(caminfo, handle, sort_keys=False)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SV3DT camInfo YAMLs from Noesis calibration.")
    parser.add_argument(
        "--cameras-config",
        default=str(REPO_ROOT / "config" / "cameras.yaml"),
        help="Path to cameras.yaml",
    )
    parser.add_argument(
        "--calibration",
        default=str(REPO_ROOT / "config" / "camera_calibration.json"),
        help="Path to camera_calibration.json",
    )
    default_pipeline = REPO_ROOT / "DS9" / "config" / "infer_v3dt.yaml"
    if not default_pipeline.exists():
        default_pipeline = REPO_ROOT / "DS9" / "config" / "infer.yaml"
    parser.add_argument(
        "--pipeline-config",
        default=str(default_pipeline),
        help="Path to pipeline config (streammux size source)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "config" / "v3dt"),
        help="Output directory for camInfo_*.yml",
    )
    parser.add_argument("--model-height", type=float, default=1.7, help="Model height in meters")
    parser.add_argument("--model-radius", type=float, default=0.35, help="Model radius in meters")
    parser.add_argument("--target-width", type=int, default=None, help="Override target width (defaults to streammux width)")
    parser.add_argument("--target-height", type=int, default=None, help="Override target height (defaults to streammux height)")
    args = parser.parse_args()

    cameras_cfg = _load_yaml(Path(args.cameras_config))
    intrinsics_models = cameras_cfg.get("intrinsics_models") or {}
    cameras = cameras_cfg.get("cameras") or {}

    with Path(args.calibration).open("r") as handle:
        calibration = json.load(handle).get("cameras") or {}

    stream_w, stream_h = _load_streammux_size(Path(args.pipeline_config))
    # Override with explicit target dimensions if provided (e.g., for tracker resolution)
    if args.target_width is not None:
        stream_w = args.target_width
    if args.target_height is not None:
        stream_h = args.target_height

    ordered = sorted(cameras.items(), key=lambda item: int(item[0]))
    axis_spec = str(os.environ.get("NOESIS_V3DT_CAMINFO_WORLD_AXES", "xzy") or "").strip().lower()
    if axis_spec and axis_spec != "xyz":
        print(f"Applying world-axis map for camInfo: {axis_spec}")
    for _, cam_info in ordered:
        name = cam_info.get("name")
        model_name = cam_info.get("model")
        if not name or not model_name:
            continue

        intr = _intrinsics_from_model(intrinsics_models, model_name)
        base_res = _resolution_from_model(intrinsics_models.get(model_name))
        fx, fy, cx, cy = (float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"]))
        fx, fy, cx, cy = _scale_intrinsics(fx, fy, cx, cy, stream_w, stream_h, base_res)

        # NVIDIA’s tracker-3d sample config uses `projectionMatrix_3x4` (not `_w2p`).
        # That variant assumes a zero-centered principal point and DeepStream internally
        # shifts by (img_w/2, img_h/2). Our camera models use a centered principal point
        # (cx≈w/2, cy≈h/2), so we can set (cx,cy) to (0,0) for `projectionMatrix_3x4`.
        matrix_type_env = str(os.environ.get("NOESIS_V3DT_CAMINFO_MATRIX_TYPE", "w2p") or "").strip().lower()
        use_w2p = matrix_type_env in ("w2p", "3x4_w2p", "projectionmatrix_3x4_w2p")
        cx_use = cx if use_w2p else 0.0
        cy_use = cy if use_w2p else 0.0
        k = np.array([[fx, 0.0, cx_use], [0.0, fy, cy_use], [0.0, 0.0, 1.0]], dtype=np.float64)

        calib = calibration.get(name)
        if not calib:
            print(f"Warning: no calibration for camera '{name}', skipping")
            continue
        pose = normalize_pose_v1(calib.get("pose")) if isinstance(calib, dict) else None
        e_col_major = pose_to_E_col_major(pose) if pose is not None else None
        if e_col_major is None:
            e_col_major = calib.get("E")
        if not e_col_major:
            print(f"Warning: calibration for '{name}' missing E, skipping")
            continue

        # Per-camera invert_e flag from calibration, fallback to env var
        invert_e_cam = calib.get("invert_e")
        if invert_e_cam is None:
            # Menon sends `E` as World→Camera (extrinsics). SV3DT expects World→Camera in camInfo
            # when building `P = K @ E[:3,:]` (for either `projectionMatrix_3x4` or `_w2p`).
            # Do NOT invert by default.
            invert_env = str(os.environ.get("NOESIS_V3DT_CAMINFO_INVERT_E", "0") or "").strip().lower()
            invert_e_cam = invert_env in ("1", "true", "yes", "y", "on")
        print(f"Generating camInfo for {name} (invert_e={invert_e_cam})")

        proj = _projection_matrix(k, e_col_major, stream_h, invert_e=invert_e_cam)
        _write_caminfo(Path(args.output_dir), name, proj, args.model_height, args.model_radius)


if __name__ == "__main__":
    main()
