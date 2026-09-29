#!/usr/bin/env python3
"""Generate camInfo YAMLs from the runtime's revision-bound world calibration.

Usage:
  python3 scripts/generate_v3dt_caminfo.py

This writes camInfo_<camera-name>.yml files under
DS9/config/v3dt/caminfo_baseline/ by default.

Contract:
  - The pipeline's existing accepted frame bindings select the target world
    revision and horizontal floor through the same CalibrationManager as DS9.
  - Projection and model dimensions use meters. Tracker Z=0 is target floor Y=0.
  - Required conventions for the rectified OpenCV pixel frame:
    - `NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p`
    - `NOESIS_V3DT_CAMINFO_INVERT_E=0` (PoseV1 produces world-to-camera E)
    - `NOESIS_V3DT_CAMINFO_Y_FLIP=0` (image Y already increases downward)
    - `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy` (swap Y/Z; SV3DT Z-up)
  - Legacy projection overrides that violate these conventions are rejected.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "DS9") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "DS9"))

from noesis.calibration.manager import create_calibration_manager
from noesis_core.scene_prior import ScenePriorSet
from noesis.v3dt_raster import scale_projection_matrix, validate_raster


def _load_yaml(path: Path) -> Dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


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
    
    # PoseV1 already produces OpenCV camera coordinates (image Y down), matching
    # the rectified nvdewarper surface. A second flip inverts the person's
    # vertical model and makes the tracker infer its ground point above its head.
    # Retain the explicit override for callers with a different image contract.
    #
    # Formula: y_flipped = h - y_original, achieved via: P_flip = [[1,0,0],[0,-1,h],[0,0,1]] @ P
    y_flip_env = str(os.environ.get("NOESIS_V3DT_CAMINFO_Y_FLIP", "0") or "").strip().lower()
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


def _write_caminfo(
    output_dir: Path,
    camera_name: str,
    proj: np.ndarray,
    model_height: float,
    model_radius: float,
    *,
    frame_binding: Optional[Dict[str, Any]] = None,
) -> None:
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
    if frame_binding is not None:
        caminfo["noesis_frame_binding"] = frame_binding
    output_path = output_dir / f"camInfo_{camera_name}.yml"
    with output_path.open("w") as handle:
        yaml.safe_dump(caminfo, handle, sort_keys=False)
    print(f"Wrote {output_path}")


def _projection_calibration(
    pipeline_path: Path,
    cameras_path: Path,
    calibration_path: Path,
    alignment_path: Path,
    *,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
):
    """Use the exact revision-bound calibration authority used by DS9."""

    pipeline = _load_yaml(pipeline_path)
    streammux = dict(pipeline.get("streammux") or {})
    if target_width is not None:
        streammux["width"] = int(target_width)
    if target_height is not None:
        streammux["height"] = int(target_height)
    pipeline["streammux"] = streammux
    scene_priors = pipeline.get("scene_priors")
    bindings = None
    if scene_priors is not None:
        if not isinstance(scene_priors, dict) or not scene_priors.get("path"):
            raise ValueError("scene_priors.path is required when scene priors are configured")
        raw_path = str(scene_priors["path"])
        catalog_path = Path(raw_path)
        if not catalog_path.is_absolute():
            if raw_path.startswith("DS9/"):
                catalog_path = REPO_ROOT / catalog_path
            elif raw_path.startswith(("config/", "pipelines/", "build/")):
                scope = REPO_ROOT / "DS9" if (REPO_ROOT / "DS9") in pipeline_path.resolve().parents else REPO_ROOT
                catalog_path = scope / catalog_path
            else:
                catalog_path = pipeline_path.parent / catalog_path
        bindings = ScenePriorSet.load(catalog_path.resolve()).frame_bindings()
    return create_calibration_manager(
        cameras_yaml_path=cameras_path,
        pipeline_config=pipeline,
        camera_calibration_json_path=calibration_path,
        ply_alignment_json_path=alignment_path,
        scene_prior_frame_bindings=bindings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SV3DT camInfo YAMLs from Noesis calibration.")
    parser.add_argument(
        "--cameras-config",
        default=str(REPO_ROOT / "DS9" / "config" / "cameras_v3dt.yaml"),
        help="Path to cameras.yaml",
    )
    parser.add_argument(
        "--calibration",
        default=os.environ.get("NOESIS_CAMERA_CALIBRATION_FILE") or str(REPO_ROOT / "config" / "camera_calibration.json"),
        help="Path to camera_calibration.json",
    )
    parser.add_argument(
        "--alignment",
        default=os.environ.get("NOESIS_PLY_ALIGNMENT_FILE") or str(REPO_ROOT / "config" / "ply_alignment.json"),
        help="Path to the runtime's ply_alignment.json",
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
        default=str(REPO_ROOT / "DS9" / "config" / "v3dt" / "caminfo_baseline"),
        help="Output directory for camInfo_*.yml",
    )
    parser.add_argument("--model-height", type=float, default=1.7, help="Model height in meters")
    parser.add_argument("--model-radius", type=float, default=0.35, help="Model radius in meters")
    parser.add_argument("--target-width", type=int, default=None, help="Override target width (defaults to streammux width)")
    parser.add_argument("--target-height", type=int, default=None, help="Override target height (defaults to streammux height)")
    args = parser.parse_args()

    # The published bbox3d contract is target-frame meters with x,z,y axes and
    # tracker Z=0 on the target floor. Reject legacy projection overrides rather
    # than labeling their coordinates with the canonical revision below.
    expected_env = {
        "NOESIS_V3DT_CAMINFO_WORLD_AXES": {"xzy"},
        "NOESIS_V3DT_CAMINFO_WORLD_SCALE": {"1", "1.0"},
        "NOESIS_V3DT_CAMINFO_MATRIX_TYPE": {"w2p", "3x4_w2p", "projectionmatrix_3x4_w2p"},
        "NOESIS_V3DT_CAMINFO_INVERT_E": {"0", "false", "off", "no"},
        "NOESIS_V3DT_CAMINFO_Y_FLIP": {"0", "false", "off", "no"},
    }
    for name, admitted in expected_env.items():
        if name in os.environ and os.environ[name].strip().lower() not in admitted:
            raise ValueError(f"{name} conflicts with the canonical camInfo frame contract")
    manager = _projection_calibration(
        Path(args.pipeline_config), Path(args.cameras_config),
        Path(args.calibration), Path(args.alignment),
        target_width=args.target_width, target_height=args.target_height,
    )
    cameras = _load_yaml(Path(args.cameras_config)).get("cameras") or {}
    pipeline = _load_yaml(Path(args.pipeline_config))
    pixel_space = (pipeline.get("v3dt") or {}).get("caminfo_pixel_space", "mux")
    if pixel_space not in ("mux", "tracker"):
        raise ValueError("caminfo_pixel_space must be mux or tracker")
    tracker = pipeline.get("tracker") or {}
    tracker_raster = None
    if pixel_space == "tracker":
        if (pipeline.get("v3dt") or {}).get("profile") != "sv3dt":
            raise ValueError("tracker camInfo pixel space requires the sv3dt profile")
        tracker_raster = validate_raster(
            (tracker.get("tracker-width"), tracker.get("tracker-height")), label="tracker"
        )
    pending = []
    for source_id, camera in sorted(cameras.items(), key=lambda item: int(item[0])):
        name = str(camera["name"])
        snapshot = manager.world_snapshot(int(source_id), name)
        if snapshot is None:
            raise ValueError(f"{name}: canonical world calibration is unavailable")
        if not np.isclose(snapshot.floor_y, 0.0, atol=1e-9):
            raise ValueError(f"{name}: tracker Z=0 requires target floor_y=0")
        proj = _projection_matrix(
            snapshot.intrinsics, snapshot.extrinsics_col_major,
            target_h=int(snapshot.image_size[1]),
        )
        binding = {
            "world_frame_id": snapshot.world_frame_id,
            "world_frame_revision": snapshot.world_frame_revision,
            "frame_transform_sha256": snapshot.frame_transform_sha256,
            "camera_calibration_sha256": snapshot.camera_calibration_sha256,
            "image_size": list(snapshot.image_size),
            "floor_y": float(snapshot.floor_y),
        }
        if tracker_raster is not None:
            proj = scale_projection_matrix(proj, snapshot.image_size, tracker_raster)
            binding["projection_pixel_space"] = "tracker"
            binding["tracker_image_size"] = list(tracker_raster)
        pending.append((name, proj, binding))
    # Validate every camera before replacing any file.
    for name, proj, binding in pending:
        _write_caminfo(
            Path(args.output_dir), name, proj, args.model_height, args.model_radius,
            frame_binding=binding,
        )


if __name__ == "__main__":
    main()
