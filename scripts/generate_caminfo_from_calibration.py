#!/usr/bin/env python3
"""
Generate DeepStream camInfo YAML files from Noesis calibration assets.

Usage: python scripts/generate_caminfo_from_calibration.py

Output: config/camInfo/camInfo_<camera-name>.yml for each camera
"""

import json
import os
import yaml
import numpy as np
from pathlib import Path

# Paths relative to repo root
REPO_ROOT = Path(__file__).parent.parent
CAMERAS_YAML = REPO_ROOT / "config" / "cameras.yaml"
CALIBRATION_JSON = REPO_ROOT / "config" / "camera_calibration.json"
OUTPUT_DIR = REPO_ROOT / "config" / "camInfo"


def load_cameras():
    with open(CAMERAS_YAML) as f:
        data = yaml.safe_load(f)
    return data["cameras"]


def load_intrinsics_models():
    with open(CAMERAS_YAML) as f:
        data = yaml.safe_load(f) or {}
    models = data.get("intrinsics_models") or {}
    if not isinstance(models, dict):
        raise TypeError(f"{CAMERAS_YAML} intrinsics_models must be a mapping")
    return models


def load_calibration():
    with open(CALIBRATION_JSON) as f:
        return json.load(f)["cameras"]


def get_k_matrix(intrinsics_data: dict, model_name: str) -> np.ndarray:
    """Get a 3x3 K matrix from a canonical cameras.yaml model."""
    entry = intrinsics_data.get(model_name)
    if not isinstance(entry, dict):
        raise KeyError(
            f"Intrinsics model '{model_name}' not found. Available: {list(intrinsics_data.keys())}"
        )
    intrinsics = entry.get("intrinsics") or {}
    if not isinstance(intrinsics, dict):
        raise TypeError(f"Intrinsics model '{model_name}' has no intrinsics mapping")
    matrix = intrinsics.get("K3x3") or entry.get("K3x3")
    if isinstance(matrix, list):
        values = np.asarray(matrix, dtype=np.float64)
        if values.size == 9:
            return values.reshape((3, 3))
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Intrinsics model '{model_name}' has no valid K parameters") from exc
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def compute_projection_matrix(K: np.ndarray, E_col_major: list) -> np.ndarray:
    """
    Compute 3x4 projection matrix P = K @ [R|t].
    E is world→camera, stored column-major.
    """
    E = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
    Rt = E[:3, :]  # 3x4 [R|t]
    P = K @ Rt
    return P


def write_caminfo(camera_name: str, P: np.ndarray, output_dir: Path):
    """Write camInfo YAML in DeepStream format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten P row-major for projectionMatrix_3x4_w2p
    # IMPORTANT: Use projectionMatrix_3x4_w2p (not projectionMatrix_3x4) because
    # our P matrix already includes the principal point (cx, cy) in the K matrix.
    # DeepStream's projectionMatrix_3x4 assumes zero-centered principal point
    # and internally adds (img_width/2, img_height/2), causing double-translation
    # which leads to off-image projections and memory explosion.
    p_flat = P.flatten(order="C").tolist()
    
    caminfo = {
        "projectionMatrix_3x4_w2p": p_flat,  # P = K @ E[:3,:], world→image with principal point
        "modelInfo": {
            "height": 1.7,   # meters
            "radius": 0.35,  # meters
        }
    }
    
    output_path = output_dir / f"camInfo_{camera_name}.yml"
    with open(output_path, "w") as f:
        yaml.dump(caminfo, f, default_flow_style=False, sort_keys=False)
    
    print(f"Wrote {output_path}")


def main():
    cameras = load_cameras()
    intrinsics = load_intrinsics_models()
    calibration = load_calibration()
    
    for cam_id, cam_info in cameras.items():
        name = cam_info["name"]
        model = cam_info["model"]
        
        # Get K matrix
        K = get_k_matrix(intrinsics, model)
        
        # Get E matrix
        if name not in calibration:
            print(f"Warning: No calibration for camera '{name}', skipping")
            continue
        E_col_major = calibration[name]["E"]
        
        # Compute projection matrix
        P = compute_projection_matrix(K, E_col_major)
        
        # Write camInfo
        write_caminfo(name, P, OUTPUT_DIR)
    
    print(f"\nGenerated {len(cameras)} camInfo files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
