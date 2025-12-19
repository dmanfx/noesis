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
INTRINSICS_JSON = REPO_ROOT / "intrinsics.json"
CALIBRATION_JSON = REPO_ROOT / "config" / "camera_calibration.json"
OUTPUT_DIR = REPO_ROOT / "config" / "camInfo"


def load_cameras():
    with open(CAMERAS_YAML) as f:
        data = yaml.safe_load(f)
    return data["cameras"]


def load_intrinsics():
    with open(INTRINSICS_JSON) as f:
        return json.load(f)


def load_calibration():
    with open(CALIBRATION_JSON) as f:
        return json.load(f)["cameras"]


def get_k_matrix(intrinsics_data: dict, model_name: str) -> np.ndarray:
    """Get 3x3 K matrix for a camera model."""
    # Handle naming variations between cameras.yaml and intrinsics.json
    # cameras.yaml uses: unifi_g3_instant, unifi_g4_instant
    # intrinsics.json uses: unifi_protect_g3_instant, unifi_protect_g4_instant
    candidates = [
        model_name,
        f"unifi_protect_{model_name}",
        model_name.replace("unifi_", "unifi_protect_"),
    ]
    for key in candidates:
        if key in intrinsics_data:
            return np.array(intrinsics_data[key]["intrinsics"]["K_matrix"], dtype=np.float64)
    raise KeyError(f"Intrinsics model '{model_name}' not found. Available: {list(intrinsics_data.keys())}")


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
    intrinsics = load_intrinsics()
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

