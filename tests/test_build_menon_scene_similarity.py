from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_menon_scene_similarity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_menon_scene_similarity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _world_to_camera_for_center(center: list[float]) -> list[float]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = -np.asarray(center, dtype=np.float64)
    return [float(value) for value in matrix.flatten(order="F")]


def test_builder_binds_exact_device_state_and_meets_metric_residual(tmp_path: Path) -> None:
    module = _load_script()
    centers = {
        "family-room": [0.0, 2.0, 10.0],
        "kitchen": [10.0, 2.0, 0.0],
        "living-room": [0.0, 2.0, 0.0],
    }
    calibration_path = tmp_path / "camera_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "cameras": {
                    camera_id: {"E": _world_to_camera_for_center(center)}
                    for camera_id, center in centers.items()
                }
            }
        ),
        encoding="utf-8",
    )
    devices_path = tmp_path / "virtual-devices.json"
    devices_path.write_text(
        json.dumps(
            [
                {
                    "id": f"device-{camera_id}",
                    "name": f"{camera_id} Camera",
                    "type": "Camera",
                    "position": {
                        "x": (100.0 * center[0]) + 12.0,
                        "y": (100.0 * center[1]) + 34.0,
                        "z": (100.0 * center[2]) + 56.0,
                    },
                }
                for camera_id, center in centers.items()
            ]
        ),
        encoding="utf-8",
    )

    result = module.build_similarity(
        camera_calibration_path=calibration_path,
        menon_devices_path=devices_path,
        anchor_residual_limit_m=0.05,
    )

    assert result["source"] == "menon_virtual_device_camera_similarity_v1"
    assert len(result["anchor_state_sha256"]) == 64
    assert len(result["camera_calibration_sha256"]) == 64
    assert result["position_rmse_m"] < 1e-9
    assert result["max_residual_m"] < 1e-9
    assert {row["anchor_id"] for row in result["correspondences"]} == {
        "device-family-room",
        "device-kitchen",
        "device-living-room",
    }
