#!/usr/bin/env python3
"""Unit tests for V3DT forensics snapshot builder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from noesis.diagnostics.v3dt_forensics import build_snapshot


@pytest.fixture
def temp_v3dt_config(tmp_path: Path) -> Path:
    cameras_yaml = tmp_path / "cameras.yaml"
    cameras_yaml.write_text(
        """
version: 1
intrinsics_models:
  test_model:
    intrinsics:
      fx: 800.0
      fy: 800.0
      cx: 640.0
      cy: 360.0
      k1: 0.0
      k2: 0.0
      k3: 0.0
cameras:
  0:
    name: test-cam
    model: test_model
    height_m: 2.0
""",
        encoding="utf-8",
    )

    calib_json = tmp_path / "camera_calibration.json"
    C_world = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    R_cw = np.eye(3, dtype=np.float64)
    t = -R_cw @ C_world
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R_cw
    E[:3, 3] = t
    calib_json.write_text(
        json.dumps({"cameras": {"test-cam": {"E": list(E.flatten(order="F"))}}}, indent=2),
        encoding="utf-8",
    )

    align_json = tmp_path / "ply_alignment.json"
    align_json.write_text(
        json.dumps({"matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "floor_y": 0.0, "units": {"s_obj_to_m": 1.0}}),
        encoding="utf-8",
    )

    pipeline_yaml = tmp_path / "pipeline.yaml"
    pipeline_yaml.write_text(
        """
streammux:
  width: 1280
  height: 720
tracker:
  config-file: config/v3dt/nvtracker_sv3dt.yml
  tracker-width: 1280
  tracker-height: 720
""",
        encoding="utf-8",
    )

    caminfo_dir = tmp_path / "v3dt"
    caminfo_dir.mkdir(parents=True, exist_ok=True)

    fx = 800.0
    fy = 800.0
    K = np.array([[fx, 0.0, 0.0], [0.0, fy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    P = K @ E[:3, :]
    caminfo = {
        "projectionMatrix_3x4": P.flatten(order="C").tolist(),
        "modelInfo": {"height": 1.7, "radius": 0.35},
    }
    (caminfo_dir / "camInfo_test-cam.yml").write_text(yaml.safe_dump(caminfo, sort_keys=False), encoding="utf-8")

    return tmp_path


def test_snapshot_caminfo_ratio(temp_v3dt_config: Path) -> None:
    snapshot = build_snapshot(
        pipeline_path=temp_v3dt_config / "pipeline.yaml",
        cameras_path=temp_v3dt_config / "cameras.yaml",
        calibration_path=temp_v3dt_config / "camera_calibration.json",
        caminfo_dir=temp_v3dt_config / "v3dt",
        alignment_path=temp_v3dt_config / "ply_alignment.json",
        tracker_config_path=None,
    )
    cams = snapshot.get("cameras", {})
    assert "test-cam" in cams
    caminfo = cams["test-cam"]["caminfo"]
    ratios = caminfo.get("translation_ratio")
    assert isinstance(ratios, list)
    assert all(abs(float(r) - 1.0) < 1e-6 for r in ratios if np.isfinite(r))
