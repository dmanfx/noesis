"""Actual configured loader -> immutable raw/world snapshot metadata."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

import noesis.calibration.manager as manager_module
import noesis.metadata.intrinsics as intrinsics_module
from noesis.calibration.manager import CalibrationManager
from noesis.metadata.intrinsics import CameraConfigLoader


ROOT = Path(__file__).resolve().parents[1]
E = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., -.25, .5, -.75, 1.)
FIELDS = ("projection_model", "distortion_model", "distortion_coefficients",
          "distortion_coefficient_names", "projection_metadata_status")


def configured(tmp_path, *, metadata=None, scalar_coefficients=False):
    intrinsics = {"fx": 800., "fy": 820., "cx": 620., "cy": 350.}
    if scalar_coefficients:
        intrinsics.update(k1=0., k2=0., k3=0.)
    model = {"resolution": [1280, 720], "intrinsics": intrinsics}
    if metadata is not None:
        model["calibration"] = metadata
    payload = {"intrinsics_models": {"configured": model},
               "cameras": {0: {"name": "fixture-camera", "model": "configured"}}}
    cameras = tmp_path / "cameras.yaml"
    cameras.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    extrinsics = tmp_path / "camera_calibration.json"
    extrinsics.write_text(json.dumps({"cameras": {"fixture-camera": {"E": E}}}))
    alignment = tmp_path / "ply_alignment.json"
    alignment.write_text(json.dumps({"matrix": np.eye(4).flatten(order="F").tolist(),
                                    "floor_y": 0., "units": {"s_obj_to_m": 1.}}))
    manager = CalibrationManager(cameras, extrinsics, alignment, streammux_size=(640, 360),
                                 raw_audit_dir=tmp_path / "private-audit")
    manager.set_camera_labels({0: "fixture-camera"})
    return cameras, payload, manager


def snapshots(manager):
    raw = manager.snapshot(0, "fixture-camera")
    world = manager.world_snapshot(0, "fixture-camera")
    assert raw is not None and world is not None
    assert world.intrinsics is raw.intrinsics
    assert raw.intrinsics.flags.writeable is False
    np.testing.assert_array_equal(raw.intrinsics, [[400., 0., 310.], [0., 410., 175.], [0., 0., 1.]])
    assert raw.extrinsics_col_major == E
    assert world.calibration_extrinsics_col_major == E
    assert world.extrinsics_col_major == raw.world_extrinsics_col_major
    assert raw.image_size == world.image_size == (640, 360)
    assert raw.unit_scale == world.unit_scale == 1.
    for name in FIELDS:
        assert getattr(raw, name) == getattr(world, name)
    return raw, world


def test_actual_runtime_loader_preserves_original_model_calibration_outside_legacy_intrinsics(tmp_path):
    # Match the canonical runtime and pytest namespace route, not an alternate
    # shared-file import that could pass while DS9's loader drops the fields.
    assert Path(manager_module.__file__).resolve() == ROOT / "noesis/calibration/manager.py"
    assert Path(intrinsics_module.__file__).resolve() == ROOT / "DS9/noesis/metadata/intrinsics.py"
    coefficients = [-.041323572296878525, -.07257812805006623, .06665786839070417, -.02667936264765735]
    path, _, manager = configured(tmp_path, scalar_coefficients=True, metadata={
        "model": "opencv_fisheye", "distortion_model": "fisheye", "distortion_coeffs": coefficients})
    intrinsics = CameraConfigLoader(path).get(0)
    assert intrinsics.projection_model == "opencv_fisheye"
    assert intrinsics.distortion_model == "fisheye"
    assert intrinsics.distortion_coefficients == tuple(coefficients)
    assert intrinsics.distortion_coefficient_names is None
    assert intrinsics.projection_metadata_status == "declared"
    assert intrinsics.as_payload() == {"fx": 800., "fy": 820., "cx": 620., "cy": 350.,
                                       "k1": 0., "k2": 0., "k3": 0., "height_m": 0.}
    raw, world = snapshots(manager)
    assert raw.distortion_coefficients == tuple(coefficients)
    assert world.distortion_coefficients is raw.distortion_coefficients
    for value in (intrinsics, raw, world):
        with pytest.raises(FrozenInstanceError):
            value.projection_model = "pinhole"
        with pytest.raises(TypeError):
            value.distortion_coefficients[0] = 9.
    assert not any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules)


def test_missing_projection_and_coefficients_remain_missing_despite_legacy_zeroes(tmp_path):
    path, _, manager = configured(tmp_path)
    intrinsics = CameraConfigLoader(path).get(0)
    assert (intrinsics.k1, intrinsics.k2, intrinsics.k3) == (0., 0., 0.)
    raw, _ = snapshots(manager)
    assert (raw.projection_model, raw.distortion_model, raw.distortion_coefficients,
            raw.distortion_coefficient_names, raw.projection_metadata_status) == (None, None, None, None, "undeclared")


def test_named_partial_coefficients_are_not_padded_or_reordered_and_unsupported_models_remain_explicit(tmp_path):
    _, _, manager = configured(tmp_path, metadata={"projection_model": "unsupported_declared_model",
        "distortion_model": "fisheye", "distortion_coeffs": {"k2": -.02, "k1": .01}})
    raw, _ = snapshots(manager)
    assert raw.projection_model == "unsupported_declared_model"
    assert raw.distortion_model == "fisheye"
    assert raw.distortion_coefficients == (-.02, .01)
    assert raw.distortion_coefficient_names == ("k2", "k1")
    assert raw.projection_metadata_status == "declared"  # declaration is not optical support


@pytest.mark.parametrize("metadata", [
    {"projection_model": {"guess": "pinhole"}},
    {"projection_model": "pinhole", "distortion_coeffs": [0., float("nan")]},
    {"projection_model": "pinhole", "distortion_coeffs": [float("inf")]},
    {"distortion_model": "x" * 65},
    {"distortion_coeffs": [0.] * 17},
    {"distortion_coeffs": {"k1": True}},
    {"distortion_coeffs": []},
])
def test_malformed_declared_metadata_is_explicitly_invalid_without_breaking_existing_K_E(tmp_path, metadata):
    _, _, manager = configured(tmp_path, scalar_coefficients=True, metadata=metadata)
    raw, _ = snapshots(manager)
    assert raw.projection_metadata_status == "invalid"
    if "distortion_coeffs" in metadata:
        assert raw.distortion_coefficients is None
    else:
        # An invalid model declaration must not erase separately supplied,
        # valid coefficient evidence. These zeroes are explicit fixture input.
        assert raw.distortion_coefficients == (0., 0., 0.)
        assert raw.distortion_coefficient_names == ("k1", "k2", "k3")


def test_metadata_only_reload_invalidates_raw_world_and_batch_cache_without_changing_K_E(tmp_path):
    path, payload, manager = configured(tmp_path, metadata={"projection_model": "pinhole", "distortion_coeffs": [.1, -.2]})
    first, first_world = snapshots(manager)
    assert manager.snapshot(0, "fixture-camera") is first
    assert manager.world_snapshot(0, "fixture-camera") is first_world
    old_mtime = path.stat().st_mtime_ns
    payload["intrinsics_models"]["configured"]["calibration"]["distortion_coeffs"] = [.3, -.4]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.utime(path, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))
    second = manager.snapshots({0: "fixture-camera"})[0]
    second_world = manager.world_snapshot(0, "fixture-camera")
    assert second is not first and second_world is not first_world
    assert first.distortion_coefficients == first_world.distortion_coefficients == (.1, -.2)
    assert second.distortion_coefficients == second_world.distortion_coefficients == (.3, -.4)
    np.testing.assert_array_equal(second.intrinsics, first.intrinsics)
    assert second.extrinsics_col_major == first.extrinsics_col_major
    assert manager.snapshots({0: "fixture-camera"})[0] is second
