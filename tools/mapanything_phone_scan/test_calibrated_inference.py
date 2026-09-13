from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.mapanything_phone_scan import da3_inference, inference
from tools.mapanything_phone_scan import windowed_da3_inference, windowed_inference
from tools.mapanything_phone_scan.calibrated_inference import (
    RECTIFIED_INTRINSICS_SCHEMA,
    calibrated_input_views,
    calibration_raw_fields,
    calibration_specs,
    rectification_valid_mask,
)


def _prepared(tmp_path: Path, count: int = 2) -> dict:
    width, height = 168, 98
    matrix = np.array([[135.0, 0.0, 81.0], [0.0, 130.0, 46.0], [0.0, 0.0, 1.0]])
    spec = {
        "schema": RECTIFIED_INTRINSICS_SCHEMA,
        "profile_id": "test-lens-exact-mode",
        "profile_sha256": "a" * 64,
        "K": matrix.tolist(),
        "resolution_px": [width, height],
        "distortion_model": "none",
        "calibration_applied": True,
        "source_K": matrix.tolist(),
        "source_distortion": [0.20, -0.03, 0.002, -0.003, 0.10],
        "source_resolution_px": [width, height],
    }
    profile = {
        "schema": "noesis.phone_camera_calibration.v1",
        "profile_id": spec["profile_id"],
        "K": matrix.tolist(),
        "D": spec["source_distortion"],
        "resolution": [width, height],
    }
    profile_bytes = (json.dumps(profile, indent=2) + "\n").encode()
    (tmp_path / "phone_camera_calibration.json").write_bytes(profile_bytes)
    spec["profile_sha256"] = hashlib.sha256(profile_bytes).hexdigest()
    frames = []
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width)[None]
    image[..., 1] = np.arange(height)[:, None]
    for index in range(count):
        path = tmp_path / f"frame_{index:04d}.png"
        assert cv2.imwrite(str(path), image)
        frames.append({
            "frame": path.name,
            "index": index,
            "timestamp_s": index * 0.5,
            "width": width,
            "height": height,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frame_id": f"test-frame-{index}",
            "camera_intrinsics": deepcopy(spec),
            "calibration_processing": {"source_frame_sha256": "b" * 64},
        })
    return {"frames": frames, "frame_count": count, "camera_calibration": {"profile_sha256": spec["profile_sha256"]}}


def _stub_artifacts(monkeypatch, module) -> None:
    def write(path, *_args):
        Path(path).write_bytes(b"test-artifact")
    monkeypatch.setattr(module, "_write_reconstruction_glb", write)
    monkeypatch.setattr(module, "_write_trajectory_preview", write)


def test_consensus_retains_calibration_limits_d5_and_source_identity(tmp_path):
    from tools.mapanything_phone_scan.build_consensus_fusion import (
        _consensus_calibration, _load_sequence, _remap_to_common_rays, _save_raw_frames,
    )
    prepared = _prepared(tmp_path)
    prepared["camera_calibration"].update(
        capture_binding_status="projection_hypothesis_review_only",
        automatic_calibration_admission=False,
        metric_vio_allowed=False,
    )
    for index, frame in enumerate(prepared["frames"]):
        frame["calibration_processing"]["source_frame_sha256"] = str(index) * 64
    (tmp_path / "prepared_frames_manifest.json").write_text(json.dumps(prepared))
    rows = prepared["frames"]
    spec = rows[0]["camera_intrinsics"]
    matrix = np.asarray(spec["K"])
    width, height = spec["resolution_px"]
    depth = np.ones((2, height, width), dtype=np.float32)
    valid = rectification_valid_mask(spec, matrix, (height, width))
    assert not valid.all()
    fused = {key: depth.copy() for key in (
        "depth", "quality", "delta", "first_evidence_weight", "second_evidence_weight",
    )}
    fused.update(
        mask=np.stack([valid, valid]),
        rgb=np.zeros((2, height, width, 3), dtype=np.uint8),
        source=np.ones_like(depth, dtype=np.uint8),
        uncertain=np.zeros_like(depth, dtype=bool),
        agreement=np.stack([valid, valid]),
    )
    matrices = np.stack([matrix, matrix])
    poses = np.stack([np.eye(4), np.eye(4)])
    _save_raw_frames(tmp_path / "output", fused, matrices, poses, rows)
    output_rows = [{
        "index": index, "raw_npz": f"raw/view_{index:04d}.npz",
        "camera_intrinsics": frame["camera_intrinsics"],
        "calibration_processing": frame["calibration_processing"],
        "source_frame_id": frame["frame_id"], "source_frame_sha256": frame["sha256"],
        "timestamp_s": frame["timestamp_s"],
    } for index, frame in enumerate(rows)]
    (tmp_path / "output/scan_outputs_manifest.json").write_text(json.dumps({"frames": output_rows}))
    sequence = _load_sequence(tmp_path / "output/raw", "calibrated")
    provenance, retained = _consensus_calibration(tmp_path, sequence, sequence)
    assert provenance["camera_calibration"] == prepared["camera_calibration"]
    assert provenance["calibration_provenance"]["calibration_admission_performed"] is False
    assert retained[0]["frame_id"] == rows[0]["frame_id"]
    assert retained[0]["camera_intrinsics"]["source_distortion"] == spec["source_distortion"]
    remapped = _remap_to_common_rays(sequence, matrices, (height, width))
    assert remapped.calibration_rows == sequence.calibration_rows
    np.testing.assert_array_equal(remapped.mask, sequence.mask)
    # Reject mixing a provider without calibration or using rows out of order.
    missing = deepcopy(sequence)
    missing.calibration_rows = None
    with pytest.raises(ValueError, match="same calibration"):
        _consensus_calibration(tmp_path, sequence, missing)
    swapped = deepcopy(sequence)
    swapped.calibration_rows.reverse()
    with pytest.raises(ValueError, match="differs from prepared calibration"):
        _consensus_calibration(tmp_path, sequence, swapped)
    # A raw profile change cannot be laundered into the fused result.
    mismatched = deepcopy(sequence)
    mismatched.calibration_rows[0]["camera_intrinsics"]["profile_id"] = "another-lens"
    with pytest.raises(ValueError, match="differs from prepared calibration"):
        _consensus_calibration(tmp_path, sequence, mismatched)
    for key, value in (("frame_id", "changed"), ("sha256", "c" * 64), ("timestamp_s", 123.0)):
        changed = deepcopy(prepared)
        changed["frames"][0][key] = value
        (tmp_path / "prepared_frames_manifest.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="prepared source identity"):
            _consensus_calibration(tmp_path, sequence, sequence)
    (tmp_path / "prepared_frames_manifest.json").write_text(json.dumps(prepared))
    (tmp_path / "phone_camera_calibration.json").write_text('{"profile_id":"changed"}')
    with pytest.raises(ValueError, match="snapshot differs"):
        _consensus_calibration(tmp_path, sequence, sequence)


@pytest.mark.parametrize("module,settings", [
    (inference, inference.MapAnythingScanSettings),
    (da3_inference, da3_inference.DA3PhoneScanSettings),
    (windowed_inference, inference.MapAnythingScanSettings),
    (windowed_da3_inference, da3_inference.DA3PhoneScanSettings),
])
def test_partial_calibration_and_unbound_anchor_fail_before_model_load(tmp_path, module, settings):
    prepared = _prepared(tmp_path)
    runner = next(getattr(module, name) for name in (
        "run_mapanything_scan", "run_da3_phone_scan", "run_windowed_mapanything_scan", "run_windowed_da3_phone_scan"
    ) if hasattr(module, name) and getattr(module, name).__module__ == module.__name__)
    mixed = deepcopy(prepared)
    del mixed["frames"][1]["camera_intrinsics"]
    with pytest.raises(RuntimeError, match="mixed calibrated"):
        runner(tmp_path, tmp_path / "out", mixed, settings(device="cpu"), lambda *_: None)
    with pytest.raises(RuntimeError, match="unbound fixed-camera"):
        runner(tmp_path, tmp_path / "out", prepared, settings(device="cpu", anchor_image=tmp_path / "missing.png"), lambda *_: None)


def test_rectification_validity_uses_k3_on_the_actual_model_rays(tmp_path):
    spec = _prepared(tmp_path)["frames"][0]["camera_intrinsics"]
    width, height = spec["resolution_px"]
    matrix = np.asarray(spec["K"])
    valid = rectification_valid_mask(spec, matrix, (height, width))
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, np.asarray(spec["source_distortion"]), None, matrix, (width, height), cv2.CV_32FC1
    )
    expected = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    np.testing.assert_array_equal(valid, expected)
    no_k3 = deepcopy(spec)
    no_k3["source_distortion"][-1] = 0.0
    assert np.count_nonzero(valid != rectification_valid_mask(no_k3, matrix, (height, width))) > 0
    identity = deepcopy(spec)
    identity["source_distortion"] = [0.0] * 5
    assert rectification_valid_mask(identity, matrix, (height, width)).all()


def test_binding_rejects_changed_frame_dimensions_and_truncated_distortion(tmp_path):
    prepared = _prepared(tmp_path)
    specs = calibration_specs(prepared["frames"])
    path = tmp_path / prepared["frames"][0]["frame"]
    assert cv2.imwrite(str(path), np.zeros((20, 20, 3), dtype=np.uint8))
    prepared["frames"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="dimensions changed"):
        calibrated_input_views([path], specs[:1], frame_rows=prepared["frames"][:1], scan_dir=tmp_path)
    prepared["frames"][0]["camera_intrinsics"]["source_distortion"] = [0.0] * 4
    with pytest.raises(ValueError, match="all five"):
        calibration_specs(prepared["frames"])


def test_frame_bytes_and_intrinsics_cannot_drift_from_saved_evidence(tmp_path):
    prepared = _prepared(tmp_path)
    row = prepared["frames"][0]
    path = tmp_path / row["frame"]
    specs = calibration_specs([row])
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="frame hash changed"):
        calibrated_input_views([path], specs, frame_rows=[row], scan_dir=tmp_path)
    path.write_bytes(original)
    row["camera_intrinsics"]["K"][0][0] += 1.0
    with pytest.raises(ValueError, match="retained profile's resize"):
        calibrated_input_views([path], specs, frame_rows=[row], scan_dir=tmp_path)


def test_mapanything_real_preprocessing_reaches_model_and_filters_raw_mask(tmp_path, monkeypatch):
    models = pytest.importorskip("mapanything.models")
    image_module = pytest.importorskip("mapanything.utils.image")
    prepared = _prepared(tmp_path)
    paths = [tmp_path / row["frame"] for row in prepared["frames"]]
    expected = image_module.preprocess_inputs(calibrated_input_views(
        paths, calibration_specs(prepared["frames"]), frame_rows=prepared["frames"], scan_dir=tmp_path
    ))
    seen = []

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def infer(self, views, **_kwargs):
            outputs = []
            for index, view in enumerate(views):
                matrix = view["intrinsics"][0].numpy()
                np.testing.assert_allclose(matrix, expected[index]["intrinsics"][0].numpy())
                seen.append(matrix)
                height, width = view["img"].shape[-2:]
                pose = np.eye(4, dtype=np.float32)
                pose[0, 3] = index * 0.1
                depth = np.full((height, width), 2.0, dtype=np.float32)
                outputs.append({
                    "pts3d": da3_inference._world_points(depth, matrix, pose)[None],
                    "depth_z": depth[None, ..., None],
                    "conf": np.ones((1, height, width, 1), dtype=np.float32),
                    "mask": np.ones((1, height, width, 1), dtype=np.uint8),
                    "img_no_norm": np.zeros((1, height, width, 3), dtype=np.uint8),
                    "camera_poses": pose[None],
                    "intrinsics": matrix[None],
                    "metric_scaling_factor": np.ones(1, dtype=np.float32),
                })
            return outputs

    monkeypatch.setattr(models.MapAnything, "from_pretrained", lambda *_a, **_k: FakeModel())
    _stub_artifacts(monkeypatch, inference)
    out = tmp_path / "out"
    out.mkdir()
    result = inference.run_mapanything_scan(tmp_path, out, prepared, inference.MapAnythingScanSettings(device="cpu"), lambda *_: None)
    assert len(seen) == 2
    assert result["inference_calibration"]["network_calibration_conditioned"] is True
    assert result["camera_calibration"] == prepared["camera_calibration"]
    with np.load(out / "raw/view_0000.npz", allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["mask"], raw["rectification_valid_mask"])
        assert 0.2 < np.mean(raw["mask"]) < 1.0
        assert json.loads(str(raw["camera_intrinsics_json"]))["source_distortion"][-1] == 0.10


@pytest.mark.parametrize("use_calibration", [True, False])
def test_da3_measured_k_reaches_metric_and_backprojection_with_network_k_retained(tmp_path, monkeypatch, use_calibration):
    api = pytest.importorskip("depth_anything_3.api")
    processor_module = pytest.importorskip("depth_anything_3.utils.io.input_processor")
    prepared = _prepared(tmp_path)
    source_K = np.stack([row["camera_intrinsics"]["K"] for row in prepared["frames"]])
    paths = [str(tmp_path / row["frame"]) for row in prepared["frames"]]
    processor = processor_module.InputProcessor()
    _, _, transformed_K = processor(paths, intrinsics=source_K, process_res=140, process_res_method="upper_bound_resize", sequential=True)
    measured = transformed_K.numpy()
    network = measured.copy()
    network[:, 0, 0] *= 1.5
    network[:, 1, 1] *= 1.5
    if not use_calibration:
        for row in prepared["frames"]:
            del row["camera_intrinsics"]
        del prepared["camera_calibration"]
    seen_metric = []

    class FakeModel:
        input_processor = processor

        def to(self, _device):
            return self

        def inference(self, images, **kwargs):
            assert "extrinsics" not in kwargs and "intrinsics" not in kwargs
            pixels, _, _ = processor(images, process_res=kwargs["process_res"], process_res_method=kwargs["process_res_method"], sequential=True)
            count, _, height, width = pixels.shape
            return SimpleNamespace(
                depth=np.ones((count, height, width), dtype=np.float32),
                conf=np.ones((count, height, width), dtype=np.float32),
                extrinsics=np.tile(np.eye(4, dtype=np.float64)[None, :3], (count, 1, 1)),
                intrinsics=network,
                processed_images=np.zeros((count, height, width, 3), dtype=np.uint8),
            )

    def fake_metric(images, intrinsics, *_args):
        seen_metric.append(intrinsics.copy())
        return np.full(images.shape[:3], 2.0, dtype=np.float32), np.ones(images.shape[:3], dtype=bool), {"test": True}

    monkeypatch.setattr(api.DepthAnything3, "from_pretrained", lambda *_a, **_k: FakeModel())
    monkeypatch.setattr(da3_inference, "_run_metric_branch", fake_metric)
    _stub_artifacts(monkeypatch, da3_inference)
    out = tmp_path / "out"
    out.mkdir()
    result = da3_inference.run_da3_phone_scan(tmp_path, out, prepared, da3_inference.DA3PhoneScanSettings(device="cpu", process_res=140), lambda *_: None)
    assert result["pose_convention"] == "opencv_cam2world_x_right_y_down_z_forward"
    assert result["coordinate_frame"] == "da3_metric_world_unaligned_to_noesis"
    trajectory = json.loads((out / "camera_trajectory.json").read_text())
    assert trajectory["pose_convention"] == result["pose_convention"]
    assert trajectory["coordinate_frame"] == result["coordinate_frame"]
    expected = measured if use_calibration else network
    np.testing.assert_allclose(seen_metric[0], expected)
    with np.load(out / "raw/view_0000.npz", allow_pickle=False) as raw:
        np.testing.assert_allclose(raw["intrinsics"], expected[0])
        np.testing.assert_allclose(raw["world_points"], da3_inference._world_points(raw["depth_z"], expected[0], np.eye(4)), atol=1e-6)
        if use_calibration:
            np.testing.assert_allclose(raw["network_intrinsics"], network[0])
            np.testing.assert_array_equal(raw["mask"], raw["rectification_valid_mask"])
            assert np.mean(raw["mask"]) < 1.0
            assert result["inference_calibration"]["network_calibration_conditioned"] is False
        else:
            assert "network_intrinsics" not in raw
            assert "rectification_valid_mask" not in raw
            assert raw["mask"].all()
            assert "inference_calibration" not in result


@pytest.mark.parametrize("provider", ["mapanything", "da3"])
def test_window_exports_preserve_bound_calibration_and_diagnostics(tmp_path, monkeypatch, provider):
    prepared = _prepared(tmp_path, count=10)
    module = windowed_inference if provider == "mapanything" else windowed_da3_inference
    settings_class = inference.MapAnythingScanSettings if provider == "mapanything" else da3_inference.DA3PhoneScanSettings
    runner = module.run_windowed_mapanything_scan if provider == "mapanything" else module.run_windowed_da3_phone_scan
    _stub_artifacts(monkeypatch, module)
    monkeypatch.setattr(module, "_estimate_bridge_similarity", lambda *_: (1.0, np.eye(3), np.zeros(3), {"test": True}))
    monkeypatch.setattr(module, "_refine_window_scale_translation", lambda *_: (1.0, np.zeros(3), {"test": True}))
    monkeypatch.setattr(module, "_window_overlap_residual", lambda *_: {"test": True})
    windows = []

    def fake_window(_scan, output, subset, _settings, _progress):
        assert subset["camera_calibration"] == prepared["camera_calibration"]
        windows.append(subset)
        (output / "raw").mkdir()
        (output / "views").mkdir()
        for index, row in enumerate(subset["frames"]):
            height, width = 14, 24
            matrix = np.array([[19.0, 0.0, 12.0], [0.0, 18.0, 7.0], [0.0, 0.0, 1.0]])
            pose = np.eye(4)
            pose[0, 3] = row["adaptive_global_index"] * 0.05
            depth = np.full((height, width), 2.0, dtype=np.float32)
            valid = np.ones((height, width), dtype=bool)
            valid[:, 0] = False
            raw_fields = calibration_raw_fields(row, valid, network_intrinsics=matrix * np.array([[1.2, 1, 1], [1, 1.2, 1], [1, 1, 1]]) if provider == "da3" else None)
            np.savez_compressed(output / "raw" / f"view_{index:04d}.npz", world_points=da3_inference._world_points(depth, matrix, pose), depth_z=depth, confidence=np.ones_like(depth), mask=valid, camera_pose=pose, intrinsics=matrix, metric_scaling_factor=np.ones(1), model_rgb=np.zeros((height, width, 3), dtype=np.uint8), **raw_fields)
            for kind in ("rgb", "depth", "confidence", "mask"):
                (output / "views" / f"view_{index:04d}_{kind}.png").write_bytes(b"preview")
        return {}

    output = tmp_path / "out"
    output.mkdir()
    result = runner(tmp_path, output, prepared, settings_class(device="cpu", max_joint_views=8, window_overlap_views=6), lambda *_: None, window_runner=fake_window)
    assert len(windows) == 2
    assert result["camera_calibration"] == prepared["camera_calibration"]
    assert result["frames"][9]["camera_intrinsics"] == prepared["frames"][9]["camera_intrinsics"]
    with np.load(output / "raw/view_0009.npz", allow_pickle=False) as raw:
        assert not raw["rectification_valid_mask"][:, 0].any()
        assert json.loads(str(raw["camera_intrinsics_json"])) == prepared["frames"][9]["camera_intrinsics"]
        if provider == "da3":
            assert "network_intrinsics" in raw
            with np.load(output / "camera_solution.npz") as solution:
                np.testing.assert_array_equal(raw["network_intrinsics"], solution["network_intrinsics"][9])
