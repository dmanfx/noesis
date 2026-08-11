from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_mapanything_intrinsics_eager.py"
)
SPEC = importlib.util.spec_from_file_location("mapanything_intrinsics_eager", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_scene_output(path: Path, values: dict[str, np.ndarray]) -> None:
    payload = []
    for name in ("conf", "depth", "mask"):
        array = np.asarray(values[name], dtype=np.float32)
        payload.append(
            {
                "name": name,
                "dimensions": "x".join(
                    str(value)
                    for value in (
                        array.shape[0],
                        1,
                        array.shape[1],
                        array.shape[2],
                    )
                ),
                "values": array.reshape(-1).tolist(),
            }
        )
    _write_json(path, payload)


def test_pixel_center_intrinsics_transform_matches_inverse_mapping() -> None:
    source_k = np.asarray(
        [
            [625.0370802939991, 0.0, 914.1592785997523],
            [0.0, 623.8566609615328, 560.8911232935495],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    network_k = MODULE.transform_intrinsics_pixel_center(
        source_k,
        source_size=(1920, 1080),
        resized_size=(518, 291),
        pad_left=0,
        pad_top=1,
    )
    np.testing.assert_allclose(
        network_k,
        np.asarray(
            [
                [168.62979125976562, 0.0, 246.2674560546875],
                [0.0, 168.09471130371094, 151.7637176513672],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-6,
    )

    sx = 518.0 / 1920.0
    sy = 291.0 / 1080.0
    for u_source, v_source in ((0.0, 0.0), (914.0, 561.0), (1919.0, 1079.0)):
        u_network = ((u_source + 0.5) * sx) - 0.5
        v_network = ((v_source + 0.5) * sy) - 0.5 + 1.0
        source_x = (u_source - source_k[0, 2]) / source_k[0, 0]
        source_y = (v_source - source_k[1, 2]) / source_k[1, 1]
        network_x = (u_network - network_k[0, 2]) / network_k[0, 0]
        network_y = (v_network - network_k[1, 2]) / network_k[1, 1]
        assert network_x == pytest.approx(source_x, abs=1e-7)
        assert network_y == pytest.approx(source_y, abs=1e-7)


def test_ray_generation_and_intrinsics_recovery_round_trip() -> None:
    intrinsics = np.asarray(
        [[205.0, 0.0, 252.5], [0.0, 204.4, 142.1], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rays = MODULE.rays_from_intrinsics_numpy(
        intrinsics,
        height=294,
        width=518,
    )[0]
    recovered = MODULE.fit_pinhole_intrinsics(rays, sample_stride=7)
    assert recovered["fx"] == pytest.approx(205.0, abs=1e-4)
    assert recovered["fy"] == pytest.approx(204.4, abs=1e-4)
    assert recovered["cx"] == pytest.approx(252.5, abs=1e-4)
    assert recovered["cy"] == pytest.approx(142.1, abs=1e-4)
    assert recovered["fit_residual_px_p95"] < 1e-4
    assert MODULE.angular_error_metrics(rays, rays)["degrees_p95"] < 1e-5


def test_model_ray_helper_parity_is_tight_and_fail_closed() -> None:
    intrinsics = np.asarray(
        [[205.0, 0.0, 252.5], [0.0, 204.4, 142.1], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    constructed = MODULE.rays_from_intrinsics_numpy(
        intrinsics,
        height=12,
        width=18,
    )
    model_helper = constructed.copy()
    model_helper[..., 0] = np.nextafter(
        model_helper[..., 0],
        np.float32(np.inf),
    )
    metrics = MODULE.verify_model_ray_helper_parity(
        constructed,
        model_helper,
    )
    assert metrics["abs_delta"]["max"] <= MODULE.RAY_HELPER_ABS_TOLERANCE
    assert metrics["angular_error_degrees"]["degrees_p95"] < 1e-4

    materially_wrong = model_helper.copy()
    materially_wrong[0, 0, 0, 0] += 1e-3
    with pytest.raises(MODULE.DiagnosticError, match="geometry helper"):
        MODULE.verify_model_ray_helper_parity(
            constructed,
            materially_wrong,
        )


def test_model_arm_forces_official_deterministic_geometry_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(MODULE, "EXPECTED_SHAPE", (1, 3, 2, 2))

    class FakeModel:
        def __init__(self) -> None:
            self.geometric_input_config = {
                "overall_prob": 0.25,
                "dropout_prob": 0.75,
                "ray_dirs_prob": 0.25,
                "depth_prob": 0.25,
                "cam_prob": 0.25,
            }
            self._original: dict[str, float] | None = None

        def _configure_geometric_input_config(
            self,
            *,
            use_calibration: bool,
            use_depth: bool,
            use_pose: bool,
            use_depth_scale: bool,
            use_pose_scale: bool,
        ) -> None:
            assert use_depth is False
            assert use_pose is False
            assert use_depth_scale is False
            assert use_pose_scale is False
            self._original = dict(self.geometric_input_config)
            self.geometric_input_config.update(
                {
                    "overall_prob": 1.0 if use_calibration else 0.0,
                    "dropout_prob": 0.0 if use_calibration else 1.0,
                    "ray_dirs_prob": 1.0 if use_calibration else 0.0,
                    "depth_prob": 0.0,
                    "cam_prob": 0.0,
                }
            )

        def _restore_original_geometric_input_config(self) -> None:
            assert self._original is not None
            self.geometric_input_config.update(self._original)

        def forward(
            self,
            views: list[dict[str, object]],
            *,
            memory_efficient_inference: bool,
        ) -> list[dict[str, object]]:
            assert memory_efficient_inference is False
            conditioned = "ray_directions_cam" in views[0]
            assert self.geometric_input_config["ray_dirs_prob"] == (
                1.0 if conditioned else 0.0
            )
            depth = torch.ones((1, 2, 2, 3), dtype=torch.float32)
            rays = torch.nn.functional.normalize(depth, dim=-1)
            return [
                {
                    "pts3d_cam": depth,
                    "ray_directions": rays,
                    "conf": torch.ones((1, 2, 2), dtype=torch.float32),
                    "non_ambiguous_mask": torch.ones(
                        (1, 2, 2),
                        dtype=torch.float32,
                    ),
                }
            ]

    model = FakeModel()
    original = dict(model.geometric_input_config)
    normalized = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    image_only = MODULE._run_model_arm(
        model=model,
        normalized_images=normalized,
        norm_type="dinov2",
        ray_directions=None,
    )
    assert image_only["depth"].shape == (1, 2, 2)
    assert model.geometric_input_config == original

    supplied = torch.ones((1, 2, 2, 3), dtype=torch.float32)
    conditioned = MODULE._run_model_arm(
        model=model,
        normalized_images=normalized,
        norm_type="dinov2",
        ray_directions=supplied,
    )
    assert conditioned["rays"].shape == (1, 2, 2, 3)
    assert model.geometric_input_config == original


def test_transform_and_ray_contracts_fail_closed() -> None:
    with pytest.raises(MODULE.DiagnosticError, match="positive"):
        MODULE.transform_intrinsics_pixel_center(
            np.eye(3),
            source_size=(0, 1080),
            resized_size=(518, 291),
            pad_left=0,
            pad_top=1,
        )
    with pytest.raises(MODULE.DiagnosticError, match="non-zero"):
        MODULE.rays_from_intrinsics_numpy(
            np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
            height=2,
            width=2,
        )
    with pytest.raises(MODULE.DiagnosticError, match="equal"):
        MODULE.angular_error_metrics(
            np.ones((2, 2, 3), dtype=np.float32),
            np.ones((2, 3, 3), dtype=np.float32),
        )


def test_layer_parity_metrics_are_exact_and_fail_closed() -> None:
    reference = np.asarray([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
    actual = np.asarray([[[0.0, 1.1], [1.8, 3.4]]], dtype=np.float32)
    metrics = MODULE.layer_parity_metrics(actual, reference)
    assert metrics["exact_equal"] is False
    assert metrics["abs_delta"]["p50"] == pytest.approx(0.15)
    assert metrics["abs_delta"]["p95"] == pytest.approx(0.37)
    assert metrics["abs_delta"]["max"] == pytest.approx(0.4)
    assert MODULE.layer_parity_metrics(reference, reference)["exact_equal"] is True

    with pytest.raises(MODULE.DiagnosticError, match="shapes differ"):
        MODULE.layer_parity_metrics(actual, reference[:, :, :1])
    invalid = actual.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(MODULE.DiagnosticError, match="finite"):
        MODULE.layer_parity_metrics(invalid, reference)


def test_resolve_hf_artifacts_locks_full_revision_and_file_hashes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    weights = tmp_path / "model.safetensors"
    config.write_bytes(b'{"model":"locked"}\n')
    weights.write_bytes(b"locked-weights")

    class FakeExporter:
        @staticmethod
        def hf_hub_download(
            *,
            repo_id: str,
            filename: str,
            revision: str,
            local_files_only: bool,
        ) -> str:
            assert repo_id == "facebook/map-anything-apache"
            assert revision == "a" * 40
            assert local_files_only is True
            return str({"config.json": config, "model.safetensors": weights}[filename])

    records = MODULE.resolve_hf_artifacts(
        exporter=FakeExporter,
        model_id="facebook/map-anything-apache",
        revision="a" * 40,
    )
    assert records["config"]["sha256"] == MODULE.sha256_file(config)
    assert records["model_safetensors"]["sha256"] == MODULE.sha256_file(weights)
    assert records["model_safetensors"]["size_bytes"] == len(b"locked-weights")

    with pytest.raises(MODULE.DiagnosticError, match="40-hex"):
        MODULE.resolve_hf_artifacts(
            exporter=FakeExporter,
            model_id="facebook/map-anything-apache",
            revision="main",
        )


def test_load_trt_reference_binds_manifest_and_rejects_input_mismatch(
    tmp_path: Path,
) -> None:
    input_raw = tmp_path / "input.raw"
    fixture_receipt = tmp_path / "fixture.json"
    output = tmp_path / "scene-output.json"
    engine = tmp_path / "engine.plan"
    manifest_path = tmp_path / "comparison.json"
    input_raw.write_bytes(b"locked-input")
    fixture_receipt.write_bytes(b'{"locked":true}\n')
    engine.write_bytes(b"engine")
    layers = {
        "depth": np.ones((1, 2, 2), dtype=np.float32),
        "conf": np.full((1, 2, 2), 2.0, dtype=np.float32),
        "mask": np.ones((1, 2, 2), dtype=np.float32),
    }
    _write_scene_output(output, layers)
    manifest = {
        "contract": MODULE.COMPARISON_MANIFEST_CONTRACT,
        "candidates": {
            MODULE.TRT_REFERENCE_CANDIDATE: {
                "include_intrinsics": False,
                "fixture_tensor": {
                    "path": str(input_raw),
                    "sha256": MODULE.sha256_file(input_raw),
                    "size_bytes": input_raw.stat().st_size,
                },
                "fixture_receipt": {
                    "path": str(fixture_receipt),
                    "sha256": MODULE.sha256_file(fixture_receipt),
                    "size_bytes": fixture_receipt.stat().st_size,
                },
                "model_output": {
                    "path": str(output),
                    "sha256": MODULE.sha256_file(output),
                    "size_bytes": output.stat().st_size,
                },
                "engine": {
                    "path": str(engine),
                    "sha256": MODULE.sha256_file(engine),
                    "size_bytes": engine.stat().st_size,
                },
            }
        },
    }
    _write_json(manifest_path, manifest)

    observed, identity = MODULE.load_trt_reference(
        comparison_manifest_path=manifest_path,
        input_raw=input_raw,
        fixture_receipt_path=fixture_receipt,
    )
    np.testing.assert_array_equal(observed["depth"], layers["depth"])
    assert identity["candidate_id"] == MODULE.TRT_REFERENCE_CANDIDATE
    assert identity["reference_output"]["sha256"] == MODULE.sha256_file(output)

    manifest["candidates"][MODULE.TRT_REFERENCE_CANDIDATE]["fixture_tensor"][
        "sha256"
    ] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(MODULE.DiagnosticError, match="fixture tensor differs"):
        MODULE.load_trt_reference(
            comparison_manifest_path=manifest_path,
            input_raw=input_raw,
            fixture_receipt_path=fixture_receipt,
        )
