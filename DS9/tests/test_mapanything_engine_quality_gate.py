from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from array import array
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = _load(
    "ds9_mapanything_quality_common_test",
    "DS9/scripts/engine_maintenance_common.py",
)


def _authority() -> dict[str, object]:
    return {
        "contract": common.MAPANYTHING_QUALITY_CONTRACT,
        "fixture": {
            "artifact_relative_path": (
                "models/engine_validation/mapanything/images-b3.raw"
            ),
            "sha256": "0" * 64,
            "size_bytes": 3 * 3 * 2 * 2 * 4,
            "tensor": {
                "name": "images",
                "dtype": "float32",
                "shape": [3, 3, 2, 2],
            },
            "identical_batch_members": True,
        },
        "trtexec_args": [
            "--iterations=1",
            "--duration=0",
            "--warmUp=0",
            "--avgRuns=1",
        ],
        "outputs": {
            "depth": {
                "dtype": "float32",
                "shape": [3, 1, 2, 2],
                "finite_fraction": {"min": 1.0, "max": 1.0},
                "positive_fraction": {"min": 1.0, "max": 1.0},
                "distribution": {
                    "min": {"min": 0.9, "max": 1.1},
                    "max": {"min": 3.9, "max": 4.1},
                    "mean": {"min": 2.4, "max": 2.6},
                    "p50": {"min": 2.4, "max": 2.6},
                },
                "batch_max_abs_delta": 0.0001,
            },
            "conf": {
                "dtype": "float32",
                "shape": [3, 1, 2, 2],
                "finite_fraction": {"min": 1.0, "max": 1.0},
                "positive_fraction": {"min": 1.0, "max": 1.0},
                "forbidden_values": [-65504.0, -3.4028235e38, 3.4028235e38],
                "distribution": {
                    "min": {"min": 0.99, "max": 1.01},
                    "max": {"min": 0.99, "max": 1.01},
                    "mean": {"min": 0.99, "max": 1.01},
                    "p50": {"min": 0.99, "max": 1.01},
                },
                "batch_max_abs_delta": 0.0001,
            },
            "mask": {
                "dtype": "float32",
                "shape": [3, 1, 2, 2],
                "finite_fraction": {"min": 1.0, "max": 1.0},
                "positive_fraction": {"min": 0.74, "max": 0.76},
                "mask_threshold": 0.5,
                "mask_coverage": {"min": 0.74, "max": 0.76},
                "distribution": {
                    "min": {"min": 0.0, "max": 0.0},
                    "max": {"min": 1.0, "max": 1.0},
                    "mean": {"min": 0.74, "max": 0.76},
                    "p50": {"min": 1.0, "max": 1.0},
                },
                "batch_max_abs_delta": 0.0001,
            },
        },
    }


def _platform() -> dict[str, str]:
    return {
        "image": "noesis-ds9-dev:test",
        "image_id": "sha256:" + "1" * 64,
        "base_digest": "sha256:" + "2" * 64,
        "tensorrt_version": "10.16.0.72",
        "cuda_version": "13.0",
        "driver_version": "595.71.05",
        "gpu_name": "NVIDIA GeForce RTX 3060",
        "gpu_uuid": "GPU-test",
        "gpu_compute_capability": "8.6",
        "gpu_memory_mib": "12288",
        "expected_trtexec_banner": common.DS9_TRTEXEC_BANNER,
    }


def _write_fixture(path: Path, authority: dict[str, object]) -> None:
    one = array("f", [float(index) / 11.0 for index in range(12)])
    values = array("f")
    for _ in range(3):
        values.extend(one)
    path.write_bytes(values.tobytes())
    os.chmod(path, 0o600)
    authority["fixture"]["sha256"] = common.sha256_file(path)


def _valid_outputs() -> dict[str, list[float]]:
    return {
        "depth": [1.0, 2.0, 3.0, 4.0] * 3,
        "conf": [1.0, 1.0, 1.0, 1.0] * 3,
        "mask": [1.0, 1.0, 1.0, 0.0] * 3,
    }


def _write_output(path: Path, outputs: dict[str, list[float]]) -> None:
    payload = [
        {
            "name": name,
            "dimensions": "3x1x2x2",
            "values": values,
        }
        for name, values in outputs.items()
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)


class MapAnythingEngineQualityGateTests(unittest.TestCase):
    def test_tracked_authority_is_strict_and_fp32(self) -> None:
        authority = common.mapanything_quality_gate_from_source_contracts(
            REPO_ROOT / "DS9/config/engine_source_contracts.json"
        )
        self.assertEqual(
            authority["fixture"]["sha256"],
            "dbe62bc2ca22077d17070bb61babb817de7f22208547372e58a0153a54bd9fc0",
        )
        self.assertEqual(authority["fixture"]["tensor"]["dtype"], "float32")
        self.assertEqual(
            {row["dtype"] for row in authority["outputs"].values()},
            {"float32"},
        )

    def test_skip_inference_is_not_functional_quality_proof(self) -> None:
        transcript = "\n".join(
            (
                "Loaded engine size: 10 MiB",
                "Engine deserialized in 0.1 sec.",
                "Skipped inference phase since --skipInference is added.",
                "&&&& PASSED TensorRT.trtexec",
            )
        )
        with self.assertRaisesRegex(
            common.EngineMaintenanceError, "skipInference"
        ):
            common.validate_trtexec_inference_transcript(transcript)

    def test_invalid_numeric_outputs_fail_closed(self) -> None:
        mutations = {
            "nan_depth": lambda rows: rows["depth"].__setitem__(0, math.nan),
            "inf_depth": lambda rows: rows["depth"].__setitem__(0, math.inf),
            "sentinel_conf": lambda rows: rows["conf"].__setitem__(0, -65504.0),
            "zero_mask": lambda rows: rows.__setitem__("mask", [0.0] * 12),
            "out_of_envelope": lambda rows: rows.__setitem__(
                "depth", [100.0] * 12
            ),
            "batch_inconsistent": lambda rows: rows["depth"].__setitem__(8, 1.5),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                authority = _authority()
                fixture = root / "images-b3.raw"
                _write_fixture(fixture, authority)
                output = root / common.MAPANYTHING_QUALITY_OUTPUT_NAME
                rows = _valid_outputs()
                mutate(rows)
                _write_output(output, rows)
                with self.assertRaises(common.EngineMaintenanceError):
                    common.evaluate_mapanything_quality_output(output, authority)

    def test_overflow_exponent_is_not_admitted_as_finite_quality_data(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            authority = _authority()
            fixture = root / "images-b3.raw"
            _write_fixture(fixture, authority)
            output = root / common.MAPANYTHING_QUALITY_OUTPUT_NAME
            output.write_text(
                '[{"name":"depth","dimensions":"3x1x2x2",'
                '"values":[1e309]}]',
                encoding="utf-8",
            )
            os.chmod(output, 0o600)

            with self.assertRaisesRegex(
                common.EngineMaintenanceError,
                "not strict JSON",
            ):
                common.evaluate_mapanything_quality_output(output, authority)

    def _prepared_run(
        self, root: Path
    ) -> tuple[object, Path, Path, Path, dict[str, object], dict[str, object]]:
        authority = _authority()
        fixture = root / "images-b3.raw"
        _write_fixture(fixture, authority)
        target = root / "engines" / "mapanything.plan"
        target.parent.mkdir()
        target.write_bytes(b"prior-engine")
        candidate = target.with_name(".mapanything.plan.building-test")
        candidate.write_bytes(b"valid-candidate-engine")
        evidence = root / "evidence"
        run = common.EngineMaintenanceRun(
            name="mapanything",
            target=target,
            evidence_root=evidence,
            inputs={"quality_fixture": fixture},
            repo_root=REPO_ROOT,
            run_id="20260712T010203000000Z",
            metadata={
                "platform": _platform(),
                "build_contract": {"precision": "fp32"},
                "quality_gate_authority": common.validate_mapanything_quality_gate_authority(
                    authority
                ),
            },
        )
        run.record_candidate(candidate)
        output = run.run_dir / common.MAPANYTHING_QUALITY_OUTPUT_NAME
        _write_output(output, _valid_outputs())
        command = common._quality_command(
            executable="/usr/bin/trtexec",
            engine_path=candidate,
            fixture_path=fixture,
            output_path=output,
            authority=common.validate_mapanything_quality_gate_authority(authority),
        )
        command_record = {
            "label": common.MAPANYTHING_QUALITY_COMMAND_LABEL,
            "command": command,
            "proof": "trtexec_inference",
            "returncode": 0,
            "timed_out": False,
            "output_exceeded": False,
            "status": "passed",
        }
        run.payload["commands"].append(command_record)
        run.persist()
        return run, target, candidate, fixture, authority, command_record

    def test_skip_inference_only_candidate_never_installs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, target, candidate, _fixture, _authority_value, _record = (
                self._prepared_run(root)
            )
            run.payload["commands"] = [
                {
                    "label": "load-candidate",
                    "command": [
                        "/usr/bin/trtexec",
                        f"--loadEngine={candidate}",
                        "--skipInference",
                    ],
                    "proof": "trtexec_load",
                    "returncode": 0,
                    "timed_out": False,
                    "output_exceeded": False,
                    "status": "passed",
                }
            ]
            run.persist()
            with self.assertRaisesRegex(
                common.EngineMaintenanceError, "requires a functional quality receipt"
            ):
                run.install_candidate(candidate)
            self.assertEqual(target.read_bytes(), b"prior-engine")
            self.assertEqual(candidate.read_bytes(), b"valid-candidate-engine")

    def test_mismatched_receipt_engine_never_installs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, target, candidate, fixture, authority, command_record = (
                self._prepared_run(root)
            )
            run.record_mapanything_functional_quality(
                candidate=candidate,
                fixture_path=fixture,
                output_path=run.run_dir / common.MAPANYTHING_QUALITY_OUTPUT_NAME,
                authority=authority,
                command_record=command_record,
            )
            candidate.write_bytes(b"changed-after-receipt")
            with self.assertRaisesRegex(
                common.EngineMaintenanceError, "engine hash/size differs"
            ):
                run.install_candidate(candidate)
            self.assertEqual(target.read_bytes(), b"prior-engine")

    def test_tampered_receipt_never_installs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, target, candidate, fixture, authority, command_record = (
                self._prepared_run(root)
            )
            run.record_mapanything_functional_quality(
                candidate=candidate,
                fixture_path=fixture,
                output_path=run.run_dir / common.MAPANYTHING_QUALITY_OUTPUT_NAME,
                authority=authority,
                command_record=command_record,
            )
            receipt = run.payload["evidence"][
                common.MAPANYTHING_QUALITY_EVIDENCE_LABEL
            ]
            receipt["outputs"]["depth"]["positive_count"] -= 1
            run.persist()
            with self.assertRaisesRegex(
                common.EngineMaintenanceError, "receipt is invalid"
            ):
                run.install_candidate(candidate)
            self.assertEqual(target.read_bytes(), b"prior-engine")
            self.assertTrue(candidate.exists())

    def test_valid_receipt_installs_atomically_and_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, target, candidate, fixture, authority, command_record = (
                self._prepared_run(root)
            )
            receipt = run.record_mapanything_functional_quality(
                candidate=candidate,
                fixture_path=fixture,
                output_path=run.run_dir / common.MAPANYTHING_QUALITY_OUTPUT_NAME,
                authority=authority,
                command_record=command_record,
            )
            installed = run.install_candidate(candidate)
            self.assertFalse(candidate.exists())
            self.assertEqual(target.read_bytes(), b"valid-candidate-engine")
            self.assertEqual(installed["sha256"], receipt["engine"]["sha256"])
            validated = common.validate_mapanything_functional_quality_receipt(
                run.payload,
                authority=authority,
                expected_engine_path=target,
                evidence_directory=run.run_dir,
                fixture_path=fixture,
            )
            self.assertEqual(validated["receipt_sha256"], receipt["receipt_sha256"])

if __name__ == "__main__":
    unittest.main()
