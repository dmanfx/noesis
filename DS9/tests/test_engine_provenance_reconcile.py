from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import onnx
import yaml
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reconciler = _load(
    "ds9_engine_provenance_reconciler",
    "DS9/scripts/reconcile_engine_provenance.py",
)
validator = _load(
    "ds9_engine_provenance_validator",
    "DS9/scripts/validate_asset_manifest.py",
)
finalizer = _load(
    "ds9_engine_finalize_transaction",
    "DS9/scripts/finalize_engine_realization.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_onnx(path: Path, output_shape: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [3, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, path)


class EngineProvenanceReconcileTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, object]:
        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        source_contracts = REPO_ROOT / "DS9/config/engine_source_contracts.json"
        contracts = json.loads(source_contracts.read_text(encoding="utf-8"))[
            "contracts"
        ]
        base = yaml.safe_load(
            (REPO_ROOT / "DS9/asset_manifest.yaml").read_text(encoding="utf-8")
        )
        for artifact in base["artifacts"]:
            if artifact.get("kind") != "tensorrt_engine":
                continue
            artifact["state"] = "missing"
            artifact["provenance"]["output_sha256"] = None
            artifact["provenance"]["built_at_utc"] = None
            artifact["provenance"]["build_host"] = None
            artifact["provenance"]["command"] = None
            artifact["provenance"].pop("maintenance", None)

        engines = {
            "yolo26_m": (
                "engine.yolo26_detect_m",
                [3, 300, 6],
                b"validated-yolo26m-engine",
            ),
            "yolo26_seg_s": (
                "engine.yolo26_seg_s",
                [3, 30, 4102],
                b"validated-yolo26s-seg-engine",
            ),
        }
        maintenance: dict[str, Path] = {}
        for name, (artifact_id, output_shape, output_bytes) in engines.items():
            artifact = next(
                row for row in base["artifacts"] if row["id"] == artifact_id
            )
            artifact["required_profiles"] = ["unit"]
            source = validator._physical_path(
                validator._relative_path(artifact["sources"][0]), artifact_root
            )
            output = validator._physical_path(
                validator._relative_path(artifact["output"]), artifact_root
            )
            _write_onnx(source, output_shape)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(output_bytes)
            artifact["provenance"]["source_sha256"] = validator._hash_sources(
                artifact["sources"], artifact_root=artifact_root
            )

            run_dir = artifact_root / f"models/engine_maintenance/run-{name}"
            run_dir.mkdir(parents=True)
            manifest = run_dir / "manifest.json"
            build_contract = copy.deepcopy(contracts[name]["maintenance_build"])
            build_contract["onnx"] = copy.deepcopy(contracts[name]["onnx"])
            platform = {
                "image": base["target"]["build_image"]["reference"],
                "image_id": base["target"]["build_image"]["image_id"],
                "base_digest": base["target"]["build_image"]["base_digest"],
                "tensorrt_version": base["target"]["build_image"]["tensorrt_version"],
                "cuda_version": base["target"]["build_image"]["cuda_version"],
                "driver_version": "595.71.05",
                "gpu_name": "NVIDIA GeForce RTX 3060",
                "gpu_uuid": "GPU-test",
                "gpu_compute_capability": "8.6",
                "gpu_memory_mib": "12288",
            }
            commands = []
            for label, proof in (
                ("probe-trtexec", "trtexec_probe"),
                ("build", "trtexec_build"),
                ("load-candidate", "trtexec_load"),
                ("load-installed", "trtexec_load"),
            ):
                commands.append(
                    {
                        "label": label,
                        "command": ["trtexec", f"--fixture={label}"],
                        "proof": proof,
                        "returncode": 0,
                        "timed_out": False,
                        "status": "passed",
                    }
                )
            payload = {
                "schema_version": 1,
                "contract": "noesis.ds9.engine_maintenance",
                "status": "complete",
                "engine": name,
                "target": str(Path("/workspace") / artifact["output"]),
                "prior": {
                    "exists": False,
                    "path": str(Path("/workspace") / artifact["output"]),
                },
                "completed_at_utc": "2026-07-10T23:00:00Z",
                "inputs_revalidated_at_utc": "2026-07-10T22:59:59Z",
                "inputs": {
                    "source_contracts": {"sha256": _sha256_file(source_contracts)}
                },
                "installed": {
                    "sha256": _sha256_file(output),
                    "size_bytes": output.stat().st_size,
                },
                "metadata": {
                    "platform": platform,
                    "build_contract": build_contract,
                },
                "commands": commands,
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            os.chmod(manifest, 0o600)
            maintenance[name] = manifest

        manifest_path = root / "asset_manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
        )
        finalizer.BASE_MANIFEST_AUTHORITY = manifest_path
        return {
            "artifact_root": artifact_root,
            "maintenance": maintenance,
            "manifest": manifest_path,
            "source_contracts": source_contracts,
        }

    def _reconcile(
        self,
        fixture: dict[str, object],
        name: str,
        *,
        expected_realization: str = "missing",
    ) -> dict[str, object]:
        manifest = fixture["manifest"]
        source_contracts = fixture["source_contracts"]
        assert isinstance(manifest, Path)
        assert isinstance(source_contracts, Path)
        maintenance, gpu_memory_guard = self._prepare_direct_guard(fixture, name)
        return reconciler.reconcile(
            engine_name=name,
            maintenance_manifest=maintenance,
            gpu_memory_guard=gpu_memory_guard,
            artifact_root=fixture["artifact_root"],
            base_manifest_path=manifest,
            expected_base_manifest_sha256=_sha256_file(manifest),
            expected_source_contracts_sha256=_sha256_file(source_contracts),
            expected_realization_sha256=expected_realization,
            dry_run=False,
        )

    def _prepare_direct_guard(
        self,
        fixture: dict[str, object],
        name: str,
    ) -> tuple[Path, dict[str, object]]:
        artifact_root = Path(fixture["artifact_root"])
        source = Path(fixture["maintenance"][name])
        payload = json.loads(source.read_text(encoding="utf-8"))
        transaction_id = finalizer.new_run_id()
        prepared_transaction_sha256 = "c" * 64
        transaction_cohort = (
            artifact_root / "models/engine_finalize" / f"{transaction_id}-{name}"
        )
        payload["run_id"] = transaction_id
        payload["created_at_utc"] = finalizer._utc_now()
        payload.setdefault("metadata", {})["host_transaction"] = {
            "transaction_id": transaction_id,
            "transaction_sha256": prepared_transaction_sha256,
            "transaction_manifest": str(
                Path("/workspace/DS9/models/engine_finalize")
                / f"{transaction_id}-{name}"
                / "transaction.json"
            ),
        }
        run_dir = (
            artifact_root / "models/engine_maintenance" / f"{transaction_id}-{name}"
        )
        run_dir.mkdir(mode=0o700)
        maintenance = run_dir / "manifest.json"
        maintenance.write_text(json.dumps(payload), encoding="utf-8")
        maintenance.chmod(0o600)
        fixture["maintenance"][name] = maintenance
        guard = self._write_gpu_guard_record(
            fixture,
            name,
            transaction_id=transaction_id,
            prepared_transaction_sha256=prepared_transaction_sha256,
            cohort=transaction_cohort,
            maintenance_manifest=maintenance,
        )
        fixture.setdefault("gpu_memory_guards", {})[name] = guard
        return maintenance, guard

    def _engine_path(self, fixture: dict[str, object], artifact_id: str) -> Path:
        manifest = fixture["manifest"]
        artifact_root = fixture["artifact_root"]
        assert isinstance(manifest, Path)
        assert isinstance(artifact_root, Path)
        base = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        artifact = next(row for row in base["artifacts"] if row["id"] == artifact_id)
        return validator._physical_path(
            validator._relative_path(artifact["output"]), artifact_root
        )

    def _snapshot(self, fixture: dict[str, object], name: str) -> dict[str, object]:
        return finalizer.snapshot(
            engine_name=name,
            artifact_root=fixture["artifact_root"],
            base_manifest_path=fixture["manifest"],
            validation_profile="engine_finalize",
        )

    def _gpu_guard_kwargs(
        self,
        fixture: dict[str, object],
        snapshot: dict[str, object],
        name: str,
        *,
        maintenance_manifest: Path | None = None,
    ) -> dict[str, object]:
        transaction_path = Path(snapshot["transaction_manifest"])
        transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
        if maintenance_manifest is None:
            maintenance_manifest = Path(fixture["maintenance"][name])
        record = self._write_gpu_guard_record(
            fixture,
            name,
            transaction_id=str(transaction["transaction_id"]),
            prepared_transaction_sha256=str(snapshot["transaction_sha256"]),
            cohort=transaction_path.parent,
            maintenance_manifest=maintenance_manifest,
        )
        return {
            "gpu_guard_evidence": Path(fixture["artifact_root"]) / str(record["path"]),
            "expected_gpu_guard_sha256": record["sha256"],
            "gpu_guard_container_id": record["container_id"],
            "gpu_guard_wrapper_pid": record["wrapper_pid"],
            "gpu_guard_wrapper_start_time_ticks": record["wrapper_start_time_ticks"],
        }

    def _write_gpu_guard_record(
        self,
        fixture: dict[str, object],
        name: str,
        *,
        transaction_id: str,
        prepared_transaction_sha256: str,
        cohort: Path,
        maintenance_manifest: Path,
    ) -> dict[str, object]:
        artifact_root = Path(fixture["artifact_root"])
        maintenance = json.loads(maintenance_manifest.read_text(encoding="utf-8"))
        expected_uuid = maintenance["metadata"]["platform"]["gpu_uuid"]
        guard_mib = finalizer.gpu_sampler.reviewed_guard_mib(name)
        wrapper_pid = os.getpid()
        wrapper_start_time_ticks = finalizer.gpu_sampler._proc_start_time_ticks(
            wrapper_pid
        )
        container_id = "d" * 64
        transaction_root = artifact_root / "models/engine_finalize"
        transaction_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        transaction_root.chmod(0o700)
        cohort.mkdir(mode=0o700, exist_ok=True)
        cohort.chmod(0o700)
        evidence = cohort / "gpu-memory.jsonl"
        used_bytes = 512 * finalizer.gpu_sampler.MIB
        total_bytes = 12288 * finalizer.gpu_sampler.MIB
        sampled_at = "2026-07-11T12:00:00.000001Z"
        rows = [
            {
                "kind": "header",
                "schema_version": 1,
                "contract": finalizer.gpu_sampler.CONTRACT,
                "device_index": 0,
                "expected_uuid": expected_uuid,
                "engine": name,
                "transaction_id": transaction_id,
                "prepared_transaction_sha256": prepared_transaction_sha256,
                "artifact_root_id": hashlib.sha256(
                    str(artifact_root).encode("utf-8")
                ).hexdigest(),
                "container_id": container_id,
                "guard_mib": guard_mib,
                "guard_bytes": guard_mib * finalizer.gpu_sampler.MIB,
                "interval_ms": finalizer.GPU_MEMORY_SAMPLE_INTERVAL_MS,
                "max_gap_ms": finalizer.GPU_MEMORY_MAX_GAP_MS,
                "parent_pid": wrapper_pid,
                "parent_start_time_ticks": wrapper_start_time_ticks,
                "sampler_pid": wrapper_pid,
                "sampler_start_time_ticks": wrapper_start_time_ticks,
                "started_at_utc": "2026-07-11T12:00:00.000000Z",
            },
            {
                "kind": "sample",
                "sequence": 1,
                "sampled_at_utc": sampled_at,
                "monotonic_ns": 1_000_000_000,
                "total_mib": 12288,
                "total_bytes": total_bytes,
                "reserved_mib": 0,
                "reserved_bytes": 0,
                "used_mib": 512,
                "used_bytes": used_bytes,
            },
            {
                "kind": "footer",
                "state": "stopped",
                "sample_count": 1,
                "peak_mib": 512,
                "peak_reserved_mib": 0,
                "first_sample_at_utc": sampled_at,
                "last_sample_at_utc": sampled_at,
                "maximum_gap_ms": 0.0,
                "ended_at_utc": "2026-07-11T12:00:00.000002Z",
            },
        ]
        evidence.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        evidence.chmod(0o600)
        summary = finalizer.gpu_sampler.summarize(
            argparse.Namespace(
                evidence=evidence,
                device_index=0,
                expected_uuid=expected_uuid,
                engine=name,
                transaction_id=transaction_id,
                prepared_transaction_sha256=prepared_transaction_sha256,
                artifact_root_id=hashlib.sha256(
                    str(artifact_root).encode("utf-8")
                ).hexdigest(),
                container_id=container_id,
                guard_mib=guard_mib,
                interval_ms=finalizer.GPU_MEMORY_SAMPLE_INTERVAL_MS,
                max_gap_ms=finalizer.GPU_MEMORY_MAX_GAP_MS,
                parent_pid=wrapper_pid,
                parent_start_time_ticks=wrapper_start_time_ticks,
                allow_active=False,
            )
        )
        return {
            "contract": finalizer.gpu_sampler.CONTRACT,
            "path": evidence.relative_to(artifact_root).as_posix(),
            "sha256": _sha256_file(evidence),
            "engine": name,
            "transaction_id": transaction_id,
            "prepared_transaction_sha256": prepared_transaction_sha256,
            "artifact_root_id": hashlib.sha256(
                str(artifact_root).encode("utf-8")
            ).hexdigest(),
            "container_id": container_id,
            "wrapper_pid": wrapper_pid,
            "wrapper_start_time_ticks": wrapper_start_time_ticks,
            "device_index": 0,
            "gpu_uuid": expected_uuid,
            "guard_mib": guard_mib,
            "sample_interval_ms": finalizer.GPU_MEMORY_SAMPLE_INTERVAL_MS,
            "maximum_gap_limit_ms": finalizer.GPU_MEMORY_MAX_GAP_MS,
            "summary": summary,
        }

    def _set_builder_prior(
        self,
        fixture: dict[str, object],
        name: str,
        prior: bytes | None,
        mode: int | None = None,
    ) -> None:
        artifact_root = Path(fixture["artifact_root"])
        transactions = []
        for path in (artifact_root / "models/engine_finalize").glob(
            "*/transaction.json"
        ):
            transaction = json.loads(path.read_text(encoding="utf-8"))
            if (
                transaction.get("state") == "prepared"
                and transaction.get("engine") == name
            ):
                transactions.append((path, transaction))
        if len(transactions) != 1:
            raise AssertionError(
                f"expected one prepared {name} transaction: {transactions}"
            )
        transaction_path, transaction = transactions[0]
        source_maintenance = fixture["maintenance"][name]
        payload = json.loads(source_maintenance.read_text(encoding="utf-8"))
        run_id = finalizer.new_run_id()
        run_dir = artifact_root / "models/engine_maintenance" / f"{run_id}-{name}"
        run_dir.mkdir(mode=0o700)
        maintenance = run_dir / "manifest.json"
        payload["run_id"] = run_id
        payload["created_at_utc"] = finalizer._utc_now()
        payload.setdefault("metadata", {})["host_transaction"] = {
            "transaction_id": transaction["transaction_id"],
            "transaction_sha256": _sha256_file(transaction_path),
            "transaction_manifest": str(
                Path("/workspace/DS9/models")
                / transaction_path.relative_to(artifact_root / "models")
            ),
        }
        fixture["maintenance"][name] = maintenance
        if prior is None:
            payload["prior"] = {"exists": False, "path": payload.get("target", "")}
        else:
            if mode is None:
                engine = self._engine_path(
                    fixture, reconciler.ENGINE_ARTIFACT_IDS[name]
                )
                mode = engine.stat().st_mode & 0o777
            payload["prior"] = {
                "exists": True,
                "path": payload.get("target", ""),
                "sha256": hashlib.sha256(prior).hexdigest(),
                "size_bytes": len(prior),
                "mode": f"{mode:04o}",
            }
            preserved = maintenance.parent / "prior" / "prior.engine"
            preserved.parent.mkdir(exist_ok=True)
            preserved.write_bytes(prior)
            preserved.chmod(0o600)
            evidence_root = Path(fixture["artifact_root"]) / "models/engine_maintenance"
            relative = preserved.relative_to(evidence_root)
            payload["preserved_prior"] = {
                "path": str(
                    Path("/workspace/DS9/models/engine_maintenance") / relative
                ),
                "sha256": hashlib.sha256(prior).hexdigest(),
                "size_bytes": len(prior),
                "mode": "0600",
            }
        maintenance.write_text(json.dumps(payload), encoding="utf-8")
        maintenance.chmod(0o600)

    def _new_maintenance_attempt(
        self,
        fixture: dict[str, object],
        snapshot: dict[str, object],
        name: str,
        *,
        run_id: str = "20990101T010203123456Z",
        candidate_bytes: bytes = b"interrupted-hidden-candidate",
        host_binding: bool = True,
        created_at_utc: str | None = None,
    ) -> tuple[Path, Path]:
        transaction_path = Path(snapshot["transaction_manifest"])
        transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
        artifact_root = Path(fixture["artifact_root"])
        run_dir = artifact_root / "models/engine_maintenance" / f"{run_id}-{name}"
        run_dir.mkdir(parents=True)
        run_dir.chmod(0o700)
        metadata: dict[str, object] = {}
        if host_binding:
            metadata["host_transaction"] = {
                "transaction_id": snapshot["transaction_id"],
                "transaction_sha256": snapshot["transaction_sha256"],
                "transaction_manifest": str(
                    Path("/workspace/DS9/models")
                    / transaction_path.relative_to(artifact_root / "models")
                ),
            }
        payload = {
            "schema_version": 1,
            "contract": "noesis.ds9.engine_maintenance",
            "run_id": run_id,
            "created_at_utc": created_at_utc or finalizer._utc_now(),
            "status": "running",
            "engine": name,
            "target": transaction["container_engine_output"],
            "metadata": metadata,
        }
        manifest = run_dir / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        manifest.chmod(0o600)
        time.sleep(0.002)
        engine = Path(transaction["engine_output"])
        candidate = engine.with_name(f".{engine.name}.building-{run_id}")
        candidate.write_bytes(candidate_bytes)
        return manifest, candidate

    def test_reconcile_writes_external_overlay_without_mutating_base(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            manifest = fixture["manifest"]
            assert isinstance(manifest, Path)
            base_before = manifest.read_bytes()
            result = self._reconcile(fixture, "yolo26_m")
            realization = Path(result["realization"])
            self.assertEqual(manifest.read_bytes(), base_before)
            self.assertEqual(realization.stat().st_mode & 0o777, 0o600)
            payload = json.loads(realization.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["artifacts"]), {"engine.yolo26_detect_m"})
            self.assertEqual(
                payload["artifacts"]["engine.yolo26_detect_m"]["provenance"][
                    "maintenance"
                ]["manifest"],
                "DS9/models/"
                + str(
                    Path(fixture["maintenance"]["yolo26_m"]).relative_to(
                        Path(fixture["artifact_root"]) / "models"
                    )
                ),
            )
            self.assertEqual(
                payload["artifacts"]["engine.yolo26_detect_m"]["provenance"][
                    "maintenance"
                ]["gpu_memory_guard"],
                fixture["gpu_memory_guards"]["yolo26_m"],
            )

    def test_reconcile_binds_caller_known_maintenance_path_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            with mock.patch.object(
                reconciler.validator,
                "_validate_engine_maintenance_proof",
                wraps=reconciler.validator._validate_engine_maintenance_proof,
            ) as proof:
                result = self._reconcile(fixture, "yolo26_m")
            maintenance = Path(fixture["maintenance"]["yolo26_m"])
            self.assertTrue(result["realization_sha256"])
            self.assertEqual(
                proof.call_args.kwargs["maintenance_manifest_path"],
                maintenance.resolve(),
            )
            self.assertEqual(
                proof.call_args.kwargs["maintenance_manifest_sha256"],
                _sha256_file(maintenance),
            )
            realization = json.loads(
                Path(result["realization"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                realization["artifacts"]["engine.yolo26_detect_m"]["provenance"][
                    "maintenance"
                ]["manifest_sha256"],
                _sha256_file(maintenance),
            )

    def test_direct_reconcile_rejects_missing_gpu_guard_before_realization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            manifest = Path(fixture["manifest"])
            source_contracts = Path(fixture["source_contracts"])
            maintenance, _guard = self._prepare_direct_guard(fixture, "yolo26_m")
            with self.assertRaisesRegex(
                ValueError, "independently validated GPU-memory guard"
            ):
                reconciler.reconcile(
                    engine_name="yolo26_m",
                    maintenance_manifest=maintenance,
                    gpu_memory_guard=None,
                    artifact_root=Path(fixture["artifact_root"]),
                    base_manifest_path=manifest,
                    expected_base_manifest_sha256=_sha256_file(manifest),
                    expected_source_contracts_sha256=_sha256_file(source_contracts),
                    expected_realization_sha256="missing",
                    dry_run=False,
                )
            self.assertFalse(
                (Path(fixture["artifact_root"]) / "asset_realization.json").exists()
            )

    def test_reconcile_cli_requires_gpu_guard_record(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "DS9/scripts/reconcile_engine_provenance.py"),
                "--engine",
                "yolo26_m",
                "--maintenance-manifest",
                "/tmp/unused-maintenance.json",
                "--artifact-root",
                "/tmp/unused-artifacts",
                "--base-manifest",
                str(REPO_ROOT / "DS9/asset_manifest.yaml"),
                "--expected-base-manifest-sha256",
                "0" * 64,
                "--expected-source-contracts-sha256",
                "0" * 64,
                "--expected-realization-sha256",
                "missing",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--gpu-memory-guard-record", result.stderr)

    def test_realized_validation_rejects_removed_null_or_mutated_gpu_guard(
        self,
    ) -> None:
        for mutation in ("removed", "null", "summary"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = self._fixture(Path(raw))
                result = self._reconcile(fixture, "yolo26_m")
                realization_path = Path(result["realization"])
                realization = json.loads(realization_path.read_text(encoding="utf-8"))
                maintenance = realization["artifacts"]["engine.yolo26_detect_m"][
                    "provenance"
                ]["maintenance"]
                if mutation == "removed":
                    maintenance.pop("gpu_memory_guard")
                elif mutation == "null":
                    maintenance["gpu_memory_guard"] = None
                else:
                    maintenance["gpu_memory_guard"]["summary"][
                        "maximum_observed_mib"
                    ] = 513
                realization_path.write_text(json.dumps(realization), encoding="utf-8")
                realization_path.chmod(0o600)
                validation = validator.validate_asset_realization(
                    fixture["manifest"],
                    realization_path,
                    fixture["artifact_root"],
                    profile="artifact:engine.yolo26_detect_m",
                    check_files=True,
                    require_provenance=True,
                )
                self.assertFalse(validation["ok"], validation)
                joined = " ".join(validation["errors"])
                if mutation == "removed":
                    self.assertIn("not an exact pre-guard trust anchor", joined)
                elif mutation == "null":
                    self.assertIn("must be a mapping", joined)
                else:
                    self.assertIn("summary differs", joined)

    def test_legacy_guard_exemption_accepts_only_exact_frozen_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            result = self._reconcile(fixture, "yolo26_m")
            realization_path = Path(result["realization"])
            realization = json.loads(realization_path.read_text(encoding="utf-8"))
            artifact_id = "engine.yolo26_detect_m"
            provenance = realization["artifacts"][artifact_id]["provenance"]
            maintenance = provenance["maintenance"]
            maintenance.pop("gpu_memory_guard")
            maintenance.pop("manifest_sha256")
            realization_path.write_text(json.dumps(realization), encoding="utf-8")
            realization_path.chmod(0o600)
            maintenance_path = Path(fixture["maintenance"]["yolo26_m"])
            exact = {
                "output_sha256": provenance["output_sha256"],
                "maintenance_manifest": maintenance["manifest"],
                "maintenance_manifest_sha256": _sha256_file(maintenance_path),
            }
            with mock.patch.dict(
                validator.LEGACY_GPU_MEMORY_GUARD_EXEMPTIONS,
                {artifact_id: exact},
                clear=True,
            ):
                accepted = validator.validate_asset_realization(
                    fixture["manifest"],
                    realization_path,
                    fixture["artifact_root"],
                    profile="artifact:engine.yolo26_detect_m",
                    check_files=True,
                    require_provenance=True,
                )
                self.assertTrue(accepted["ok"], accepted)
                validator.LEGACY_GPU_MEMORY_GUARD_EXEMPTIONS[artifact_id] = {
                    **exact,
                    "maintenance_manifest_sha256": "0" * 64,
                }
                rejected = validator.validate_asset_realization(
                    fixture["manifest"],
                    realization_path,
                    fixture["artifact_root"],
                    profile="artifact:engine.yolo26_detect_m",
                    check_files=True,
                    require_provenance=True,
                )
                self.assertFalse(rejected["ok"], rejected)
                self.assertIn(
                    "not an exact pre-guard trust anchor",
                    " ".join(rejected["errors"]),
                )

    def test_authoritative_realized_provenance_cannot_skip_file_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            result = self._reconcile(fixture, "yolo26_m")
            validation = validator.validate_asset_realization(
                fixture["manifest"],
                result["realization"],
                fixture["artifact_root"],
                profile="unit",
                check_files=False,
                require_provenance=True,
            )
            self.assertFalse(validation["ok"], validation)
            self.assertIn("requires check_files", " ".join(validation["errors"]))

    def test_incremental_reconcile_preserves_prior_entries_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            first = self._reconcile(fixture, "yolo26_m")
            second = self._reconcile(
                fixture,
                "yolo26_seg_s",
                expected_realization=str(first["realization_sha256"]),
            )
            realization = Path(second["realization"])
            payload = json.loads(realization.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload["artifacts"]),
                {"engine.yolo26_detect_m", "engine.yolo26_seg_s"},
            )
            validation = validator.validate_asset_realization(
                fixture["manifest"],
                realization,
                fixture["artifact_root"],
                profile="unit",
                check_files=True,
                require_provenance=True,
            )
            self.assertTrue(validation["ok"], validation)

    def test_reconcile_rejects_output_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            maintenance = fixture["maintenance"]["yolo26_m"]
            payload = json.loads(maintenance.read_text(encoding="utf-8"))
            payload["installed"]["sha256"] = "0" * 64
            maintenance.write_text(json.dumps(payload), encoding="utf-8")
            os.chmod(maintenance, 0o600)
            with self.assertRaisesRegex(ValueError, "hash/size differs"):
                self._reconcile(fixture, "yolo26_m")

    def test_reconcile_enforces_base_contract_and_realization_cas(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            manifest = fixture["manifest"]
            source_contracts = fixture["source_contracts"]
            assert isinstance(manifest, Path)
            assert isinstance(source_contracts, Path)
            maintenance, gpu_memory_guard = self._prepare_direct_guard(
                fixture, "yolo26_m"
            )
            common = {
                "engine_name": "yolo26_m",
                "maintenance_manifest": maintenance,
                "gpu_memory_guard": gpu_memory_guard,
                "artifact_root": fixture["artifact_root"],
                "base_manifest_path": manifest,
                "expected_base_manifest_sha256": _sha256_file(manifest),
                "expected_source_contracts_sha256": _sha256_file(source_contracts),
                "expected_realization_sha256": "missing",
                "dry_run": False,
            }
            with self.assertRaisesRegex(ValueError, "base manifest changed"):
                reconciler.reconcile(
                    **{**common, "expected_base_manifest_sha256": "0" * 64}
                )
            with self.assertRaisesRegex(ValueError, "source contracts changed"):
                reconciler.reconcile(
                    **{**common, "expected_source_contracts_sha256": "0" * 64}
                )
            first = reconciler.reconcile(**common)
            with self.assertRaisesRegex(ValueError, "realization changed"):
                reconciler.reconcile(**common)
            self.assertTrue(first["realization_sha256"])

    def test_dangling_realization_and_symlinked_maintenance_are_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            artifact_root = fixture["artifact_root"]
            assert isinstance(artifact_root, Path)
            realization = artifact_root / "asset_realization.json"
            realization.symlink_to(artifact_root / "missing.json")
            with self.assertRaisesRegex(ValueError, "realization changed"):
                self._reconcile(fixture, "yolo26_m")
            realization.unlink()
            maintenance = fixture["maintenance"]["yolo26_m"]
            saved = maintenance.with_name("saved.json")
            maintenance.rename(saved)
            maintenance.symlink_to(saved)
            with self.assertRaisesRegex(ValueError, "symlink"):
                reconciler.reconcile(
                    engine_name="yolo26_m",
                    maintenance_manifest=maintenance,
                    gpu_memory_guard=fixture["gpu_memory_guards"]["yolo26_m"],
                    artifact_root=artifact_root,
                    base_manifest_path=Path(fixture["manifest"]),
                    expected_base_manifest_sha256=_sha256_file(
                        Path(fixture["manifest"])
                    ),
                    expected_source_contracts_sha256=_sha256_file(
                        Path(fixture["source_contracts"])
                    ),
                    expected_realization_sha256="missing",
                    dry_run=False,
                )

    def test_validator_rejects_forged_private_maintenance(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            result = self._reconcile(fixture, "yolo26_m")
            maintenance = fixture["maintenance"]["yolo26_m"]
            payload = json.loads(maintenance.read_text(encoding="utf-8"))
            payload["metadata"]["platform"]["image_id"] = "sha256:" + "0" * 64
            maintenance.write_text(json.dumps(payload), encoding="utf-8")
            os.chmod(maintenance, 0o600)
            validation = validator.validate_asset_realization(
                fixture["manifest"],
                result["realization"],
                fixture["artifact_root"],
                profile="unit",
                check_files=True,
                require_provenance=True,
            )
            self.assertFalse(validation["ok"])
            self.assertIn("build-image authority", " ".join(validation["errors"]))

    def test_validator_rejects_current_authority_toctou_at_final_cas(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            result = self._reconcile(fixture, "yolo26_m")
            secure_read = validator._read_regular_owned_bytes

            def inject_final_source_drift(path: Path, label: str) -> bytes:
                raw = secure_read(path, label)
                if label == "source-contract final CAS":
                    return raw + b" "
                return raw

            with mock.patch.object(
                validator,
                "_read_regular_owned_bytes",
                side_effect=inject_final_source_drift,
            ):
                validation = validator.validate_asset_realization(
                    fixture["manifest"],
                    result["realization"],
                    fixture["artifact_root"],
                    profile="unit",
                    check_files=True,
                    require_provenance=True,
                )
            self.assertFalse(validation["ok"])
            self.assertIn(
                "source-contract authority changed during validation",
                " ".join(validation["errors"]),
            )

    def test_host_finalize_commits_engine_and_overlay_without_base_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.write_bytes(b"prior-engine")
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", b"prior-engine")
            engine.write_bytes(candidate)
            manifest = fixture["manifest"]
            assert isinstance(manifest, Path)
            base_before = manifest.read_bytes()
            result = finalizer.commit(
                transaction_manifest=Path(snapshot["transaction_manifest"]),
                expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                **self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m"),
            )
            self.assertEqual(result["status"], "committed")
            self.assertEqual(engine.read_bytes(), candidate)
            self.assertEqual(manifest.read_bytes(), base_before)
            transaction = json.loads(
                Path(result["transaction_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(transaction["state"], "committed")
            guard = transaction["commit"]["gpu_memory_guard"]
            self.assertEqual(guard["contract"], finalizer.gpu_sampler.CONTRACT)
            self.assertEqual(guard["summary"]["state"], "stopped")
            self.assertTrue(guard["summary"]["guard_ok"])
            self.assertEqual(guard["summary"]["maximum_observed_mib"], 512)
            self.assertEqual(guard["sha256"], guard["summary"]["evidence_sha256"])
            realization = json.loads(
                (Path(fixture["artifact_root"]) / "asset_realization.json").read_text(
                    encoding="utf-8"
                )
            )
            realized_guard = realization["artifacts"]["engine.yolo26_detect_m"][
                "provenance"
            ]["maintenance"]["gpu_memory_guard"]
            self.assertEqual(realized_guard, guard)
            self.assertTrue((Path(fixture["artifact_root"]) / guard["path"]).is_file())
            self.assertEqual(transaction["cleanup"]["errors"], [])
            self.assertGreater(transaction["cleanup"]["freed_bytes"], 0)

    def test_host_finalize_missing_gpu_guard_rolls_back_prior_engine(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.write_bytes(b"prior-engine")
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", b"prior-engine")
            engine.write_bytes(candidate)
            guard = self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m")
            Path(guard["gpu_guard_evidence"]).unlink()
            with self.assertRaisesRegex(finalizer.FinalizeError, "rolled_back"):
                finalizer.commit(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                    **guard,
                )
            self.assertEqual(engine.read_bytes(), b"prior-engine")
            transaction = json.loads(
                Path(snapshot["transaction_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(transaction["state"], "rolled_back")

    def test_host_finalize_mutated_gpu_guard_rolls_back_prior_engine(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.write_bytes(b"prior-engine")
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", b"prior-engine")
            engine.write_bytes(candidate)
            guard = self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m")
            evidence = Path(guard["gpu_guard_evidence"])
            evidence.write_bytes(evidence.read_bytes().replace(b"\n", b" \n", 1))
            evidence.chmod(0o600)
            with self.assertRaisesRegex(finalizer.FinalizeError, "rolled_back"):
                finalizer.commit(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                    **guard,
                )
            self.assertEqual(engine.read_bytes(), b"prior-engine")
            transaction = json.loads(
                Path(snapshot["transaction_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(transaction["state"], "rolled_back")

    def test_host_finalize_validation_failure_restores_prior_engine_and_realization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            first = self._reconcile(fixture, "yolo26_m")
            realization = Path(first["realization"])
            realization_before = realization.read_bytes()
            engine = self._engine_path(fixture, "engine.yolo26_seg_s")
            candidate = engine.read_bytes()
            engine.write_bytes(b"prior-seg-engine")
            snapshot = self._snapshot(fixture, "yolo26_seg_s")
            self._set_builder_prior(fixture, "yolo26_seg_s", b"prior-seg-engine")
            engine.write_bytes(candidate)
            with mock.patch.object(
                finalizer.validator,
                "validate_asset_realization",
                return_value={
                    "ok": False,
                    "errors": ["forged validation failure"],
                    "blockers": [],
                },
            ):
                with self.assertRaisesRegex(finalizer.FinalizeError, "rolled_back"):
                    finalizer.commit(
                        transaction_manifest=Path(snapshot["transaction_manifest"]),
                        expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                        maintenance_manifest=fixture["maintenance"]["yolo26_seg_s"],
                        **self._gpu_guard_kwargs(fixture, snapshot, "yolo26_seg_s"),
                    )
            self.assertEqual(engine.read_bytes(), b"prior-seg-engine")
            self.assertEqual(realization.read_bytes(), realization_before)
            transaction = json.loads(
                Path(snapshot["transaction_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(transaction["state"], "rolled_back")
            self.assertEqual(transaction["retention"]["state"], "cohort_pruned")
            self.assertEqual(transaction["retention"]["errors"], [])
            self.assertFalse(
                (
                    Path(snapshot["transaction_manifest"]).parent / "prior_engine.bin"
                ).exists()
            )
            self.assertFalse(
                (
                    Path(snapshot["transaction_manifest"]).parent
                    / "prior_realization.json"
                ).exists()
            )

    def test_host_finalize_failure_removes_candidate_when_no_prior_engine(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.unlink()
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", None)
            engine.write_bytes(candidate)
            with mock.patch.object(
                finalizer.validator,
                "validate_asset_realization",
                return_value={"ok": False, "errors": ["forced"], "blockers": []},
            ):
                with self.assertRaisesRegex(finalizer.FinalizeError, "rolled_back"):
                    finalizer.commit(
                        transaction_manifest=Path(snapshot["transaction_manifest"]),
                        expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                        maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                        **self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m"),
                    )
            self.assertFalse(engine.exists())
            self.assertFalse(
                (Path(fixture["artifact_root"]) / "asset_realization.json").exists()
            )

    def test_host_finalize_reconcile_cas_failure_restores_prior_engine(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.write_bytes(b"prior-engine")
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", b"prior-engine")
            engine.write_bytes(candidate)
            manifest = fixture["manifest"]
            assert isinstance(manifest, Path)
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(finalizer.FinalizeError, "rolled_back"):
                finalizer.commit(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                    **self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m"),
                )
            self.assertEqual(engine.read_bytes(), b"prior-engine")

    def test_host_finalize_restores_transaction_owned_realization_after_cas_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.write_bytes(b"prior-engine")
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", b"prior-engine")
            engine.write_bytes(candidate)
            realization = Path(fixture["artifact_root"]) / "asset_realization.json"
            realization.write_text('{"external":"writer"}\n', encoding="utf-8")
            realization.chmod(0o600)
            with self.assertRaisesRegex(finalizer.FinalizeError, "rolled_back"):
                finalizer.commit(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                    **self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m"),
                )
            self.assertEqual(engine.read_bytes(), b"prior-engine")
            self.assertFalse(realization.exists())

    def test_host_abort_without_maintenance_restores_prelaunch_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            engine.write_bytes(b"prior-engine")
            snapshot = self._snapshot(fixture, "yolo26_m")
            engine.write_bytes(b"unknown-post-launch-candidate")
            result = finalizer.rollback(
                transaction_manifest=Path(snapshot["transaction_manifest"]),
                expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                maintenance_manifest=None,
                reason="host watchdog terminated detached build",
            )
            self.assertEqual(result["status"], "rolled_back")
            self.assertEqual(engine.read_bytes(), b"prior-engine")

    def test_host_abort_refuses_symlinked_engine_target(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            engine.write_bytes(b"prior-engine")
            snapshot = self._snapshot(fixture, "yolo26_m")
            engine.unlink()
            outside = Path(raw_root) / "outside.engine"
            outside.write_bytes(b"external")
            engine.symlink_to(outside)
            with self.assertRaisesRegex(finalizer.FinalizeError, "tamper_refused"):
                finalizer.rollback(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=None,
                    reason="test symlink tamper",
                )
            self.assertEqual(outside.read_bytes(), b"external")

    def test_prior_disagreement_fails_commit_and_restores_host_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.write_bytes(b"host-snapshot-prior")
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", b"different-builder-prior")
            engine.write_bytes(candidate)
            with self.assertRaisesRegex(
                finalizer.FinalizeError, "prior sha256 differs"
            ):
                finalizer.commit(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                    **self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m"),
                )
            self.assertEqual(engine.read_bytes(), b"host-snapshot-prior")

    def test_startup_recovery_audits_then_aborts_prepared_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            engine.write_bytes(b"prior-engine")
            self._snapshot(fixture, "yolo26_m")
            engine.write_bytes(b"interrupted-candidate")
            audit = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"],
                apply=False,
                reason="audit",
            )
            self.assertFalse(audit["ok"])
            self.assertEqual(engine.read_bytes(), b"interrupted-candidate")
            recovery = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"],
                apply=True,
                reason="startup",
            )
            self.assertTrue(recovery["ok"], recovery)
            self.assertEqual(engine.read_bytes(), b"prior-engine")

    def test_startup_recovery_does_not_reauthorize_sealed_history(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.write_bytes(b"last-known-good")
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", b"last-known-good")
            engine.write_bytes(candidate)
            committed = finalizer.commit(
                transaction_manifest=Path(snapshot["transaction_manifest"]),
                expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                **self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m"),
            )

            manifest = Path(fixture["manifest"])
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "\n# later reviewed authority\n",
                encoding="utf-8",
            )
            audit = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"],
                apply=False,
                reason="post-authority-update audit",
            )
            self.assertTrue(audit["ok"], audit)
            self.assertEqual(
                [row["state_before"] for row in audit["transactions"]],
                ["committed"],
            )
            self.assertEqual(
                Path(committed["transaction_manifest"]).stat().st_mode & 0o777,
                0o600,
            )

    def test_startup_recovery_rejects_forged_engine_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            engine.write_bytes(b"prior-engine")
            snapshot = self._snapshot(fixture, "yolo26_m")
            outside = Path(raw_root) / "outside.engine"
            outside.write_bytes(b"must-survive")
            transaction_path = Path(snapshot["transaction_manifest"])
            transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
            transaction["engine_output"] = str(outside)
            transaction_path.write_text(json.dumps(transaction), encoding="utf-8")
            transaction_path.chmod(0o600)
            recovery = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"],
                apply=True,
                reason="adversarial recovery",
            )
            self.assertFalse(recovery["ok"])
            self.assertIn("host engine path differs", " ".join(recovery["blockers"]))
            self.assertEqual(outside.read_bytes(), b"must-survive")

    def test_startup_recovery_rejects_transaction_supplied_alternate_base(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            original = Path(fixture["manifest"])
            alternate = Path(raw_root) / "forged-base.yaml"
            alternate.write_bytes(original.read_bytes())
            transaction_path = Path(snapshot["transaction_manifest"])
            transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
            transaction["base_manifest"] = {
                "path": str(alternate),
                "sha256": _sha256_file(alternate),
            }
            transaction_path.write_text(json.dumps(transaction), encoding="utf-8")
            transaction_path.chmod(0o600)
            recovery = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"],
                apply=True,
                reason="adversarial alternate base",
            )
            self.assertFalse(recovery["ok"])
            self.assertIn(
                "base-manifest path is not the tracked authority",
                " ".join(recovery["blockers"]),
            )

    def test_committed_canary_revert_restores_retained_host_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.write_bytes(b"last-known-good")
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", b"last-known-good")
            engine.write_bytes(candidate)
            committed = finalizer.commit(
                transaction_manifest=Path(snapshot["transaction_manifest"]),
                expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                **self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m"),
            )
            reverted = finalizer.revert_committed(
                transaction_manifest=Path(committed["transaction_manifest"]),
                expected_transaction_sha256=str(committed["transaction_sha256"]),
                reason="live canary failed",
            )
            self.assertEqual(reverted["status"], "reverted")
            self.assertEqual(engine.read_bytes(), b"last-known-good")
            self.assertFalse(
                (Path(fixture["artifact_root"]) / "asset_realization.json").exists()
            )
            reverted_transaction = json.loads(
                Path(reverted["transaction_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(reverted_transaction["state"], "reverted")
            self.assertEqual(
                reverted_transaction["retention"]["state"], "cohort_pruned"
            )
            self.assertFalse(
                (
                    Path(reverted["transaction_manifest"]).parent / "prior_engine.bin"
                ).exists()
            )

    def test_exit_rollback_is_idempotent_after_internal_commit_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            candidate = engine.read_bytes()
            engine.write_bytes(b"prior-engine")
            snapshot = self._snapshot(fixture, "yolo26_m")
            self._set_builder_prior(fixture, "yolo26_m", b"prior-engine")
            engine.write_bytes(candidate)
            with mock.patch.object(
                finalizer.validator,
                "validate_asset_realization",
                return_value={"ok": False, "errors": ["forced"], "blockers": []},
            ):
                with self.assertRaises(finalizer.FinalizeError):
                    finalizer.commit(
                        transaction_manifest=Path(snapshot["transaction_manifest"]),
                        expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                        maintenance_manifest=fixture["maintenance"]["yolo26_m"],
                        **self._gpu_guard_kwargs(fixture, snapshot, "yolo26_m"),
                    )
            second = finalizer.rollback(
                transaction_manifest=Path(snapshot["transaction_manifest"]),
                expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                maintenance_manifest=None,
                reason="EXIT trap",
            )
            self.assertTrue(second["idempotent"])

    def test_transaction_bound_hidden_candidate_is_hashed_removed_and_recorded(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            _manifest, candidate = self._new_maintenance_attempt(
                fixture, snapshot, "yolo26_m"
            )
            candidate_hash = _sha256_file(candidate)
            result = finalizer.rollback(
                transaction_manifest=Path(snapshot["transaction_manifest"]),
                expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                maintenance_manifest=None,
                reason="host monitor rejected builder",
            )
            self.assertEqual(result["status"], "rolled_back")
            self.assertFalse(candidate.exists())
            transaction = json.loads(
                Path(result["transaction_manifest"]).read_text(encoding="utf-8")
            )
            cleanup = transaction["candidate_cleanup"]
            self.assertEqual(cleanup["state"], "complete")
            self.assertEqual(cleanup["reason"], "host monitor rejected builder")
            self.assertEqual(cleanup["removed"][0]["sha256"], candidate_hash)
            self.assertEqual(
                cleanup["authority"]["transaction_sha256"],
                snapshot["transaction_sha256"],
            )

    def test_preexisting_hidden_candidate_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            foreign = engine.with_name(f".{engine.name}.building-FOREIGN")
            foreign.write_bytes(b"preexisting-foreign")
            snapshot = self._snapshot(fixture, "yolo26_m")
            _manifest, candidate = self._new_maintenance_attempt(
                fixture, snapshot, "yolo26_m"
            )
            result = finalizer.rollback(
                transaction_manifest=Path(snapshot["transaction_manifest"]),
                expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                maintenance_manifest=None,
                reason="preserve preexisting",
            )
            self.assertEqual(result["status"], "rolled_back")
            self.assertFalse(candidate.exists())
            self.assertEqual(foreign.read_bytes(), b"preexisting-foreign")

    def test_symlink_nonregular_hardlink_and_unknown_residue_are_never_deleted(
        self,
    ) -> None:
        cases = ("symlink", "fifo", "hardlink", "unknown")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw_root:
                fixture = self._fixture(Path(raw_root))
                snapshot = self._snapshot(fixture, "yolo26_m")
                if case == "unknown":
                    engine = self._engine_path(fixture, "engine.yolo26_detect_m")
                    candidate = engine.with_name(
                        f".{engine.name}.building-20990101T111111111111Z"
                    )
                    candidate.write_bytes(b"unknown")
                else:
                    _manifest, candidate = self._new_maintenance_attempt(
                        fixture, snapshot, "yolo26_m"
                    )
                    if case == "symlink":
                        target = candidate.with_name("foreign-target")
                        target.write_bytes(b"foreign")
                        candidate.unlink()
                        candidate.symlink_to(target)
                    elif case == "fifo":
                        candidate.unlink()
                        os.mkfifo(candidate)
                    else:
                        os.link(candidate, candidate.with_name("foreign-hardlink"))
                with self.assertRaises(finalizer.FinalizeError):
                    finalizer.rollback(
                        transaction_manifest=Path(snapshot["transaction_manifest"]),
                        expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                        maintenance_manifest=None,
                        reason=f"reject {case}",
                    )
                self.assertTrue(candidate.exists() or candidate.is_symlink())

    def test_explicit_residue_cas_cleans_preinventory_rollback_and_is_idempotent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            manifest, candidate = self._new_maintenance_attempt(
                fixture,
                snapshot,
                "yolo26_m",
                host_binding=False,
            )
            transaction_path = Path(snapshot["transaction_manifest"])
            transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
            transaction.pop("hidden_candidate_inventory")
            transaction["state"] = "rolled_back"
            transaction_path.write_text(json.dumps(transaction), encoding="utf-8")
            transaction_path.chmod(0o600)
            transaction_sha = _sha256_file(transaction_path)
            candidate_sha = _sha256_file(candidate)
            realization = Path(fixture["artifact_root"]) / "asset_realization.json"
            realization_before = (
                realization.read_bytes() if realization.exists() else None
            )
            result = finalizer.cleanup_residue(
                transaction_manifest=transaction_path,
                expected_transaction_sha256=transaction_sha,
                maintenance_manifest=manifest,
                expected_maintenance_sha256=_sha256_file(manifest),
                expected_candidate_sha256=candidate_sha,
                expected_candidate_size_bytes=candidate.stat().st_size,
                reason="reviewed pre-binding residue cleanup",
            )
            self.assertEqual(result["status"], "residue_cleaned")
            self.assertFalse(candidate.exists())
            cleanup = result["candidate_cleanup"]
            self.assertEqual(cleanup["reason"], "reviewed pre-binding residue cleanup")
            self.assertEqual(
                cleanup["authority"]["rolled_back_transaction_sha256"],
                transaction_sha,
            )
            self.assertEqual(cleanup["removed"][0]["sha256"], candidate_sha)
            recovery = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"],
                apply=True,
                reason="idempotence",
            )
            self.assertTrue(recovery["ok"], recovery)
            self.assertEqual(
                realization.read_bytes() if realization.exists() else None,
                realization_before,
            )
            with self.assertRaisesRegex(finalizer.FinalizeError, "already sealed"):
                finalizer.cleanup_residue(
                    transaction_manifest=transaction_path,
                    expected_transaction_sha256=_sha256_file(transaction_path),
                    maintenance_manifest=manifest,
                    expected_maintenance_sha256=_sha256_file(manifest),
                    expected_candidate_sha256=candidate_sha,
                    expected_candidate_size_bytes=cleanup["removed"][0]["size_bytes"],
                    reason="repeat must use recover",
                )

            removed_path = Path(cleanup["removed"][0]["path"])
            removed_path.write_bytes(b"reappeared")
            reappeared = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"],
                apply=True,
                reason="reappeared residue",
            )
            self.assertFalse(reappeared["ok"])
            self.assertIn("reappeared", " ".join(reappeared["blockers"]))
            self.assertEqual(removed_path.read_bytes(), b"reappeared")

    def test_explicit_residue_cas_mismatch_preserves_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            manifest, candidate = self._new_maintenance_attempt(
                fixture, snapshot, "yolo26_m", host_binding=False
            )
            transaction_path = Path(snapshot["transaction_manifest"])
            transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
            transaction.pop("hidden_candidate_inventory")
            transaction["state"] = "rolled_back"
            transaction_path.write_text(json.dumps(transaction), encoding="utf-8")
            transaction_path.chmod(0o600)
            with self.assertRaises(finalizer.FinalizeError):
                finalizer.cleanup_residue(
                    transaction_manifest=transaction_path,
                    expected_transaction_sha256=_sha256_file(transaction_path),
                    maintenance_manifest=manifest,
                    expected_maintenance_sha256=_sha256_file(manifest),
                    expected_candidate_sha256="0" * 64,
                    expected_candidate_size_bytes=candidate.stat().st_size,
                    reason="wrong CAS",
                )
            self.assertTrue(candidate.exists())

    def test_explicit_residue_rejects_transaction_and_manifest_cas(self) -> None:
        for mismatch in ("transaction", "manifest"):
            with (
                self.subTest(mismatch=mismatch),
                tempfile.TemporaryDirectory() as raw_root,
            ):
                fixture = self._fixture(Path(raw_root))
                snapshot = self._snapshot(fixture, "yolo26_m")
                manifest, candidate = self._new_maintenance_attempt(
                    fixture, snapshot, "yolo26_m", host_binding=False
                )
                transaction_path = Path(snapshot["transaction_manifest"])
                transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
                transaction.pop("hidden_candidate_inventory")
                transaction["state"] = "rolled_back"
                transaction_path.write_text(json.dumps(transaction), encoding="utf-8")
                transaction_path.chmod(0o600)
                with self.assertRaises(finalizer.FinalizeError):
                    finalizer.cleanup_residue(
                        transaction_manifest=transaction_path,
                        expected_transaction_sha256=(
                            "0" * 64
                            if mismatch == "transaction"
                            else _sha256_file(transaction_path)
                        ),
                        maintenance_manifest=manifest,
                        expected_maintenance_sha256=(
                            "0" * 64
                            if mismatch == "manifest"
                            else _sha256_file(manifest)
                        ),
                        expected_candidate_sha256=_sha256_file(candidate),
                        expected_candidate_size_bytes=candidate.stat().st_size,
                        reason=f"wrong {mismatch} CAS",
                    )
                self.assertTrue(candidate.exists())

    def test_prepared_cleanup_resumes_after_crash_before_unlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            _manifest, candidate = self._new_maintenance_attempt(
                fixture, snapshot, "yolo26_m"
            )
            original = finalizer._hidden_candidate_record
            count = 0

            def interrupt_second(path: Path):
                nonlocal count
                if path == candidate:
                    count += 1
                    if count == 2:
                        raise KeyboardInterrupt("crash before unlink")
                return original(path)

            with mock.patch.object(
                finalizer, "_hidden_candidate_record", side_effect=interrupt_second
            ):
                with self.assertRaises(KeyboardInterrupt):
                    finalizer.rollback(
                        transaction_manifest=Path(snapshot["transaction_manifest"]),
                        expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                        maintenance_manifest=None,
                        reason="simulated crash",
                    )
            prepared = json.loads(
                Path(snapshot["transaction_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(prepared["candidate_cleanup"]["state"], "prepared")
            self.assertEqual(
                prepared["prepared_transaction_sha256"], snapshot["transaction_sha256"]
            )
            recovery = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"], apply=True, reason="resume"
            )
            self.assertTrue(recovery["ok"], recovery)
            self.assertFalse(candidate.exists())

    def test_prepared_cleanup_rejects_changed_preserved_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            preserved = engine.with_name(f".{engine.name}.building-FOREIGN")
            preserved.write_bytes(b"pre-snapshot")
            snapshot = self._snapshot(fixture, "yolo26_m")
            _manifest, candidate = self._new_maintenance_attempt(
                fixture, snapshot, "yolo26_m"
            )
            original = finalizer._hidden_candidate_record
            count = 0

            def interrupt_candidate(path: Path):
                nonlocal count
                if path == candidate:
                    count += 1
                    if count == 2:
                        raise KeyboardInterrupt("leave prepared")
                return original(path)

            with mock.patch.object(
                finalizer, "_hidden_candidate_record", side_effect=interrupt_candidate
            ):
                with self.assertRaises(KeyboardInterrupt):
                    finalizer.rollback(
                        transaction_manifest=Path(snapshot["transaction_manifest"]),
                        expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                        maintenance_manifest=None,
                        reason="prepare then mutate",
                    )
            preserved.write_bytes(b"external-change")
            recovery = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"], apply=True, reason="resume"
            )
            self.assertFalse(recovery["ok"])
            self.assertIn("changed", " ".join(recovery["blockers"]))
            self.assertTrue(candidate.exists())

    def test_prepared_cleanup_retains_plan_after_crash_after_unlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            _manifest, candidate = self._new_maintenance_attempt(
                fixture, snapshot, "yolo26_m"
            )
            candidate_sha = _sha256_file(candidate)
            original_write = finalizer._write_json_atomic

            def interrupt_complete(path: Path, payload: dict[str, object]) -> None:
                cleanup = payload.get("candidate_cleanup")
                if isinstance(cleanup, dict) and cleanup.get("state") == "complete":
                    raise KeyboardInterrupt("crash after unlink")
                original_write(path, payload)

            with mock.patch.object(
                finalizer, "_write_json_atomic", side_effect=interrupt_complete
            ):
                with self.assertRaises(KeyboardInterrupt):
                    finalizer.rollback(
                        transaction_manifest=Path(snapshot["transaction_manifest"]),
                        expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                        maintenance_manifest=None,
                        reason="simulated post-unlink crash",
                    )
            self.assertFalse(candidate.exists())
            recovery = finalizer.recover_transactions(
                artifact_root=fixture["artifact_root"], apply=True, reason="resume"
            )
            self.assertTrue(recovery["ok"], recovery)
            transaction = json.loads(
                Path(snapshot["transaction_manifest"]).read_text(encoding="utf-8")
            )
            removed = transaction["candidate_cleanup"]["removed"][0]
            self.assertEqual(removed["sha256"], candidate_sha)
            self.assertEqual(
                removed["removal_status"], "recovered_after_interrupted_unlink"
            )

    def test_candidate_hash_open_never_follows_swap_to_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            candidate = root / ".engine.building-run"
            candidate.write_bytes(b"candidate")
            target = root / "secret"
            target.write_bytes(b"must-not-be-read")
            original_open = finalizer.os.open
            swapped = False

            def swap_before_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if Path(path) == candidate and not swapped:
                    swapped = True
                    candidate.unlink()
                    candidate.symlink_to(target)
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(finalizer.os, "open", side_effect=swap_before_open):
                with self.assertRaisesRegex(
                    finalizer.FinalizeError, "without following"
                ):
                    finalizer._hidden_candidate_record(candidate)
            self.assertTrue(candidate.is_symlink())
            self.assertEqual(target.read_bytes(), b"must-not-be-read")

    def test_commit_rejects_manifest_without_exact_host_transaction_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            engine = self._engine_path(fixture, "engine.yolo26_detect_m")
            self._set_builder_prior(fixture, "yolo26_m", engine.read_bytes())
            maintenance = fixture["maintenance"]["yolo26_m"]
            payload = json.loads(maintenance.read_text(encoding="utf-8"))
            payload["metadata"].pop("host_transaction")
            maintenance.write_text(json.dumps(payload), encoding="utf-8")
            maintenance.chmod(0o600)
            with self.assertRaisesRegex(finalizer.FinalizeError, "exact host binding"):
                finalizer.commit(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=maintenance,
                    **self._gpu_guard_kwargs(
                        fixture,
                        snapshot,
                        "yolo26_m",
                        maintenance_manifest=maintenance,
                    ),
                )

    def test_commit_rejects_mutated_presnapshot_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            maintenance = fixture["maintenance"]["yolo26_m"]
            payload = json.loads(maintenance.read_text(encoding="utf-8"))
            payload["metadata"]["host_transaction"] = {
                "transaction_id": snapshot["transaction_id"],
                "transaction_sha256": snapshot["transaction_sha256"],
                "transaction_manifest": "forged-pre-snapshot-path",
            }
            maintenance.write_text(json.dumps(payload), encoding="utf-8")
            maintenance.chmod(0o600)
            with self.assertRaisesRegex(
                finalizer.FinalizeError, "pre-snapshot maintenance manifest changed"
            ):
                finalizer.commit(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=maintenance,
                    **self._gpu_guard_kwargs(
                        fixture,
                        snapshot,
                        "yolo26_m",
                        maintenance_manifest=maintenance,
                    ),
                )

    def test_deleted_presnapshot_manifest_blocks_candidate_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            _manifest, candidate = self._new_maintenance_attempt(
                fixture, snapshot, "yolo26_m"
            )
            fixture["maintenance"]["yolo26_m"].unlink()
            with self.assertRaisesRegex(finalizer.FinalizeError, "manual_recovery"):
                finalizer.rollback(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=None,
                    reason="deleted prior evidence",
                )
            transaction = json.loads(
                Path(snapshot["transaction_manifest"]).read_text(encoding="utf-8")
            )
            self.assertIn(
                "disappeared",
                " ".join(transaction["candidate_cleanup"]["blockers"]),
            )
            self.assertTrue(candidate.exists())

    def test_candidate_predating_manifest_is_preserved_and_blocks_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            fixture = self._fixture(Path(raw_root))
            snapshot = self._snapshot(fixture, "yolo26_m")
            _manifest, candidate = self._new_maintenance_attempt(
                fixture,
                snapshot,
                "yolo26_m",
                created_at_utc="2099-01-01T01:02:03Z",
            )
            with self.assertRaises(finalizer.FinalizeError):
                finalizer.rollback(
                    transaction_manifest=Path(snapshot["transaction_manifest"]),
                    expected_transaction_sha256=str(snapshot["transaction_sha256"]),
                    maintenance_manifest=None,
                    reason="future manifest",
                )
            transaction = json.loads(
                Path(snapshot["transaction_manifest"]).read_text(encoding="utf-8")
            )
            self.assertIn(
                "predates",
                " ".join(transaction["candidate_cleanup"]["blockers"]),
            )
            self.assertTrue(candidate.exists())


if __name__ == "__main__":
    unittest.main()
