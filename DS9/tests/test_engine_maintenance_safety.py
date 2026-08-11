from __future__ import annotations

import configparser
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "DS9" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import engine_maintenance_common as maintenance_common  # noqa: E402

from engine_maintenance_common import (  # noqa: E402
    EngineMaintenanceRun,
    EngineMaintenanceError,
    engine_maintenance_lock,
    ensure_private_directory,
    input_bundle_record,
    onnx_contract,
    require_absent_candidate_path,
)


def _load_module(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fake_trtexec(root: Path) -> Path:
    executable = root / "fake-trtexec"
    executable.write_text(
        textwrap.dedent(
            '''\
            #!/usr/bin/env python3
            import os
            import sys
            from pathlib import Path

            mode = os.environ.get("FAKE_TRT_MODE", "ok")
            state = Path(os.environ["FAKE_TRT_STATE"])
            args = sys.argv[1:]

            if "--help" in args:
                print("TensorRT v101401 [fake]")
                if mode == "probe_failure":
                    print("[E] Model missing")
                    print("&&&& FAILED TensorRT.trtexec")
                    raise SystemExit(1)
                raise SystemExit(0)

            save = next((value.split("=", 1)[1] for value in args if value.startswith("--saveEngine=")), None)
            if save is not None:
                Path(save).write_bytes(b"new-candidate-engine")
                if mode == "partial_build_failure":
                    print("[E] Error[4]: partial serialization")
                    print("&&&& FAILED TensorRT.trtexec")
                    raise SystemExit(1)
                print("&&&& PASSED TensorRT.trtexec")
                raise SystemExit(0)

            load = next((value.split("=", 1)[1] for value in args if value.startswith("--loadEngine=")), None)
            if load is None:
                print("[E] Model missing")
                print("&&&& FAILED TensorRT.trtexec")
                raise SystemExit(1)
            count = int(state.read_text() or "0") if state.exists() else 0
            count += 1
            state.write_text(str(count))
            fail = mode == "false_zero_candidate" or (
                mode in {"final_load_failure", "final_load_tamper"} and count == 2
            )
            if mode == "final_load_tamper" and count == 2:
                Path(load).write_bytes(b"external-tamper")
            print("Loaded engine size: 1 MiB")
            print("Engine deserialized in 0.001 sec.")
            print("Skipped inference phase since --skipInference is added.")
            if fail:
                print("[E] Error[6]: engine deserialization failed")
                print("Engine deserialization failed")
            print("&&&& PASSED TensorRT.trtexec")
            raise SystemExit(0)
            '''
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _fake_gpp_with_partial_builder(root: Path) -> Path:
    executable = root / "fake-g++"
    builder_script = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import sys
        from pathlib import Path

        candidate = Path(sys.argv[sys.argv.index("--output") + 1])
        candidate.write_bytes(b"authorized-partial-wholebody-engine")
        print("[NOESIS_TRT_BUILDER] status=FAIL reason=simulated partial write")
        raise SystemExit(1)
        """
    )
    executable.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import os
            import sys
            from pathlib import Path

            if "--version" in sys.argv:
                print("g++ (fake DS9 toolchain) 13.3.0")
                raise SystemExit(0)
            output = Path(sys.argv[sys.argv.index("-o") + 1])
            output.write_text({builder_script!r}, encoding="utf-8")
            os.chmod(output, 0o755)
            """
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _write_minimal_onnx(path: Path) -> None:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "maintenance-fixture",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path.write_bytes(model.SerializeToString())


class EngineMaintenanceSafetyTests(unittest.TestCase):
    def test_unique_candidate_guard_preserves_regular_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            regular = root / ".model.engine.building-fixed"
            regular.write_bytes(b"foreign-candidate")
            with self.assertRaisesRegex(
                EngineMaintenanceError, "refusing to delete"
            ):
                require_absent_candidate_path(regular)
            self.assertEqual(regular.read_bytes(), b"foreign-candidate")

            target = root / "target"
            target.write_bytes(b"foreign-target")
            linked = root / ".model.engine.building-linked"
            linked.symlink_to(target)
            with self.assertRaisesRegex(
                EngineMaintenanceError, "refusing to delete"
            ):
                require_absent_candidate_path(linked)
            self.assertTrue(linked.is_symlink())
            self.assertEqual(target.read_bytes(), b"foreign-target")

    def _adoption_fixture(self, root: Path) -> tuple[EngineMaintenanceRun, Path, Path]:
        target = root / "engines" / "model.engine"
        target.parent.mkdir()
        target.write_bytes(b"known-prior-engine")
        source = root / "source.onnx"
        _write_minimal_onnx(source)
        run = EngineMaintenanceRun(
            name="adoption-test",
            target=target,
            evidence_root=root / "evidence",
            inputs={"source": source},
            repo_root=root,
        )
        workspace = root / "sdk-source"
        workspace.mkdir(mode=0o700)
        workspace.chmod(0o700)
        derived = workspace / "source.etlt_b32_gpu0_fp16.engine"
        return run, workspace, derived

    def test_derived_candidate_adoption_is_private_verified_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run, workspace, derived = self._adoption_fixture(root)
            derived.write_bytes(b"sdk-derived-engine")
            candidate = run.target.with_name(
                f".{run.target.name}.building-{run.run_id}"
            )

            adopted = run.adopt_derived_candidate(
                derived,
                candidate,
                workspace_root=workspace,
            )

            self.assertEqual(derived.read_bytes(), b"sdk-derived-engine")
            self.assertEqual(candidate.read_bytes(), b"sdk-derived-engine")
            self.assertEqual(_mode(candidate), 0o600)
            self.assertEqual(adopted["sha256"], hashlib.sha256(candidate.read_bytes()).hexdigest())
            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            adoption = manifest["candidate_adoption"]
            self.assertEqual(adoption["status"], "verified")
            self.assertEqual(
                adoption["failure_cleanup_authority"],
                "host_engine_finalize_transaction",
            )
            self.assertEqual(
                adoption["copy"]["method"], "exclusive_nofollow_stream_copy"
            )
            self.assertEqual(
                adoption["copy"]["durability"],
                "file_and_destination_directory_fsynced",
            )
            self.assertEqual(run.target.read_bytes(), b"known-prior-engine")

    def test_derived_candidate_adoption_preserves_preexisting_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run, workspace, derived = self._adoption_fixture(root)
            derived.write_bytes(b"sdk-derived-engine")
            candidate = run.target.with_name(
                f".{run.target.name}.building-{run.run_id}"
            )
            candidate.write_bytes(b"foreign-candidate")

            with self.assertRaisesRegex(EngineMaintenanceError, "refusing to delete"):
                run.adopt_derived_candidate(
                    derived,
                    candidate,
                    workspace_root=workspace,
                )
            self.assertEqual(candidate.read_bytes(), b"foreign-candidate")
            self.assertEqual(run.target.read_bytes(), b"known-prior-engine")

    def test_derived_candidate_adoption_rejects_symlink_and_hardlink_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run, workspace, derived = self._adoption_fixture(root)
            real = workspace / "real.engine"
            real.write_bytes(b"sdk-derived-engine")
            derived.symlink_to(real)
            candidate = run.target.with_name(
                f".{run.target.name}.building-{run.run_id}"
            )
            with self.assertRaisesRegex(EngineMaintenanceError, "symlink"):
                run.adopt_derived_candidate(
                    derived,
                    candidate,
                    workspace_root=workspace,
                )
            derived.unlink()
            os.link(real, derived)
            with self.assertRaisesRegex(EngineMaintenanceError, "single-link"):
                run.adopt_derived_candidate(
                    derived,
                    candidate,
                    workspace_root=workspace,
                )
            self.assertFalse(candidate.exists())

    def test_derived_candidate_adoption_rejects_escape_and_wrong_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run, workspace, derived = self._adoption_fixture(root)
            outside = root / "outside.engine"
            outside.write_bytes(b"sdk-derived-engine")
            candidate = run.target.with_name(
                f".{run.target.name}.building-{run.run_id}"
            )
            with self.assertRaisesRegex(EngineMaintenanceError, "escaped"):
                run.adopt_derived_candidate(
                    outside,
                    candidate,
                    workspace_root=workspace,
                )

            derived.write_bytes(b"sdk-derived-engine")
            wrong_candidate = run.target.with_name(
                f".{run.target.name}.building-wrong-run"
            )
            with self.assertRaisesRegex(EngineMaintenanceError, "exact host-authorized"):
                run.adopt_derived_candidate(
                    derived,
                    wrong_candidate,
                    workspace_root=workspace,
                )
            self.assertFalse(candidate.exists())
            self.assertFalse(wrong_candidate.exists())

    def test_derived_candidate_tamper_leaves_authorized_partial_for_host(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run, workspace, derived = self._adoption_fixture(root)
            derived.write_bytes(b"sdk-derived-engine")
            candidate = run.target.with_name(
                f".{run.target.name}.building-{run.run_id}"
            )
            original_read = maintenance_common.os.read
            tampered = False

            def tampering_read(descriptor: int, size: int) -> bytes:
                nonlocal tampered
                block = original_read(descriptor, size)
                if block and not tampered:
                    tampered = True
                    with derived.open("ab") as handle:
                        handle.write(b"-tampered")
                return block

            with (
                mock.patch.object(
                    maintenance_common.os, "read", side_effect=tampering_read
                ),
                self.assertRaisesRegex(EngineMaintenanceError, "source changed"),
            ):
                run.adopt_derived_candidate(
                    derived,
                    candidate,
                    workspace_root=workspace,
                )

            self.assertTrue(candidate.is_file())
            self.assertEqual(
                candidate,
                run.target.with_name(f".{run.target.name}.building-{run.run_id}"),
            )
            self.assertEqual(run.target.read_bytes(), b"known-prior-engine")
            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["candidate_adoption"]["status"],
                "failed_authorized_partial",
            )
            self.assertEqual(
                manifest["candidate_adoption"]["failure_cleanup_authority"],
                "host_engine_finalize_transaction",
            )

    def test_both_builders_delegate_failed_candidate_cleanup_to_host(self) -> None:
        for relative in (
            "DS9/scripts/rebuild_engines.py",
            "DS9/scripts/build_v3dt_tracker_engine.py",
        ):
            with self.subTest(relative=relative):
                source = (REPO_ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("require_absent_candidate_path(temporary_engine)", source)
                self.assertNotIn("temporary_engine.unlink", source)
        wholebody_source = (
            REPO_ROOT
            / "DS9/csrc/wholebody49_engine_builder/wholebody49_engine_builder.cpp"
        ).read_text(encoding="utf-8")
        self.assertNotIn("::unlink", wholebody_source)

    def test_canonical_wrapper_explicitly_maps_external_model_root(self) -> None:
        wrapper = (
            REPO_ROOT / "DS9/scripts/run_canonical_engine_maintenance.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "--env NOESIS_MODEL_DIR=/workspace/DS9/models",
            wrapper,
        )

    def test_wholebody_partial_build_is_left_for_host_transaction_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            module, base_spec, engine, evidence = self._fixture(root)
            spec = module.EngineSpec(
                name=base_spec.name,
                source_onnx=base_spec.source_onnx,
                staged_onnx=base_spec.staged_onnx,
                engine=base_spec.engine,
                trtexec_args=(),
                builder=module.WHOLEBODY_BUILDER_CONTRACT,
                builder_variant="s_masks",
            )
            fake_trtexec = _fake_trtexec(root)
            fake_gpp = _fake_gpp_with_partial_builder(root)
            state = root / "fake-state"
            with (
                mock.patch.dict(
                    os.environ,
                    {"FAKE_TRT_MODE": "ok", "FAKE_TRT_STATE": str(state)},
                    clear=False,
                ),
                mock.patch.object(module.shutil, "which", return_value=str(fake_gpp)),
                self.assertRaisesRegex(EngineMaintenanceError, "build exited 1"),
            ):
                module._build(
                    spec,
                    trtexec=str(fake_trtexec),
                    evidence_root=evidence,
                    build_timeout_seconds=10,
                    load_timeout_seconds=10,
                    inspect_onnx_contract=False,
                )

            self.assertEqual(engine.read_bytes(), b"known-good-engine")
            residues = list(engine.parent.glob(".*.building-*"))
            self.assertEqual(len(residues), 1)
            self.assertEqual(
                residues[0].read_bytes(), b"authorized-partial-wholebody-engine"
            )
            _path, manifest = self._manifest(evidence)
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("wholebody_builder_source", manifest["inputs"])
            executable = manifest["evidence"]["wholebody_builder_executable"]
            self.assertEqual(executable["mode"], "0700")
            self.assertEqual(
                manifest["commands"][-1]["proof"], "wholebody_builder_s_masks"
            )

    def _fixture(self, root: Path):
        module = _load_module(
            f"ds9_rebuild_safety_{id(root)}", "DS9/scripts/rebuild_engines.py"
        )
        source = root / "model.onnx"
        engine = root / "engines" / "model.engine"
        evidence = root / "evidence"
        _write_minimal_onnx(source)
        engine.parent.mkdir()
        engine.write_bytes(b"known-good-engine")
        engine.chmod(0o640)
        spec = module.EngineSpec(
            name="maintenance_test",
            source_onnx=source,
            staged_onnx=source,
            engine=engine,
            trtexec_args=(),
        )
        return module, spec, engine, evidence

    def _run(self, root: Path, mode: str):
        module, spec, engine, evidence = self._fixture(root)
        fake = _fake_trtexec(root)
        state = root / "fake-state"
        with mock.patch.dict(
            os.environ,
            {"FAKE_TRT_MODE": mode, "FAKE_TRT_STATE": str(state)},
            clear=False,
        ):
            module._build(
                spec,
                trtexec=str(fake),
                validate_load=True,
                evidence_root=evidence,
                build_timeout_seconds=10,
                load_timeout_seconds=10,
                inspect_onnx_contract=False,
            )
        return engine, evidence

    @staticmethod
    def _manifest(evidence: Path) -> tuple[Path, dict[str, object]]:
        manifests = list(evidence.glob("*/manifest.json"))
        if len(manifests) != 1:
            raise AssertionError(f"expected one manifest, found {manifests}")
        return manifests[0], json.loads(manifests[0].read_text(encoding="utf-8"))

    def test_false_zero_pass_candidate_load_preserves_previous_engine(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            with self.assertRaisesRegex(EngineMaintenanceError, "fail-closed marker"):
                self._run(root, "false_zero_candidate")
            engine = root / "engines" / "model.engine"
            self.assertEqual(engine.read_bytes(), b"known-good-engine")
            residues = list(engine.parent.glob(".*.building-*"))
            self.assertEqual(len(residues), 1)
            self.assertEqual(residues[0].read_bytes(), b"new-candidate-engine")
            _path, manifest = self._manifest(root / "evidence")
            self.assertEqual(manifest["status"], "failed")

    def test_partial_failed_build_preserves_previous_engine(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            with self.assertRaisesRegex(EngineMaintenanceError, "build exited 1"):
                self._run(root, "partial_build_failure")
            engine = root / "engines" / "model.engine"
            self.assertEqual(engine.read_bytes(), b"known-good-engine")
            residues = list(engine.parent.glob(".*.building-*"))
            self.assertEqual(len(residues), 1)
            self.assertGreater(residues[0].stat().st_size, 0)
            _path, manifest = self._manifest(root / "evidence")
            self.assertEqual(manifest["status"], "failed")

    def test_final_load_failure_atomically_restores_prior(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            with self.assertRaisesRegex(EngineMaintenanceError, "fail-closed marker"):
                self._run(root, "final_load_failure")
            engine = root / "engines" / "model.engine"
            self.assertEqual(engine.read_bytes(), b"known-good-engine")
            self.assertEqual(_mode(engine), 0o640)
            _path, manifest = self._manifest(root / "evidence")
            rollback = manifest["rollback_after_install_failure"]
            self.assertEqual(rollback["status"], "restored_prior")
            self.assertEqual(manifest["status"], "failed")

    def test_final_load_tamper_refuses_to_overwrite_external_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            with self.assertRaisesRegex(EngineMaintenanceError, "fail-closed marker"):
                self._run(root, "final_load_tamper")
            engine = root / "engines" / "model.engine"
            self.assertEqual(engine.read_bytes(), b"external-tamper")
            _path, manifest = self._manifest(root / "evidence")
            rollback = manifest["rollback_after_install_failure"]
            self.assertEqual(rollback["status"], "manual_recovery_required")
            self.assertIn("refusing to overwrite", rollback["reason"])
            preserved = Path(manifest["preserved_prior"]["path"])
            self.assertEqual(preserved.read_bytes(), b"known-good-engine")

    def test_success_requires_both_loads_and_private_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            engine, evidence = self._run(root, "ok")
            self.assertEqual(engine.read_bytes(), b"new-candidate-engine")
            manifest_path, manifest = self._manifest(evidence)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(
                [row["label"] for row in manifest["commands"]],
                ["probe-trtexec", "build", "load-candidate", "load-installed"],
            )
            run_dir = manifest_path.parent
            for directory in (evidence, run_dir, run_dir / "logs", run_dir / "prior"):
                self.assertEqual(_mode(directory), 0o700, directory)
            for private_file in (
                manifest_path,
                run_dir / "prior" / "model.engine",
                *(run_dir / "logs").iterdir(),
            ):
                self.assertEqual(_mode(private_file), 0o600, private_file)

    def test_private_evidence_rejects_symlink_and_file_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            real = root / "real"
            real.mkdir()
            symlink = root / "linked"
            symlink.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(EngineMaintenanceError, "symlink"):
                ensure_private_directory(symlink / "evidence")
            blocker = root / "not-a-directory"
            blocker.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(EngineMaintenanceError, "not a directory"):
                ensure_private_directory(blocker / "evidence")

    def test_install_refuses_noncooperative_prior_target_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            target = root / "engines/model.engine"
            target.parent.mkdir()
            target.write_bytes(b"known-prior")
            source = root / "source.onnx"
            _write_minimal_onnx(source)
            run = EngineMaintenanceRun(
                name="cas-test",
                target=target,
                evidence_root=root / "evidence",
                inputs={"source": source},
                repo_root=root,
            )
            candidate = root / "candidate.engine"
            candidate.write_bytes(b"candidate")
            target.write_bytes(b"external-writer")
            with self.assertRaisesRegex(EngineMaintenanceError, "changed after prior"):
                run.install_candidate(candidate)
            self.assertEqual(target.read_bytes(), b"external-writer")
            self.assertEqual(candidate.read_bytes(), b"candidate")

    def test_install_refuses_target_appearing_after_recorded_absence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            target = root / "engines/model.engine"
            source = root / "source.onnx"
            _write_minimal_onnx(source)
            run = EngineMaintenanceRun(
                name="absence-cas-test",
                target=target,
                evidence_root=root / "evidence",
                inputs={"source": source},
                repo_root=root,
            )
            candidate = root / "candidate.engine"
            candidate.write_bytes(b"candidate")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"external-writer")
            with self.assertRaisesRegex(EngineMaintenanceError, "appeared after prior"):
                run.install_candidate(candidate)
            self.assertEqual(target.read_bytes(), b"external-writer")

    def test_maintenance_lock_rejects_preexisting_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            real = root / "real.lock"
            real.write_text("", encoding="utf-8")
            real.chmod(0o600)
            linked = root / "linked.lock"
            linked.symlink_to(real)
            with self.assertRaisesRegex(EngineMaintenanceError, "symlink"):
                with engine_maintenance_lock(linked, dry_run=False):
                    self.fail("symlinked lock must never be acquired")

    def test_build_contract_drift_fails_before_trtexec_or_install(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            module, spec, engine, evidence = self._fixture(root)
            record = input_bundle_record(spec.source_onnx)
            contracts = root / "contracts.json"
            contracts.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "contracts": {
                            spec.name: {
                                "raw_sha256": record["sha256"],
                                "bundle_sha256": record["bundle_sha256"],
                                "onnx": onnx_contract(spec.source_onnx),
                                "maintenance_build": {
                                    "precision_arg": "--fp16",
                                    "trtexec_args": [],
                                    "plugin_args": [],
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            drifted = module.EngineSpec(
                name=spec.name,
                source_onnx=spec.source_onnx,
                staged_onnx=spec.staged_onnx,
                engine=spec.engine,
                trtexec_args=spec.trtexec_args,
                precision_arg="--bf16",
            )
            transaction = root / "transaction.json"
            transaction.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "contract": "noesis.ds9.engine_finalize_transaction",
                        "state": "prepared",
                        "transaction_id": "unit-transaction",
                        "engine": spec.name,
                        "container_engine_output": str(engine.absolute()),
                        "prior_engine": {
                            "exists": True,
                            "sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                            "size_bytes": engine.stat().st_size,
                            "mode": f"{_mode(engine):04o}",
                        },
                    }
                ),
                encoding="utf-8",
            )
            transaction.chmod(0o600)
            transaction_sha = hashlib.sha256(transaction.read_bytes()).hexdigest()
            fake = _fake_trtexec(root)
            with mock.patch.object(module, "SOURCE_CONTRACTS", contracts):
                with self.assertRaisesRegex(
                    EngineMaintenanceError, "maintenance build contract drift"
                ):
                    module._build(
                        drifted,
                        trtexec=str(fake),
                        evidence_root=evidence,
                        build_timeout_seconds=10,
                        load_timeout_seconds=10,
                        transaction_manifest=transaction,
                        expected_transaction_sha256=transaction_sha,
                    )
            self.assertEqual(engine.read_bytes(), b"known-good-engine")
            self.assertFalse(evidence.exists())

    def test_preflight_help_probe_rejects_correct_banner_plus_failure(self) -> None:
        preflight = _load_module(
            "ds9_preflight_trtexec_failure_test", "DS9/scripts/ds9_preflight.py"
        )
        result = SimpleNamespace(
            returncode=1,
            stdout="TensorRT v101401 [fake]\n",
            stderr="[E] Model missing\n&&&& FAILED TensorRT.trtexec\n",
        )
        with (
            mock.patch.object(preflight.shutil, "which", return_value="/fake/trtexec"),
            mock.patch.object(preflight.subprocess, "run", return_value=result) as run,
        ):
            self.assertFalse(preflight._trtexec_ok())
        run.assert_called_once_with(
            ["/fake/trtexec", "--help"], text=True, capture_output=True
        )

    def test_preflight_help_probe_requires_exact_ds9_banner(self) -> None:
        preflight = _load_module(
            "ds9_preflight_trtexec_success_test", "DS9/scripts/ds9_preflight.py"
        )
        result = SimpleNamespace(
            returncode=0,
            stdout="TensorRT v101401 [fake]\nhelp text\n",
            stderr="",
        )
        with (
            mock.patch.object(preflight.shutil, "which", return_value="/fake/trtexec"),
            mock.patch.object(preflight.subprocess, "run", return_value=result),
        ):
            self.assertTrue(preflight._trtexec_ok())

    def test_v3dt_and_all_wrapper_profiles_use_shared_guarded_paths(self) -> None:
        v3dt = (SCRIPTS_ROOT / "build_v3dt_tracker_engine.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "EngineMaintenanceRun",
            '"load-candidate"',
            '"load-installed"',
            "rollback_after_install_failure",
        ):
            self.assertIn(token, v3dt)
        wrapper = (SCRIPTS_ROOT / "run_canonical_engine_maintenance.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("engine_maintenance", wrapper)
        self.assertIn("--evidence-root", wrapper)
        self.assertIn(
            "CANONICAL_ORDER=(yolo26_m reid_swin yolo26_pose_n depth_anything_v2_tracking mapanything)",
            wrapper,
        )
        self.assertIn('[yolo26_seg_s]="yolo26s-seg_fused_b3_fp16.engine"', wrapper)
        self.assertIn(
            '[mapanything]="mapanything_images_294x518_b3_fp32.plan"', wrapper
        )
        self.assertIn('"mapanything-functional-quality"', wrapper)
        self.assertIn('"trtexec_inference"', wrapper)
        self.assertIn('"--dumpOutput"', wrapper)
        rebuild = _load_module(
            "ds9_rebuild_profile_route_test", "DS9/scripts/rebuild_engines.py"
        )
        specs = {row.name for row in rebuild._specs(include_mapanything=True)}
        self.assertTrue(
            {
                "mapanything",
                "wholebody49_s_masks",
                "wholebody49_x_boxes",
                "bodypose3dnet",
            }.issubset(specs)
        )

    def test_static_baseline_and_v3dt_configs_match_canonical_pgies(self) -> None:
        baseline = yaml.safe_load(
            (REPO_ROOT / "DS9/config/infer.yaml").read_text(encoding="utf-8")
        )
        v3dt = yaml.safe_load(
            (REPO_ROOT / "DS9/config/infer_v3dt.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            baseline["models"]["pgie"]["engine"],
            "DS9/models/engines/yolo26m_b3_fp16.engine",
        )
        self.assertEqual(
            baseline["preprocess"]["config-file"],
            "DS9/pipelines/config_preproc_yolo26_m.ini",
        )
        self.assertEqual(
            v3dt["models"]["pgie"]["engine"],
            "DS9/models/engines/yolo26s-seg_fused_b3_fp16.engine",
        )
        self.assertEqual(
            v3dt["preprocess"]["config-file"],
            "DS9/pipelines/config_preproc_yolo26_seg_s.ini",
        )
        for relative in (
            "DS9/pipelines/config_infer_primary_yolo26_m.ini",
            "DS9/pipelines/config_infer_primary_yolo26_seg_s.ini",
        ):
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(REPO_ROOT / relative)
            self.assertNotIn("onnx-file", parser["property"])
            self.assertTrue(parser["property"]["model-engine-file"].startswith("DS9/models/engines/"))
        for relative in (
            "DS9/pipelines/config_preproc_yolo26_m.ini",
            "DS9/pipelines/config_preproc_yolo26_seg_s.ini",
        ):
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(REPO_ROOT / relative)
            self.assertEqual(parser["property"]["tensor-name"], "images")
            self.assertEqual(parser["group-0"]["src-ids"], "0;1;2")

    def test_staging_owns_canonical_baseline_and_v3dt_sources(self) -> None:
        staging = _load_module(
            "ds9_stage_canonical_pgie_test", "DS9/scripts/stage_canonical_sources.py"
        )
        rows = {destination.as_posix(): source for source, destination in staging.CANONICAL_SOURCES}
        self.assertEqual(
            rows["models/onnx/yolo26m.onnx"],
            (REPO_ROOT / "models/onnx/yolo26m.onnx").resolve(),
        )
        self.assertEqual(
            rows["models/onnx/yolo26s-seg_fused.onnx"],
            (REPO_ROOT / "models/yolo26s-seg_fused.onnx").resolve(),
        )
        self.assertEqual(
            rows["models/onnx/yolo26n-pose_b3.onnx"],
            (REPO_ROOT / "models/onnx/yolo26n-pose_b3.onnx").resolve(),
        )

        pose_export = _load_module(
            "ds9_pose_export_storage_test",
            "DS9/scripts/export_pose_b3_cpu.py",
        )
        self.assertEqual(
            pose_export.DEFAULT_OUTPUT,
            (REPO_ROOT / "models/onnx/yolo26n-pose_b3.onnx").resolve(),
        )
        self.assertEqual(
            pose_export.DEFAULT_PROVENANCE_OUTPUT,
            REPO_ROOT
            / "DS9/models/onnx/yolo26n-pose_b3.onnx.provenance.json",
        )
        for unsafe in (
            Path("/"),
            REPO_ROOT,
            REPO_ROOT / "DS9",
            REPO_ROOT / "DS9" / "models" / "pose-export-workspace",
        ):
            with self.subTest(unsafe=unsafe):
                with mock.patch.dict(
                    os.environ,
                    {"NOESIS_DS9_ARTIFACT_ROOT": str(unsafe)},
                    clear=False,
                ):
                    with self.assertRaises(ValueError):
                        pose_export._artifact_root()


if __name__ == "__main__":
    unittest.main()
