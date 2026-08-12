from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_rebaser():
    path = REPO_ROOT / "DS9/scripts/rebase_asset_realization.py"
    spec = importlib.util.spec_from_file_location("ds9_asset_realization_rebaser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebaser = _load_rebaser()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_private(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    os.chmod(path, 0o600)


class AssetRealizationRebaseTests(unittest.TestCase):
    def _fixture(self, base: Path) -> dict[str, Any]:
        artifact_root = base / "artifacts"
        artifact_root.mkdir(mode=0o750)
        lock = artifact_root / rebaser.LOCK_FILENAME
        _write_private(lock, b"")

        snapshots = artifact_root / "manifest_rebase"
        snapshots.mkdir(mode=0o755)
        snapshot_dir = snapshots / "20260711T031717Z"
        snapshot_dir.mkdir(mode=0o700)

        engine_one = {
            "id": "engine.one",
            "kind": "tensorrt_engine",
            "role": "canonical detector",
            "output": "DS9/models/engines/one.engine",
            "sources": ["DS9/models/onnx/one.onnx"],
            "builder": "DS9/scripts/build_one.sh",
            "required_profiles": ["canonical"],
            "state": "missing",
            "compatibility": {"tensorrt": "10.16.0.72"},
            "provenance": {"source_sha256": "1" * 64},
        }
        engine_two = {
            "id": "engine.two",
            "kind": "tensorrt_engine",
            "role": "future detector",
            "output": "DS9/models/engines/two.engine",
            "sources": ["DS9/models/onnx/two.onnx"],
            "builder": "DS9/scripts/build_two.sh",
            "required_profiles": ["alternate"],
            "state": "missing",
            "compatibility": {"tensorrt": "10.16.0.72"},
            "provenance": {"source_sha256": "2" * 64},
        }
        plugin = {
            "id": "plugin.roi_exclude",
            "kind": "gstreamer_plugin",
            "role": "ROI exclusion",
            "output": "DS9/gst-plugins/libgstnvdsroiexclude.so",
            "sources": ["DS9/gst-plugins/gstnvdsroiexclude.cpp"],
            "builder": "DS9/scripts/build_gst_plugins.sh",
            "required_profiles": ["runtime_common"],
            "state": "staged_unverified",
            "compatibility": {"tensorrt": None},
            "provenance": {"source_sha256": "3" * 64},
        }
        old_manifest = {
            "schema_version": 2,
            "manifest_id": "noesis-ds9-artifacts",
            "schema": "DS9/docs/asset_manifest.schema.json",
            "updated_at": "2026-07-10",
            "target": {
                "deepstream": {"major": 9, "version": "9.1"},
                "cuda": "13.2",
                "tensorrt": "10.16.0.72",
                "build_image": {"image_id": "sha256:" + "a" * 64},
            },
            "runtime": {"entrypoint": "DS9/noesis/ds9_runtime.py"},
            "policy": {"ds8_binary_reuse": "forbidden"},
            "artifacts": [plugin, engine_one, engine_two],
            "shared_app_data": [],
            "packaging_boundary": {"external": True},
        }
        new_manifest = copy.deepcopy(old_manifest)
        new_plugin = new_manifest["artifacts"][0]
        new_plugin["builder"] = "DS9/scripts/build_nvdsroiexclude_ds9.sh"
        new_plugin["provenance"]["source_sha256"] = "4" * 64

        old_manifest_path = snapshot_dir / "asset_manifest.before.yaml"
        _write_private(
            old_manifest_path,
            yaml.safe_dump(old_manifest, sort_keys=False).encode("utf-8"),
        )

        repo = base / "repo"
        new_manifest_path = repo / "DS9/asset_manifest.yaml"
        new_manifest_path.parent.mkdir(parents=True)
        new_manifest_path.write_text(
            yaml.safe_dump(new_manifest, sort_keys=False), encoding="utf-8"
        )
        os.chmod(new_manifest_path, 0o644)
        contracts_path = repo / "DS9/config/engine_source_contracts.json"
        contracts_path.parent.mkdir(parents=True)
        contracts_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "contracts": {
                        "one": {"raw_sha256": "5" * 64},
                        "two": {"raw_sha256": "6" * 64},
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.chmod(contracts_path, 0o644)

        realization_path = artifact_root / rebaser.REALIZATION_FILENAME
        realization = {
            "schema_version": 1,
            "contract": rebaser.REALIZATION_CONTRACT,
            "base_manifest": {
                "path": "DS9/asset_manifest.yaml",
                "sha256": _sha256(old_manifest_path),
            },
            "source_contracts": {
                "path": "DS9/config/engine_source_contracts.json",
                "sha256": _sha256(contracts_path),
            },
            "created_at_utc": "2026-07-11T01:00:00Z",
            "updated_at_utc": "2026-07-11T02:00:00Z",
            "artifacts": {
                "engine.one": {
                    "state": "staged_unverified",
                    "provenance": {
                        "output_sha256": "7" * 64,
                        "private_marker": "must-not-enter-evidence",
                    },
                }
            },
        }
        _write_private(
            realization_path,
            (json.dumps(realization, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        return {
            "artifact_root": artifact_root,
            "lock": lock,
            "repo": repo,
            "old_manifest_path": old_manifest_path,
            "new_manifest_path": new_manifest_path,
            "contracts_path": contracts_path,
            "realization_path": realization_path,
            "old_manifest": old_manifest,
            "new_manifest": new_manifest,
            "realization": realization,
        }

    def _arguments(self, fixture: dict[str, Any]) -> dict[str, Any]:
        return {
            "artifact_root": fixture["artifact_root"],
            "old_manifest_snapshot": fixture["old_manifest_path"],
            "expected_old_manifest_sha256": _sha256(fixture["old_manifest_path"]),
            "expected_new_manifest_sha256": _sha256(fixture["new_manifest_path"]),
            "expected_source_contracts_sha256": _sha256(fixture["contracts_path"]),
            "expected_old_realization_sha256": _sha256(fixture["realization_path"]),
            "expected_new_realization_sha256": None,
            "updated_at_utc": "2026-07-11T04:00:00Z",
            "dry_run": True,
        }

    def _call(self, fixture: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        arguments = self._arguments(fixture)
        arguments.update(overrides)
        repo = fixture["repo"]
        with mock.patch.multiple(
            rebaser,
            REPO_ROOT=repo,
            NEW_MANIFEST_AUTHORITY=fixture["new_manifest_path"],
            SOURCE_CONTRACTS_AUTHORITY=fixture["contracts_path"],
        ):
            return rebaser.rebase_realization(**arguments)

    def _rewrite_new_manifest(
        self,
        fixture: dict[str, Any],
        mutate: Callable[[dict[str, Any]], None],
    ) -> None:
        payload = copy.deepcopy(fixture["new_manifest"])
        mutate(payload)
        fixture["new_manifest"] = payload
        fixture["new_manifest_path"].write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        os.chmod(fixture["new_manifest_path"], 0o644)

    def _rewrite_realization(
        self,
        fixture: dict[str, Any],
        mutate: Callable[[dict[str, Any]], None],
    ) -> None:
        payload = copy.deepcopy(fixture["realization"])
        mutate(payload)
        fixture["realization"] = payload
        _write_private(
            fixture["realization_path"],
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    def test_dry_run_proves_narrow_rebase_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            realization_before = fixture["realization_path"].read_bytes()
            evidence_before = sorted(
                path.relative_to(fixture["artifact_root"]).as_posix()
                for path in (fixture["artifact_root"] / "manifest_rebase").rglob("*")
            )

            result = self._call(fixture)

            self.assertTrue(result["ok"])
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["realized_engine_count"], 1)
            self.assertEqual(result["engine_artifact_count"], 2)
            self.assertEqual(
                result["manifest_diff"]["changed_non_engine_artifact_ids"],
                ["plugin.roi_exclude"],
            )
            self.assertNotIn("must-not-enter-evidence", json.dumps(result))
            self.assertEqual(fixture["realization_path"].read_bytes(), realization_before)
            self.assertEqual(
                sorted(
                    path.relative_to(fixture["artifact_root"]).as_posix()
                    for path in (
                        fixture["artifact_root"] / "manifest_rebase"
                    ).rglob("*")
                ),
                evidence_before,
            )

    def test_exact_runtime_image_authority_addition_is_accepted_and_evidenced(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            self._rewrite_new_manifest(
                fixture,
                lambda payload: payload["runtime"].update(
                    {
                        "image": copy.deepcopy(
                            rebaser.REVIEWED_RUNTIME_IMAGE_AUTHORITY
                        )
                    }
                ),
            )

            dry_run = self._call(fixture)

            self.assertTrue(dry_run["runtime_image_authority"]["changed"])
            self.assertIsNone(dry_run["runtime_image_authority"]["before"])
            self.assertEqual(
                dry_run["runtime_image_authority"]["after"],
                rebaser.REVIEWED_RUNTIME_IMAGE_AUTHORITY,
            )
            self.assertTrue(
                all(
                    path.startswith("runtime.image")
                    for path in dry_run["manifest_diff"]["changed_paths"]
                    if path.startswith("runtime")
                )
            )

            result = self._call(
                fixture,
                dry_run=False,
                expected_new_realization_sha256=dry_run[
                    "new_realization_sha256"
                ],
            )
            evidence = json.loads(
                (fixture["artifact_root"] / result["evidence"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                evidence["runtime_image_authority"],
                dry_run["runtime_image_authority"],
            )
            self.assertTrue(
                evidence["semantic_checks"][
                    "runtime_fields_outside_image_unchanged"
                ]
            )

    def test_yaml_merge_defaults_are_supported_but_explicit_duplicates_fail(self) -> None:
        merged = b"""
defaults: &defaults
  id: inherited
  kind: tensorrt_engine
artifacts:
  - <<: *defaults
    id: engine.one
"""
        payload = rebaser._parse_yaml_mapping(merged, "fixture")
        self.assertEqual(payload["artifacts"][0]["id"], "engine.one")
        self.assertEqual(payload["artifacts"][0]["kind"], "tensorrt_engine")
        with self.assertRaisesRegex(rebaser.RebaseError, "duplicate mapping key"):
            rebaser._parse_yaml_mapping(b"key: one\nkey: two\n", "fixture")

    def test_json_rebase_authority_rejects_duplicates_and_overflow(self) -> None:
        with self.assertRaisesRegex(rebaser.RebaseError, "duplicate mapping key"):
            rebaser._parse_json_mapping(
                b'{"outer":{"selector":"one","selector":"two"}}',
                "fixture",
            )
        with self.assertRaisesRegex(rebaser.RebaseError, "non-finite number"):
            rebaser._parse_json_mapping(b'{"value":1e309}', "fixture")

    def test_apply_is_compare_and_swap_with_private_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            before = copy.deepcopy(fixture["realization"])
            dry_run = self._call(fixture)

            result = self._call(
                fixture,
                dry_run=False,
                expected_new_realization_sha256=dry_run[
                    "new_realization_sha256"
                ],
            )

            self.assertEqual(result["state"], "committed")
            self.assertEqual(
                _sha256(fixture["realization_path"]),
                dry_run["new_realization_sha256"],
            )
            after = json.loads(fixture["realization_path"].read_text(encoding="utf-8"))
            before["base_manifest"]["sha256"] = _sha256(
                fixture["new_manifest_path"]
            )
            before["updated_at_utc"] = "2026-07-11T04:00:00Z"
            self.assertEqual(after, before)
            self.assertEqual(
                stat.S_IMODE(fixture["realization_path"].stat().st_mode), 0o600
            )

            evidence_path = fixture["artifact_root"] / result["evidence"]
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["state"], "committed")
            self.assertEqual(evidence["realization"]["old_sha256"], result["old_realization_sha256"])
            self.assertEqual(evidence["realization"]["new_sha256"], result["new_realization_sha256"])
            self.assertEqual(
                evidence["mutation_paths"],
                ["base_manifest.sha256", "updated_at_utc"],
            )
            self.assertNotIn(
                "must-not-enter-evidence", evidence_path.read_text(encoding="utf-8")
            )
            self.assertEqual(stat.S_IMODE(evidence_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(evidence_path.parent.stat().st_mode), 0o700)

    def test_post_rename_directory_fsync_failure_is_recovered_as_commit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            dry_run = self._call(fixture)
            original_fsync = rebaser._fsync_directory
            injected = False

            def fail_realization_fsync_once(path: Path) -> None:
                nonlocal injected
                if path == fixture["artifact_root"] and not injected:
                    injected = True
                    raise OSError("simulated post-rename directory fsync failure")
                original_fsync(path)

            with mock.patch.object(
                rebaser,
                "_fsync_directory",
                side_effect=fail_realization_fsync_once,
            ):
                result = self._call(
                    fixture,
                    dry_run=False,
                    expected_new_realization_sha256=dry_run[
                        "new_realization_sha256"
                    ],
                )

            self.assertTrue(injected)
            self.assertEqual(result["state"], "committed")
            self.assertEqual(
                _sha256(fixture["realization_path"]),
                dry_run["new_realization_sha256"],
            )
            evidence = json.loads(
                (fixture["artifact_root"] / result["evidence"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(evidence["state"], "committed")
            self.assertEqual(
                evidence["realization_replace_recovery"]["outcome"],
                "proposal_exact_bytes_refsynced",
            )

    def test_post_rename_read_verification_failure_is_recovered_as_commit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            dry_run = self._call(fixture)
            original_read = rebaser._read_file
            injected = False

            def fail_first_proposal_read(
                path: Path,
                label: str,
                *,
                require_private: bool,
            ) -> bytes:
                nonlocal injected
                observed = original_read(
                    path,
                    label,
                    require_private=require_private,
                )
                if (
                    path == fixture["realization_path"]
                    and not injected
                    and hashlib.sha256(observed).hexdigest()
                    == dry_run["new_realization_sha256"]
                ):
                    injected = True
                    raise OSError("simulated post-rename read verification failure")
                return observed

            with mock.patch.object(
                rebaser,
                "_read_file",
                side_effect=fail_first_proposal_read,
            ):
                result = self._call(
                    fixture,
                    dry_run=False,
                    expected_new_realization_sha256=dry_run[
                        "new_realization_sha256"
                    ],
                )

            self.assertTrue(injected)
            self.assertEqual(result["state"], "committed")
            self.assertEqual(
                _sha256(fixture["realization_path"]),
                dry_run["new_realization_sha256"],
            )
            evidence = json.loads(
                (fixture["artifact_root"] / result["evidence"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(evidence["state"], "committed")
            self.assertEqual(
                evidence["realization_replace_recovery"]["outcome"],
                "proposal_exact_bytes_refsynced",
            )

    def test_pre_rename_realization_failure_records_terminal_abort(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            old_bytes = fixture["realization_path"].read_bytes()
            dry_run = self._call(fixture)
            original_replace = rebaser._atomic_private_replace

            def fail_before_realization_replace(
                path: Path,
                content: bytes,
                *,
                expected_current_sha256: str | None,
                label: str,
            ) -> str:
                if path == fixture["realization_path"]:
                    raise OSError("simulated pre-rename realization failure")
                return original_replace(
                    path,
                    content,
                    expected_current_sha256=expected_current_sha256,
                    label=label,
                )

            with mock.patch.object(
                rebaser,
                "_atomic_private_replace",
                side_effect=fail_before_realization_replace,
            ), self.assertRaisesRegex(rebaser.RebaseError, "before commit"):
                self._call(
                    fixture,
                    dry_run=False,
                    expected_new_realization_sha256=dry_run[
                        "new_realization_sha256"
                    ],
                )

            self.assertEqual(fixture["realization_path"].read_bytes(), old_bytes)
            evidence_paths = list(
                (fixture["artifact_root"] / "manifest_rebase").glob(
                    "*/rebase_evidence.json"
                )
            )
            self.assertEqual(len(evidence_paths), 1)
            self.assertEqual(
                json.loads(evidence_paths[0].read_text(encoding="utf-8"))["state"],
                "aborted_before_commit",
            )

    def test_third_realization_state_keeps_nonterminal_recovery_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            dry_run = self._call(fixture)
            original_replace = rebaser._atomic_private_replace
            foreign = b'{"unexpected":"third-state"}\n'

            def install_third_state(
                path: Path,
                content: bytes,
                *,
                expected_current_sha256: str | None,
                label: str,
            ) -> str:
                if path == fixture["realization_path"]:
                    _write_private(path, foreign)
                    raise OSError("simulated ambiguous post-rename state")
                return original_replace(
                    path,
                    content,
                    expected_current_sha256=expected_current_sha256,
                    label=label,
                )

            with mock.patch.object(
                rebaser,
                "_atomic_private_replace",
                side_effect=install_third_state,
            ), self.assertRaisesRegex(rebaser.RebaseError, "ambiguous exact-byte"):
                self._call(
                    fixture,
                    dry_run=False,
                    expected_new_realization_sha256=dry_run[
                        "new_realization_sha256"
                    ],
                )

            self.assertEqual(fixture["realization_path"].read_bytes(), foreign)
            evidence_paths = list(
                (fixture["artifact_root"] / "manifest_rebase").glob(
                    "*/rebase_evidence.json"
                )
            )
            self.assertEqual(len(evidence_paths), 1)
            evidence = json.loads(evidence_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(
                evidence["state"],
                "recovery_required_after_realization_replace",
            )
            self.assertEqual(
                evidence["realization_replace_observed_state"],
                "ambiguous",
            )
            with self.assertRaisesRegex(rebaser.RebaseError, "requires recovery"):
                self._call(fixture)

    def test_post_rename_committed_evidence_read_failure_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            dry_run = self._call(fixture)
            original_read = rebaser._read_file
            injected = False

            def fail_first_committed_evidence_read(
                path: Path,
                label: str,
                *,
                require_private: bool,
            ) -> bytes:
                nonlocal injected
                observed = original_read(
                    path,
                    label,
                    require_private=require_private,
                )
                if path.name == "rebase_evidence.json" and not injected:
                    payload = json.loads(observed.decode("utf-8"))
                    if payload.get("state") == "committed":
                        injected = True
                        raise OSError(
                            "simulated committed-evidence verification failure"
                        )
                return observed

            with mock.patch.object(
                rebaser,
                "_read_file",
                side_effect=fail_first_committed_evidence_read,
            ):
                result = self._call(
                    fixture,
                    dry_run=False,
                    expected_new_realization_sha256=dry_run[
                        "new_realization_sha256"
                    ],
                )

            self.assertTrue(injected)
            self.assertEqual(result["state"], "committed")
            evidence = json.loads(
                (fixture["artifact_root"] / result["evidence"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(evidence["state"], "committed")
            self.assertEqual(
                _sha256(fixture["realization_path"]),
                dry_run["new_realization_sha256"],
            )

    def test_target_or_top_level_authority_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            self._rewrite_new_manifest(
                fixture,
                lambda payload: payload["target"].update({"cuda": "13.2"}),
            )
            with self.assertRaisesRegex(rebaser.RebaseError, "target authority drifted"):
                self._call(fixture)

        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            self._rewrite_new_manifest(
                fixture,
                lambda payload: payload["target"]["build_image"].update(
                    {"image_id": "sha256:" + "0" * 64}
                ),
            )
            with self.assertRaisesRegex(rebaser.RebaseError, "target authority drifted"):
                self._call(fixture)

        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            self._rewrite_new_manifest(
                fixture,
                lambda payload: payload["runtime"].update({"entrypoint": "other.py"}),
            )
            with self.assertRaisesRegex(rebaser.RebaseError, "runtime authority drift"):
                self._call(fixture)

    def test_unreviewed_runtime_image_authority_is_rejected(self) -> None:
        for mutation in (
            {"image_id": "sha256:" + "0" * 64},
            {"parent_image_id": "sha256:" + "1" * 64},
            {"unexpected": "field"},
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = self._fixture(Path(raw))

                def mutate(payload: dict[str, Any]) -> None:
                    authority = copy.deepcopy(
                        rebaser.REVIEWED_RUNTIME_IMAGE_AUTHORITY
                    )
                    authority.update(mutation)
                    payload["runtime"]["image"] = authority

                self._rewrite_new_manifest(fixture, mutate)
                with self.assertRaisesRegex(
                    rebaser.RebaseError, "exact reviewed runtime-image authority"
                ):
                    self._call(fixture)

    def test_identical_unreviewed_runtime_image_authority_is_rejected(self) -> None:
        authority = copy.deepcopy(rebaser.REVIEWED_RUNTIME_IMAGE_AUTHORITY)
        authority["image_id"] = "sha256:" + "0" * 64
        old_manifest = {
            "runtime": {
                "entrypoint": "DS9/noesis/ds9_runtime.py",
                "image": copy.deepcopy(authority),
            }
        }
        new_manifest = copy.deepcopy(old_manifest)

        with self.assertRaisesRegex(
            rebaser.RebaseError, "exact reviewed runtime-image authority"
        ):
            rebaser._runtime_image_transition(old_manifest, new_manifest)

    def test_realized_and_unrealized_engine_drift_are_rejected(self) -> None:
        for artifact_id in ("engine.one", "engine.two"):
            with self.subTest(artifact_id=artifact_id), tempfile.TemporaryDirectory() as raw:
                fixture = self._fixture(Path(raw))

                def mutate(payload: dict[str, Any]) -> None:
                    selected = next(
                        row for row in payload["artifacts"] if row["id"] == artifact_id
                    )
                    selected["builder"] = "DS9/scripts/drifted_builder.sh"

                self._rewrite_new_manifest(fixture, mutate)
                with self.assertRaisesRegex(
                    rebaser.RebaseError, "engine artifact record drifted"
                ):
                    self._call(fixture)

    def test_source_contract_and_expected_output_hash_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            fixture["contracts_path"].write_text(
                '{"schema_version":1,"contracts":{"drift":{}}}',
                encoding="utf-8",
            )
            os.chmod(fixture["contracts_path"], 0o644)
            with self.assertRaisesRegex(rebaser.RebaseError, "source contract.*hash mismatch"):
                self._call(
                    fixture,
                    expected_source_contracts_sha256="0" * 64,
                )

        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            dry_run = self._call(fixture)
            with self.assertRaisesRegex(rebaser.RebaseError, "new realization hash mismatch"):
                self._call(
                    fixture,
                    dry_run=False,
                    expected_new_realization_sha256=(
                        "0" * 64
                        if dry_run["new_realization_sha256"] != "0" * 64
                        else "1" * 64
                    ),
                )
            self.assertEqual(
                _sha256(fixture["realization_path"]),
                self._arguments(fixture)["expected_old_realization_sha256"],
            )

    def test_realization_source_contract_binding_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            self._rewrite_realization(
                fixture,
                lambda payload: payload["source_contracts"].update(
                    {"sha256": "8" * 64}
                ),
            )
            with self.assertRaisesRegex(rebaser.RebaseError, "authority drifted"):
                self._call(fixture)

    def test_private_no_follow_inputs_and_lock_ownership_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            os.chmod(fixture["old_manifest_path"], 0o644)
            with self.assertRaisesRegex(rebaser.RebaseError, "mode 0600"):
                self._call(fixture)

        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            held = os.open(fixture["lock"], os.O_RDWR)
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(rebaser.RebaseError, "owns the lock"):
                    self._call(fixture)
            finally:
                os.close(held)

        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            original = fixture["old_manifest_path"]
            target = original.with_name("foreign.yaml")
            target.write_bytes(original.read_bytes())
            original.unlink()
            original.symlink_to(target.name)
            with self.assertRaisesRegex(rebaser.RebaseError, "symlink"):
                self._call(fixture)

        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            linked_root = Path(raw) / "artifact-root-link"
            linked_root.symlink_to(fixture["artifact_root"], target_is_directory=True)
            with self.assertRaisesRegex(rebaser.RebaseError, "symlink"):
                self._call(fixture, artifact_root=linked_root)

    def test_evidence_finalization_failure_rolls_back_realization(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            old_bytes = fixture["realization_path"].read_bytes()
            dry_run = self._call(fixture)
            original_replace = rebaser._atomic_private_replace

            def fail_committed_evidence(
                path: Path,
                content: bytes,
                *,
                expected_current_sha256: str | None,
                label: str,
            ) -> str:
                if path.name == "rebase_evidence.json":
                    payload = json.loads(content.decode("utf-8"))
                    if payload.get("state") == "committed":
                        raise OSError("simulated evidence fsync failure")
                return original_replace(
                    path,
                    content,
                    expected_current_sha256=expected_current_sha256,
                    label=label,
                )

            with mock.patch.object(
                rebaser,
                "_atomic_private_replace",
                side_effect=fail_committed_evidence,
            ), self.assertRaisesRegex(rebaser.RebaseError, "realization was restored"):
                self._call(
                    fixture,
                    dry_run=False,
                    expected_new_realization_sha256=dry_run[
                        "new_realization_sha256"
                    ],
                )
            self.assertEqual(fixture["realization_path"].read_bytes(), old_bytes)
            evidence_paths = list(
                (fixture["artifact_root"] / "manifest_rebase").glob(
                    "*/rebase_evidence.json"
                )
            )
            self.assertEqual(len(evidence_paths), 1)
            self.assertEqual(
                json.loads(evidence_paths[0].read_text(encoding="utf-8"))["state"],
                "rolled_back_after_evidence_failure",
            )

    def test_unresolved_prepared_evidence_blocks_another_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self._fixture(Path(raw))
            unresolved = (
                fixture["artifact_root"]
                / "manifest_rebase/20260711T031717Z/rebase_evidence.json"
            )
            _write_private(
                unresolved,
                json.dumps(
                    {
                        "schema_version": 1,
                        "contract": rebaser.REBASE_CONTRACT,
                        "state": "prepared",
                    }
                ).encode("utf-8"),
            )
            with self.assertRaisesRegex(rebaser.RebaseError, "requires recovery"):
                self._call(fixture)


if __name__ == "__main__":
    unittest.main()
