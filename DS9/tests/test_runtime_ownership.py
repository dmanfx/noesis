from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPO_ROOT / "DS9" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ownership = _load_script("validate_runtime_ownership")


class RuntimeOwnershipTests(unittest.TestCase):
    def test_live_matrix_is_structurally_valid(self) -> None:
        matrix = yaml.safe_load(
            (REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(encoding="utf-8")
        )
        result = ownership.validate_matrix(matrix)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(matrix["schema_version"], 2)
        self.assertEqual(result["duplicate_count"], 22)
        self.assertEqual(result["evidence_counts"]["repository_source"], 67)
        self.assertEqual(result["evidence_counts"]["asset_realization"], 0)
        self.assertEqual(result["evidence_counts"]["runtime_session"], 0)
        self.assertGreaterEqual(len(result["blockers"]), 1)

    def test_tracked_matrix_is_selector_free_normative_policy(self) -> None:
        matrix = yaml.safe_load(
            (REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(
                encoding="utf-8"
            )
        )
        for capability in matrix["capabilities"]:
            self.assertEqual(
                set(capability["evidence"]),
                {"repository_source"},
                capability["id"],
            )

    def test_ds9_pose_evidence_uses_owned_native_contract_not_tensor_wrapper(self) -> None:
        matrix = yaml.safe_load(
            (REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(
                encoding="utf-8"
            )
        )
        capability = next(
            row
            for row in matrix["capabilities"]
            if row["id"] == "metadata.ds9_pose_path"
        )
        evidence = capability["evidence"]["repository_source"]["ds9"]
        required = set(evidence["contains_all"])
        self.assertEqual(evidence["path"], "DS9/noesis/pipelines/hooks.py")
        self.assertTrue(
            {
                "NOESIS_DS9_ALLOW_PYDS_COMPAT",
                "noesis_pose_meta_ext",
                "extract_pose_keypoints",
                "attach_pose_features",
                "_record_quarantined_compat_path",
            }.issubset(required)
        )
        self.assertNotIn("tensor_items", required)
        result = ownership.validate_matrix(matrix)
        self.assertTrue(result["ok"], result["errors"])

    def test_bev_ownership_uses_semantic_forced_ds9_import_origin(self) -> None:
        matrix = yaml.safe_load(
            (REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(
                encoding="utf-8"
            )
        )
        capability = next(
            row for row in matrix["capabilities"] if row["id"] == "telemetry.bev_contract"
        )
        self.assertEqual(capability["status"], "shared")
        subjects = capability["evidence"]["repository_source"]
        self.assertEqual(set(subjects), {"canonical"})
        evidence = subjects["canonical"]
        self.assertNotIn("contains_all", evidence)
        self.assertEqual(
            evidence["python_import_origin"],
            {
                "module": "noesis.telemetry.bev",
                "expected_path": "noesis/telemetry/bev.py",
                "prepend_paths": ["DS9", "."],
            },
        )
        self.assertFalse((REPO_ROOT / "DS9/noesis/telemetry/bev.py").exists())
        result = ownership.validate_matrix(matrix)
        self.assertTrue(result["ok"], result["errors"])

        poisoned = copy.deepcopy(matrix)
        target = next(
            row
            for row in poisoned["capabilities"]
            if row["id"] == "telemetry.bev_contract"
        )
        target["evidence"]["repository_source"]["canonical"][
            "python_import_origin"
        ]["expected_path"] = "DS9/noesis/telemetry/__init__.py"
        rejected = ownership.validate_matrix(poisoned)
        self.assertFalse(rejected["ok"])
        self.assertTrue(
            any(
                "expected_path must equal repository evidence path" in error
                for error in rejected["errors"]
            ),
            rejected["errors"],
        )

    def test_strict_mode_surfaces_declared_cutover_blockers(self) -> None:
        matrix = yaml.safe_load(
            (REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(encoding="utf-8")
        )
        result = ownership.validate_matrix(matrix, require_parity=True)
        self.assertFalse(result["ok"])
        blocker_ids = {blocker.split(":", 1)[0] for blocker in result["blockers"]}
        self.assertEqual(
            blocker_ids,
            {
                "model.wholebody49_profile",
                "model.reid_profile",
                "model.mapanything_validated_fp32_builder",
                "tracking.v3dt",
                "artifacts.canonical_graph",
            },
        )
        self.assertFalse(any("api.household_identity" in error for error in result["errors"]))

    def test_schema_version_requires_exact_integer_two(self) -> None:
        live = yaml.safe_load(
            (REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(
                encoding="utf-8"
            )
        )
        for invalid in (True, False, 1, 2.0, "2", None):
            with self.subTest(schema_version=invalid):
                matrix = copy.deepcopy(live)
                matrix["schema_version"] = invalid
                result = ownership.validate_matrix(matrix)
                self.assertFalse(result["ok"])
                self.assertTrue(
                    any("exact integer 2" in error for error in result["errors"])
                )

    def test_yaml_duplicate_alias_anchor_and_merge_are_rejected(self) -> None:
        invalid_documents = {
            "duplicate": "schema_version: 2\nschema_version: 2\n",
            "alias": "schema_version: 2\npolicy: &policy {}\ncopy: *policy\n",
            "merge": "schema_version: 2\nbase: &base {x: 1}\ncopy: {<<: *base}\n",
        }
        for name, raw in invalid_documents.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                ownership._parse_ownership_yaml(raw, "fixture")

    def test_missing_dynamic_evidence_blocks_otherwise_parity_capabilities(self) -> None:
        matrix = yaml.safe_load(
            (REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            "artifacts.canonical_graph",
            "model.mapanything_validated_fp32_builder",
            "model.reid_profile",
            "model.wholebody49_profile",
            "tracking.v3dt",
        }
        result = ownership.validate_matrix(matrix, require_parity=True)
        self.assertFalse(result["ok"])
        blocker_ids = {row.split(":", 1)[0] for row in result["blockers"]}
        self.assertEqual(blocker_ids, expected)
        self.assertTrue(
            all(
                capability["status"] == "parity"
                for capability in matrix["capabilities"]
                if capability["id"] in expected
            )
        )
        self.assertTrue(
            all("validator requires" in blocker for blocker in result["blockers"])
        )

    def test_dynamic_evidence_cannot_override_normative_blocked_status(self) -> None:
        matrix = yaml.safe_load(
            (REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(
                encoding="utf-8"
            )
        )
        target = next(
            row
            for row in matrix["capabilities"]
            if row["id"] == "cli.core_runtime_options"
        )
        target["status"] = "blocked"
        target["gap"] = {
            "owner": "fixture",
            "reason": "real static contract gap",
            "exit_criteria": "fix the implementation contract",
        }

        normal = ownership.validate_matrix(matrix)
        strict = ownership.validate_matrix(matrix, require_parity=True)

        self.assertTrue(normal["ok"], normal["errors"])
        self.assertTrue(
            any(
                blocker.startswith("cli.core_runtime_options:")
                for blocker in normal["blockers"]
            )
        )
        self.assertFalse(strict["ok"])
        self.assertTrue(
            any(
                "strict parity blocker: cli.core_runtime_options:" in error
                for error in strict["errors"]
            )
        )

    def test_forbidden_ds8_import_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "noesis").mkdir()
            (root / "DS9" / "noesis").mkdir(parents=True)
            (root / "DS9" / "scripts").mkdir(parents=True)
            (root / "noesis" / "shared.py").write_text("MARKER = True\n", encoding="utf-8")
            (root / "DS9" / "noesis" / "shared.py").write_text(
                "from noesis import ds8_runtime\nMARKER = True\n", encoding="utf-8"
            )
            matrix = {
                "schema_version": 2,
                "matrix_id": ownership.EXPECTED_MATRIX_ID,
                "policy": {
                    "allowed_module_classifications": sorted(
                        ownership.ALLOWED_MODULE_CLASSIFICATIONS
                    ),
                    "allowed_capability_statuses": sorted(
                        ownership.ALLOWED_CAPABILITY_STATUSES
                    ),
                    "strict_blocking_statuses": sorted(
                        ownership.STRICT_BLOCKING_STATUSES
                    ),
                    "forbidden_ds9_imports": [
                        {
                            "id": "ds8",
                            "module": "noesis.ds8_runtime",
                            "regex": r"(?m)^\s*from\s+noesis\.ds8_runtime\b",
                        }
                    ],
                    "scan_roots": ["DS9/noesis"],
                    "scan_suffixes": [".py"],
                },
                "modules": [
                    {
                        "id": "duplicate.shared",
                        "classification": "duplicated_pending_convergence",
                        "ds8_path": "noesis/shared.py",
                        "ds9_path": "DS9/noesis/shared.py",
                        "reason": "fixture",
                        "convergence_target": "shared",
                    }
                ],
                "capabilities": [
                    {
                        "id": "fixture",
                        "surface": "test",
                        "status": "parity",
                        "evidence": {
                            "repository_source": {
                                "ds8": {
                                    "path": "noesis/shared.py",
                                    "contains_all": ["MARKER"],
                                },
                                "ds9": {
                                    "path": "DS9/noesis/shared.py",
                                    "contains_all": ["MARKER"],
                                },
                            }
                        },
                    }
                ],
            }
            previous = ownership.REPO_ROOT
            ownership.REPO_ROOT = root
            try:
                result = ownership.validate_matrix(matrix)
            finally:
                ownership.REPO_ROOT = previous
            self.assertFalse(result["ok"])
            self.assertTrue(any("forbidden DS9 dependency ds8" in error for error in result["errors"]))

    def test_repository_source_yaml_rejects_duplicate_explicit_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.yaml"
            source.write_text("config:\n  lane: baseline\n  lane: v3dt\n", encoding="utf-8")
            errors: list[str] = []
            with mock.patch.object(ownership, "REPO_ROOT", root):
                ownership._validate_evidence(
                    "fixture",
                    {"path": "source.yaml", "yaml_keys": ["config.lane"]},
                    errors,
                )
            self.assertTrue(any("duplicate repository source YAML key" in row for row in errors))

    def test_repository_source_descriptor_read_rejects_filename_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.yaml"
            source.write_text("config:\n  lane: baseline\n", encoding="utf-8")
            swapped = False

            def rename_swap(label: str, _root: Path, _relative: Path) -> None:
                nonlocal swapped
                if label != "repository source fixture" or swapped:
                    return
                swapped = True
                source.rename(root / "source-opened.yaml")
                source.write_text("config:\n  lane: v3dt\n", encoding="utf-8")

            errors: list[str] = []
            with (
                mock.patch.object(ownership, "REPO_ROOT", root),
                mock.patch.object(ownership, "_OPENAT_TEST_HOOK", rename_swap),
            ):
                ownership._validate_evidence(
                    "fixture",
                    {"path": "source.yaml", "yaml_equals": {"config.lane": "baseline"}},
                    errors,
                )
            self.assertTrue(swapped)
            self.assertTrue(
                any("replaced" in row or "changed while it was read" in row for row in errors),
                errors,
            )


if __name__ == "__main__":
    unittest.main()
