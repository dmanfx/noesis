from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "driver_rollback_rehearsal.py"
SPEC = REPO_ROOT / "DS9" / "config" / "driver_rollback_580.json"


def _load_module():
    module_spec = importlib.util.spec_from_file_location(
        "driver_rollback_rehearsal_test_module", SCRIPT
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


rehearsal = _load_module()


class DriverRollbackRehearsalTests(unittest.TestCase):
    def test_exact_spec_is_structurally_valid(self) -> None:
        spec = rehearsal.load_spec(SPEC)
        records = rehearsal.expected_packages(spec)
        self.assertEqual(len(records), 33)
        self.assertEqual(
            {key.architecture for key in records}, {"all", "amd64", "i386"}
        )
        self.assertEqual(
            spec["simulation"], {"upgraded": 0, "newly_installed": 25, "removed": 21}
        )
        self.assertEqual(len(spec["explicit_removals"]), 21)

    def test_every_non_595_removal_has_solver_necessity_evidence(self) -> None:
        spec = rehearsal.load_spec(SPEC)
        non_595 = {name for name in spec["explicit_removals"] if "595" not in name}
        self.assertEqual(non_595, {"nvidia-prime"})
        evidence = spec["solver_required_non_595_removals"]["nvidia-prime"]
        self.assertEqual(evidence["required_by"], "nvidia-driver-580-open")
        self.assertEqual(evidence["relationship"], "Conflicts")

    def test_undocumented_generic_removal_is_rejected(self) -> None:
        raw = json.loads(SPEC.read_text(encoding="utf-8"))
        raw["explicit_removals"].append("unrelated-desktop-package")
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "rollback.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(
                rehearsal.RehearsalError, "solver-necessity evidence"
            ):
                rehearsal.load_spec(path)

    def test_relationship_digest_covers_dependency_and_conflict_fields(self) -> None:
        control = {field: "" for field in rehearsal.RELATIONSHIP_FIELDS}
        control["Depends"] = "driver-core (= 580.1)"
        before = rehearsal.relationship_sha256(control, rehearsal.RELATIONSHIP_FIELDS)
        control["Conflicts"] = "driver-core-595"
        after = rehearsal.relationship_sha256(control, rehearsal.RELATIONSHIP_FIELDS)
        self.assertNotEqual(before, after)

    def test_checkpoint_hashes_every_declared_byte(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root) / "checkpoint"
            required = {
                "README.md": b"checkpoint\n",
                "VALIDATION.md": b"validation\n",
                "host/state/package-cache-inventory.tsv": b"inventory\n",
                "host/state/dpkg-status": b"status\n",
            }
            root.mkdir(mode=0o700)
            for relative, content in required.items():
                path = root / relative
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                path.write_bytes(content)
                path.chmod(0o600)
            lines = [
                f"{hashlib.sha256(content).hexdigest()}  ./{relative}"
                for relative, content in sorted(required.items())
            ]
            sums = root / "SHA256SUMS"
            sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
            sums.chmod(0o600)
            authority = hashlib.sha256(sums.read_bytes()).hexdigest()

            result = rehearsal.verify_checkpoint(root, authority)
            self.assertEqual(result["verified_file_count"], 4)
            self.assertEqual(result["verified_bytes"], sum(map(len, required.values())))

            (root / "README.md").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(rehearsal.RehearsalError, "hash mismatch"):
                rehearsal.verify_checkpoint(root, authority)

    def test_checkpoint_rejects_manifest_traversal(self) -> None:
        with self.assertRaisesRegex(rehearsal.RehearsalError, "unsafe"):
            rehearsal._safe_manifest_relative("../outside")
        with self.assertRaisesRegex(rehearsal.RehearsalError, "unsafe"):
            rehearsal._safe_manifest_relative("nested/../outside")

    def test_simulation_is_local_dry_run_with_exact_actions(self) -> None:
        package = rehearsal.PackageKey("nvidia-driver-580-open", "580.1", "amd64")
        record = rehearsal.PackageRecord(
            key=package,
            sha256="a" * 64,
            relationships_sha256="b" * 64,
            path=Path("/checkpoint/rollback/nvidia-driver-580-open.deb"),
        )
        spec = {
            "explicit_removals": ["nvidia-driver-595-open"],
            "simulation": {"upgraded": 0, "newly_installed": 1, "removed": 1},
        }
        apt_output = "\n".join(
            (
                "The following NEW packages will be installed:",
                "  nvidia-driver-580-open",
                "0 upgraded, 1 newly installed, 1 to remove and 0 not upgraded.",
                "Remv nvidia-driver-595-open [595.1]",
                "Inst nvidia-driver-580-open (580.1 local-deb [amd64])",
                "Conf nvidia-driver-580-open (580.1 local-deb [amd64])",
            )
        )
        completed = SimpleNamespace(returncode=0, stdout=apt_output, stderr="")
        with (
            mock.patch.object(rehearsal, "installed_packages", return_value={}),
            mock.patch.object(rehearsal, "sha256_file", return_value="c" * 64),
            mock.patch.object(rehearsal, "_run", return_value=completed) as run,
        ):
            result = rehearsal.simulate_rollback({package: record}, spec)
        command = run.call_args.args[0]
        self.assertIn("--simulate", command)
        self.assertIn("--no-download", command)
        self.assertNotIn("-y", command)
        self.assertEqual(result["install_actions_verified"], 1)
        self.assertEqual(result["removal_actions_verified"], 1)
        self.assertFalse(result["network_downloads_allowed"])

    def test_recovery_plan_never_claims_execution(self) -> None:
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        plan = rehearsal.recovery_plan(spec)
        self.assertIn("--simulate", plan[2]["action"])
        self.assertIn("-y", plan[3]["action"])
        self.assertIn("reboot", " ".join(item["action"] for item in plan))
        self.assertTrue(
            all(item["order"] == str(index + 1) for index, item in enumerate(plan))
        )

    def test_output_is_atomic_private_and_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            root.chmod(0o700)
            output = root / "rollback-evidence.json"
            payload = {"status": "validated_rehearsal", "rollback_executed": False}
            digest = rehearsal.write_private_atomic_json(output, payload)
            self.assertEqual(oct(output.stat().st_mode & 0o777), "0o600")
            self.assertEqual(output.stat().st_nlink, 1)
            self.assertEqual(digest, hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            with self.assertRaisesRegex(
                rehearsal.RehearsalError, "refusing to replace"
            ):
                rehearsal.write_private_atomic_json(output, payload)

    def test_output_rejects_symlink_and_shared_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            root.chmod(0o700)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            linked = root / "linked.json"
            linked.symlink_to(target)
            with self.assertRaisesRegex(rehearsal.RehearsalError, "is a symlink"):
                rehearsal.write_private_atomic_json(linked, {"status": "test"})

            shared = root / "shared"
            shared.mkdir(mode=0o755)
            with self.assertRaisesRegex(rehearsal.RehearsalError, "owner-only"):
                rehearsal.write_private_atomic_json(
                    shared / "evidence.json", {"status": "test"}
                )


if __name__ == "__main__":
    unittest.main()
