from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "DS9/scripts/verify_gpu_process_ownership.py"
SPEC = importlib.util.spec_from_file_location("ds9_gpu_process_ownership", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {MODULE_PATH}")
ownership = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ownership
SPEC.loader.exec_module(ownership)


def _write_stat(root: Path, pid: int, parent_pid: int, start_time: int) -> None:
    process = root / str(pid)
    process.mkdir(parents=True)
    # Fields after comm begin with state (field 3); starttime is field 22.
    fields = ["S", str(parent_pid), *("0" for _ in range(17)), str(start_time)]
    (process / "stat").write_text(
        f"{pid} (fixture worker) " + " ".join(fields) + "\n",
        encoding="utf-8",
    )


class GPUProcessOwnershipTests(unittest.TestCase):
    def test_same_container_grandchild_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            proc = Path(raw)
            _write_stat(proc, 100, 1, 1000)
            _write_stat(proc, 110, 100, 1100)
            _write_stat(proc, 120, 110, 1200)
            result = ownership.prove_descendant(
                owner_pid=120,
                container_init_pid=100,
                expected_container_init_start_time_ticks=1000,
                proc_root=proc,
            )
            self.assertEqual(result["status"], "descendant")
            self.assertEqual(
                [entry["pid"] for entry in result["chain"]], [120, 110, 100]
            )

    def test_container_init_itself_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            proc = Path(raw)
            _write_stat(proc, 100, 1, 1000)
            result = ownership.prove_descendant(
                owner_pid=100,
                container_init_pid=100,
                expected_container_init_start_time_ticks=1000,
                proc_root=proc,
            )
            self.assertEqual([entry["pid"] for entry in result["chain"]], [100])

    def test_live_foreign_owner_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            proc = Path(raw)
            _write_stat(proc, 100, 1, 1000)
            _write_stat(proc, 200, 1, 2000)
            with self.assertRaisesRegex(ownership.ProcessProofError, "escapes"):
                ownership.prove_descendant(
                    owner_pid=200,
                    container_init_pid=100,
                    expected_container_init_start_time_ticks=1000,
                    proc_root=proc,
                )

    def test_vanished_owner_requests_nvml_requery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            proc = Path(raw)
            _write_stat(proc, 100, 1, 1000)
            result = ownership.prove_owners(
                owner_pids=[120],
                container_init_pid=100,
                expected_container_init_start_time_ticks=1000,
                proc_root=proc,
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["vanished"][0]["owner_pid"], 120)

    def test_pid_reuse_during_walk_is_rejected(self) -> None:
        init = ownership.ProcessIdentity(100, 1, 1000)
        owner_first = ownership.ProcessIdentity(120, 100, 1200)
        owner_reused = ownership.ProcessIdentity(120, 1, 9999)
        with mock.patch.object(
            ownership,
            "_read_process_identity",
            side_effect=[init, owner_first, owner_reused],
        ):
            with self.assertRaisesRegex(ownership.ProcessProofError, "identity changed"):
                ownership.prove_descendant(
                    owner_pid=120,
                    container_init_pid=100,
                    expected_container_init_start_time_ticks=1000,
                )

    def test_unreadable_or_malformed_process_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            proc = Path(raw)
            _write_stat(proc, 100, 1, 1000)
            process = proc / "120"
            process.mkdir()
            (process / "stat").write_text("malformed\n", encoding="utf-8")
            with self.assertRaisesRegex(ownership.ProcessProofError, "malformed"):
                ownership.prove_descendant(
                    owner_pid=120,
                    container_init_pid=100,
                    expected_container_init_start_time_ticks=1000,
                    proc_root=proc,
                )

    def test_init_pid_reuse_across_probes_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            proc = Path(raw)
            _write_stat(proc, 100, 1, 9000)
            with self.assertRaisesRegex(
                ownership.ProcessProofError, "differs from the inspected start time"
            ):
                ownership.prove_owners(
                    owner_pids=[],
                    container_init_pid=100,
                    expected_container_init_start_time_ticks=1000,
                    proc_root=proc,
                )

    def test_comm_name_containing_closing_parenthesis_parses(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            proc = Path(raw)
            _write_stat(proc, 100, 1, 1000)
            process = proc / "120"
            process.mkdir()
            fields = ["S", "100", *("0" for _ in range(17)), "1200"]
            (process / "stat").write_text(
                "120 (fixture ) worker) " + " ".join(fields) + "\n",
                encoding="utf-8",
            )
            result = ownership.prove_descendant(
                owner_pid=120,
                container_init_pid=100,
                expected_container_init_start_time_ticks=1000,
                proc_root=proc,
            )
            self.assertEqual(result["status"], "descendant")


if __name__ == "__main__":
    unittest.main()
