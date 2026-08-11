from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import time
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "DS9/scripts/nvml_gpu_memory_sampler.py"
SPEC = importlib.util.spec_from_file_location("ds9_nvml_sampler_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sampler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampler)


def _args(path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "evidence": path,
        "device_index": 0,
        "expected_uuid": "GPU-fixture",
        "engine": "wholebody49_s_masks",
        "transaction_id": "txn-fixture",
        "prepared_transaction_sha256": "a" * 64,
        "artifact_root_id": "b" * 64,
        "container_id": "c" * 64,
        "guard_mib": 9000,
        "interval_ms": 25,
        "max_gap_ms": 250,
        "parent_pid": os.getpid(),
        "parent_start_time_ticks": sampler._proc_start_time_ticks(os.getpid()),
        "max_samples": 1,
        "allow_active": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class _Backend:
    samples: list[tuple[int, int, int]] = []
    sleep_after_sample = 0.0

    def __init__(self, _device_index: int, _expected_uuid: str) -> None:
        self.index = 0

    def sample(self) -> tuple[int, int, int]:
        value = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        if self.sleep_after_sample:
            time.sleep(self.sleep_after_sample)
        return value

    def close(self) -> None:
        return None


def _run_sample(
    tmp_path: Path,
    used_bytes: list[int],
    *,
    max_samples: int | None = None,
    sleep_after_sample: float = 0.0,
    **overrides: object,
) -> tuple[int, Path]:
    evidence = tmp_path / "guard.jsonl"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.parent.chmod(0o700)

    class Backend(_Backend):
        samples = [
            (12288 * sampler.MIB, 375 * sampler.MIB, value) for value in used_bytes
        ]
        pass

    Backend.sleep_after_sample = sleep_after_sample
    args = _args(
        evidence,
        max_samples=max_samples or len(used_bytes),
        **overrides,
    )
    with mock.patch.object(sampler, "_notify_exact_parent"):
        status = sampler.sample(args, backend_type=Backend)
    return status, evidence


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_raw_guard_rejects_one_byte_over_and_accepts_exact_boundary(
    tmp_path: Path,
) -> None:
    boundary = 9000 * sampler.MIB
    status, evidence = _run_sample(tmp_path / "over", [boundary + 1])
    assert status == sampler.EXIT_GUARD_BREACH
    rows = _rows(evidence)
    assert rows[1]["used_bytes"] == boundary + 1
    assert rows[-1]["state"] == "guard_breached"

    status, evidence = _run_sample(tmp_path / "exact", [boundary])
    assert status == sampler.EXIT_SAMPLE_LIMIT
    assert _rows(evidence)[-1]["state"] == "sample_limit"


def test_sampler_fails_on_excessive_inter_sample_gap(tmp_path: Path) -> None:
    status, evidence = _run_sample(
        tmp_path,
        [512 * sampler.MIB, 512 * sampler.MIB],
        max_samples=2,
        sleep_after_sample=0.3,
    )
    assert status == sampler.EXIT_CADENCE_GAP
    assert _rows(evidence)[-1]["state"] == "cadence_gap"


def test_sampler_fails_when_bound_parent_identity_is_gone(tmp_path: Path) -> None:
    status, evidence = _run_sample(
        tmp_path,
        [512 * sampler.MIB],
        parent_pid=999_999_999,
        parent_start_time_ticks=1,
    )
    assert status == sampler.EXIT_PARENT_LOST
    assert _rows(evidence)[-1]["state"] == "parent_lost"


def test_sampler_fails_closed_when_nvml_shutdown_fails(tmp_path: Path) -> None:
    evidence = tmp_path / "guard.jsonl"
    evidence.parent.chmod(0o700)
    args = _args(evidence, max_samples=2)

    class Backend(_Backend):
        samples = [(12288 * sampler.MIB, 375 * sampler.MIB, 512 * sampler.MIB)]

        def sample(self) -> tuple[int, int, int]:
            os.kill(os.getpid(), signal.SIGTERM)
            return super().sample()

        def close(self) -> None:
            raise sampler.SamplerError("nvmlShutdown failed: fixture")

    previous = signal.getsignal(signal.SIGTERM)
    try:
        with mock.patch.object(sampler, "_notify_exact_parent"):
            status = sampler.sample(args, backend_type=Backend)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert status == sampler.EXIT_SAMPLER_ERROR
    assert _rows(evidence)[-1]["state"] == "sampler_error"
    assert "nvmlShutdown failed" in str(_rows(evidence)[-1]["error"])


def test_sampler_evidence_creation_is_exclusive(tmp_path: Path) -> None:
    evidence = tmp_path / "guard.jsonl"
    evidence.parent.chmod(0o700)
    evidence.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sampler.sample(_args(evidence), backend_type=_Backend)


def _clean_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    start = 1_000_000_000
    header = {
        "kind": "header",
        "schema_version": 1,
        "contract": sampler.CONTRACT,
        "device_index": args.device_index,
        "expected_uuid": args.expected_uuid,
        "engine": args.engine,
        "transaction_id": args.transaction_id,
        "prepared_transaction_sha256": args.prepared_transaction_sha256,
        "artifact_root_id": args.artifact_root_id,
        "container_id": args.container_id,
        "guard_mib": args.guard_mib,
        "guard_bytes": args.guard_mib * sampler.MIB,
        "interval_ms": args.interval_ms,
        "max_gap_ms": args.max_gap_ms,
        "parent_pid": args.parent_pid,
        "parent_start_time_ticks": args.parent_start_time_ticks,
        "sampler_pid": 123,
        "sampler_start_time_ticks": 456,
        "started_at_utc": "2026-07-11T13:00:00.000000Z",
    }
    samples = []
    for sequence, used_mib in enumerate((512, 513), start=1):
        samples.append(
            {
                "kind": "sample",
                "sequence": sequence,
                "sampled_at_utc": f"2026-07-11T13:00:00.0{sequence}0000Z",
                "monotonic_ns": start + sequence * 25_000_000,
                "total_mib": 12288,
                "total_bytes": 12288 * sampler.MIB,
                "reserved_mib": 375,
                "reserved_bytes": 375 * sampler.MIB,
                "used_mib": used_mib,
                "used_bytes": used_mib * sampler.MIB,
            }
        )
    footer = {
        "kind": "footer",
        "state": "stopped",
        "sample_count": 2,
        "peak_mib": 513,
        "peak_reserved_mib": 375,
        "first_sample_at_utc": samples[0]["sampled_at_utc"],
        "last_sample_at_utc": samples[-1]["sampled_at_utc"],
        "maximum_gap_ms": 25.0,
        "ended_at_utc": "2026-07-11T13:00:01.000000Z",
    }
    return [header, *samples, footer]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_summary_reconstructs_clean_maximum_and_digest(tmp_path: Path) -> None:
    evidence = tmp_path / "guard.jsonl"
    args = _args(evidence)
    _write_rows(evidence, _clean_rows(args))
    result = sampler.summarize(args)
    assert result["guard_ok"] is True
    assert result["maximum_observed_mib"] == 513
    assert result["sample_count"] == 2
    assert len(result["evidence_sha256"]) == 64


@pytest.mark.parametrize(
    "mutate,message",
    (
        (lambda rows: rows.pop(), "terminal footer"),
        (lambda rows: rows[-1].__setitem__("peak_mib", 1), "count/peak"),
        (
            lambda rows: rows[-1].__setitem__("maximum_gap_ms", 24.0),
            "sample summary",
        ),
        (
            lambda rows: rows[-1].__setitem__("peak_reserved_mib", 1),
            "sample summary",
        ),
        (lambda rows: rows.insert(2, dict(rows[0])), "row order/shape"),
        (lambda rows: rows[2].__setitem__("sequence", 9), "sequence"),
        (
            lambda rows: rows[2].__setitem__(
                "monotonic_ns", rows[1]["monotonic_ns"] + 300_000_000
            ),
            "cadence gap",
        ),
    ),
)
def test_summary_rejects_truncated_or_forged_evidence(
    tmp_path: Path, mutate, message: str
) -> None:
    evidence = tmp_path / "guard.jsonl"
    args = _args(evidence)
    rows = _clean_rows(args)
    mutate(rows)
    _write_rows(evidence, rows)
    with pytest.raises(sampler.SamplerError, match=message):
        sampler.summarize(args)


def test_summary_rejects_wrong_transaction_gpu_or_wrapper_identity(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "guard.jsonl"
    args = _args(evidence)
    _write_rows(evidence, _clean_rows(args))
    for key, value in (
        ("transaction_id", "wrong"),
        ("expected_uuid", "GPU-wrong"),
        ("parent_start_time_ticks", args.parent_start_time_ticks + 1),
        ("container_id", "d" * 64),
    ):
        with pytest.raises(sampler.SamplerError, match="header differs"):
            sampler.summarize(_args(evidence, **{key: value}))


def test_summary_cannot_conceal_transient_guard_breach(tmp_path: Path) -> None:
    evidence = tmp_path / "guard.jsonl"
    args = _args(evidence)
    rows = _clean_rows(args)
    rows[1]["used_bytes"] = 512 * sampler.MIB
    rows[1]["used_mib"] = 512
    rows[2]["used_bytes"] = 9441 * sampler.MIB
    rows[2]["used_mib"] = 9441
    rows[-1]["peak_mib"] = 9441
    _write_rows(evidence, rows)
    with pytest.raises(sampler.SamplerError, match="conceals a guard breach"):
        sampler.summarize(args)


def test_summary_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    evidence = tmp_path / "guard.jsonl"
    args = _args(evidence)
    rows = _clean_rows(args)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    lines[0] = lines[0].replace("{", '{"kind":"header",', 1)
    evidence.write_text("\n".join(lines) + "\n", encoding="utf-8")
    evidence.chmod(0o600)
    with pytest.raises(sampler.SamplerError, match="duplicate JSON key"):
        sampler.summarize(args)


def test_summary_rejects_unsafe_parent_and_hardlinked_evidence(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "guard.jsonl"
    args = _args(evidence)
    _write_rows(evidence, _clean_rows(args))
    evidence.parent.chmod(0o755)
    with pytest.raises(sampler.SamplerError, match="parent directory is not private"):
        sampler.summarize(args)
    evidence.parent.chmod(0o700)
    os.link(evidence, tmp_path / "guard-hardlink.jsonl")
    with pytest.raises(sampler.SamplerError, match="ownership/mode/size is unsafe"):
        sampler.summarize(args)


def test_summary_rejects_non_integer_sampler_identity(tmp_path: Path) -> None:
    evidence = tmp_path / "guard.jsonl"
    args = _args(evidence)
    rows = _clean_rows(args)
    rows[0]["sampler_pid"] = True
    _write_rows(evidence, rows)
    with pytest.raises(sampler.SamplerError, match="process identity"):
        sampler.summarize(args)
