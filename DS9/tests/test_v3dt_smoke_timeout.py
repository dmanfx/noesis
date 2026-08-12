from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from noesis_core.v3dt_validation import V3DTBBoxTimeoutContract


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "sv3dt_meta_smoke_test.py"
SPEC = importlib.util.spec_from_file_location("ds9_v3dt_bbox_timeout_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def test_timeout_contract_has_one_exact_runner_envelope() -> None:
    contract = V3DTBBoxTimeoutContract(duration_seconds=20.0, attempts=8)
    assert contract.collection_envelope_seconds == 167.0
    assert contract.runner_timeout_seconds == 187.0


@pytest.mark.parametrize(
    ("duration", "attempts", "message"),
    (
        (0.0, 1, "duration"),
        (float("nan"), 1, "duration"),
        (1.0, 0, "attempts"),
        (1.0, 33, "attempts"),
        (1.0, True, "attempts"),
    ),
)
def test_timeout_contract_rejects_invalid_values(
    duration: float, attempts: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        V3DTBBoxTimeoutContract(
            duration_seconds=duration,
            attempts=attempts,  # type: ignore[arg-type]
        )


def test_bbox_attempts_stop_early_without_a_final_sleep() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    async def collect(_uri: str, _auth: object, *, duration: float) -> bool:
        assert duration == 4.0
        calls.append(len(calls) + 1)
        return len(calls) == 2

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    result = asyncio.run(
        smoke._run_bbox3d_attempts(
            "ws://127.0.0.1:6008",
            object(),
            V3DTBBoxTimeoutContract(duration_seconds=4.0, attempts=4),
            collect=collect,
            sleep=sleep,
        )
    )

    assert result == (True, None)
    assert calls == [1, 2]
    assert sleeps == [1.0]


def test_bbox_attempts_fail_deterministically_and_never_sleep_after_last() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    async def collect(_uri: str, _auth: object, *, duration: float) -> bool:
        assert duration == 3.0
        calls.append(len(calls) + 1)
        return False

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    ok, error = asyncio.run(
        smoke._run_bbox3d_attempts(
            "ws://127.0.0.1:6008",
            object(),
            V3DTBBoxTimeoutContract(duration_seconds=3.0, attempts=3),
            collect=collect,
            sleep=sleep,
        )
    )

    assert ok is False
    assert isinstance(error, RuntimeError)
    assert "bbox3d was missing" in str(error)
    assert calls == [1, 2, 3]
    assert sleeps == [1.0, 1.0]


def test_smoke_cli_exposes_the_shared_default_attempt_count() -> None:
    args = smoke._parse_args([])
    assert args.attempts == 8
