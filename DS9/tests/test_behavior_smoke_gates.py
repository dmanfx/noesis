from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path) -> ModuleType:
    original_sys_path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_sys_path


bridge = _load_module(
    "ds9_bridge_contract_smoke_gate_test",
    REPO_ROOT / "DS9" / "scripts" / "ds9_bridge_contract_smoke_test.py",
)
root_zero_copy = _load_module(
    "root_zero_copy_smoke_gate_test",
    REPO_ROOT / "scripts" / "zero_copy_smoke_test.py",
)
ds9_zero_copy = _load_module(
    "ds9_zero_copy_smoke_gate_test",
    REPO_ROOT / "DS9" / "scripts" / "zero_copy_smoke_test.py",
)


def _bridge_args() -> SimpleNamespace:
    return SimpleNamespace(
        ws="ws://127.0.0.1:6008",
        duration=10.0,
        min_depth_tensor_frames=1,
        min_object_depth_roi_copies=1,
        min_object_depth_attaches=1,
        min_object_depth_ok=1,
        min_reid_extractions=1,
        max_zero_copy_violations=0,
        require_embedding_track=True,
    )


def _passing_bridge_evidence(pipeline_errors: list[str]) -> dict[str, object]:
    return bridge._evaluate_bridge_evidence(
        _bridge_args(),
        max_counters={
            "depth_tracking_device_frames_total": 1,
            "object_depth_gpu_roi_copies_total": 1,
            "object_depth_attach_total": 1,
            "object_depth_status_total.ok": 1,
            "tensor_host_copies_total.reid": 1,
            "core_path.cpu_copy_violation.total": 0,
        },
        stats_samples=1,
        tracking_messages=1,
        tracks_seen=1,
        depth_ok_tracks=1,
        embedding_tracks=1,
        pipeline_errors=pipeline_errors,
    )


def test_bridge_pipeline_errors_are_cumulative_and_gate_success() -> None:
    errors = bridge._merge_pipeline_errors([], ["first error", "", "first error"])
    errors = bridge._merge_pipeline_errors(errors, ["second error"])

    assert errors == ["first error", "second error"]
    clean = _passing_bridge_evidence([])
    failed = _passing_bridge_evidence(errors)
    assert clean["ok"] is True
    assert clean["checks"]["pipeline_errors_absent"] is True
    assert failed["ok"] is False
    assert failed["checks"]["pipeline_errors_absent"] is False
    assert failed["pipeline_errors"] == errors


@pytest.mark.parametrize(
    ("attempts", "successes", "last_status", "expected"),
    (
        (0, 0, "none", "no_rest_refresh_attempts"),
        (1, 0, "ok", "no_successful_rest_refresh"),
        (2, 1, "http_503", "rest_refresh_final_status_not_ok"),
        (1, 1, "ok", None),
    ),
)
def test_zero_copy_rest_refresh_contract_is_fail_closed_in_both_copies(
    attempts: int, successes: int, last_status: str, expected: str | None
) -> None:
    kwargs = {
        "attempts": attempts,
        "successes": successes,
        "last_status": last_status,
    }
    assert root_zero_copy._rest_refresh_contract_error(**kwargs) == expected
    assert ds9_zero_copy._rest_refresh_contract_error(**kwargs) == expected
