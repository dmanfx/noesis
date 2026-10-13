"""Bounded CPU-only tests; systemd and inference never reach the host."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock, create_autospec

import pytest

from tools.mapanything_phone_scan import run_highres_reconstruction_lane as lane


@pytest.fixture(autouse=True)
def block_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("lane tests must not execute subprocesses, inference, or main()")

    monkeypatch.setattr(lane.subprocess, "run", forbidden)
    monkeypatch.setattr(lane.subprocess, "Popen", forbidden)
    monkeypatch.setattr(lane, "run_adaptive_da3_phone_scan", forbidden)
    monkeypatch.setattr(lane, "run_stage", forbidden)
    monkeypatch.setattr(lane, "main", forbidden)


@pytest.fixture
def lease_file(tmp_path):
    def create(**overrides):
        lease = {
            "configured": True,
            "appliance_target": lane.TARGET,
            "appliance_was_active": True,
            "noesis_was_active": True,
            "pause_requested": True,
            "appliance_paused": True,
            "restore_passed": None,
            "created_at": "2026-10-07T10:00:00+00:00",
            "input_binding": {"sha256": "unchanged-source"},
            **overrides,
        }
        path = tmp_path / "runtime_lease.json"
        lane.save(path, lease)
        return tmp_path, path, lease

    return create


@pytest.fixture
def runtime_mocks(monkeypatch):
    calls = []
    control = Mock(side_effect=lambda action: calls.append(("control", action)))

    def runtime_active(unit):
        calls.append(("active", unit))
        return True

    active = Mock(side_effect=runtime_active)
    monkeypatch.setattr(lane, "control", control)
    monkeypatch.setattr(lane, "active", active)
    monkeypatch.setattr(lane, "utc", lambda: "2026-10-07T11:00:00+00:00")
    save = Mock(wraps=lane.save)
    monkeypatch.setattr(lane, "save", save)
    return control, active, save, calls


@pytest.mark.parametrize(
    ("original_active", "pause_requested"),
    [(False, False), (False, True), (True, False)],
    ids=["originally-inactive", "inactive-pause-recorded", "active-not-paused"],
)
def test_restore_never_starts_an_inactive_or_unleased_appliance(
    lease_file, runtime_mocks, original_active, pause_requested
):
    root, path, original = lease_file(
        appliance_was_active=original_active, pause_requested=pause_requested
    )
    control, active, save, _ = runtime_mocks
    save.reset_mock()  # Ignore fixture creation of the lease.
    before = path.read_bytes()

    assert lane.restore(root) == original

    control.assert_not_called()
    active.assert_not_called()
    save.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.parametrize("appliance_paused", [False, True])
@pytest.mark.parametrize("restore_passed", [None, False])
def test_requested_active_lease_starts_target_then_checks_runtime_and_saves(
    lease_file, runtime_mocks, appliance_paused, restore_passed
):
    root, path, original = lease_file(
        appliance_paused=appliance_paused, restore_passed=restore_passed,
        restore_error="previous attempt failed",
    )
    control, active, save, calls = runtime_mocks
    save.reset_mock()

    result = lane.restore(root)

    control.assert_called_once_with("start")
    active.assert_called_once_with(lane.RUNTIME)
    assert calls == [("control", "start"), ("active", lane.RUNTIME)]
    expected = {
        **original, "restore_passed": True,
        "restored_at": "2026-10-07T11:00:00+00:00", "restore_error": None,
    }
    assert result == expected
    save.assert_called_once_with(path, expected)
    assert json.loads(path.read_text()) == expected


def test_successful_restoration_is_idempotent_without_restart_or_recheck(
    lease_file, runtime_mocks
):
    root, path, _ = lease_file()
    control, active, save, _ = runtime_mocks
    save.reset_mock()
    first = lane.restore(root)
    before = path.read_bytes()

    assert lane.restore(root) == first

    control.assert_called_once_with("start")
    active.assert_called_once_with(lane.RUNTIME)
    save.assert_called_once_with(path, first)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "error",
    [subprocess.CalledProcessError(
        1, ["systemctl", "--user", "start", lane.TARGET, lane.RUNTIME]
     ),
     subprocess.TimeoutExpired(
        ["systemctl", "--user", "start", lane.TARGET, lane.RUNTIME], 300
     ),
     KeyboardInterrupt("interrupted restoration")],
    ids=["start-failed", "start-timed-out", "start-interrupted"],
)
def test_failed_start_persists_failure_and_rethrows_same_exception(
    lease_file, runtime_mocks, error
):
    root, path, original = lease_file()
    control, active, save, _ = runtime_mocks
    save.reset_mock()
    control.side_effect = error

    with pytest.raises(type(error)) as caught:
        lane.restore(root)

    assert caught.value is error
    control.assert_called_once_with("start")
    active.assert_not_called()
    expected = {
        **original, "restore_passed": False,
        "restore_error": f"{type(error).__name__}: {error}",
    }
    save.assert_called_once_with(path, expected)
    assert json.loads(path.read_text()) == expected


@pytest.mark.parametrize("check_raises", [False, True], ids=["inactive", "check-error"])
def test_failed_runtime_check_persists_failure_and_rethrows(
    lease_file, runtime_mocks, check_raises
):
    root, path, original = lease_file()
    control, active, save, _ = runtime_mocks
    save.reset_mock()
    error = OSError("mock active check failed") if check_raises else None
    active.side_effect = error
    active.return_value = False

    with pytest.raises(OSError if check_raises else RuntimeError) as caught:
        lane.restore(root)

    if check_raises:
        assert caught.value is error
    else:
        assert str(caught.value) == "Noesis was not active after appliance restoration"
    control.assert_called_once_with("start")
    active.assert_called_once_with(lane.RUNTIME)
    expected = {
        **original, "restore_passed": False,
        "restore_error": f"{type(caught.value).__name__}: {caught.value}",
    }
    save.assert_called_once_with(path, expected)
    assert json.loads(path.read_text()) == expected


def test_failed_lease_can_retry_and_clear_persisted_error(lease_file, runtime_mocks):
    root, path, _ = lease_file()
    control, active, save, _ = runtime_mocks
    save.reset_mock()
    control.side_effect = [OSError("mock start failure"), None]
    with pytest.raises(OSError, match="mock start failure"):
        lane.restore(root)
    assert json.loads(path.read_text())["restore_passed"] is False

    result = lane.restore(root)

    assert control.call_count == save.call_count == 2
    active.assert_called_once_with(lane.RUNTIME)
    assert result["restore_passed"] is True
    assert result["restore_error"] is None
    assert json.loads(path.read_text()) == result


@pytest.mark.parametrize("payload", [b"", b"\x00bounded source\xff\n"])
def test_binding_reports_exact_bytes_hash_and_resolved_path_without_mutation(
    tmp_path, payload
):
    source = tmp_path / "source.bin"
    source.write_bytes(payload)

    assert lane.binding(source) == {
        "path": str(source.resolve()), "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert source.read_bytes() == payload


def test_missing_binding_fails_without_creating_a_file(tmp_path):
    source = tmp_path / "missing.bin"
    with pytest.raises(FileNotFoundError):
        lane.binding(source)
    assert not source.exists()


def test_save_writes_complete_sibling_before_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / "lease.json"
    lane.save(path, {"old": True})
    before = path.read_bytes()
    temporary = path.with_suffix(".json.tmp")
    payload = {"restore_passed": True, "source": Path("prepared/manifest.json")}
    expected = {"restore_passed": True, "source": "prepared/manifest.json"}
    original_replace = Path.replace

    def checked_replace(source, destination):
        assert source == temporary
        assert destination == path
        assert path.read_bytes() == before
        assert json.loads(source.read_text()) == expected
        return original_replace(source, destination)

    replace = create_autospec(Path.replace, side_effect=checked_replace)
    monkeypatch.setattr(Path, "replace", replace)

    assert lane.save(path, payload) is None

    replace.assert_called_once_with(temporary, path)
    assert json.loads(path.read_text()) == expected
    assert path.read_text().endswith("\n")
    assert not temporary.exists()


@pytest.mark.parametrize("failure_at", ["write", "replace"])
def test_atomic_save_failure_preserves_previous_destination(
    tmp_path, monkeypatch, failure_at
):
    path = tmp_path / "lease.json"
    lane.save(path, {"old": True})
    before = path.read_bytes()
    error = OSError(f"mock {failure_at} failure")
    replace = create_autospec(Path.replace, side_effect=error)
    monkeypatch.setattr(Path, "replace", replace)
    if failure_at == "write":
        monkeypatch.setattr(
            Path, "write_text", create_autospec(Path.write_text, side_effect=error)
        )

    with pytest.raises(OSError) as caught:
        lane.save(path, {"new": True})

    assert caught.value is error
    assert path.read_bytes() == before
    temporary = path.with_suffix(".json.tmp")
    if failure_at == "write":
        replace.assert_not_called()
    else:
        replace.assert_called_once_with(temporary, path)
        assert json.loads(temporary.read_text()) == {"new": True}


@pytest.mark.parametrize("returncode", [0, 3])
def test_active_uses_mocked_systemctl_for_exact_runtime_unit(monkeypatch, returncode):
    systemctl = Mock(return_value=subprocess.CompletedProcess([], returncode))
    monkeypatch.setattr(lane.subprocess, "run", systemctl)

    assert lane.active(lane.RUNTIME) is (returncode == 0)

    systemctl.assert_called_once_with(
        ["systemctl", "--user", "is-active", "--quiet", lane.RUNTIME], check=False
    )


def test_control_starts_exact_appliance_units_with_bounded_mocked_systemctl(
    monkeypatch,
):
    systemctl = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(lane.subprocess, "run", systemctl)

    lane.control("start")

    systemctl.assert_called_once_with(
        ["systemctl", "--user", "start", lane.TARGET, lane.RUNTIME],
        check=True, timeout=300,
    )
