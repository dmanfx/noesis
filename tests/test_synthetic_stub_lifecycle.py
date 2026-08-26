from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from noesis.pipelines import deepstream_pipeline as pipeline_builder
from noesis_core.servicemaker_shutdown import (
    OrderlyEosError,
    SyntheticStubEosMessage,
    is_synthetic_stub_pipeline,
    request_orderly_eos,
    synthetic_stub_lifecycle_evidence,
    validate_synthetic_stub_eos_message,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> Any:
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    original_sys_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


def test_synthetic_stub_orderly_eos_is_monotonic_and_unblocks_wait() -> None:
    module = pipeline_builder
    backend = module._NoopDSPipeline("lifecycle-test")
    backend.add("noesiseos", "orderly_eos_control")
    callbacks: list[SyntheticStubEosMessage] = []
    backend.prepare(callbacks.append)
    backend.activate()
    pipeline = SimpleNamespace(
        shutdown_eos_component_name="orderly_eos_control",
        components={
            "orderly_eos_control": SimpleNamespace(element="noesiseos")
        },
        ds_pipeline=backend,
        lifecycle_evidence=synthetic_stub_lifecycle_evidence(),
    )

    wait_returned = threading.Event()

    def _wait() -> None:
        backend.wait()
        wait_returned.set()

    wait_thread = threading.Thread(target=_wait, daemon=True)
    wait_thread.start()
    assert not wait_returned.wait(0.02)

    evidence = request_orderly_eos(pipeline, timeout_s=1.0, poll_interval_s=0.001)
    assert wait_returned.wait(1.0)
    wait_thread.join(timeout=1.0)
    assert not wait_thread.is_alive()
    assert evidence.request_sequence == evidence.accepted_sequence == 1
    assert evidence.last_request_ok is True
    assert callbacks == [
        SyntheticStubEosMessage(
            request_sequence=1,
            backend="synthetic_stub",
            native_runtime=False,
            promotable=False,
        )
    ]
    assert validate_synthetic_stub_eos_message(backend, callbacks[0]) == 1
    assert backend.lifecycle_evidence == {
        "backend": "synthetic_stub",
        "native_runtime": False,
        "promotable": False,
    }
    assert is_synthetic_stub_pipeline(backend)

    node = backend["orderly_eos_control"]
    node.set({"request-sequence": 1})
    time.sleep(0.02)
    assert len(callbacks) == 1
    second = request_orderly_eos(pipeline, timeout_s=1.0, poll_interval_s=0.001)
    assert second.request_sequence == second.accepted_sequence == 2
    assert [message.request_sequence for message in callbacks] == [1, 2]


def test_synthetic_stub_eos_callback_failure_is_not_acknowledged() -> None:
    module = pipeline_builder
    backend = module._NoopDSPipeline("lifecycle-test")
    backend.add("noesiseos", "orderly_eos_control")

    def _reject(_message: SyntheticStubEosMessage) -> None:
        raise RuntimeError("callback rejected")

    backend.prepare(_reject)
    backend.activate()
    pipeline = SimpleNamespace(
        shutdown_eos_component_name="orderly_eos_control",
        components={
            "orderly_eos_control": SimpleNamespace(element="noesiseos")
        },
        ds_pipeline=backend,
    )

    with pytest.raises(OrderlyEosError, match="acknowledgement timed out"):
        request_orderly_eos(pipeline, timeout_s=0.03, poll_interval_s=0.001)
    node = backend["orderly_eos_control"]
    assert node.get("accepted-sequence") == 0
    assert node.get("last-request-ok") is False


def test_synthetic_stub_node_get_and_analytics_receipt(tmp_path: Path) -> None:
    module = pipeline_builder
    config_path = tmp_path / "exclude.ini"
    config_path.write_text("[property]\nenable=1\n", encoding="utf-8")
    backend = module._NoopDSPipeline("analytics-test")
    backend.add(
        "nvdsroiexclude",
        "analytics_exclude",
        {"config-file": str(config_path)},
    )
    node = backend["analytics_exclude"]
    initial_sha256 = node.get("active-config-sha256")

    assert len(initial_sha256) == 64
    assert node.get("last-reload-ok") is True
    with pytest.raises(KeyError):
        node.get("not-a-real-property")

    config_path.write_text("[property]\nenable=0\n", encoding="utf-8")
    expected_sha256 = module._NoopPipelineNode._sha256_file(config_path)
    node.set({"expected-config-sha256": expected_sha256})
    node.set({"reload-request-sequence": 1})
    assert node.get("reload-request-sequence") == 1
    assert node.get("reload-accepted-sequence") == 1
    assert node.get("active-config-sha256") == expected_sha256
    assert node.get("last-reload-ok") is True


def test_synthetic_marker_requires_exact_non_promotable_pipeline_evidence() -> None:
    assert is_synthetic_stub_pipeline(
        SimpleNamespace(lifecycle_evidence=synthetic_stub_lifecycle_evidence())
    )
    for drifted in (
        {"backend": "native_servicemaker", "native_runtime": False, "promotable": False},
        {"backend": "synthetic_stub", "native_runtime": True, "promotable": False},
        {"backend": "synthetic_stub", "native_runtime": False, "promotable": True},
        {
            "backend": "synthetic_stub",
            "native_runtime": False,
            "promotable": False,
            "status": "pass",
        },
    ):
        assert not is_synthetic_stub_pipeline(
            SimpleNamespace(lifecycle_evidence=drifted)
        )

    backend = SimpleNamespace(lifecycle_evidence=synthetic_stub_lifecycle_evidence())
    with pytest.raises(OrderlyEosError, match="lifecycle marker drifted"):
        validate_synthetic_stub_eos_message(
            backend,
            SyntheticStubEosMessage(
                request_sequence=1,
                backend="native_servicemaker",
            ),
        )
    with pytest.raises(OrderlyEosError, match="without exact stub evidence"):
        validate_synthetic_stub_eos_message(
            SimpleNamespace(lifecycle_evidence={}),
            SyntheticStubEosMessage(request_sequence=1),
        )


def test_ds9_builder_uses_only_the_ds9_stub_selector() -> None:
    source = (
        ROOT / "DS9" / "noesis" / "pipelines" / "deepstream_pipeline.py"
    ).read_text(encoding="utf-8")
    selector_block = source[
        source.index("under_pytest =") : source.index("pipeline = DeepStreamPipeline(")
    ]
    assert 'use_stub = _env_truthy("NOESIS_DS9_STUB_PIPELINE"' in selector_block
    assert "NOESIS_DS9_FORCE_NATIVE_TEST_PIPELINE" in selector_block
    assert "NOESIS_DS8_STUB_PIPELINE" not in selector_block
    assert "NOESIS_DS8_FORCE_NATIVE_TEST_PIPELINE" not in selector_block


def test_runtime_handles_only_the_typed_validated_stub_eos_before_native_guard() -> None:
    runtime_path = ROOT / "DS9" / "noesis" / "ds9_runtime_core.py"
    source = runtime_path.read_text(encoding="utf-8")
    start = source.index("def _on_pyservicemaker_message(")
    end = source.index("\ndef _start_pyservicemaker_wait_loop(", start)
    handler = source[start:end]

    typed = "if type(message) is SyntheticStubEosMessage:"
    validate = "validate_synthetic_stub_eos_message(ds_pipeline, message)"
    native_guard = "if not _PYSERVICEMAKER_MSGS:"
    assert handler.index(typed) < handler.index(validate) < handler.index(native_guard)
    assert 'state["pipeline_eos_seen"] = True' in handler
    assert 'state["synthetic_eos_request_sequence"] = sequence' in handler
    assert 'state["pipeline_failed"] = True' in handler
    assert 'logger.info("EOS received on pipeline (reason=%s)", eos_reason)' in handler


def test_ds9_launcher_stub_selector_is_ds9_owned_and_skips_native_preflight(
) -> None:
    code = r'''
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from types import ModuleType

root = Path.cwd()
path = root / "DS9" / "noesis" / "ds9_runtime.py"
spec = importlib.util.spec_from_file_location("ds9_synthetic_stub_launcher", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

os.environ.pop("NOESIS_DS9_STUB_PIPELINE", None)
os.environ["NOESIS_DS8_STUB_PIPELINE"] = "1"
assert module._synthetic_stub_requested() is False
os.environ["NOESIS_DS9_STUB_PIPELINE"] = "true"
assert module._synthetic_stub_requested() is True

module._ensure_runtime_sys_path = lambda: None
module._set_ds9_environment = lambda: None
module._selected_launch_paths = lambda: (
    root / "unused-infer.yaml",
    root / "unused-cameras.yaml",
    True,
    True,
)
def must_not_run(*args, **kwargs):
    raise AssertionError("native DS9 preparation ran for the synthetic backend")
module._run_preflight = must_not_run
module._preload_ds9_python_runtime = must_not_run
runtime_core = ModuleType("noesis.ds9_runtime_core")
runtime_core.main = lambda: 0
sys.modules["noesis.ds9_runtime_core"] = runtime_core

stderr = io.StringIO()
with contextlib.redirect_stderr(stderr):
    assert module.main() == 0
print(stderr.getvalue().strip())
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout)
    assert evidence == {
        "event": "pipeline_backend_selected",
        "backend": "synthetic_stub",
        "native_runtime": False,
        "promotable": False,
    }


@pytest.mark.parametrize(
    ("name", "path", "requires_auth"),
    [
        (
            "root_zero_copy_stats_stub_result",
            ROOT / "scripts" / "zero_copy_stats_smoke_test.py",
            True,
        ),
        (
            "ds9_zero_copy_stats_stub_result",
            ROOT / "DS9" / "scripts" / "zero_copy_stats_smoke_test.py",
            True,
        ),
    ],
)
def test_zero_copy_stats_stub_result_is_explicitly_non_promotable(
    name: str,
    path: Path,
    requires_auth: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module(name, path)
    args = SimpleNamespace(
        auth_token_file=tmp_path / "unused-token",
        stats_ws="ws://127.0.0.1:6008",
        log_path=tmp_path / "stub.log",
        no_spawn=False,
        pipeline_config=tmp_path / "infer.yaml",
        cameras_config=tmp_path / "cameras.yaml",
        stub=True,
        skip_cuda_preflight=True,
        duration_s=3.0,
        startup_timeout=1.0,
        max_p99_ms=3.0,
        max_violations=0,
        depth_camera="__stub__",
    )
    monkeypatch.setattr(module, "_parse_args", lambda: args)
    if requires_auth:
        monkeypatch.setattr(module, "load_required_internal_auth", lambda _path: object())

    async def _result(**_kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "samples": 1}

    class _PortLock:
        def close(self) -> None:
            return None

    fake_process = SimpleNamespace()

    def _spawn(**_kwargs: Any) -> Any:
        args.log_path.write_text(
            "\n".join(module._SYNTHETIC_LIFECYCLE_LOG_MARKERS),
            encoding="utf-8",
        )
        return fake_process

    monkeypatch.setattr(module, "_reserve_local_port", lambda: (61008, _PortLock()))
    monkeypatch.setattr(module, "_spawn_runtime", _spawn)
    monkeypatch.setattr(
        module,
        "_terminate_process",
        lambda _proc: {
            "signal_requested": True,
            "forced_kill": False,
            "exit_code": 0,
            "graceful": True,
        },
    )
    monkeypatch.setattr(module, "_collect_stats", _result)
    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["test_backend_only"] is True
    assert payload["synthetic_lifecycle"]["ok"] is True
    assert payload["lifecycle_evidence"] == {
        "backend": "synthetic_stub",
        "native_runtime": False,
        "promotable": False,
    }


def test_zero_copy_stats_stub_receipt_rejects_forced_or_incomplete_shutdown(
    tmp_path: Path,
) -> None:
    module = _load_module(
        "root_zero_copy_stats_failed_lifecycle",
        ROOT / "scripts" / "zero_copy_stats_smoke_test.py",
    )
    log_path = tmp_path / "forced.log"
    log_path.write_text(module._SYNTHETIC_BACKEND_LOG_MARKER, encoding="utf-8")
    receipt = module._synthetic_lifecycle_receipt(
        log_path,
        {
            "signal_requested": True,
            "forced_kill": True,
            "exit_code": -9,
            "graceful": False,
        },
    )
    assert receipt["ok"] is False
    assert receipt["termination"]["forced_kill"] is True
    assert "Shutdown complete" in receipt["missing_markers"]


@pytest.mark.parametrize(
    ("name", "path"),
    [
        (
            "root_zero_copy_stats_stub_state",
            ROOT / "scripts" / "zero_copy_stats_smoke_test.py",
        ),
        (
            "ds9_zero_copy_stats_stub_state",
            ROOT / "DS9" / "scripts" / "zero_copy_stats_smoke_test.py",
        ),
    ],
)
def test_zero_copy_stub_state_is_isolated_from_operator_home(
    name: str,
    path: Path,
    tmp_path: Path,
) -> None:
    module = _load_module(name, path)
    operator_home = tmp_path / "operator-home"
    camera_secrets = operator_home / ".local/state/noesis/secrets/camera_sources.json"
    mapanything_key = operator_home / ".local/state/noesis/secrets/mapanything_rpc.key"
    env = {"HOME": str(operator_home)}
    state_dir = module._configure_synthetic_runtime_environment(env)
    try:
        assert state_dir != operator_home
        assert state_dir.stat().st_mode & 0o777 == 0o700
        assert env["HOME"] == str(state_dir)
        assert env["PYTHONUSERBASE"] == str(operator_home / ".local")
        assert env["NOESIS_CAMERA_SECRETS_FILE"] == str(camera_secrets)
        assert env["NOESIS_MAPANYTHING_API_KEY_FILE"] == str(mapanything_key)
        for key in (
            "NOESIS_ANALYTICS_EXCLUDE_CONFIG",
            "NOESIS_IDENTITY_V2_STORE",
            "NOESIS_WORLD_JOURNAL_PATH",
            "NOESIS_BUILD_DIR",
        ):
            assert Path(env[key]).is_relative_to(state_dir)
        assert env["NOESIS_HOUSEHOLD_ARCHIVE_STATE"] == "0"
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)
