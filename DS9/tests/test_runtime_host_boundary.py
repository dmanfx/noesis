from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
from pathlib import Path
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "run_canonical_runtime_host.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ds9_runtime_host_boundary_test_module", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


host = _load_module()


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.chmod(0o700)
    return path


def _private_file(path: Path, value: str = "sentinel") -> Path:
    _private_directory(path.parent)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    return path


def _config_env(tmp_path: Path) -> dict[str, str]:
    native = _private_directory(tmp_path / "native")
    venv_python = native / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.chmod(0o700)
    artifacts = _private_directory(tmp_path / "artifacts")
    runtime = _private_directory(tmp_path / "runtime")
    payload = _private_directory(tmp_path / "state" / "payload")
    analytics = _private_directory(payload / "analytics")
    scene = _private_directory(payload / "scene")
    virtual_twin = _private_directory(payload / "virtual_twin")
    secrets = _private_directory(tmp_path / "secrets")
    return {
        "NOESIS_DS91_NATIVE_ROOT": str(native),
        "NOESIS_DS9_ARTIFACT_ROOT": str(artifacts),
        "NOESIS_DS9_RUNTIME_ROOT": str(runtime),
        "NOESIS_WORLD_JOURNAL_PATH": str(_private_file(payload / "world.db")),
        "NOESIS_IDENTITY_V2_STORE": str(_private_file(payload / "identity.db")),
        "NOESIS_ANALYTICS_CONFIG": str(
            _private_file(analytics / "nvdsanalytics.yaml")
        ),
        "NOESIS_ANALYTICS_EXCLUDE_CONFIG": str(
            _private_file(analytics / "config_nvdsanalytics_exclude.ini")
        ),
        "NOESIS_SCENE_STORE_PATH": str(
            _private_file(scene / "scene_releases.sqlite3")
        ),
        "NOESIS_VIRTUAL_TWIN_ROOT": str(virtual_twin),
        "NOESIS_CAMERA_SECRETS_FILE": str(_private_file(secrets / "camera_sources.json")),
        "NOESIS_MAPANYTHING_API_KEY_FILE": str(
            _private_file(secrets / "mapanything_rpc.key")
        ),
        "NOESIS_INTERNAL_AUTH_TOKEN_FILE": str(_private_file(secrets / "gateway-token")),
        "NOESIS_DEPLOYMENT_ID": "deploy-test-native-host",
        "NOESIS_HEALTH_SELECTOR_SHA256": "a" * 64,
        "NOESIS_STATE_RELEASE_ID": "state-test-native-host",
        "NOESIS_SOFTWARE_REVISION": "b" * 40,
    }


def test_host_path_materialization_uses_env_and_not_docker(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    with mock.patch.dict(os.environ, env, clear=False):
        config = host.load_native_config(env)
    assert config.artifacts == Path(env["NOESIS_DS9_ARTIFACT_ROOT"])
    assert config.world_store == Path(env["NOESIS_WORLD_JOURNAL_PATH"])
    assert config.secrets.cameras == Path(env["NOESIS_CAMERA_SECRETS_FILE"])
    assert not hasattr(config, "docker")


def test_missing_artifact_root_is_rejected(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    env.pop("NOESIS_DS9_ARTIFACT_ROOT")
    with pytest.raises(host.NativeRuntimeError, match="NOESIS_DS9_ARTIFACT_ROOT"):
        host.load_native_config(env)


def test_relative_secret_path_is_rejected(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    env["NOESIS_CAMERA_SECRETS_FILE"] = "secrets/camera_sources.json"
    with pytest.raises(host.NativeRuntimeError, match="absolute"):
        host.load_native_config(env)


def test_canonical_command_uses_host_storage_and_baseline_lane(tmp_path: Path) -> None:
    storage = tmp_path / "depth"
    argv = host.canonical_runtime_arguments(storage_base=storage)
    assert argv[0] == "DS9/noesis/ds9_runtime.py"
    assert "--pgie-profile" in argv
    assert argv[argv.index("--pgie-profile") + 1] == "yolo26"
    assert argv[argv.index("--tracking-mode") + 1] == "baseline"
    assert argv[argv.index("--storage-base") + 1] == str(storage)
    assert argv[argv.index("--ws-port") + 1] == "6008"
    assert argv[argv.index("--rest-port") + 1] == "8080"


def test_mv3dt_command_is_an_explicit_native_host_lane(tmp_path: Path) -> None:
    storage = tmp_path / "depth"
    argv = host.canonical_runtime_arguments(
        storage_base=storage,
        tracking_mode="mv3dt",
    )
    assert argv[argv.index("--pipeline-config") + 1] == "DS9/config/infer_mv3dt.yaml"
    assert argv[argv.index("--cameras-config") + 1] == "DS9/config/cameras_v3dt.yaml"
    assert argv[argv.index("--pgie-profile") + 1] == "yolo26"
    assert argv[argv.index("--size") + 1] == "m"
    assert argv[argv.index("--tracking-mode") + 1] == "mv3dt"


def test_run_environment_strips_selector_and_docker(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    env.update(
        {
            "NOESIS_APPLIANCE_RUNTIME_CONTEXT": "selector-json",
            "NOESIS_APPLIANCE_RUNTIME_CONTEXT_SHA256": "c" * 64,
            "NOESIS_DEPLOYMENT_SELECTOR_FILE": "/tmp/selector.json",
            "NOESIS_DS9_DOCKER_ROOT": "/tmp/docker-root",
            "DOCKER_HOST": "unix:///tmp/docker.sock",
        }
    )
    with mock.patch.dict(os.environ, env, clear=False):
        config = host.load_native_config(env)
        built = host.build_run_environment(
            config,
            session_id="sess",
            storage_base=tmp_path / "state",
            evidence_root=tmp_path / "evidence",
            build_root=tmp_path / "build",
        )
    assert built["NOESIS_APPLIANCE_RUNTIME_CONTEXT"] != "selector-json"
    context = json.loads(built["NOESIS_APPLIANCE_RUNTIME_CONTEXT"])
    assert context["contract"] == "noesis.appliance.runtime_context"
    assert context["runtime_family"] == "ds9"
    assert context["runtime_variant"] == "ds9:baseline"
    assert context["deployment_id"] == env["NOESIS_DEPLOYMENT_ID"]
    assert context["selector_sha256"] == env["NOESIS_HEALTH_SELECTOR_SHA256"]
    assert "NOESIS_DEPLOYMENT_SELECTOR_FILE" not in built
    assert "NOESIS_DS9_DOCKER_ROOT" not in built
    assert "DOCKER_HOST" not in built
    assert built["NOESIS_PGIE_PROFILE"] == "yolo26"
    assert built["NOESIS_MOSAIC_RTSP_ENABLED"] == "0"
    assert built["NOESIS_MOSAIC_WEBRTC_ENABLED"] == "1"
    assert built["NOESIS_WORLD_JOURNAL_PATH"] == env["NOESIS_WORLD_JOURNAL_PATH"]
    assert built["NOESIS_CPU_MATH_THREADS"] == "1"


def test_mv3dt_run_environment_preserves_explicit_opt_in(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    with mock.patch.dict(os.environ, env, clear=False):
        config = host.load_native_config(env)
        built = host.build_run_environment(
            config,
            session_id="sess-mv3dt",
            storage_base=tmp_path / "state",
            evidence_root=tmp_path / "evidence",
            build_root=tmp_path / "build",
            tracking_mode="mv3dt",
        )
    assert built["NOESIS_TRACKING_MODE"] == "mv3dt"
    assert built["NOESIS_PGIE_PROFILE"] == "yolo26"
    context = json.loads(built["NOESIS_APPLIANCE_RUNTIME_CONTEXT"])
    assert context["runtime_variant"] == "ds9:v3dt"


def test_run_environment_requires_health_identity(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    env.pop("NOESIS_DEPLOYMENT_ID")
    merged = {**os.environ, **env}
    merged.pop("NOESIS_DEPLOYMENT_ID", None)
    with mock.patch.dict(os.environ, merged, clear=True):
        config = host.load_native_config(env)
        with pytest.raises(host.NativeRuntimeError, match="NOESIS_DEPLOYMENT_ID"):
            host.build_run_environment(
                config,
                session_id="sess",
                storage_base=tmp_path / "state",
                evidence_root=tmp_path / "evidence",
                build_root=tmp_path / "build",
            )


def test_port_occupancy_connects_and_does_not_bind() -> None:
    with mock.patch.object(socket, "create_connection") as connect:
        connect.side_effect = OSError("closed")
        occupancy = host.port_occupancy()
    assert occupancy["probe"] == "connect"
    assert occupancy["occupied"] == []
    assert occupancy["ports"] == [6008, 8080]
    assert connect.call_count == 2
    assert connect.call_args_list[0].args[0] == ("127.0.0.1", 6008)


def test_wrong_cuda_version_is_rejected(tmp_path: Path) -> None:
    def fake_run(command):
        joined = " ".join(str(part) for part in command)
        if "--query-gpu" in joined or "nvidia-smi" in joined:
            return 0, "NVIDIA GeForce RTX 3060, 595.71.05\n"
        if "nvcc" in joined:
            return 0, "Cuda compilation tools, release 13.0, V13.0.0\n"
        if "trtexec" in joined:
            return 0, "&&&& RUNNING TensorRT.trtexec [TensorRT v101600] [b72]\n"
        if "deepstream-app" in joined:
            return 0, "DeepStreamSDK 9.1.0\nTensorRT Version: 10.16\n"
        if "gst-launch" in joined:
            return 0, "gst-launch-1.0 version 1.24.2\nGStreamer 1.24.2\n"
        return 0, ""

    with mock.patch.object(host, "_run_text", side_effect=fake_run):
        with pytest.raises(host.NativeRuntimeError, match="CUDA 13.2"):
            host.verify_platform_versions()


def test_run_refuses_occupied_canonical_ports(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    config = host.load_native_config(env)
    with mock.patch.object(
        host, "port_occupancy", return_value={"occupied": [6008], "ports": [6008, 8080]}
    ):
        with pytest.raises(host.NativeRuntimeError, match="ports are occupied"):
            host.run_native(config)


def test_run_command_executes_ds9_runtime_with_venv_python(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    with mock.patch.dict(os.environ, env, clear=False):
        config = host.load_native_config(env)
        command = host.build_run_command(
            config,
            session_id="sess",
            storage_base=tmp_path / "depth",
        )
        assert command[0] == str(config.venv_python)
        assert command[1].endswith("DS9/noesis/ds9_runtime.py")
        assert "--tracking-mode" in command
        with mock.patch.object(host, "port_occupancy", return_value={"occupied": []}):
            with mock.patch.object(host, "prepare_session_directories") as prepare:
                prepare.return_value = {
                    "storage_base": tmp_path / "depth",
                    "session_state": tmp_path / "state",
                    "evidence_root": tmp_path / "evidence",
                    "build_root": tmp_path / "build",
                }
                with mock.patch.object(os, "execve") as execve:
                    host.run_native(config)
    execve.assert_called_once()
    executed = execve.call_args.args[1]
    assert executed[1].endswith("DS9/noesis/ds9_runtime.py")
    env_arg = execve.call_args.args[2]
    assert "DOCKER_HOST" not in env_arg
    assert env_arg["NOESIS_TRACKING_MODE"] == "baseline"
    assert env_arg["NOESIS_APPLIANCE_RUNTIME_CONTEXT"]
    assert json.loads(env_arg["NOESIS_APPLIANCE_RUNTIME_CONTEXT"])["runtime_family"] == "ds9"


def test_ds9_core_does_not_shadow_the_native_venv_service_maker() -> None:
    source = (REPO_ROOT / "DS9" / "noesis" / "ds9_runtime_core.py").read_text(
        encoding="utf-8"
    )
    assert "system_site=service_maker_system_site()" in source
    assert 'f"/usr/local/lib/python{sys.version_info.major}' not in source


def test_native_dependency_constraints_accept_exact_versions(tmp_path: Path) -> None:
    constraints = _private_file(
        tmp_path / "constraints.txt",
        "starlette==0.52.1\nanyio==4.12.1\n",
    )
    completed = mock.Mock(
        returncode=0,
        stdout=json.dumps({"starlette": "0.52.1", "anyio": "4.12.1"}),
    )
    with mock.patch.object(host.subprocess, "run", return_value=completed):
        result = host.verify_native_dependency_constraints(
            Path(sys.executable), constraints
        )
    assert result["count"] == 2


def test_native_dependency_constraints_reject_version_drift(tmp_path: Path) -> None:
    constraints = _private_file(tmp_path / "constraints.txt", "starlette==0.52.1\n")
    completed = mock.Mock(
        returncode=0,
        stdout=json.dumps({"starlette": "1.6.0"}),
    )
    with mock.patch.object(host.subprocess, "run", return_value=completed):
        with pytest.raises(host.NativeRuntimeError, match="starlette expected=0.52.1"):
            host.verify_native_dependency_constraints(Path(sys.executable), constraints)


def test_secret_mode_must_remain_owner_only(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    config = host.load_native_config(env)
    world_readable = Path(env["NOESIS_CAMERA_SECRETS_FILE"])
    world_readable.chmod(0o644)
    with pytest.raises((host.NativeRuntimeError, Exception)):
        host.validate_secret_files(config.secrets)
    world_readable.chmod(0o600)


def test_artifact_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    config = host.load_native_config(env)
    realization = Path(env["NOESIS_DS9_ARTIFACT_ROOT"]) / "asset_realization.json"
    realization.write_text("{}", encoding="utf-8")
    realization.chmod(0o600)
    with pytest.raises(host.NativeRuntimeError, match="digest mismatch"):
        host.verify_artifact_realization(config)
