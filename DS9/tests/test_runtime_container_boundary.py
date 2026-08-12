from __future__ import annotations

import importlib.util
import fcntl
import hashlib
import json
import math
import os
import socket
import stat
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "run_canonical_runtime_container.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ds9_runtime_container_boundary_test_module", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime = _load_module()


def test_port_free_probe_reuses_addresses_before_binding() -> None:
    ipv4 = mock.Mock()
    ipv6 = mock.Mock()
    with mock.patch.object(socket, "socket", side_effect=[ipv4, ipv6]):
        assert runtime._port_is_free(6008) is True

    for probe in (ipv4, ipv6):
        probe.setsockopt.assert_any_call(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        probe.close.assert_called_once_with()


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.chmod(0o700)
    return path


def _private_file(path: Path, value: str) -> Path:
    _private_directory(path.parent)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    return path


def _roots(tmp_path: Path):
    docker = _private_directory(tmp_path / "docker")
    _private_directory(docker / "data")
    _private_directory(docker / "run")
    artifacts = _private_directory(tmp_path / "artifacts")
    _private_directory(artifacts / "models" / "engines")
    return runtime.HostRoots(
        docker=docker,
        artifacts=artifacts,
        runtime=tmp_path / "runtime",
    )


def _secrets(tmp_path: Path):
    secret_root = _private_directory(tmp_path / "secrets")
    return runtime.SecretFiles(
        cameras=_private_file(secret_root / "camera.json", "camera-secret-sentinel"),
        mapanything=_private_file(secret_root / "map.key", "map-secret-sentinel"),
        internal_auth=_private_file(secret_root / "auth.key", "auth-secret-sentinel"),
    )


def _bind_unix_socket(path: Path) -> socket.socket:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    return listener


def _fake_docker(bin_dir: Path) -> Path:
    executable = bin_dir / "docker"
    executable.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            log = Path(os.environ["FAKE_DOCKER_LOG"])
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(args) + "\\n")
            if args[:1] == ["--host"]:
                args = args[2:]
            scenario = os.environ.get("FAKE_DOCKER_SCENARIO", "ok")
            state = Path(os.environ["FAKE_DOCKER_STATE"])

            if args[:1] == ["info"]:
                payload = {
                    "ID": "secondary-daemon-id",
                    "DockerRootDir": os.environ["FAKE_DOCKER_DATA_ROOT"],
                    "DefaultRuntime": "nvidia" if scenario == "default_nvidia" else "runc",
                    "Runtimes": {"runc": {}, "nvidia": {}},
                }
                print(json.dumps(payload))
                raise SystemExit(0)
            if args[:2] == ["network", "ls"]:
                print(json.dumps({"Name": "host", "Driver": "host"}))
                print(json.dumps({"Name": "none", "Driver": "null"}))
                if scenario == "bridge_network":
                    print(json.dumps({"Name": "bridge", "Driver": "bridge"}))
                raise SystemExit(0)
            if args[:2] == ["image", "inspect"]:
                target = args[2]
                parent_layers = ["sha256:" + "1" * 64, "sha256:" + "2" * 64]
                runtime_layers = parent_layers + [
                    "sha256:" + "3" * 64,
                    "sha256:" + "4" * 64,
                ]
                if scenario == "rootfs_prefix_mismatch":
                    runtime_layers[0] = "sha256:" + "9" * 64
                if scenario == "rootfs_delta_mismatch":
                    runtime_layers.append("sha256:" + "5" * 64)
                is_parent = target == os.environ["FAKE_PARENT_IMAGE_ID"]
                image_id = (
                    "sha256:" + "d" * 64
                    if is_parent and scenario == "parent_inspect_id_mismatch"
                    else os.environ["FAKE_PARENT_IMAGE_ID"]
                    if is_parent
                    else
                    "sha256:" + "f" * 64
                    if scenario == "image_mismatch"
                    else os.environ["FAKE_IMAGE_ID"]
                )
                payload = {
                    "Id": image_id,
                    "Config": {"Labels": {
                        "org.opencontainers.image.base.digest": os.environ["FAKE_BASE_DIGEST"],
                        "com.noesis.engine-build.image.reference": os.environ["FAKE_PARENT_IMAGE_REF"],
                        "com.noesis.engine-build.image.id": (
                            "sha256:" + "e" * 64
                            if scenario == "parent_image_mismatch"
                            else os.environ["FAKE_PARENT_IMAGE_ID"]
                        ),
                    }},
                    "RootFS": {
                        "Type": "layers",
                        "Layers": parent_layers if is_parent else runtime_layers,
                    },
                }
                if scenario == "rootfs_missing" and not is_parent:
                    payload.pop("RootFS")
                print(json.dumps(payload))
                raise SystemExit(0)
            if args[:1] == ["ps"]:
                if scenario == "existing_runtime" and "label=com.noesis.role=ds9-runtime" in args:
                    print("deadbeefcafe noesis-ds9-runtime-old Exited")
                elif state.exists() and state.read_text(encoding="utf-8") != "removed":
                    print("a" * 12)
                raise SystemExit(0)
            if args[:1] == ["run"]:
                state.write_text("running", encoding="utf-8")
                print("a" * 64)
                raise SystemExit(0)
            if args[:1] == ["inspect"]:
                if "--format" in args:
                    print("true 0")
                else:
                    print(Path(os.environ["FAKE_INSPECT_JSON"]).read_text(encoding="utf-8"))
                raise SystemExit(0)
            if args[:1] == ["kill"]:
                state.write_text("stopped", encoding="utf-8")
                print(args[-1])
                raise SystemExit(0)
            if args[:1] == ["wait"]:
                print("7" if scenario == "bad_exit" else "0")
                raise SystemExit(0)
            if args[:1] == ["logs"]:
                print(Path(os.environ["FAKE_RUNTIME_LOG"]).read_text(encoding="utf-8"), end="")
                raise SystemExit(0)
            if args[:1] == ["rm"]:
                state.write_text("removed", encoding="utf-8")
                print(args[-1])
                raise SystemExit(0)
            print("unsupported fake docker command", args, file=sys.stderr)
            raise SystemExit(91)
            """
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _fake_environment(tmp_path: Path, roots, *, scenario: str = "ok"):
    bin_dir = _private_directory(tmp_path / "bin")
    _fake_docker(bin_dir)
    log = tmp_path / "docker-calls.jsonl"
    state = tmp_path / "docker-state"
    inspect_path = tmp_path / "inspect.json"
    runtime_log = tmp_path / "runtime.log"
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(log),
        "FAKE_DOCKER_STATE": str(state),
        "FAKE_DOCKER_DATA_ROOT": str(roots.docker_data),
        "FAKE_DOCKER_SCENARIO": scenario,
        "FAKE_IMAGE_ID": runtime.IMAGE_ID,
        "FAKE_BASE_DIGEST": runtime.BASE_DIGEST,
        "FAKE_PARENT_IMAGE_REF": runtime.PARENT_BUILD_IMAGE_REF,
        "FAKE_PARENT_IMAGE_ID": runtime.PARENT_BUILD_IMAGE_ID,
        "FAKE_INSPECT_JSON": str(inspect_path),
        "FAKE_RUNTIME_LOG": str(runtime_log),
    }


def _docker_calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _inspect_payload(roots, secrets_, session, lane=runtime.BASELINE_LANE):
    command = runtime.build_container_command(roots, secrets_, session, lane)
    env_values = []
    mounts = []
    index = 0
    while index < len(command):
        if command[index] == "--env":
            env_values.append(command[index + 1])
            index += 2
            continue
        if command[index] == "--mount":
            fields = dict(
                field.split("=", 1)
                for field in command[index + 1].split(",")
                if "=" in field
            )
            mounts.append(
                {
                    "Type": "bind",
                    "Source": fields["src"],
                    "Destination": fields["dst"],
                    "RW": "readonly" not in command[index + 1].split(","),
                }
            )
            index += 2
            continue
        index += 1
    return {
        "AppArmorProfile": runtime.EXPECTED_APPARMOR_PROFILE,
        "Image": runtime.IMAGE_ID,
        "State": {"Pid": 4242},
        "Config": {
            "User": f"{os.geteuid()}:{os.getegid()}",
            "Entrypoint": ["python3"],
            "Cmd": runtime.canonical_runtime_arguments(lane),
            "Env": env_values,
            "Labels": {
                "com.noesis.role": "ds9-runtime",
                runtime.SESSION_LABEL_KEY: session.session_id,
                runtime.LANE_LABEL_KEY: lane.name,
            },
        },
        "HostConfig": {
            "ReadonlyRootfs": True,
            "Privileged": False,
            "CapAdd": None,
            "CapDrop": ["ALL"],
            "Memory": runtime.RUNTIME_MEMORY_BYTES,
            "MemorySwap": runtime.RUNTIME_MEMORY_BYTES,
            "MemorySwappiness": None,
            "SecurityOpt": list(runtime.EXPECTED_SECURITY_OPTIONS),
            **dict(runtime.EXPECTED_SENSITIVE_HOST_DEFAULTS),
            "Devices": [],
            "Binds": None,
            "PidMode": "",
            "UsernsMode": "",
            "ReadonlyPaths": list(runtime.EXPECTED_READONLY_PATHS),
            "MaskedPaths": list(runtime.EXPECTED_MASKED_PATHS),
            "NetworkMode": "host",
            "IpcMode": "host",
            "Runtime": "nvidia",
            "DeviceRequests": [{"DeviceIDs": ["0"], "Capabilities": [["gpu"]]}],
            "Tmpfs": {
                "/tmp": "rw,exec,nosuid,nodev,size=2147483648,mode=0700",
                str(
                    runtime.CONTAINER_SECRET_ROOT
                ): "rw,noexec,nosuid,nodev,size=65536,mode=0700",
            },
        },
        "Mounts": mounts,
    }


def _snapshot(digest: str = "a" * 64):
    return runtime.CheckoutSnapshot(
        digest=digest,
        file_count=1,
        byte_count=3,
        entries={"source.py": f"file:0644:3:{digest}"},
    )


def _resource_samples(
    memory_at,
    *,
    duration_seconds: int = 300,
    interval_seconds: int = 5,
    oom_at_seconds: int | None = None,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    peak = 0
    for elapsed in range(0, duration_seconds + 1, interval_seconds):
        current = int(memory_at(elapsed))
        peak = max(peak, current)
        oom = int(oom_at_seconds is not None and elapsed >= oom_at_seconds)
        samples.append(
            {
                "elapsed_seconds": float(elapsed),
                "captured_at_utc": (
                    f"2026-07-11T06:{(elapsed % 3600) // 60:02d}:"
                    f"{elapsed % 60:02d}Z"
                ),
                "memory_current_bytes": current,
                "memory_peak_bytes": peak,
                "memory_events_low": 0,
                "memory_events_high": 0,
                "memory_events_max": 0,
                "memory_events_oom": oom,
                "memory_events_oom_kill": oom,
                "pids_current": 32,
                "gpu_compute_owner_count": 1,
                "gpu_used_memory_mib": 4096,
                "gpu_largest_process_memory_mib": 4096,
                "gpu_owner_verified": True,
            }
        )
    return samples


def _resource_binding(
    lane=runtime.V3DT_LANE,
    *,
    container_id: str = "a" * 64,
) -> dict[str, str]:
    return {
        "container_id": container_id,
        "runtime_image_id": runtime.IMAGE_ID,
        "checkout_sha256": "a" * 64,
        "realization_sha256": "b" * 64,
        "primary_engine_artifact_id": runtime.RESOURCE_SOAK_PRIMARY_ENGINE_ID[
            lane.name
        ],
        "primary_engine_sha256": "c" * 64,
        "pipeline_config": lane.pipeline_config,
        "pipeline_config_sha256": runtime._sha256_file(
            runtime.REPO_ROOT / lane.pipeline_config
        ),
        "cameras_config": lane.cameras_config,
        "cameras_config_sha256": runtime._sha256_file(
            runtime.REPO_ROOT / lane.cameras_config
        ),
    }


def _execution_plan(session, lane=runtime.BASELINE_LANE) -> dict[str, object]:
    return {
        "ready_for_explicit_run": True,
        "blockers": [],
        "runtime_lane": lane.name,
        "analytics_state": runtime.analytics_state_contract(session, lane),
    }


def _observed_runtime_identity(session, lane=runtime.BASELINE_LANE) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": runtime.RUNTIME_IDENTITY_CONTRACT,
        "contract_version": 1,
        "session_id": session.session_id,
        "runtime_lane": lane.name,
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
        "health_generated_at_us": 1_000_000,
        "observed_at_utc": "2026-07-11T00:00:01Z",
        "endpoints": dict(runtime.CANONICAL_ENDPOINTS),
    }


def _host_gpu() -> dict[str, object]:
    return {
        "index": 0,
        "name": "NVIDIA GeForce RTX 3060",
        "uuid": "GPU-test",
        "compute_capability": "8.6",
        "memory_mib": 12288,
        "driver_version": "595.71.05",
    }


def _realization_fixture(
    path: Path,
    engine_ids: frozenset[str] = runtime.CANONICAL_ENGINE_ARTIFACT_IDS,
) -> tuple[dict[str, object], dict[str, object]]:
    base = {
        "artifacts": [
            {"id": artifact_id, "kind": "tensorrt_engine"}
            for artifact_id in sorted(engine_ids)
        ]
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract": runtime.REALIZATION_CONTRACT,
        "base_manifest": {
            "path": runtime.BASE_MANIFEST_RELATIVE,
            "sha256": runtime._sha256_file(
                runtime.REPO_ROOT / runtime.BASE_MANIFEST_RELATIVE
            ),
        },
        "source_contracts": {
            "path": runtime.SOURCE_CONTRACTS_RELATIVE,
            "sha256": runtime._sha256_file(
                runtime.REPO_ROOT / runtime.SOURCE_CONTRACTS_RELATIVE
            ),
        },
        "created_at_utc": "2026-07-10T22:00:00Z",
        "updated_at_utc": "2026-07-10T22:01:00Z",
        "artifacts": {
            artifact_id: {
                "state": "staged_unverified",
                "provenance": {
                    "source_sha256": "a" * 64,
                    "output_sha256": "b" * 64,
                    "maintenance": {
                        "driver_version": "595.71.05",
                        "gpu": {
                            "name": "NVIDIA GeForce RTX 3060",
                            "uuid": "GPU-test",
                            "compute_capability": "8.6",
                            "memory_mib": 12288,
                        },
                    },
                },
            }
            for artifact_id in sorted(engine_ids)
        },
    }
    _private_file(path, json.dumps(payload))
    return base, payload


def test_readiness_profile_is_fail_closed() -> None:
    runtime.validate_readiness_profile(
        tracking_mode="baseline", profile="yolo26", size="m"
    )
    runtime.validate_readiness_profile(
        tracking_mode="v3dt", profile="yolo26_seg", size="s"
    )
    with pytest.raises(runtime.RuntimeContainerError, match="alternate-only"):
        runtime.validate_readiness_profile(
            tracking_mode="baseline", profile="yolo11_seg", size="m"
        )
    with pytest.raises(runtime.RuntimeContainerError, match="mismatch"):
        runtime.validate_readiness_profile(
            tracking_mode="baseline", profile="yolo26_seg", size="s"
        )
    with pytest.raises(runtime.RuntimeContainerError, match="mismatch"):
        runtime.validate_readiness_profile(
            tracking_mode="v3dt", profile="yolo26", size="m"
        )


def test_canonical_config_and_command_are_yolo26_detect_m() -> None:
    config = runtime.validate_canonical_config()
    assert config["pgie_profile"] == "yolo26"
    assert config["model_size"] == "m"
    assert config["source_ids"] == ["0", "1", "2"]
    assert config["profile_policy"] == {
        "baseline_readiness": {"profile": "yolo26", "size": "m"},
        "v3dt_readiness": {"profile": "yolo26_seg", "size": "s"},
        "alternate_only": ["yolo11_seg"],
    }
    args = runtime.canonical_runtime_arguments()
    assert args[args.index("--pgie-profile") + 1] == "yolo26"
    assert args[args.index("--size") + 1] == "m"
    assert "yolo11_seg" not in args


def test_v3dt_lane_is_exact_and_never_falls_back_to_baseline() -> None:
    lane = runtime.resolve_runtime_lane("v3dt")
    config = runtime.validate_canonical_config(lane)
    assert config["lane"] == "v3dt"
    assert config["pipeline"] == "DS9/config/infer_v3dt.yaml"
    assert config["cameras"] == "DS9/config/cameras_v3dt.yaml"
    assert config["pgie_profile"] == "yolo26_seg"
    assert config["model_size"] == "s"
    assert config["tracking_mode"] == "v3dt"
    assert config["source_ids"] == ["0", "1", "2"]
    assert set(config["required_engine_ids"]) == runtime.V3DT_ENGINE_ARTIFACT_IDS
    v3dt_config = yaml.safe_load(
        (runtime.REPO_ROOT / runtime.V3DT_LANE.pipeline_config).read_text(
            encoding="utf-8"
        )
    )["v3dt"]
    assert v3dt_config["world_frame"] == "backend_world_m"
    assert v3dt_config["caminfo_world_axes"] == "xzy"

    args = runtime.canonical_runtime_arguments(lane)
    assert args[args.index("--pipeline-config") + 1] == "DS9/config/infer_v3dt.yaml"
    assert args[args.index("--cameras-config") + 1] == "DS9/config/cameras_v3dt.yaml"
    assert args[args.index("--pgie-profile") + 1] == "yolo26_seg"
    assert args[args.index("--size") + 1] == "s"
    assert args[args.index("--tracking-mode") + 1] == "v3dt"
    assert "DS9/config/infer.yaml" not in args

    with pytest.raises(runtime.RuntimeContainerError, match="unsupported"):
        runtime.resolve_runtime_lane("v3dtt")
    with pytest.raises(runtime.RuntimeContainerError, match="selection drifted"):
        runtime.validate_lane_selection(
            lane,
            tracking_mode="baseline",
            profile="yolo26",
            size="m",
        )


@pytest.mark.parametrize(
    ("lane_name", "size", "engine_id"),
    (
        ("wholebody49-s", "s", "engine.wholebody49_s_masks"),
        ("wholebody49-x", "x", "engine.wholebody49_x_boxes"),
    ),
)
def test_wholebody49_lanes_use_only_reviewed_runtime_materialization(
    lane_name: str, size: str, engine_id: str
) -> None:
    lane = runtime.resolve_runtime_lane(lane_name)
    config = runtime.validate_canonical_config(lane)
    assert config["pipeline"] == "DS9/config/infer.yaml"
    assert config["pgie_profile"] == "wholebody49"
    assert config["model_size"] == size
    assert engine_id in config["required_engine_ids"]
    assert "engine.yolo26_detect_m" in config["required_engine_ids"]
    assert "artifact:parser.yolo_detect" in config["artifact_profiles"]
    assert f"artifact:{engine_id}" in config["artifact_profiles"]
    args = runtime.canonical_runtime_arguments(lane)
    assert args[args.index("--pgie-profile") + 1] == "wholebody49"
    assert args[args.index("--size") + 1] == size
    assert args[args.index("--tracking-mode") + 1] == "baseline"


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("default_nvidia", "default runtime must remain runc"),
        ("bridge_network", "network isolation drifted"),
        ("image_mismatch", "image ID drifted"),
        ("parent_image_mismatch", "parent image ID drifted"),
        ("parent_inspect_id_mismatch", "parent image ID drifted"),
        ("rootfs_missing", "RootFS type"),
        ("rootfs_prefix_mismatch", "not based on the exact engine-build image"),
        ("rootfs_delta_mismatch", "RootFS layer delta drifted"),
    ],
)
def test_fake_docker_rejects_daemon_and_image_drift(
    tmp_path: Path, scenario: str, message: str
) -> None:
    roots = _roots(tmp_path)
    listener = _bind_unix_socket(roots.docker_socket)
    env = _fake_environment(tmp_path, roots, scenario=scenario)
    try:
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(runtime.RuntimeContainerError, match=message):
                runtime.inspect_docker_state(roots, runtime.CommandRunner())
    finally:
        listener.close()


def test_fake_docker_plan_inspection_never_invokes_run(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    listener = _bind_unix_socket(roots.docker_socket)
    env = _fake_environment(tmp_path, roots)
    try:
        with mock.patch.dict(os.environ, env, clear=False):
            state = runtime.inspect_docker_state(roots, runtime.CommandRunner())
        assert state.image_id == runtime.IMAGE_ID
        calls = _docker_calls(Path(env["FAKE_DOCKER_LOG"]))
        assert not any("run" in call[2:4] for call in calls)
        assert [call[2] for call in calls] == ["info", "network", "image", "image"]
    finally:
        listener.close()


def test_full_plan_preflight_is_write_free_and_never_invokes_docker_run(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    listener = _bind_unix_socket(roots.docker_socket)
    secret_root = _private_directory(tmp_path / "plan-secrets")
    cameras = _private_file(
        secret_root / "cameras.json",
        json.dumps(
            {
                "version": 1,
                "sources": {"camera-one": "rtsp://user:pass@127.0.0.1/stream"},
            }
        ),
    )
    map_key = _private_file(secret_root / "map.key", "a" * 43)
    auth = _private_file(secret_root / "auth.key", "b" * 43 + "\n")
    env = {
        **_fake_environment(tmp_path, roots),
        "NOESIS_DS9_DOCKER_ROOT": str(roots.docker),
        "NOESIS_DS9_ARTIFACT_ROOT": str(roots.artifacts),
        "NOESIS_DS9_RUNTIME_ROOT": str(roots.runtime),
        "NOESIS_CAMERA_SECRETS_FILE": str(cameras),
        "NOESIS_MAPANYTHING_API_KEY_FILE": str(map_key),
        "NOESIS_INTERNAL_AUTH_TOKEN_FILE": str(auth),
    }
    try:
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                runtime,
                "validate_runtime_artifacts",
                return_value={"ok": True, "profile": "canonical"},
            ),
            mock.patch.object(runtime, "snapshot_checkout", return_value=_snapshot()),
            mock.patch.object(
                runtime, "existing_host_runtime_processes", return_value=[]
            ),
            mock.patch.object(runtime, "gpu_compute_owners", return_value=[]),
            mock.patch.object(runtime, "host_gpu_identity", return_value=_host_gpu()),
            mock.patch.object(runtime, "unavailable_canonical_ports", return_value=[]),
        ):
            _roots_result, _secrets_result, _session, _docker, _source, plan = (
                runtime.preflight(
                    env=os.environ,
                    session_id="write-free-plan-01",
                    runner=runtime.CommandRunner(),
                )
            )
        assert plan["ready_for_explicit_run"] is True
        assert not roots.runtime.exists()
        calls = _docker_calls(Path(env["FAKE_DOCKER_LOG"]))
        assert not any(call[2:3] == ["run"] for call in calls)
    finally:
        listener.close()


def test_fake_docker_detects_existing_runtime_owner(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    env = _fake_environment(tmp_path, roots, scenario="existing_runtime")
    with mock.patch.dict(os.environ, env, clear=False):
        owners = runtime.existing_runtime_containers(roots, runtime.CommandRunner())
    assert owners == ["deadbeefcafe noesis-ds9-runtime-old Exited"]


def test_runtime_command_exposes_no_secret_bytes_and_locks_mounts(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "secure-session-01")
    command = runtime.build_container_command(roots, secrets_, session)
    joined = "\n".join(command)
    assert "camera-secret-sentinel" not in joined
    assert "map-secret-sentinel" not in joined
    assert "auth-secret-sentinel" not in joined
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--security-opt=label=disable" in command
    assert [item.removeprefix("--security-opt=") for item in command if item.startswith("--security-opt=")] == list(
        runtime.EXPECTED_SECURITY_OPTIONS
    )
    assert "--network=host" in command
    assert "--ipc=host" in command
    assert command[command.index("--memory") + 1] == str(runtime.RUNTIME_MEMORY_BYTES)
    assert command[command.index("--memory-swap") + 1] == str(
        runtime.RUNTIME_MEMORY_BYTES
    )
    assert "--memory-swappiness" not in command
    assert command[command.index("--runtime=nvidia") + 1] == "--gpus"
    assert command[command.index("--gpus") + 1] == "device=0"
    mounts = [
        command[index + 1] for index, value in enumerate(command) if value == "--mount"
    ]
    assert any("dst=/workspace" in value and "readonly" in value for value in mounts)
    assert any(
        "dst=/opt/noesis/ds9-artifacts" in value and "readonly" in value
        for value in mounts
    )
    assert sum("dst=/run/noesis-secrets/" in value for value in mounts) == 3
    state_mount_index = next(
        index
        for index, value in enumerate(mounts)
        if f"dst={runtime.CONTAINER_STATE_ROOT}" in value
    )
    analytics_mount_index = next(
        index
        for index, value in enumerate(mounts)
        if f"dst={runtime.CONTAINER_ANALYTICS_ROOT}" in value
    )
    assert state_mount_index < analytics_mount_index
    assert (
        f"src={session.persistent_analytics},dst={runtime.CONTAINER_ANALYTICS_ROOT}"
        in mounts[analytics_mount_index]
    )
    assert "readonly" not in mounts[analytics_mount_index]
    env_values = {
        command[index + 1].split("=", 1)[0]: command[index + 1].split("=", 1)[1]
        for index, value in enumerate(command)
        if value == "--env"
    }
    assert env_values["NOESIS_ANALYTICS_CONFIG"] == str(
        runtime.CONTAINER_ANALYTICS_CONFIG
    )
    assert env_values["NOESIS_ANALYTICS_EXCLUDE_CONFIG"] == str(
        runtime.CONTAINER_ANALYTICS_EXCLUDE_CONFIG
    )
    assert env_values["NOESIS_MOSAIC_RTSP_ENABLED"] == "0"
    assert env_values["NOESIS_MOSAIC_WEBRTC_ENABLED"] == "1"
    assert runtime.CANONICAL_PORTS == (
        runtime.CANONICAL_WS_PORT,
        runtime.CANONICAL_REST_PORT,
    )


def test_analytics_state_seeds_private_files_with_production_render_bytes(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "analytics-seed-01")
    runtime.prepare_session_paths(roots, session)
    checkout_config = runtime.REPO_ROOT / runtime.ANALYTICS_SEED_RELATIVE
    checkout_exclude = runtime.REPO_ROOT / "config/config_nvdsanalytics_exclude.ini"
    checkout_before = {
        path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
        for path in (checkout_config, checkout_exclude)
    }

    result = runtime.prepare_analytics_state(session)

    state_root = session.persistent_analytics
    config_path = state_root / runtime.ANALYTICS_CONFIG_FILENAME
    exclude_path = state_root / runtime.ANALYTICS_EXCLUDE_FILENAME
    assert result["action"] == "seeded"
    assert config_path.read_bytes() == checkout_config.read_bytes()
    assert stat.S_IMODE(state_root.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(state_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(exclude_path.stat().st_mode) == 0o600

    from DS9.noesis.server import analytics_api

    expected_directory = _private_directory(tmp_path / "expected-render")
    expected_path = _private_file(expected_directory / "exclude.ini", "")
    expected_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    analytics_api._normalize_stream_keys(expected_config)
    analytics_api._persist_exclude_ini(
        expected_config["analytics"]["stages"]["exclude"],
        expected_path,
    )
    assert exclude_path.read_bytes() == expected_path.read_bytes()
    for path, before in checkout_before.items():
        assert (
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_mtime_ns,
        ) == before


def test_checkout_analytics_yaml_matches_checked_in_exclusion_ini() -> None:
    """Keep runtime reloads from moving exclusion ROIs into another image space."""
    from DS9.noesis.server import analytics_api

    config_path = runtime.REPO_ROOT / runtime.ANALYTICS_SEED_RELATIVE
    exclude_path = runtime.REPO_ROOT / "config/config_nvdsanalytics_exclude.ini"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    analytics_api._normalize_stream_keys(config)
    rendered = analytics_api._render_exclude_ini(
        config["analytics"]["stages"]["exclude"]
    )

    assert rendered.strip() == exclude_path.read_text(encoding="utf-8").strip()


def test_analytics_seed_second_write_failure_leaves_no_partial_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _roots(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "analytics-seed-fail-01")
    runtime.prepare_session_paths(roots, session)
    real_write = runtime.atomic_write_private_file
    writes = 0

    def fail_second_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise runtime.PrivatePathError("forced second analytics seed write failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(runtime, "atomic_write_private_file", fail_second_write)
    with pytest.raises(runtime.RuntimeContainerError, match="forced second"):
        runtime.prepare_analytics_state(session)

    assert not session.persistent_analytics.exists()
    assert not session.persistent_analytics.is_symlink()
    assert not any(
        path.name.startswith(runtime.ANALYTICS_SEED_DIRECTORY_PREFIX)
        for path in session.persistent_analytics.parent.iterdir()
    )

    monkeypatch.setattr(runtime, "atomic_write_private_file", real_write)
    result = runtime.prepare_analytics_state(session)
    assert result["action"] == "seeded"


def test_analytics_seed_rejects_unresolved_candidate_directory(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "analytics-seed-residue-01")
    runtime.prepare_session_paths(roots, session)
    residue = _private_directory(
        session.persistent_analytics.parent
        / f"{runtime.ANALYTICS_SEED_DIRECTORY_PREFIX}interrupted"
    )
    _private_file(residue / runtime.ANALYTICS_CONFIG_FILENAME, "partial")

    with pytest.raises(runtime.RuntimeContainerError, match="requires recovery"):
        runtime.prepare_analytics_state(session)

    assert not session.persistent_analytics.exists()


def test_analytics_state_preserves_valid_preexisting_writable_pair(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "analytics-preserve-01")
    runtime.prepare_session_paths(roots, session)
    seeded = runtime.prepare_analytics_state(session)
    assert seeded["action"] == "seeded"
    state_root = session.persistent_analytics
    config_path = state_root / runtime.ANALYTICS_CONFIG_FILENAME
    exclude_path = state_root / runtime.ANALYTICS_EXCLUDE_FILENAME

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    updated["analytics"]["stages"]["exclude"]["streams"]["1"]["roi_filtering"][
        "enable"
    ] = False
    updated_payload = yaml.safe_dump(updated, sort_keys=False).encode("utf-8")
    normalized = runtime._load_analytics_config(
        updated_payload,
        label="updated analytics test state",
    )
    updated_exclude = runtime._render_analytics_exclude_ini(
        normalized,
        private_directory=state_root,
    )
    runtime.atomic_write_private_file(
        config_path,
        updated_payload,
        label="test analytics config",
    )
    runtime.atomic_write_private_file(
        exclude_path,
        updated_exclude,
        label="test analytics exclusion config",
    )
    before = {
        path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
        for path in (config_path, exclude_path)
    }
    with pytest.raises(runtime.RuntimeContainerError, match="refusing to reuse"):
        runtime.prepare_session_paths(roots, session)

    second_session = runtime.SessionPaths.from_root(
        roots.runtime,
        "analytics-preserve-02",
    )
    runtime.prepare_session_paths(roots, second_session)
    preserved = runtime.prepare_analytics_state(second_session)

    assert preserved["action"] == "preserved"
    assert second_session.persistent_analytics == state_root
    assert not (second_session.state / runtime.ANALYTICS_STATE_DIRECTORY).exists()
    assert (
        config_path.read_bytes()
        != (runtime.REPO_ROOT / runtime.ANALYTICS_SEED_RELATIVE).read_bytes()
    )
    for path, identity in before.items():
        assert (
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_mtime_ns,
        ) == identity


def test_analytics_state_rejects_corrupt_partial_and_mismatched_state(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path / "corrupt")
    corrupt = runtime.SessionPaths.from_root(roots.runtime, "analytics-corrupt-01")
    runtime.prepare_session_paths(roots, corrupt)
    runtime.prepare_analytics_state(corrupt)
    corrupt_root = corrupt.persistent_analytics
    runtime.atomic_write_private_file(
        corrupt_root / runtime.ANALYTICS_CONFIG_FILENAME,
        b"version: [",
        label="corrupt analytics test config",
    )
    with pytest.raises(runtime.RuntimeContainerError, match="invalid YAML"):
        runtime.prepare_analytics_state(corrupt)

    partial_roots = _roots(tmp_path / "partial")
    partial = runtime.SessionPaths.from_root(
        partial_roots.runtime,
        "analytics-partial-01",
    )
    runtime.prepare_session_paths(partial_roots, partial)
    runtime.prepare_analytics_state(partial)
    partial_root = partial.persistent_analytics
    (partial_root / runtime.ANALYTICS_EXCLUDE_FILENAME).unlink()
    with pytest.raises(
        runtime.RuntimeContainerError, match="both exist or both be absent"
    ):
        runtime.prepare_analytics_state(partial)

    mismatch_roots = _roots(tmp_path / "mismatch")
    mismatch = runtime.SessionPaths.from_root(
        mismatch_roots.runtime,
        "analytics-mismatch-01",
    )
    runtime.prepare_session_paths(mismatch_roots, mismatch)
    runtime.prepare_analytics_state(mismatch)
    mismatch_root = mismatch.persistent_analytics
    runtime.atomic_write_private_file(
        mismatch_root / runtime.ANALYTICS_EXCLUDE_FILENAME,
        b"[property]\nenable = 0\n",
        label="mismatched analytics test exclusion config",
    )
    with pytest.raises(runtime.RuntimeContainerError, match="does not match YAML"):
        runtime.prepare_analytics_state(mismatch)


def test_analytics_state_rejects_canonical_stream_key_collisions() -> None:
    ambiguous = b"""\
version: 1
analytics:
  stages:
    exclude:
      streams:
        0: {roi_filtering: {enable: false}}
        '0': {roi_filtering: {enable: true}}
"""
    with pytest.raises(runtime.RuntimeContainerError, match="ambiguous stream keys"):
        runtime._load_analytics_config(
            ambiguous,
            label="ambiguous analytics test state",
        )


def test_analytics_state_rejects_symlink_escape_and_nonregular_files(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path / "directory-link")
    linked = runtime.SessionPaths.from_root(roots.runtime, "analytics-linked-01")
    runtime.prepare_session_paths(roots, linked)
    outside = _private_directory(tmp_path / "outside-analytics")
    linked.persistent_analytics.symlink_to(outside, target_is_directory=True)
    with pytest.raises(runtime.RuntimeContainerError, match="symlink"):
        runtime.prepare_analytics_state(linked)
    assert list(outside.iterdir()) == []

    file_roots = _roots(tmp_path / "file-link")
    file_linked = runtime.SessionPaths.from_root(
        file_roots.runtime,
        "analytics-file-linked-01",
    )
    runtime.prepare_session_paths(file_roots, file_linked)
    file_linked_root = _private_directory(file_linked.persistent_analytics)
    outside_config = _private_file(outside / "outside.yaml", "outside-config")
    outside_exclude = _private_file(outside / "outside.ini", "outside-exclude")
    (file_linked_root / runtime.ANALYTICS_CONFIG_FILENAME).symlink_to(outside_config)
    (file_linked_root / runtime.ANALYTICS_EXCLUDE_FILENAME).symlink_to(outside_exclude)
    with pytest.raises(runtime.RuntimeContainerError, match="symlink"):
        runtime.prepare_analytics_state(file_linked)
    assert outside_config.read_text(encoding="utf-8") == "outside-config"
    assert outside_exclude.read_text(encoding="utf-8") == "outside-exclude"

    nonregular_roots = _roots(tmp_path / "nonregular")
    nonregular = runtime.SessionPaths.from_root(
        nonregular_roots.runtime,
        "analytics-nonregular-01",
    )
    runtime.prepare_session_paths(nonregular_roots, nonregular)
    state_root = _private_directory(nonregular.persistent_analytics)
    _private_directory(state_root / runtime.ANALYTICS_CONFIG_FILENAME)
    _private_file(state_root / runtime.ANALYTICS_EXCLUDE_FILENAME, "not-used")
    with pytest.raises(runtime.RuntimeContainerError, match="regular file"):
        runtime.prepare_analytics_state(nonregular)

    mode_roots = _roots(tmp_path / "unsafe-mode")
    unsafe_mode = runtime.SessionPaths.from_root(
        mode_roots.runtime,
        "analytics-unsafe-mode-01",
    )
    runtime.prepare_session_paths(mode_roots, unsafe_mode)
    runtime.prepare_analytics_state(unsafe_mode)
    (
        unsafe_mode.persistent_analytics / runtime.ANALYTICS_CONFIG_FILENAME
    ).chmod(0o644)
    with pytest.raises(runtime.RuntimeContainerError, match="mode must be 0600"):
        runtime.prepare_analytics_state(unsafe_mode)


def test_analytics_state_rejects_unexpected_persistent_entries(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "analytics-extra-01")
    runtime.prepare_session_paths(roots, session)
    _private_directory(session.persistent_analytics)
    _private_file(session.persistent_analytics / "unreviewed.json", "{}")
    with pytest.raises(runtime.RuntimeContainerError, match="unexpected entries"):
        runtime.prepare_analytics_state(session)
    assert not (
        session.persistent_analytics / runtime.ANALYTICS_CONFIG_FILENAME
    ).exists()


@pytest.mark.parametrize(
    ("removed", "added", "expected"),
    (
        ("2", None, "missing=\\['2'\\]"),
        (None, "3", "unexpected=\\['3'\\]"),
    ),
)
def test_analytics_state_requires_exact_exclusion_source_coverage(
    removed: str | None,
    added: str | None,
    expected: str,
) -> None:
    config = yaml.safe_load(
        (runtime.REPO_ROOT / runtime.ANALYTICS_SEED_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    streams = config["analytics"]["stages"]["exclude"]["streams"]
    if removed is not None:
        streams.pop(removed)
    if added is not None:
        streams[added] = {
            "label": "unexpected",
            "roi_filtering": {"enable": False, "rois": []},
        }
    with pytest.raises(runtime.RuntimeContainerError, match=expected):
        runtime._validate_analytics_stream_coverage(
            config,
            ("0", "1", "2"),
            label="coverage test",
        )


def test_lane_source_ids_require_exact_pipeline_camera_equality() -> None:
    pipeline = {
        "sources": [
            {"source-id": 0, "uri_secret": "living-room"},
            {"source-id": 1, "uri_secret": "kitchen"},
        ]
    }
    cameras = {"cameras": {0: {"name": "living-room"}, 1: {"name": "kitchen"}}}
    assert runtime._canonical_source_ids_from_configs(
        pipeline,
        cameras,
        label="source test",
    ) == ("0", "1")

    cameras["cameras"][2] = cameras["cameras"].pop(1)
    with pytest.raises(runtime.RuntimeContainerError, match="coverage drifted"):
        runtime._canonical_source_ids_from_configs(
            pipeline,
            cameras,
            label="source test",
        )


def test_analytics_max_shape_larger_than_one_mib_survives_next_session(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    first = runtime.SessionPaths.from_root(roots.runtime, "analytics-max-01")
    runtime.prepare_session_paths(roots, first)
    runtime.prepare_analytics_state(first)

    config_path = first.persistent_analytics / runtime.ANALYTICS_CONFIG_FILENAME
    exclude_path = first.persistent_analytics / runtime.ANALYTICS_EXCLUDE_FILENAME
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    update_streams = []
    for stream_id in ("0", "1", "2"):
        rois = []
        for roi_index in range(64):
            radius = 350 - (roi_index % 8) * 5
            points = [
                [
                    960 + radius * math.cos(2 * math.pi * point_index / 128),
                    540 + radius * math.sin(2 * math.pi * point_index / 128),
                ]
                for point_index in range(128)
            ]
            rois.append(
                {
                    "id": f"R{roi_index:02d}",
                    "description": "x" * 512,
                    "points_px": points,
                }
            )
        config["analytics"]["stages"]["exclude"]["streams"][stream_id][
            "roi_filtering"
        ] = {"enable": True, "rois": rois}
        update_streams.append(
            {
                "stream_id": stream_id,
                "label": f"camera-{stream_id}",
                "enable": True,
                "rois": rois,
            }
        )

    from DS9.noesis.server import analytics_api

    analytics_api.ROIUpdateRequest(stage="exclude", streams=update_streams)

    payload = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
    assert 1024 * 1024 < len(payload) <= runtime.ANALYTICS_YAML_MAX_BYTES
    normalized = runtime._load_analytics_config(
        payload,
        label="max-shape analytics state",
        canonical_source_ids=("0", "1", "2"),
    )
    exclude_payload = runtime._render_analytics_exclude_ini(
        normalized,
        private_directory=first.persistent_analytics,
    )
    assert len(exclude_payload) <= runtime.ANALYTICS_EXCLUDE_INI_MAX_BYTES
    runtime.atomic_write_private_file(config_path, payload, label="max-shape YAML")
    runtime.atomic_write_private_file(
        exclude_path,
        exclude_payload,
        label="max-shape exclusion INI",
    )

    second = runtime.SessionPaths.from_root(roots.runtime, "analytics-max-02")
    runtime.prepare_session_paths(roots, second)
    preserved = runtime.prepare_analytics_state(second)
    assert preserved["action"] == "preserved"
    assert preserved["config_sha256"] == hashlib.sha256(payload).hexdigest()


def test_analytics_size_contract_matches_api_and_rejects_oversize_yaml() -> None:
    from DS9.noesis.server import analytics_api

    assert runtime.ANALYTICS_YAML_MAX_BYTES == analytics_api.ANALYTICS_YAML_MAX_BYTES
    assert (
        runtime.ANALYTICS_EXCLUDE_INI_MAX_BYTES
        == analytics_api.ANALYTICS_EXCLUDE_INI_MAX_BYTES
    )
    with pytest.raises(runtime.RuntimeContainerError, match="exceeds"):
        runtime._load_analytics_config(
            b"x" * (runtime.ANALYTICS_YAML_MAX_BYTES + 1),
            label="oversize analytics state",
        )


def test_plan_fails_readiness_when_canonical_artifact_gate_is_not_ready(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "artifact-session-01")
    docker = runtime.DockerState(
        daemon_id="secondary",
        docker_root=str(roots.docker_data),
        default_runtime="runc",
        image_id=runtime.IMAGE_ID,
        base_digest=runtime.BASE_DIGEST,
        parent_build_image_reference=runtime.PARENT_BUILD_IMAGE_REF,
        parent_build_image_id=runtime.PARENT_BUILD_IMAGE_ID,
        parent_rootfs_layer_count=70,
        runtime_rootfs_layer_count=72,
        parent_rootfs_sha256="d" * 64,
        runtime_rootfs_sha256="e" * 64,
        networks={"host": "host", "none": "null"},
    )
    plan = runtime.build_plan(
        roots=roots,
        secrets_=secrets_,
        session=session,
        docker=docker,
        source=_snapshot(),
        repo_provenance={
            "runtime_dockerfile": "a" * 64,
            "build_dockerfile": "b" * 64,
            "requirements_lock": "c" * 64,
        },
        artifact_readiness={"ok": False, "errors": ["engine missing"]},
        config={"pgie_profile": "yolo26", "model_size": "m"},
        runtime_containers=[],
        runtime_processes=[],
        gpu_owners=[],
        unavailable_ports=[],
        artifact_lock={"available": True, "exists": False},
    )
    assert plan["ready_for_explicit_run"] is False
    assert "DS9 baseline artifact/provenance gate is not ready" in plan["blockers"]
    assert plan["docker_run_invoked"] is False
    assert plan["analytics_state"] == runtime.analytics_state_contract(session)
    assert plan["analytics_state"]["host_paths"]["config"] == str(
        session.persistent_analytics / runtime.ANALYTICS_CONFIG_FILENAME
    )
    assert plan["analytics_state"]["canonical_source_ids"] == ["0", "1", "2"]
    authorized = plan["authorized_run_command"]
    assert (
        "NOESIS_ANALYTICS_CONFIG=" + str(runtime.CONTAINER_ANALYTICS_CONFIG)
        in authorized
    )
    assert (
        "NOESIS_ANALYTICS_EXCLUDE_CONFIG="
        + str(runtime.CONTAINER_ANALYTICS_EXCLUDE_CONFIG)
        in authorized
    )


def test_execution_rejects_missing_or_drifted_analytics_plan_contract(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "analytics-plan-01")
    with pytest.raises(runtime.RuntimeContainerError, match="missing or drifted"):
        runtime._validate_plan_analytics_state(
            {
                "analytics_state": {
                    **runtime.analytics_state_contract(session),
                    "container_environment": {},
                }
            },
            session,
        )


def test_artifact_transaction_lock_contention_is_fail_closed(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    lock_path = roots.artifacts / runtime.ARTIFACT_TRANSACTION_LOCK_FILENAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = runtime.artifact_transaction_lock_status(roots.artifacts)
        assert status["available"] is False
        with pytest.raises(
            runtime.RuntimeContainerError, match="artifact transaction owns"
        ):
            runtime.acquire_artifact_transaction_lock(roots.artifacts)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert (
        runtime.artifact_transaction_lock_status(roots.artifacts)["available"] is True
    )


def test_evidence_setup_failure_releases_artifact_transaction_lock(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "evidence-failure")
    before = _snapshot()
    with (
        mock.patch.object(
            runtime,
            "assert_immediate_run_preconditions",
            return_value=runtime.secret_identities(secrets_),
        ),
        mock.patch.object(
            runtime,
            "_write_private_json",
            side_effect=OSError("evidence write failed"),
        ),
        pytest.raises(OSError, match="evidence write failed"),
    ):
        runtime.execute(
            roots=roots,
            secrets_=secrets_,
            session=session,
            source_before=before,
            plan=_execution_plan(session),
            runner=mock.create_autospec(runtime.CommandRunner, instance=True),
            duration_seconds=0.01,
            startup_timeout_seconds=1.0,
            shutdown_timeout_seconds=35.0,
        )

    descriptor, _ = runtime.acquire_artifact_transaction_lock(roots.artifacts)
    runtime.release_artifact_transaction_lock(descriptor)


def test_realization_requires_private_file_and_exact_trust_anchors(
    tmp_path: Path,
) -> None:
    path = tmp_path / runtime.REALIZATION_FILENAME
    base, payload = _realization_fixture(path)
    realization, anchors, realization_sha256 = runtime._realization_structure(
        path, base_manifest=base
    )
    assert realization["contract"] == runtime.REALIZATION_CONTRACT
    assert realization_sha256 == runtime._sha256_file(path)
    assert anchors["base_manifest"] == payload["base_manifest"]["sha256"]
    assert anchors["source_contracts"] == payload["source_contracts"]["sha256"]

    path.chmod(0o644)
    with pytest.raises(runtime.RuntimeContainerError, match="mode must be 0600"):
        runtime._realization_structure(path, base_manifest=base)

    path.chmod(0o600)
    payload["base_manifest"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(runtime.RuntimeContainerError, match="base_manifest is stale"):
        runtime._realization_structure(path, base_manifest=base)

    payload["base_manifest"]["sha256"] = runtime._sha256_file(
        runtime.REPO_ROOT / runtime.BASE_MANIFEST_RELATIVE
    )
    payload["source_contracts"]["sha256"] = "1" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(
        runtime.RuntimeContainerError, match="source_contracts is stale"
    ):
        runtime._realization_structure(path, base_manifest=base)


def test_realization_rejects_missing_unknown_and_immutable_overrides(
    tmp_path: Path,
) -> None:
    path = tmp_path / runtime.REALIZATION_FILENAME
    base, payload = _realization_fixture(path)
    artifacts = payload["artifacts"]
    missing_id = sorted(runtime.CANONICAL_ENGINE_ARTIFACT_IDS)[0]
    missing = artifacts.pop(missing_id)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(runtime.RuntimeContainerError, match="lacks required engine"):
        runtime._realization_structure(path, base_manifest=base)

    artifacts[missing_id] = missing
    artifacts["engine.unknown"] = missing
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(runtime.RuntimeContainerError, match="unknown artifact ID"):
        runtime._realization_structure(path, base_manifest=base)

    artifacts.pop("engine.unknown")
    artifacts[missing_id]["output"] = "malicious.engine"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(runtime.RuntimeContainerError, match="keys drifted"):
        runtime._realization_structure(path, base_manifest=base)


def test_realization_missing_is_a_hard_failure(tmp_path: Path) -> None:
    with pytest.raises(runtime.RuntimeContainerError, match="is missing"):
        runtime._realization_structure(
            tmp_path / runtime.REALIZATION_FILENAME,
            base_manifest={"artifacts": []},
        )


def test_authoritative_realization_gate_is_required_and_reports_exact_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / runtime.REALIZATION_FILENAME
    _base, payload = _realization_fixture(path)
    calls: list[tuple[object, ...]] = []

    def validate(*args, **kwargs):
        calls.append((*args, kwargs))
        return {"ok": True, "profile": "canonical", "selected_count": 5}

    fake_validator = SimpleNamespace(validate_asset_realization=validate)
    with mock.patch.dict(sys.modules, {"validate_asset_manifest": fake_validator}):
        result = runtime.validate_runtime_artifacts(tmp_path, host_gpu=_host_gpu())
    assert result["ok"] is True
    assert result["realization_sha256"] == runtime._sha256_file(path)
    assert result["base_manifest_sha256"] == payload["base_manifest"]["sha256"]
    assert (
        result["engine_source_contracts_sha256"]
        == payload["source_contracts"]["sha256"]
    )
    assert result["host_compatibility"]["ok"] is True
    assert result["host_compatibility"]["host_gpu"] == _host_gpu()
    assert calls
    kwargs = calls[0][-1]
    assert kwargs == {
        "profile": "canonical",
        "check_files": True,
        "require_provenance": True,
    }


def test_v3dt_realization_gate_selects_only_the_reviewed_v3dt_profile(
    tmp_path: Path,
) -> None:
    path = tmp_path / runtime.REALIZATION_FILENAME
    _realization_fixture(path, runtime.V3DT_ENGINE_ARTIFACT_IDS)
    calls: list[dict[str, object]] = []

    def validate(*_args, **kwargs):
        calls.append(dict(kwargs))
        return {"ok": True, "profile": kwargs["profile"], "selected_count": 1}

    fake_validator = SimpleNamespace(validate_asset_realization=validate)
    with mock.patch.dict(sys.modules, {"validate_asset_manifest": fake_validator}):
        result = runtime.validate_runtime_artifacts(
            tmp_path,
            host_gpu=_host_gpu(),
            lane=runtime.V3DT_LANE,
        )
    assert result["ok"] is True
    assert result["lane"] == "v3dt"
    assert result["profiles"] == ["v3dt"]
    assert set(result["required_engine_ids"]) == runtime.V3DT_ENGINE_ARTIFACT_IDS
    assert calls == [
        {
            "profile": "v3dt",
            "check_files": True,
            "require_provenance": True,
        }
    ]


def test_immediate_prelaunch_rejects_realization_mutation(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    before = _snapshot()
    with (
        mock.patch.object(runtime, "existing_runtime_containers", return_value=[]),
        mock.patch.object(runtime, "existing_host_runtime_processes", return_value=[]),
        mock.patch.object(runtime, "gpu_compute_owners", return_value=[]),
        mock.patch.object(runtime, "host_gpu_identity", return_value=_host_gpu()),
        mock.patch.object(runtime, "unavailable_canonical_ports", return_value=[]),
        mock.patch.object(runtime, "snapshot_checkout", return_value=before),
        mock.patch.object(
            runtime,
            "validate_runtime_artifacts",
            return_value={"ok": True, "realization_sha256": "b" * 64},
        ),
        pytest.raises(runtime.RuntimeContainerError, match="changed between planning"),
    ):
        runtime.assert_immediate_run_preconditions(
            roots=roots,
            secrets_=secrets_,
            source_before=before,
            planned_artifact_readiness={
                "ok": True,
                "realization_sha256": "a" * 64,
                "host_compatibility": {"ok": True, "host_gpu": _host_gpu()},
            },
            runner=runtime.CommandRunner(),
        )


def test_host_gpu_identity_and_engine_compatibility_are_exact() -> None:
    runner = mock.create_autospec(runtime.CommandRunner, instance=True)
    runner.run.return_value = subprocess.CompletedProcess(
        [],
        0,
        stdout=("0, NVIDIA GeForce RTX 3060, GPU-test, 8.6, 12288, 595.71.05\n"),
        stderr="",
    )
    assert runtime.host_gpu_identity(runner) == _host_gpu()

    realization = {
        "artifacts": {
            artifact_id: {
                "provenance": {
                    "maintenance": {
                        "driver_version": "595.71.05",
                        "gpu": {
                            "name": "NVIDIA GeForce RTX 3060",
                            "uuid": "GPU-test",
                            "compute_capability": "8.6",
                            "memory_mib": 12288,
                        },
                    }
                }
            }
            for artifact_id in runtime.CANONICAL_ENGINE_ARTIFACT_IDS
        }
    }
    compatibility = runtime._canonical_engine_host_compatibility(
        realization,
        _host_gpu(),
    )
    assert compatibility["ok"] is True

    changed_host = {**_host_gpu(), "driver_version": "596.00.00"}
    with pytest.raises(
        runtime.RuntimeContainerError, match="differs from current GPU 0"
    ):
        runtime._canonical_engine_host_compatibility(realization, changed_host)


def test_container_inspect_rejects_writable_checkout(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "inspect-session-01")
    payload = _inspect_payload(roots, secrets_, session)
    for mount in payload["Mounts"]:
        if mount["Destination"] == "/workspace":
            mount["RW"] = True
    with pytest.raises(runtime.RuntimeContainerError, match="writability drifted"):
        runtime.validate_container_inspect(
            payload, roots=roots, secrets_=secrets_, session=session
        )


def test_container_inspect_rejects_profile_or_gpu_drift(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "inspect-session-02")
    payload = _inspect_payload(roots, secrets_, session)
    payload["Config"]["Cmd"][payload["Config"]["Cmd"].index("yolo26")] = "yolo11_seg"
    with pytest.raises(runtime.RuntimeContainerError, match="arguments drifted"):
        runtime.validate_container_inspect(
            payload, roots=roots, secrets_=secrets_, session=session
        )


def test_container_inspect_requires_exact_analytics_environment(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "analytics-inspect-01")
    payload = _inspect_payload(roots, secrets_, session)
    runtime.validate_container_inspect(
        payload,
        roots=roots,
        secrets_=secrets_,
        session=session,
    )

    analytics_mount = next(
        mount
        for mount in payload["Mounts"]
        if mount["Destination"] == str(runtime.CONTAINER_ANALYTICS_ROOT)
    )
    analytics_mount["Source"] = str(session.state / "analytics")
    with pytest.raises(runtime.RuntimeContainerError, match="mount source drifted"):
        runtime.validate_container_inspect(
            payload,
            roots=roots,
            secrets_=secrets_,
            session=session,
        )


    analytics_mount["Source"] = str(session.persistent_analytics)
    payload["Mounts"].append(dict(analytics_mount))
    with pytest.raises(runtime.RuntimeContainerError, match="ambiguous mount"):
        runtime.validate_container_inspect(
            payload,
            roots=roots,
            secrets_=secrets_,
            session=session,
        )

    payload["Mounts"].pop()

    payload["Config"]["Env"] = [
        value
        for value in payload["Config"]["Env"]
        if not value.startswith("NOESIS_ANALYTICS_EXCLUDE_CONFIG=")
    ]
    with pytest.raises(
        runtime.RuntimeContainerError,
        match="NOESIS_ANALYTICS_EXCLUDE_CONFIG",
    ):
        runtime.validate_container_inspect(
            payload,
            roots=roots,
            secrets_=secrets_,
            session=session,
        )


def test_container_inspect_rejects_ambiguous_or_nested_lease_environment(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "env-inspect-01")
    payload = _inspect_payload(roots, secrets_, session)
    payload["Config"]["Env"].append("NOESIS_TRACKING_MODE=baseline")
    with pytest.raises(runtime.RuntimeContainerError, match="duplicate key"):
        runtime.validate_container_inspect(
            payload,
            roots=roots,
            secrets_=secrets_,
            session=session,
        )

    payload = _inspect_payload(roots, secrets_, session)
    payload["Config"]["Env"].append(
        "NOESIS_STATE_RELEASE_LEASE_FILE=/tmp/forbidden.lease"
    )
    with pytest.raises(runtime.RuntimeContainerError, match="forbidden host secret"):
        runtime.validate_container_inspect(
            payload,
            roots=roots,
            secrets_=secrets_,
            session=session,
        )


def test_v3dt_container_inspect_requires_exact_lane_label_command_and_env(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "v3dt-inspect-01")
    payload = _inspect_payload(roots, secrets_, session, runtime.V3DT_LANE)
    runtime.validate_container_inspect(
        payload,
        roots=roots,
        secrets_=secrets_,
        session=session,
        lane=runtime.V3DT_LANE,
    )

    payload["Config"]["Labels"][runtime.LANE_LABEL_KEY] = "baseline"
    with pytest.raises(runtime.RuntimeContainerError, match="lane label drifted"):
        runtime.validate_container_inspect(
            payload,
            roots=roots,
            secrets_=secrets_,
            session=session,
            lane=runtime.V3DT_LANE,
        )


def test_container_inspect_rejects_memory_ceiling_drift(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "memory-session-01")
    payload = _inspect_payload(roots, secrets_, session)
    payload["HostConfig"]["MemorySwap"] = runtime.RUNTIME_MEMORY_BYTES * 2
    with pytest.raises(runtime.RuntimeContainerError, match=r"memory\+swap ceiling"):
        runtime.validate_container_inspect(
            payload, roots=roots, secrets_=secrets_, session=session
        )
    payload = _inspect_payload(roots, secrets_, session)
    payload["HostConfig"]["MemorySwappiness"] = 60
    with pytest.raises(
        runtime.RuntimeContainerError, match="memory swappiness drifted"
    ):
        runtime.validate_container_inspect(
            payload, roots=roots, secrets_=secrets_, session=session
        )
    payload = _inspect_payload(roots, secrets_, session)
    payload["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["0", "1"]
    with pytest.raises(runtime.RuntimeContainerError, match="GPU device 0"):
        runtime.validate_container_inspect(
            payload, roots=roots, secrets_=secrets_, session=session
        )


def test_container_inspect_requires_observed_docker_29_default_shape(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "docker-default-shape")
    payload = _inspect_payload(roots, secrets_, session)
    host = payload["HostConfig"]
    assert "Dns" in host and host["Dns"] is None
    assert "StorageOpt" not in host
    assert "Sysctls" not in host
    runtime.validate_container_inspect(
        payload, roots=roots, secrets_=secrets_, session=session
    )

    del host["Dns"]
    with pytest.raises(runtime.RuntimeContainerError, match="Dns"):
        runtime.validate_container_inspect(
            payload, roots=roots, secrets_=secrets_, session=session
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("SecurityOpt", ["no-new-privileges", "seccomp=unconfined"], "security-option"),
        ("Devices", [{"PathOnHost": "/dev/mem"}], "device/bind"),
        ("Binds", [], "device/bind"),
        ("PidMode", "host", "PID/user namespace"),
        ("UsernsMode", "host", "PID/user namespace"),
        ("ReadonlyPaths", [], "read-only path"),
        ("MaskedPaths", [], "masked path"),
        ("CgroupnsMode", "host", "CgroupnsMode"),
        ("OomKillDisable", True, "OomKillDisable"),
        ("Sysctls", None, "Sysctls"),
        ("Sysctls", {}, "Sysctls"),
        ("Sysctls", {"net.ipv4.ip_forward": "1"}, "Sysctls"),
        ("DeviceCgroupRules", ["c 1:3 rwm"], "DeviceCgroupRules"),
        ("CgroupParent", "system.slice", "CgroupParent"),
        ("Dns", [], "Dns"),
        ("Dns", ["8.8.8.8"], "Dns"),
        ("DnsOptions", ["use-vc"], "DnsOptions"),
        ("DnsSearch", ["example.test"], "DnsSearch"),
        ("StorageOpt", None, "StorageOpt"),
        ("StorageOpt", {}, "StorageOpt"),
        ("StorageOpt", {"size": "20G"}, "StorageOpt"),
        ("ContainerIDFile", "/tmp/container.id", "ContainerIDFile"),
        ("VolumeDriver", "local", "VolumeDriver"),
        ("OomScoreAdj", 1, "OomScoreAdj"),
        ("Isolation", "hyperv", "Isolation"),
        ("Cgroup", "custom", "Cgroup"),
        ("ShmSize", 128 * 1024 * 1024, "ShmSize"),
    ),
)
def test_container_inspect_rejects_sensitive_hostconfig_default_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "security-inspect-test")
    payload = _inspect_payload(roots, secrets_, session)
    payload["HostConfig"][field] = value
    with pytest.raises(runtime.RuntimeContainerError, match=message):
        runtime.validate_container_inspect(
            payload, roots=roots, secrets_=secrets_, session=session
        )


def test_container_inspect_rejects_apparmor_profile_drift(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "apparmor-inspect-test")
    payload = _inspect_payload(roots, secrets_, session)
    payload["AppArmorProfile"] = ""
    with pytest.raises(runtime.RuntimeContainerError, match="AppArmor"):
        runtime.validate_container_inspect(
            payload, roots=roots, secrets_=secrets_, session=session
        )


def test_supervisor_observes_authenticated_runtime_identity(tmp_path: Path) -> None:
    token_file = _private_file(tmp_path / "auth" / "token", "secret-token")
    response = SimpleNamespace(
        status=200,
        read=lambda _limit: json.dumps(
            {
                "contract": "noesis.capability.health",
                "contract_version": 1,
                "instance_id": "runtime-instance-test",
                "run_id": "runtime-run-test",
                "generated_at_us": 1_752_000_000_000_000,
            }
        ).encode("utf-8"),
    )
    connection = mock.Mock()
    connection.getresponse.return_value = response
    with (
        mock.patch.object(runtime, "load_internal_token", return_value="secret-token"),
        mock.patch.object(runtime.http.client, "HTTPConnection", return_value=connection),
        mock.patch.object(runtime, "utc_now", return_value="2026-07-11T06:00:01Z"),
    ):
        observed = runtime.observe_runtime_identity(
            token_file,
            session_id="identity-observation-test",
            runtime_lane="baseline",
        )

    connection.request.assert_called_once_with(
        "GET",
        "/api/v1/health/capabilities",
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer secret-token",
            "Connection": "close",
        },
    )
    assert observed == {
        "schema_version": 1,
        "contract": runtime.RUNTIME_IDENTITY_CONTRACT,
        "contract_version": 1,
        "session_id": "identity-observation-test",
        "runtime_lane": "baseline",
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
        "health_generated_at_us": 1_752_000_000_000_000,
        "observed_at_utc": "2026-07-11T06:00:01Z",
        "endpoints": runtime.CANONICAL_ENDPOINTS,
    }


def test_supervisor_authority_json_rejects_duplicates_and_nonfinite_numbers() -> None:
    with pytest.raises(runtime.RuntimeContainerError, match="duplicate_key"):
        runtime._strict_json_value(
            '{"instance_id":"one","instance_id":"two"}',
            label="supervisor authority fixture",
        )
    with pytest.raises(runtime.RuntimeContainerError, match="nonfinite_number"):
        runtime._strict_json_value(
            '{"generated_at_us":NaN}',
            label="supervisor authority fixture",
        )


@pytest.mark.parametrize(
    "failure_line",
    (
        "WARNING Failed to persist exclusion config: denied",
        "ERROR Failed to write analytics config to state",
        "OSError: [Errno 30] Read-only file system: '/var/lib/noesis/state/analytics/nvdsanalytics.yaml'",
        "analytics state path EROFS",
    ),
)
def test_shutdown_log_rejects_analytics_persistence_failures(
    failure_line: str,
) -> None:
    text = "\n".join(
        [failure_line, *(f"INFO {marker}" for marker in runtime.SHUTDOWN_MARKERS)]
    )
    result = runtime.validate_shutdown_log(text)
    assert result["ok"] is False
    assert result["failure_signatures"]


def test_shutdown_log_accepts_eos_callback_before_request_receipt_log() -> None:
    text = "\n".join(
        (
            f"INFO {runtime.SHUTDOWN_MARKERS[0]}",
            f"INFO {runtime.SHUTDOWN_MARKERS[2]}",
            f"INFO {runtime.SHUTDOWN_MARKERS[1]}",
            f"INFO {runtime.SHUTDOWN_MARKERS[3]}",
            f"INFO {runtime.SHUTDOWN_MARKERS[4]}",
        )
    )
    result = runtime.validate_shutdown_log(text)
    assert result["ok"] is True
    assert result["ordered"] is True


def test_shutdown_log_rejects_callback_before_request_initiation() -> None:
    text = "\n".join(
        (
            f"INFO {runtime.SHUTDOWN_MARKERS[2]}",
            f"INFO {runtime.SHUTDOWN_MARKERS[0]}",
            f"INFO {runtime.SHUTDOWN_MARKERS[1]}",
            f"INFO {runtime.SHUTDOWN_MARKERS[3]}",
            f"INFO {runtime.SHUTDOWN_MARKERS[4]}",
        )
    )
    result = runtime.validate_shutdown_log(text)
    assert result["ok"] is False
    assert result["ordered"] is False


def test_fake_docker_lifecycle_is_term_exit_zero_remove_and_seal(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "runtime-session-01")
    env = _fake_environment(tmp_path, roots)
    Path(env["FAKE_INSPECT_JSON"]).write_text(
        json.dumps([_inspect_payload(roots, secrets_, session)]), encoding="utf-8"
    )
    Path(env["FAKE_RUNTIME_LOG"]).write_text(
        "\n".join(f"INFO {marker}" for marker in runtime.SHUTDOWN_MARKERS) + "\n",
        encoding="utf-8",
    )
    plan = _execution_plan(session)
    before = _snapshot()
    runner = runtime.CommandRunner()

    def confirm_while_transaction_is_locked(
        inspect: Mapping[str, object],
        command_runner: runtime.CommandRunner,
    ) -> dict[str, object]:
        del inspect, command_runner
        lock_status = runtime.artifact_transaction_lock_status(roots.artifacts)
        assert lock_status["available"] is False
        return {
            "container_init_pid": 4242,
            "compute_owners": ["4243, python3, 1"],
        }

    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch.object(
            runtime,
            "assert_immediate_run_preconditions",
            return_value=runtime.secret_identities(secrets_),
        ),
        mock.patch.object(runtime, "_port_listening", return_value=True),
        mock.patch.object(runtime, "_port_is_free", return_value=True),
        mock.patch.object(runtime, "snapshot_checkout", return_value=before),
        mock.patch.object(
            runtime,
            "observe_runtime_identity",
            return_value=_observed_runtime_identity(session),
        ),
        mock.patch.object(
            runtime,
            "confirm_container_gpu_ownership",
            side_effect=confirm_while_transaction_is_locked,
        ),
    ):
        summary = runtime.execute(
            roots=roots,
            secrets_=secrets_,
            session=session,
            source_before=before,
            plan=plan,
            runner=runner,
            duration_seconds=0.01,
            startup_timeout_seconds=1.0,
            shutdown_timeout_seconds=35.0,
        )
    assert summary["ok"] is True
    assert summary["container"]["term_sent"] is True
    assert summary["container"]["exit_code"] == 0
    assert summary["container"]["forced_removal"] is False
    assert summary["container"]["absent_after"] is True
    assert (
        summary["artifact_transaction_lock"][
            "held_through_readiness_and_gpu_confirmation"
        ]
        is True
    )
    assert (
        runtime.artifact_transaction_lock_status(roots.artifacts)["available"] is True
    )
    calls = _docker_calls(Path(env["FAKE_DOCKER_LOG"]))
    verbs = [call[2] for call in calls]
    assert "run" in verbs
    assert "kill" in verbs
    assert "wait" in verbs
    assert "rm" in verbs
    kill_call = next(call for call in calls if call[2] == "kill")
    assert "--signal=TERM" in kill_call
    launcher = session.launcher_evidence
    assert (launcher / "runtime.log").is_file()
    assert (launcher / "container-inspect.json").is_file()
    assert (launcher / "summary.json").is_file()
    assert (launcher / "SHA256SUMS").is_file()
    for phase in ("before", "after"):
        assert (
            launcher / f"analytics-{phase}-{runtime.ANALYTICS_CONFIG_FILENAME}"
        ).is_file()
        assert (
            launcher / f"analytics-{phase}-{runtime.ANALYTICS_EXCLUDE_FILENAME}"
        ).is_file()
        metadata_path = launcher / f"analytics-state-{phase}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["canonical_source_ids"] == ["0", "1", "2"]
        assert metadata["persistence"] == "appliance"
    assert (
        summary["analytics_state"]["before"]["config_sha256"]
        == summary["analytics_state"]["after"]["config_sha256"]
    )
    with pytest.raises(runtime.RuntimeContainerError, match="immutable analytics evidence"):
        runtime.capture_analytics_state_evidence(
            session,
            runtime.BASELINE_LANE,
            phase="before",
            action="must-not-replace",
        )
    for path in launcher.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_fake_v3dt_resource_soak_is_opt_in_and_sealed(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "v3dt-soak-session")
    env = _fake_environment(tmp_path, roots)
    Path(env["FAKE_INSPECT_JSON"]).write_text(
        json.dumps([_inspect_payload(roots, secrets_, session, runtime.V3DT_LANE)]),
        encoding="utf-8",
    )
    Path(env["FAKE_RUNTIME_LOG"]).write_text(
        "\n".join(f"INFO {marker}" for marker in runtime.SHUTDOWN_MARKERS) + "\n",
        encoding="utf-8",
    )
    before = _snapshot()
    samples = _resource_samples(lambda _elapsed: 4 * 1024 * 1024 * 1024)

    def synthetic_soak(
        _roots_value,
        _runner,
        _container_name,
        _inspect,
        launcher_evidence,
        *,
        session_id,
        runtime_lane,
        runtime_instance_id,
        runtime_run_id,
        duration_seconds,
        runtime_binding,
    ):
        report = runtime.evaluate_resource_soak_samples(
            samples,
            session_id=session_id,
            runtime_lane=runtime_lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            requested_duration_seconds=duration_seconds,
            stop_reason="duration_complete",
            runtime_binding=runtime_binding,
        )
        persisted = runtime.persist_resource_soak_evidence(
            launcher_evidence,
            samples,
            report,
        )
        return "duration_complete", persisted

    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch.object(
            runtime,
            "assert_immediate_run_preconditions",
            return_value=runtime.secret_identities(secrets_),
        ),
        mock.patch.object(runtime, "_port_listening", return_value=True),
        mock.patch.object(runtime, "_port_is_free", return_value=True),
        mock.patch.object(runtime, "snapshot_checkout", return_value=before),
        mock.patch.object(
            runtime,
            "observe_runtime_identity",
            return_value=_observed_runtime_identity(session, runtime.V3DT_LANE),
        ),
        mock.patch.object(
            runtime,
            "confirm_container_gpu_ownership",
            return_value={
                "container_init_pid": 4242,
                "compute_owners": ["4243, python3, 4096"],
            },
        ),
        mock.patch.object(runtime, "run_resource_soak", side_effect=synthetic_soak),
        mock.patch.object(
            runtime,
            "build_resource_runtime_binding",
            return_value=_resource_binding(runtime.V3DT_LANE),
        ),
    ):
        summary = runtime.execute(
            roots=roots,
            secrets_=secrets_,
            session=session,
            source_before=before,
            plan=_execution_plan(session, runtime.V3DT_LANE),
            runner=runtime.CommandRunner(),
            duration_seconds=runtime.RESOURCE_SOAK_MIN_DURATION_SECONDS,
            startup_timeout_seconds=1.0,
            shutdown_timeout_seconds=35.0,
            lane=runtime.V3DT_LANE,
            resource_soak=True,
        )
    assert summary["ok"] is True
    assert summary["resource_soak"]["ok"] is True
    sums = (session.launcher_evidence / "SHA256SUMS").read_text(encoding="utf-8")
    assert runtime.RESOURCE_SOAK_SAMPLES_FILENAME in sums
    assert runtime.RESOURCE_SOAK_REPORT_FILENAME in sums


def test_container_gpu_confirmation_rejects_foreign_owner() -> None:
    inspect = {"State": {"Pid": 4242}}
    with (
        mock.patch.object(
            runtime,
            "gpu_compute_owners",
            return_value=["4343, python3, 512"],
        ),
        mock.patch.object(runtime, "_pid_descends_from", return_value=False),
        pytest.raises(runtime.RuntimeContainerError, match="foreign GPU owner"),
    ):
        runtime.confirm_container_gpu_ownership(
            inspect,
            mock.create_autospec(runtime.CommandRunner, instance=True),
            timeout_seconds=0.1,
        )


def test_v3dt_resource_soak_plateau_passes_exact_thresholds() -> None:
    base = 4 * 1024 * 1024 * 1024
    report = runtime.evaluate_resource_soak_samples(
        _resource_samples(lambda _elapsed: base),
        session_id="v3dt-resource-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=runtime.RESOURCE_SOAK_MIN_DURATION_SECONDS,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(),
    )
    assert report["ok"] is True
    assert report["failed_checks"] == []
    assert report["metrics"]["sample_count"] == 61
    assert report["metrics"]["post_warmup_slope_bytes_per_second"] == 0.0
    assert report["thresholds"] == runtime._resource_soak_thresholds("v3dt")


@pytest.mark.parametrize(
    ("lane", "expected_gpu_limit"),
    (
        (runtime.V3DT_LANE, 11_000),
        (runtime.WHOLEBODY49_S_LANE, 10_000),
        (runtime.WHOLEBODY49_X_LANE, 11_000),
    ),
)
def test_resource_soak_lane_gpu_and_pid_ceilings_are_exact(
    lane, expected_gpu_limit: int
) -> None:
    samples = _resource_samples(lambda _elapsed: 4 * 1024 * 1024 * 1024)
    for sample in samples:
        sample["gpu_used_memory_mib"] = expected_gpu_limit - 1
        sample["gpu_largest_process_memory_mib"] = expected_gpu_limit - 1
        sample["pids_current"] = 4095
    passing = runtime.evaluate_resource_soak_samples(
        samples,
        session_id="resource-lane-policy-test",
        runtime_lane=lane.name,
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(lane),
    )
    assert passing["ok"] is True
    assert (
        passing["thresholds"]["maximum_gpu_process_memory_mib_exclusive"]
        == expected_gpu_limit
    )

    samples[-1]["gpu_used_memory_mib"] = expected_gpu_limit
    gpu_failed = runtime.evaluate_resource_soak_samples(
        samples,
        session_id="resource-lane-policy-test",
        runtime_lane=lane.name,
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(lane),
    )
    assert gpu_failed["checks"]["gpu_process_memory_below_limit"] is False

    samples[-1]["gpu_used_memory_mib"] = expected_gpu_limit - 1
    samples[-1]["pids_current"] = 4096
    pids_failed = runtime.evaluate_resource_soak_samples(
        samples,
        session_id="resource-lane-policy-test",
        runtime_lane=lane.name,
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(lane),
    )
    assert pids_failed["checks"]["pids_below_limit"] is False


def test_resource_soak_rejects_unreviewed_baseline_and_binding_splice() -> None:
    samples = _resource_samples(lambda _elapsed: 4 * 1024 * 1024 * 1024)
    with pytest.raises(runtime.RuntimeContainerError, match="requires v3dt"):
        runtime.evaluate_resource_soak_samples(
            samples,
            session_id="resource-binding-test",
            runtime_lane="baseline",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            requested_duration_seconds=300,
            stop_reason="duration_complete",
            runtime_binding=_resource_binding(),
        )
    binding = _resource_binding(runtime.WHOLEBODY49_S_LANE)
    binding["primary_engine_artifact_id"] = "engine.wholebody49_x_boxes"
    spliced = runtime.evaluate_resource_soak_samples(
        samples,
        session_id="resource-binding-test",
        runtime_lane="wholebody49-s",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=binding,
    )
    assert spliced["ok"] is False
    assert spliced["checks"]["runtime_binding_complete"] is False


def test_v3dt_resource_soak_startup_ramp_then_plateau_passes() -> None:
    base = 2 * 1024 * 1024 * 1024
    ramp_per_second = 48 * 1024 * 1024
    samples = _resource_samples(
        lambda elapsed: base + min(elapsed, 60) * ramp_per_second
    )
    report = runtime.evaluate_resource_soak_samples(
        samples,
        session_id="v3dt-resource-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(),
    )
    assert report["ok"] is True
    assert report["metrics"]["post_warmup_growth_bytes"] == 0
    assert report["metrics"]["post_warmup_slope_bytes_per_second"] == 0.0


def test_v3dt_resource_soak_linear_leak_fails_slope() -> None:
    base = 4 * 1024 * 1024 * 1024
    leak_per_second = 2 * 1024 * 1024
    report = runtime.evaluate_resource_soak_samples(
        _resource_samples(lambda elapsed: base + elapsed * leak_per_second),
        session_id="v3dt-resource-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(),
    )
    assert report["ok"] is False
    assert report["checks"]["post_warmup_slope_within_limit"] is False
    assert report["metrics"]["post_warmup_slope_bytes_per_second"] == leak_per_second


def test_v3dt_resource_soak_late_jump_fails_growth() -> None:
    base = 4 * 1024 * 1024 * 1024
    jump = 600 * 1024 * 1024
    report = runtime.evaluate_resource_soak_samples(
        _resource_samples(lambda elapsed: base + (jump if elapsed >= 290 else 0)),
        session_id="v3dt-resource-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(),
    )
    assert report["ok"] is False
    assert report["checks"]["post_warmup_growth_within_limit"] is False
    assert report["metrics"]["post_warmup_growth_bytes"] == jump


def test_v3dt_resource_soak_rejects_memory_at_absolute_limit() -> None:
    report = runtime.evaluate_resource_soak_samples(
        _resource_samples(
            lambda _elapsed: runtime.RESOURCE_SOAK_MAX_MEMORY_BYTES
        ),
        session_id="v3dt-resource-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(),
    )
    assert report["ok"] is False
    assert report["checks"]["absolute_memory_below_limit"] is False


def test_v3dt_resource_soak_oom_event_fails() -> None:
    base = 4 * 1024 * 1024 * 1024
    report = runtime.evaluate_resource_soak_samples(
        _resource_samples(lambda _elapsed: base, oom_at_seconds=200),
        session_id="v3dt-resource-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(),
    )
    assert report["ok"] is False
    assert report["checks"]["oom_increment_zero"] is False
    assert report["checks"]["oom_kill_increment_zero"] is False
    assert report["metrics"]["oom_increment"] == 1
    assert report["metrics"]["oom_kill_increment"] == 1


@pytest.mark.parametrize(
    ("samples", "failed_check"),
    (
        (
            _resource_samples(
                lambda _elapsed: 4 * 1024 * 1024 * 1024,
                duration_seconds=295,
            ),
            "observed_duration_sufficient",
        ),
        (
            _resource_samples(lambda _elapsed: 4 * 1024 * 1024 * 1024)[::2],
            "sample_count_sufficient",
        ),
    ),
)
def test_v3dt_resource_soak_rejects_insufficient_duration_or_samples(
    samples: list[dict[str, object]], failed_check: str
) -> None:
    report = runtime.evaluate_resource_soak_samples(
        samples,
        session_id="v3dt-resource-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(),
    )
    assert report["ok"] is False
    assert report["checks"][failed_check] is False


def _cgroup_v2_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    container_id = "a" * 64
    init_pid = 4242
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    (cgroup_root / "cgroup.controllers").write_text(
        "cpu memory pids\n", encoding="ascii"
    )
    proc_root = tmp_path / "proc"
    membership = proc_root / str(init_pid) / "cgroup"
    membership.parent.mkdir(parents=True)
    relative = Path("system.slice") / f"docker-{container_id}.scope"
    membership.write_text(f"0::/{relative.as_posix()}\n", encoding="ascii")
    inspect = {"Id": container_id, "State": {"Pid": init_pid}}
    return inspect, cgroup_root, proc_root, cgroup_root / relative


def test_v3dt_resource_soak_reads_exact_container_cgroup_v2_files(
    tmp_path: Path,
) -> None:
    inspect, cgroup_root, proc_root, container_cgroup = _cgroup_v2_fixture(tmp_path)
    container_cgroup.mkdir(parents=True)
    (container_cgroup / "memory.current").write_text("4096\n", encoding="ascii")
    (container_cgroup / "memory.peak").write_text("8192\n", encoding="ascii")
    (container_cgroup / "memory.events").write_text(
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
        encoding="ascii",
    )
    (container_cgroup / "pids.current").write_text("32\n", encoding="ascii")
    resolved = runtime.resolve_container_cgroup_v2_path(
        inspect,
        cgroup_root=cgroup_root,
        proc_root=proc_root,
    )
    assert resolved == container_cgroup
    assert runtime.read_container_cgroup_v2_sample(resolved) == {
        "memory_current_bytes": 4096,
        "memory_peak_bytes": 8192,
        "memory_events_low": 0,
        "memory_events_high": 0,
        "memory_events_max": 0,
        "memory_events_oom": 0,
        "memory_events_oom_kill": 0,
        "pids_current": 32,
    }


def test_v3dt_resource_soak_missing_cgroup_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    inspect, cgroup_root, proc_root, _container_cgroup = _cgroup_v2_fixture(tmp_path)
    with pytest.raises(runtime.RuntimeContainerError, match="unavailable"):
        runtime.resolve_container_cgroup_v2_path(
            inspect,
            cgroup_root=cgroup_root,
            proc_root=proc_root,
        )


def test_v3dt_resource_soak_report_is_private_and_sealed(tmp_path: Path) -> None:
    launcher = _private_directory(tmp_path / "launcher")
    samples = _resource_samples(lambda _elapsed: 4 * 1024 * 1024 * 1024)
    report = runtime.evaluate_resource_soak_samples(
        samples,
        session_id="v3dt-resource-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        requested_duration_seconds=300,
        stop_reason="duration_complete",
        runtime_binding=_resource_binding(),
    )
    persisted = runtime.persist_resource_soak_evidence(launcher, samples, report)
    manifest_sha = runtime._seal_evidence(launcher)
    samples_path = launcher / runtime.RESOURCE_SOAK_SAMPLES_FILENAME
    report_path = launcher / runtime.RESOURCE_SOAK_REPORT_FILENAME
    sums_path = launcher / "SHA256SUMS"
    assert persisted["ok"] is True
    assert persisted["schema_version"] == 2
    assert persisted["contract_version"] == 2
    assert persisted["session_id"] == "v3dt-resource-test"
    assert persisted["runtime_lane"] == "v3dt"
    assert persisted["samples_evidence"]["sha256"] == runtime._sha256_file(samples_path)
    samples_payload = json.loads(samples_path.read_text(encoding="utf-8"))
    assert samples_payload["session_id"] == "v3dt-resource-test"
    assert samples_payload["runtime_lane"] == "v3dt"
    assert manifest_sha == runtime._sha256_file(sums_path)
    sums = sums_path.read_text(encoding="utf-8")
    assert runtime.RESOURCE_SOAK_SAMPLES_FILENAME in sums
    assert runtime.RESOURCE_SOAK_REPORT_FILENAME in sums
    for path in (samples_path, report_path, sums_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_fake_docker_bad_shutdown_forces_removal_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    secrets_ = _secrets(tmp_path)
    session = runtime.SessionPaths.from_root(roots.runtime, "runtime-session-02")
    env = _fake_environment(tmp_path, roots)
    Path(env["FAKE_INSPECT_JSON"]).write_text(
        json.dumps([_inspect_payload(roots, secrets_, session)]), encoding="utf-8"
    )
    Path(env["FAKE_RUNTIME_LOG"]).write_text(
        "INFO Shutdown complete\n", encoding="utf-8"
    )
    before = _snapshot()
    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch.object(
            runtime,
            "assert_immediate_run_preconditions",
            return_value=runtime.secret_identities(secrets_),
        ),
        mock.patch.object(runtime, "_port_listening", return_value=True),
        mock.patch.object(runtime, "_port_is_free", return_value=True),
        mock.patch.object(runtime, "snapshot_checkout", return_value=before),
        mock.patch.object(
            runtime,
            "observe_runtime_identity",
            return_value=_observed_runtime_identity(session),
        ),
        mock.patch.object(
            runtime,
            "confirm_container_gpu_ownership",
            return_value={
                "container_init_pid": 4242,
                "compute_owners": ["4243, python3, 1"],
            },
        ),
        pytest.raises(runtime.RuntimeContainerError, match="evidence preserved"),
    ):
        runtime.execute(
            roots=roots,
            secrets_=secrets_,
            session=session,
            source_before=before,
            plan=_execution_plan(session),
            runner=runtime.CommandRunner(),
            duration_seconds=0.01,
            startup_timeout_seconds=1.0,
            shutdown_timeout_seconds=35.0,
        )
    calls = _docker_calls(Path(env["FAKE_DOCKER_LOG"]))
    forced = [call for call in calls if call[2] == "rm" and "--force" in call]
    assert forced
    summary = json.loads(
        (session.launcher_evidence / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["ok"] is False
    assert summary["container"]["forced_removal"] is True
    assert summary["shutdown_lifecycle"]["ok"] is False


def test_checkout_snapshot_detects_tracked_and_untracked_mutation(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.py").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.py"], check=True)
    (repo / "untracked.py").write_text("untracked\n", encoding="utf-8")
    before = runtime.snapshot_checkout(repo)
    (repo / "tracked.py").write_text("after\n", encoding="utf-8")
    after = runtime.snapshot_checkout(repo)
    comparison = runtime.compare_snapshots(before, after)
    assert comparison["unchanged"] is False
    assert comparison["changed"] == ["tracked.py"]
    assert before.file_count == 2


def test_checkout_snapshot_descriptor_read_rejects_filename_swap(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / "tracked.py"
    source.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.py"], check=True)
    swapped = False

    def rename_swap(label: str, _repo: Path, _relative: Path) -> None:
        nonlocal swapped
        if label != "tracked.py" or swapped:
            return
        swapped = True
        source.rename(repo / "tracked-opened.py")
        source.write_text("replacement\n", encoding="utf-8")

    with (
        mock.patch.object(runtime, "_SNAPSHOT_OPENAT_TEST_HOOK", rename_swap),
        pytest.raises(runtime.RuntimeContainerError, match="changed|replaced"),
    ):
        runtime.snapshot_checkout(repo)
    assert swapped is True


def test_checkout_snapshot_covers_ignored_manifest_owned_runtime_binaries(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("*.so\n*.plan\n", encoding="utf-8")
    manifest = {
        "artifacts": [
            {
                "id": "native.fixture",
                "kind": "native_extension",
                "output": "DS9/native_extensions/native_fixture*.so",
            },
            {
                "id": "gst.fixture",
                "kind": "gstreamer_plugin",
                "output": "DS9/gst-plugins/libgstfixture.so",
            },
            {
                "id": "trt.fixture",
                "kind": "tensorrt_plugin",
                "output": "DS9/plugins/libtrtfixture.so",
            },
            {
                "id": "parser.fixture",
                "kind": "nvinfer_parser",
                "output": "DS9/pipelines/fixture/libparserfixture.so",
            },
            {
                "id": "engine.fixture",
                "kind": "tensorrt_engine",
                "output": "DS9/models/engines/fixture.plan",
            },
        ]
    }
    manifest_path = repo / runtime.BASE_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    binary_paths = (
        "DS9/gst-plugins/libgstfixture.so",
        "DS9/native_extensions/native_fixture.cpython-312-x86_64-linux-gnu.so",
        "DS9/pipelines/fixture/libparserfixture.so",
        "DS9/plugins/libtrtfixture.so",
    )
    for relative in (*binary_paths, "DS9/models/engines/fixture.plan"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"before\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", ".gitignore", runtime.BASE_MANIFEST_RELATIVE],
        check=True,
    )

    before = runtime.snapshot_checkout(repo)
    assert before.manifest_binary_paths == binary_paths
    assert before.summary()["manifest_binary_count"] == 4
    assert before.summary()["manifest_binary_paths"] == list(binary_paths)
    assert "DS9/models/engines/fixture.plan" not in before.entries

    for relative in (*binary_paths, "DS9/models/engines/fixture.plan"):
        (repo / relative).write_bytes(b"after\n")
    after = runtime.snapshot_checkout(repo)
    comparison = runtime.compare_snapshots(before, after)
    assert comparison["unchanged"] is False
    assert comparison["changed"] == list(binary_paths)


def test_checkout_snapshot_rejects_unsafe_manifest_binary_pattern(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    manifest_path = repo / runtime.BASE_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "artifacts": [
                    {
                        "id": "native.escape",
                        "kind": "native_extension",
                        "output": "../escape.so",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(runtime.RuntimeContainerError, match="safe DS9-relative"):
        runtime.snapshot_checkout(repo)


def test_run_mode_requires_explicit_gpu_authorization() -> None:
    assert runtime.main(["run", "--session-id", "authorization-test"]) == 2


def test_resource_soak_requires_v3dt_and_minimum_duration() -> None:
    assert (
        runtime.main(
            [
                "run",
                "--session-id",
                "resource-soak-baseline",
                "--authorize-gpu-runtime",
                "--resource-soak",
                "--duration-seconds",
                "300",
            ]
        )
        == 2
    )
    assert (
        runtime.main(
            [
                "run",
                "--session-id",
                "resource-soak-short",
                "--lane",
                "v3dt",
                "--authorize-gpu-runtime",
                "--resource-soak",
                "--duration-seconds",
                "299",
            ]
        )
        == 2
    )
