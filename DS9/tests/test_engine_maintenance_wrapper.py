from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "DS9" / "scripts" / "run_canonical_engine_maintenance.sh"
IMAGE_REF = "noesis-ds9-dev:9.1-20260812"
IMAGE_ID = "sha256:88d80ad35f12ec3a574cf2555a8242d33ac4110abdcc5f88a6cbdee40dfcf872"
BASE_DIGEST = "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)


def _process_identity(pid: int) -> dict[str, Any]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing <= 0:
        raise RuntimeError(f"malformed process identity for PID {pid}")
    fields = raw[closing + 1 :].split()
    cmdline = [
        value.decode("utf-8", errors="surrogateescape")
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    return {
        "start_time_ticks": int(fields[19]),
        "process_group_id": os.getpgid(pid),
        "cmdline": cmdline,
    }


def _pidfd_terminate_fixture_process(
    pid: object,
    expected: object,
    *,
    helper: Path,
) -> bool:
    """Signal one exact helper identity without trusting a recycled PID/PGID."""

    if not isinstance(pid, int) or pid <= 1 or not isinstance(expected, Mapping):
        return False
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        try:
            observed = _process_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            return False
        expected_identity = {
            "start_time_ticks": expected.get("start_time_ticks"),
            "process_group_id": expected.get("process_group_id"),
            "cmdline": expected.get("cmdline"),
        }
        if observed != expected_identity or str(helper) not in observed["cmdline"]:
            return False
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        return True
    finally:
        os.close(pidfd)


class WrapperFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bin = root / "bin"
        self.docker_root = root / "docker"
        self.artifact_root = root / "artifacts"
        self.docker_log = root / "docker.jsonl"
        self.python_log = root / "python.jsonl"
        self.nvidia_log = root / "nvidia.jsonl"
        self.nvidia_state = root / "nvidia-state"
        self.owner_state = root / "owner-state"
        self.state = root / "container-state"
        self.container_child = root / "container-child.pid"
        self.container_child_marker = root / "spawn-container-child"
        self.artifact_root_id = hashlib.sha256(
            str(self.artifact_root).encode("utf-8")
        ).hexdigest()
        self.bin.mkdir()
        (self.docker_root / "data").mkdir(parents=True)
        (self.artifact_root / "models" / "onnx").mkdir(parents=True)
        (self.artifact_root / "models" / "engines").mkdir()
        maintenance = self.artifact_root / "models" / "engine_maintenance"
        maintenance.mkdir(mode=0o700)
        logs = self.artifact_root / "logs"
        logs.mkdir(mode=0o700)
        lock = self.artifact_root / ".noesis-ds9-artifact-transaction.lock"
        lock.touch(mode=0o600)
        self._write_python_shim()
        self._write_container_init()
        self._write_df()
        self._write_nvidia_smi()
        self._write_docker()

    def _write_df(self) -> None:
        _write_executable(
            self.bin / "df",
            """\
            #!/usr/bin/env python3
            print("Avail")
            print(1099511627776)
            """,
        )

    def _write_container_init(self) -> None:
        _write_executable(
            self.bin / "fake-container-init",
            f"""\
            #!{sys.executable}
            import signal
            import subprocess
            import sys
            import time
            from pathlib import Path

            marker = Path(sys.argv[1])
            child_path = Path(sys.argv[2])
            fixture_root = marker.parent
            child = None
            stopping = False

            def stop(_signum, _frame):
                global stopping
                stopping = True

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            try:
                while not stopping and fixture_root.exists():
                    if marker.exists() and child is None:
                        child = subprocess.Popen(
                            [sys.executable, "-c", "import time; time.sleep(3600)"],
                        )
                        child_path.write_text(str(child.pid), encoding="utf-8")
                    time.sleep(0.01)
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=2)
            """,
        )

    def _write_python_shim(self) -> None:
        _write_executable(
            self.bin / "python3",
            f"""\
            #!{sys.executable}
            import fcntl
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            log = Path(os.environ["FAKE_PYTHON_LOG"])
            if args and Path(args[0]).name == "nvml_gpu_memory_sampler.py" and args[1:2] == ["sample"]:
                import datetime
                import signal
                import time

                def value(flag):
                    return args[args.index(flag) + 1]

                def utc_now():
                    return datetime.datetime.now(datetime.timezone.utc).isoformat(
                        timespec="microseconds"
                    ).replace("+00:00", "Z")

                evidence = Path(value("--evidence"))
                evidence.parent.mkdir(parents=True, exist_ok=True)
                evidence.parent.chmod(0o700)
                interval_ms = int(value("--interval-ms"))
                guard_mib = int(value("--guard-mib"))
                parent_pid = int(value("--parent-pid"))
                stopping = False

                def stop(_signum, _frame):
                    nonlocal_stopping[0] = True

                nonlocal_stopping = [False]
                signal.signal(signal.SIGTERM, stop)
                header = {{
                    "kind": "header",
                    "schema_version": 1,
                    "contract": "noesis.ds9.nvml_gpu_memory_guard.v1",
                    "device_index": int(value("--device-index")),
                    "expected_uuid": value("--expected-uuid"),
                    "engine": value("--engine"),
                    "transaction_id": value("--transaction-id"),
                    "prepared_transaction_sha256": value("--prepared-transaction-sha256"),
                    "artifact_root_id": value("--artifact-root-id"),
                    "container_id": value("--container-id"),
                    "guard_mib": guard_mib,
                    "guard_bytes": guard_mib * 1024 * 1024,
                    "interval_ms": interval_ms,
                    "max_gap_ms": int(value("--max-gap-ms")),
                    "parent_pid": parent_pid,
                    "parent_start_time_ticks": int(value("--parent-start-time-ticks")),
                    "sampler_pid": os.getpid(),
                    "sampler_start_time_ticks": int(Path(f"/proc/{{os.getpid()}}/stat").read_text().split()[21]),
                    "started_at_utc": utc_now(),
                }}
                count = 0
                peak = 0
                first = None
                last = None
                previous = None
                maximum_gap = 0.0
                mode = os.environ.get("FAKE_DOCKER_MODE", "plan")
                with evidence.open("x", encoding="utf-8") as handle:
                    os.chmod(evidence, 0o600)

                    def write(row, durable=False):
                        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\\n")
                        handle.flush()
                        if durable:
                            os.fsync(handle.fileno())

                    write(header, True)
                    while not nonlocal_stopping[0]:
                        parent_stat = Path(f"/proc/{{parent_pid}}/stat")
                        parent_dead = not parent_stat.exists()
                        if not parent_dead:
                            raw_parent = parent_stat.read_text(encoding="utf-8")
                            parent_dead = raw_parent[raw_parent.rfind(")") + 1:].split()[0] == "Z"
                        if parent_dead:
                            write({{
                                "kind": "footer", "state": "parent_lost",
                                "sample_count": count, "peak_mib": peak,
                                "ended_at_utc": utc_now(),
                            }}, True)
                            raise SystemExit(43)
                        now_mono = time.monotonic_ns()
                        now_utc = utc_now()
                        used_mib = 512
                        container_state = Path(os.environ["FAKE_CONTAINER_STATE"])
                        started = False
                        if container_state.exists():
                            started = json.loads(container_state.read_text()).get("init_pid") is not None
                        if started and mode in {{"memory_guard", "sampler_transient_breach"}}:
                            used_mib = 10441
                        count += 1
                        peak = max(peak, used_mib)
                        first = first or now_utc
                        last = now_utc
                        if previous is not None:
                            maximum_gap = max(maximum_gap, (now_mono - previous) / 1_000_000)
                        previous = now_mono
                        write({{
                            "kind": "sample",
                            "sequence": count,
                            "sampled_at_utc": now_utc,
                            "monotonic_ns": now_mono,
                            "total_mib": 12288,
                            "total_bytes": 12288 * 1024 * 1024,
                            "reserved_mib": 375,
                            "reserved_bytes": 375 * 1024 * 1024,
                            "used_mib": used_mib,
                            "used_bytes": used_mib * 1024 * 1024,
                        }})
                        if used_mib > guard_mib:
                            write({{
                                "kind": "footer", "state": "guard_breached",
                                "sample_count": count, "peak_mib": peak,
                                "peak_reserved_mib": 375, "breach_mib": used_mib,
                                "breach_bytes": used_mib * 1024 * 1024,
                                "breach_at_utc": now_utc,
                                "first_sample_at_utc": first,
                                "last_sample_at_utc": last,
                                "maximum_gap_ms": maximum_gap,
                                "ended_at_utc": utc_now(),
                            }}, True)
                            os.kill(parent_pid, signal.SIGUSR1)
                            raise SystemExit(42)
                        if started and mode == "sampler_death":
                            write({{
                                "kind": "footer", "state": "sampler_error",
                                "error": "fixture death", "sample_count": count,
                                "peak_mib": peak, "ended_at_utc": utc_now(),
                            }}, True)
                            raise SystemExit(45)
                        time.sleep(interval_ms / 1000)
                    write({{
                        "kind": "footer", "state": "stopped",
                        "sample_count": count, "peak_mib": peak,
                        "peak_reserved_mib": 375,
                        "first_sample_at_utc": first,
                        "last_sample_at_utc": last,
                        "maximum_gap_ms": maximum_gap,
                        "ended_at_utc": utc_now(),
                    }}, True)
                raise SystemExit(0)
            if args and Path(args[0]).name in {{
                "finalize_engine_realization.py",
                "validate_asset_manifest.py",
                "stage_canonical_sources.py",
                "reconcile_engine_provenance.py",
            }}:
                name = Path(args[0]).name
                lock_held = None
                if name in {{"finalize_engine_realization.py", "reconcile_engine_provenance.py"}}:
                    lock_path = Path(os.environ["NOESIS_DS9_ARTIFACT_ROOT"]) / ".noesis-ds9-artifact-transaction.lock"
                    with lock_path.open("a") as handle:
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            lock_held = True
                        else:
                            lock_held = False
                            fcntl.flock(handle, fcntl.LOCK_UN)
                    if not lock_held:
                        raise SystemExit("transaction operation ran without the lock")
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({{"name": name, "args": args[1:], "lock_held": lock_held}}) + "\\n")
                if name == "finalize_engine_realization.py":
                    operation = args[1]
                    mode = os.environ.get("FAKE_DOCKER_MODE", "plan")
                    artifact = Path(os.environ["NOESIS_DS9_ARTIFACT_ROOT"])
                    if operation == "recover":
                        if mode == "recovery_blocked":
                            print(json.dumps({{"status": "blocked", "blockers": ["manual"]}}))
                            raise SystemExit(3)
                        print(json.dumps({{"status": "clean", "recovered": []}}))
                        raise SystemExit(0)
                    if operation == "snapshot":
                        engine = args[args.index("--engine") + 1]
                        run_dir = artifact / "models" / "engine_finalize" / f"fake-{{engine}}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        run_dir.chmod(0o700)
                        transaction = run_dir / "transaction.json"
                        transaction.write_text(json.dumps({{"engine": engine, "state": "prepared"}}), encoding="utf-8")
                        transaction.chmod(0o600)
                        digest = __import__("hashlib").sha256(transaction.read_bytes()).hexdigest()
                        print(json.dumps({{
                            "transaction_manifest": str(transaction),
                            "transaction_sha256": digest,
                            "transaction_id": f"fake-{{engine}}",
                            "prior_realization": "missing",
                        }}))
                        raise SystemExit(0)
                    transaction = Path(args[args.index("--transaction-manifest") + 1])
                    payload = json.loads(transaction.read_text(encoding="utf-8"))
                    engine = payload["engine"]
                    outputs = {{
                        "yolo26_m": "yolo26m_b3_fp16.engine",
                        "yolo26_seg_s": "yolo26s-seg_fused_b3_fp16.engine",
                    }}
                    if operation == "rollback":
                        output_name = outputs.get(engine)
                        if output_name:
                            (artifact / "models" / "engines" / output_name).unlink(missing_ok=True)
                        print(json.dumps({{"status": "rolled_back", "transaction_manifest": str(transaction)}}))
                        raise SystemExit(0)
                    if operation == "commit":
                        if mode == "finalizer_commit_failure":
                            print("forced finalizer failure", file=sys.stderr)
                            raise SystemExit(2)
                        print(json.dumps({{
                            "status": "committed",
                            "transaction_manifest": str(transaction),
                            "transaction_sha256": "b" * 64,
                        }}))
                        raise SystemExit(0)
                raise SystemExit(0)
            os.execv({sys.executable!r}, [{sys.executable!r}, *args])
            """,
        )

    def _write_nvidia_smi(self) -> None:
        _write_executable(
            self.bin / "nvidia-smi",
            """\
            #!/usr/bin/env python3
            import json
            import os
            import sys
            import time
            from pathlib import Path

            query = " ".join(sys.argv[1:])
            mode = os.environ.get("FAKE_DOCKER_MODE", "plan")
            with open(os.environ["FAKE_NVIDIA_LOG"], "a", encoding="utf-8") as handle:
                handle.write(__import__("json").dumps(sys.argv[1:]) + "\\n")
            if "--id=0" not in sys.argv[1:]:
                print("unscoped GPU query", file=sys.stderr)
                raise SystemExit(9)
            if "index,name,uuid,compute_cap,memory.total,driver_version" in query:
                print("0, NVIDIA GeForce RTX 4070, GPU-fake-uuid, 8.9, 12288, 595.71.05")
            elif "memory.used" in query:
                print("9441" if mode == "memory_guard" else "512")
            elif "pid,process_name,used_memory" in query:
                state = os.environ["FAKE_NVIDIA_STATE"]
                try:
                    count = int(open(state, encoding="utf-8").read())
                except FileNotFoundError:
                    count = 0
                count += 1
                open(state, "w", encoding="utf-8").write(str(count))
                if mode == "preexisting_owner":
                    print("777, intruder, 256 MiB")
                elif mode == "owner_after_snapshot" and count >= 3:
                    print("778, late-intruder, 128 MiB")
            elif "query-compute-apps=pid" in query:
                count_path = Path(os.environ["FAKE_OWNER_STATE"])
                count = int(count_path.read_text() or "0") if count_path.exists() else 0
                count_path.write_text(str(count + 1), encoding="utf-8")
                container = json.loads(
                    Path(os.environ["FAKE_CONTAINER_STATE"]).read_text(encoding="utf-8")
                )
                if mode == "owner_intrusion":
                    print(container["foreign_pid"])
                elif mode == "signal_wait":
                    print(container["init_pid"])
                elif mode == "same_container_late_child":
                    marker = Path(os.environ["FAKE_CONTAINER_CHILD_MARKER"])
                    child_path = Path(os.environ["FAKE_CONTAINER_CHILD"])
                    marker.touch()
                    deadline = time.monotonic() + 2
                    while not child_path.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    if not child_path.exists():
                        print("container child did not appear", file=sys.stderr)
                        raise SystemExit(8)
                    print(child_path.read_text(encoding="utf-8").strip())
                elif mode == "vanished_at_completion":
                    # A stable NVML row whose process has already exited forces
                    # four fail-closed ancestry retries. The wrapper may accept
                    # that churn only after independently observing completion.
                    print("999999999")
            """,
        )

    def _write_docker(self) -> None:
        _write_executable(
            self.bin / "docker",
            """\
            #!/usr/bin/env python3
            import hashlib
            import json
            import os
            import signal
            import stat
            import subprocess
            import sys
            import time
            from pathlib import Path

            IMAGE_REF = "noesis-ds9-dev:9.1-20260812"
            IMAGE_ID = "sha256:88d80ad35f12ec3a574cf2555a8242d33ac4110abdcc5f88a6cbdee40dfcf872"
            BASE = "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994"
            CONTAINER_ID = "c" * 64
            args = sys.argv[1:]
            raw_args = list(args)
            log = Path(os.environ["FAKE_DOCKER_LOG"])
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(raw_args) + "\\n")
            if args[:1] == ["--host"]:
                args = args[2:]
            mode = os.environ.get("FAKE_DOCKER_MODE", "plan")
            state = Path(os.environ["FAKE_CONTAINER_STATE"])
            artifact = Path(os.environ["NOESIS_DS9_ARTIFACT_ROOT"])
            docker_root = Path(os.environ["NOESIS_DS9_DOCKER_ROOT"])

            def value_after(flag):
                return args[args.index(flag) + 1]

            def mounts_from_run():
                rows = []
                index = 0
                while index < len(args):
                    if args[index] == "--mount":
                        fields = dict(item.split("=", 1) for item in args[index + 1].split(",") if "=" in item)
                        rows.append({
                            "Type": fields.get("type"),
                            "Source": fields.get("src"),
                            "Destination": fields.get("dst"),
                            "RW": "readonly" not in args[index + 1].split(","),
                        })
                        index += 2
                    else:
                        index += 1
                return rows

            def labels_from_run():
                labels = {}
                for index, value in enumerate(args[:-1]):
                    if value == "--label" and "=" in args[index + 1]:
                        key, raw = args[index + 1].split("=", 1)
                        labels[key] = raw
                return labels

            def process_identity(pid):
                raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
                closing = raw.rfind(")")
                if closing <= 0:
                    raise RuntimeError(f"malformed process identity for PID {pid}")
                fields = raw[closing + 1:].split()
                cmdline = [
                    value.decode("utf-8", errors="surrogateescape")
                    for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\\0")
                    if value
                ]
                return {
                    "start_time_ticks": int(fields[19]),
                    "process_group_id": os.getpgid(pid),
                    "cmdline": cmdline,
                }

            def await_helper_identity(pid):
                helper = os.environ["FAKE_CONTAINER_INIT_HELPER"]
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        identity = process_identity(pid)
                    except (FileNotFoundError, ProcessLookupError):
                        time.sleep(0.01)
                        continue
                    if helper in identity["cmdline"]:
                        return identity
                    time.sleep(0.01)
                raise RuntimeError(f"fake helper identity did not stabilize: {pid}")

            def process_matches(pid, expected):
                if not isinstance(pid, int) or pid <= 1 or not isinstance(expected, dict):
                    return False
                try:
                    return process_identity(pid) == expected
                except (FileNotFoundError, ProcessLookupError):
                    return False

            def terminate_process(pid, expected):
                if not isinstance(pid, int) or pid <= 1 or not isinstance(expected, dict):
                    return False
                try:
                    pidfd = os.pidfd_open(pid)
                except ProcessLookupError:
                    return False
                try:
                    if not process_matches(pid, expected):
                        return False
                    helper = os.environ["FAKE_CONTAINER_INIT_HELPER"]
                    if helper not in expected.get("cmdline", []):
                        return False
                    signal.pidfd_send_signal(pidfd, signal.SIGTERM)
                    deadline = time.monotonic() + 2
                    while process_matches(pid, expected) and time.monotonic() < deadline:
                        time.sleep(0.01)
                    if process_matches(pid, expected):
                        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    return True
                finally:
                    os.close(pidfd)

            def write_state(payload):
                temporary = state.with_name(
                    f".{state.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
                )
                try:
                    temporary.write_text(json.dumps(payload), encoding="utf-8")
                    os.replace(temporary, state)
                finally:
                    temporary.unlink(missing_ok=True)

            if args[:1] == ["info"]:
                template = value_after("--format") if "--format" in args else ""
                if "DockerRootDir" in template:
                    print(docker_root / "data")
                elif "DefaultRuntime" in template:
                    print("runc")
                raise SystemExit(0)
            if args[:2] == ["network", "ls"]:
                raise SystemExit(0)
            if args[:1] == ["images"]:
                raise SystemExit(0)
            if args[:1] == ["ps"]:
                requested = value_after("--filter") if "--filter" in args else ""
                if mode == "foreign_name_collision" and requested.startswith("name="):
                    print("d" * 64)
                    raise SystemExit(0)
                if state.exists():
                    stored = json.loads(state.read_text(encoding="utf-8"))
                    if requested.startswith("label="):
                        key, expected = requested.removeprefix("label=").split("=", 1)
                        if stored.get("labels", {}).get(key) == expected:
                            print(CONTAINER_ID)
                    elif requested.startswith("name=") and stored.get("name") in requested:
                        print(CONTAINER_ID)
                raise SystemExit(0)
            if args[:2] == ["image", "inspect"]:
                target = args[2]
                template = value_after("--format")
                if target == IMAGE_REF:
                    count_path = state.with_suffix(".tag-count")
                    count = int(count_path.read_text() or "0") if count_path.exists() else 0
                    count_path.write_text(str(count + 1), encoding="utf-8")
                    if mode == "retag_toctou" and count:
                        print("sha256:attacker")
                    else:
                        print(IMAGE_ID)
                elif target == IMAGE_ID and "base.digest" in template:
                    print(BASE)
                elif target == IMAGE_ID and "tensorrt.version" in template:
                    print("10.16.0.72")
                elif target == IMAGE_ID and "Config.Env" in template:
                    print("CUDA_VERSION=13.2.0.046")
                else:
                    raise SystemExit(81)
                raise SystemExit(0)
            if args[:1] == ["create"]:
                env = {}
                for index, value in enumerate(args[:-1]):
                    if value == "--env" and "=" in args[index + 1]:
                        key, raw = args[index + 1].split("=", 1)
                        env[key] = raw
                write_state({
                    "id": CONTAINER_ID,
                    "name": value_after("--name"),
                    "mounts": mounts_from_run(),
                    "labels": labels_from_run(),
                    "engine": value_after("--only"),
                    "env": env,
                    "init_pid": None,
                    "init_identity": None,
                    "foreign_pid": None,
                    "foreign_identity": None,
                })
                print(CONTAINER_ID)
                raise SystemExit(0)
            if args[:1] == ["start"]:
                stored = json.loads(state.read_text(encoding="utf-8"))
                if "--detach" not in args:
                    pass
                init = subprocess.Popen(
                    [
                        os.environ["FAKE_CONTAINER_INIT_HELPER"],
                        os.environ["FAKE_CONTAINER_CHILD_MARKER"],
                        os.environ["FAKE_CONTAINER_CHILD"],
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                foreign = None
                if mode == "owner_intrusion":
                    foreign = subprocess.Popen(
                        [
                            os.environ["FAKE_CONTAINER_INIT_HELPER"],
                            str(state.with_suffix(".foreign-marker")),
                            str(state.with_suffix(".foreign-child")),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                init_identity = await_helper_identity(init.pid)
                foreign_identity = (
                    await_helper_identity(foreign.pid) if foreign is not None else None
                )
                stored.update({
                    "init_pid": init.pid,
                    "init_identity": init_identity,
                    "foreign_pid": foreign.pid if foreign is not None else None,
                    "foreign_identity": foreign_identity,
                })
                write_state(stored)
                engine = stored["engine"]
                outputs = {
                    "yolo26_m": "yolo26m_b3_fp16.engine",
                    "yolo26_seg_s": "yolo26s-seg_fused_b3_fp16.engine",
                }
                if mode in {
                    "success",
                    "same_container_late_child",
                    "vanished_at_completion",
                    "manifest_no_build",
                    "finalizer_commit_failure",
                }:
                    output = artifact / "models" / "engines" / outputs[engine]
                    output.write_bytes(b"fake-engine")
                    evidence = artifact / "models" / "engine_maintenance" / "fake-run"
                    evidence.mkdir(parents=True, exist_ok=True)
                    evidence.chmod(0o700)
                    labels = ["probe-trtexec", "load-candidate", "load-installed"]
                    if mode != "manifest_no_build":
                        labels.insert(1, "build")
                    env = stored["env"]
                    manifest = {
                        "contract": "noesis.ds9.engine_maintenance",
                        "status": "complete",
                        "engine": engine,
                        "commands": [{"label": label, "status": "passed"} for label in labels],
                        "installed": {
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                            "size_bytes": output.stat().st_size,
                        },
                        "metadata": {"platform": {
                            "image": env["NOESIS_DS9_MAINT_IMAGE"],
                            "image_id": env["NOESIS_DS9_MAINT_IMAGE_ID"],
                            "base_digest": env["NOESIS_DS9_MAINT_BASE_DIGEST"],
                            "tensorrt_version": env["NOESIS_DS9_MAINT_TRT_VERSION"],
                            "cuda_version": env["NOESIS_DS9_MAINT_CUDA_VERSION"],
                            "driver_version": env["NOESIS_DS9_MAINT_DRIVER_VERSION"],
                            "gpu_name": env["NOESIS_DS9_MAINT_GPU_NAME"],
                            "gpu_uuid": env["NOESIS_DS9_MAINT_GPU_UUID"],
                            "gpu_compute_capability": env["NOESIS_DS9_MAINT_GPU_COMPUTE_CAPABILITY"],
                            "gpu_memory_mib": env["NOESIS_DS9_MAINT_GPU_MEMORY_MIB"],
                            "expected_trtexec_banner": "TensorRT v101600",
                        }},
                    }
                    manifest_path = evidence / "manifest.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    manifest_path.chmod(0o600)
                print(CONTAINER_ID)
                raise SystemExit(0)
            if args[:1] == ["run"]:
                raise SystemExit(0)
            if args[:1] == ["inspect"]:
                if "--format" in args:
                    template = value_after("--format")
                    if not state.exists():
                        raise SystemExit(1)
                    stored = json.loads(state.read_text(encoding="utf-8"))
                    if "noesis.ds9.role" in template:
                        labels = stored.get("labels", {})
                        print(
                            "|".join(
                                (
                                    labels.get("noesis.ds9.role", ""),
                                    labels.get("noesis.ds9.transaction", ""),
                                    labels.get(
                                        "noesis.ds9.artifact-root-sha256", ""
                                    ),
                                    f"/{stored.get('name', '')}",
                                )
                            )
                        )
                    elif "State.Running" in template:
                        init_pid = stored.get("init_pid")
                        owner_count_path = Path(os.environ["FAKE_OWNER_STATE"])
                        owner_count = (
                            int(owner_count_path.read_text() or "0")
                            if owner_count_path.exists()
                            else 0
                        )
                        if "State.Pid" in template:
                            running = mode != "cleanup_still_running"
                            observed_pid = init_pid
                            if mode == "init_identity_drift":
                                observed_pid = init_pid + 1
                            elif mode == "vanished_at_completion" and owner_count >= 8:
                                running = False
                            print(f"{'true' if running else 'false'}|{observed_pid}")
                        else:
                            running = mode in {
                                "owner_intrusion",
                                "signal_wait",
                                "same_container_late_child",
                                "vanished_at_completion",
                                "memory_guard",
                            }
                            if mode == "same_container_late_child" and owner_count >= 3:
                                running = False
                            elif mode == "vanished_at_completion" and owner_count >= 8:
                                running = False
                            print("true" if running else "false")
                        if not running:
                            if terminate_process(
                                init_pid, stored.get("init_identity")
                            ):
                                stored["init_identity"] = None
                                write_state(stored)
                    elif "State.ExitCode" in template:
                        print("0")
                    elif "noesis.ds9.artifact-root-sha256" in template:
                        print(
                            stored.get("labels", {}).get(
                                "noesis.ds9.artifact-root-sha256", ""
                            )
                        )
                    raise SystemExit(0)
                if not state.exists():
                    raise SystemExit(1)
                if mode == "orphan_inspect_failure":
                    raise SystemExit(42)
                stored = json.loads(state.read_text(encoding="utf-8"))
                memory = 25769803776
                cap_drop = ["ALL"]
                security_opt = ["no-new-privileges"]
                readonly = True
                mounts = stored["mounts"]
                if mode == "inspect_mount_drift":
                    next(row for row in mounts if row["Destination"] == "/workspace")["RW"] = True
                elif mode == "inspect_resource_drift":
                    memory = 1
                elif mode == "inspect_security_drift":
                    cap_drop = []
                elif mode == "inspect_extra_security":
                    security_opt.append("seccomp=unconfined")
                elif mode == "inspect_extra_mount":
                    mounts.append({
                        "Type": "bind",
                        "Source": str(artifact / "extra"),
                        "Destination": "/unexpected",
                        "RW": False,
                    })
                row = {
                    "Id": stored.get("id", CONTAINER_ID),
                    "Name": f"/{stored.get('name', '')}",
                    "Image": IMAGE_ID,
                    "State": {
                        "Running": process_matches(
                            stored.get("init_pid"), stored.get("init_identity")
                        ),
                        "Pid": stored.get("init_pid", 0),
                        "ExitCode": 0,
                    },
                    "Config": {
                        "User": f"{os.getuid()}:{os.getgid()}",
                        "Labels": stored.get("labels", {}),
                    },
                    "HostConfig": {
                        "Init": True,
                        "ReadonlyRootfs": readonly,
                        "NetworkMode": "none",
                        "CapDrop": cap_drop,
                        "SecurityOpt": security_opt,
                        "Runtime": "nvidia",
                        "Memory": memory,
                        "MemorySwap": 25769803776,
                        "MemorySwappiness": None,
                        "PidsLimit": 512,
                        "LogConfig": {"Type": "local", "Config": {"max-file": "2", "max-size": "16m"}},
                        "DeviceRequests": [{"DeviceIDs": ["0"], "Capabilities": [["gpu"]]}],
                    },
                    "Mounts": mounts,
                }
                print(json.dumps([row]))
                raise SystemExit(0)
            if args[:1] == ["top"]:
                print("PID")
                print("1234")
                raise SystemExit(0)
            if args[:1] == ["logs"]:
                if mode in {
                    "success",
                    "same_container_late_child",
                    "vanished_at_completion",
                    "manifest_no_build",
                    "finalizer_commit_failure",
                }:
                    print("[EVIDENCE] /workspace/DS9/models/engine_maintenance/fake-run/manifest.json")
                else:
                    print("fake container log", file=sys.stderr)
                raise SystemExit(0)
            if args[:1] == ["rm"]:
                if mode == "cleanup_still_running" and state.exists():
                    raise SystemExit(2)
                if state.exists():
                    stored = json.loads(state.read_text(encoding="utf-8"))
                    terminate_process(
                        stored.get("init_pid"), stored.get("init_identity")
                    )
                    terminate_process(
                        stored.get("foreign_pid"), stored.get("foreign_identity")
                    )
                state.unlink(missing_ok=True)
                raise SystemExit(0)
            raise SystemExit(f"unsupported fake docker invocation: {args}")
            """,
        )

    def env(self, mode: str = "plan") -> dict[str, str]:
        return {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "NOESIS_DS9_DOCKER_ROOT": str(self.docker_root),
            "NOESIS_DS9_ARTIFACT_ROOT": str(self.artifact_root),
            "NOESIS_DS9_MIN_FREE_HEADROOM_BYTES": "0",
            "FAKE_DOCKER_LOG": str(self.docker_log),
            "FAKE_PYTHON_LOG": str(self.python_log),
            "FAKE_NVIDIA_LOG": str(self.nvidia_log),
            "FAKE_NVIDIA_STATE": str(self.nvidia_state),
            "FAKE_OWNER_STATE": str(self.owner_state),
            "FAKE_CONTAINER_STATE": str(self.state),
            "FAKE_CONTAINER_INIT_HELPER": str(self.bin / "fake-container-init"),
            "FAKE_CONTAINER_CHILD": str(self.container_child),
            "FAKE_CONTAINER_CHILD_MARKER": str(self.container_child_marker),
            "FAKE_DOCKER_MODE": mode,
        }

    def run(
        self,
        *arguments: str,
        mode: str = "plan",
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = self.env(mode)
        env.update(extra_env or {})
        try:
            return subprocess.run(
                [str(WRAPPER), *arguments],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
        except BaseException:
            self.cleanup_processes()
            raise

    def docker_calls(self) -> list[list[str]]:
        if not self.docker_log.exists():
            return []
        return [
            json.loads(line)
            for line in self.docker_log.read_text(encoding="utf-8").splitlines()
        ]

    def python_calls(self) -> list[dict[str, object]]:
        if not self.python_log.exists():
            return []
        return [
            json.loads(line)
            for line in self.python_log.read_text(encoding="utf-8").splitlines()
        ]

    def nvidia_calls(self) -> list[list[str]]:
        if not self.nvidia_log.exists():
            return []
        return [
            json.loads(line)
            for line in self.nvidia_log.read_text(encoding="utf-8").splitlines()
        ]

    def cleanup_processes(self) -> None:
        if not self.state.exists():
            return
        stored = json.loads(self.state.read_text(encoding="utf-8"))
        helper = self.bin / "fake-container-init"
        for key in ("init", "foreign"):
            _pidfd_terminate_fixture_process(
                stored.get(f"{key}_pid"),
                stored.get(f"{key}_identity"),
                helper=helper,
            )


class EngineMaintenanceWrapperTests(unittest.TestCase):
    def test_selection_modes_are_mutually_exclusive(self) -> None:
        result = subprocess.run(
            [str(WRAPPER), "--all", "--v3dt", "--plan"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)

    def test_plan_executes_exact_selected_dry_run_by_immutable_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            artifact_state_before = {
                path.relative_to(fixture.artifact_root).as_posix(): (
                    path.lstat().st_mode,
                    path.lstat().st_uid,
                    path.lstat().st_gid,
                    path.lstat().st_size,
                    path.lstat().st_mtime_ns,
                    path.lstat().st_ctime_ns,
                )
                for path in sorted(
                    (fixture.artifact_root, *fixture.artifact_root.rglob("*"))
                )
            }
            result = fixture.run("--only", "yolo26_m", "--plan", mode="retag_toctou")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            artifact_state_after = {
                path.relative_to(fixture.artifact_root).as_posix(): (
                    path.lstat().st_mode,
                    path.lstat().st_uid,
                    path.lstat().st_gid,
                    path.lstat().st_size,
                    path.lstat().st_mtime_ns,
                    path.lstat().st_ctime_ns,
                )
                for path in sorted(
                    (fixture.artifact_root, *fixture.artifact_root.rglob("*"))
                )
            }
            self.assertEqual(artifact_state_after, artifact_state_before)
            runs = [call for call in fixture.docker_calls() if call[:1] == ["run"]]
            self.assertEqual(len(runs), 1)
            command = runs[0]
            for token in (
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--init",
                "--memory",
                "25769803776",
                "--memory-swap",
                "--pids-limit",
                "512",
                "--log-driver",
                "local",
                "--runtime=runc",
                "NVIDIA_VISIBLE_DEVICES=void",
                IMAGE_ID,
                "--dry-run",
            ):
                self.assertIn(token, command)
            self.assertNotIn("--memory-swappiness", command)
            self.assertNotIn("--gpus", command)
            self.assertNotIn("--detach", command)
            self.assertLess(
                command.index(IMAGE_ID), command.index("DS9/scripts/rebuild_engines.py")
            )
            tag_inspects = [
                call
                for call in fixture.docker_calls()
                if len(call) > 2
                and call[:2] == ["image", "inspect"]
                and call[2] == IMAGE_REF
            ]
            self.assertEqual(len(tag_inspects), 1)
            stage = next(
                call
                for call in fixture.python_calls()
                if call["name"] == "stage_canonical_sources.py"
            )
            self.assertEqual(stage["args"], ["--verify-only", "--only", "yolo26_m"])
            recover = next(
                call
                for call in fixture.python_calls()
                if call["name"] == "finalize_engine_realization.py"
            )
            self.assertEqual(recover["args"][0], "recover")
            self.assertNotIn("--apply", recover["args"])
            self.assertTrue(fixture.nvidia_calls())
            self.assertTrue(all("--id=0" in call for call in fixture.nvidia_calls()))

    def test_grouped_plans_use_profile_scoped_source_verification_and_run_all_members(
        self,
    ) -> None:
        expected = {
            "--all": ("canonical", 5),
            "--v3dt": ("v3dt", 3),
        }
        for mode, (profile, run_count) in expected.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as raw:
                fixture = WrapperFixture(Path(raw))
                result = fixture.run(mode, "--plan")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                runs = [call for call in fixture.docker_calls() if call[:1] == ["run"]]
                self.assertEqual(len(runs), run_count)
                stage = next(
                    call
                    for call in fixture.python_calls()
                    if call["name"] == "stage_canonical_sources.py"
                )
                self.assertEqual(stage["args"], ["--verify-only", "--profile", profile])

    def test_lock_contention_fails_before_validation_or_container_launch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            lock = fixture.artifact_root / ".noesis-ds9-artifact-transaction.lock"
            lock.touch(mode=0o600)
            with lock.open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = fixture.run("--only", "yolo26_m", "--plan")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "another host DS9 engine-maintenance invocation", result.stderr
            )
            self.assertFalse(fixture.python_calls())
            self.assertFalse(
                any(call[:1] == ["run"] for call in fixture.docker_calls())
            )

    def test_unsafe_lock_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            target = fixture.root / "attacker-lock"
            target.touch()
            lock = fixture.artifact_root / ".noesis-ds9-artifact-transaction.lock"
            lock.unlink()
            lock.symlink_to(target)
            result = fixture.run("--only", "yolo26_m", "--plan")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("regular non-symlink", result.stderr)

    def test_plan_refuses_missing_paths_without_creating_them(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            lock = fixture.artifact_root / ".noesis-ds9-artifact-transaction.lock"
            maintenance = fixture.artifact_root / "models" / "engine_maintenance"
            logs = fixture.artifact_root / "logs"
            lock.unlink()
            maintenance.rmdir()
            logs.rmdir()
            result = fixture.run("--only", "yolo26_m", "--plan")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("plan mode requires the pre-existing artifact directory", result.stderr)
            self.assertFalse(maintenance.exists())
            self.assertFalse(logs.exists())
            self.assertFalse(lock.exists())

    def test_plan_refuses_unsafe_modes_without_repairing_them(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            maintenance = fixture.artifact_root / "models" / "engine_maintenance"
            logs = fixture.artifact_root / "logs"
            lock = fixture.artifact_root / ".noesis-ds9-artifact-transaction.lock"
            maintenance.chmod(0o755)
            logs.chmod(0o755)
            lock.chmod(0o644)
            result = fixture.run("--only", "yolo26_m", "--plan")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must already be owner-private mode 0700", result.stderr)
            self.assertEqual(maintenance.stat().st_mode & 0o777, 0o755)
            self.assertEqual(logs.stat().st_mode & 0o777, 0o755)
            self.assertEqual(lock.stat().st_mode & 0o777, 0o644)

    def test_docker_and_artifact_roots_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            nested = fixture.docker_root / "artifacts"
            (nested / "models" / "onnx").mkdir(parents=True)
            result = fixture.run(
                "--only",
                "yolo26_m",
                "--plan",
                extra_env={"NOESIS_DS9_ARTIFACT_ROOT": str(nested)},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be disjoint", result.stderr)

    def test_detached_launch_failure_removes_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run("--only", "yolo26_m", mode="orphan_inspect_failure")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(fixture.state.exists())
            self.assertTrue(
                any(call[:2] == ["rm", "-f"] for call in fixture.docker_calls())
            )
            self.assertIn(
                "rollback",
                [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ],
            )

    def test_inspect_rejects_mount_resource_and_security_drift(self) -> None:
        expectations = {
            "inspect_mount_drift": "mount boundary mismatch",
            "inspect_resource_drift": "memory/no-swap bound mismatch",
            "inspect_security_drift": "capability drop set is not exact",
            "inspect_extra_security": "security options are not exact",
            "inspect_extra_mount": "mount destination set is not exact",
        }
        for mode, message in expectations.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as raw:
                fixture = WrapperFixture(Path(raw))
                result = fixture.run("--only", "yolo26_m", mode=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(fixture.state.exists())
                self.assertIn(
                    "rollback",
                    [
                        call["args"][0]
                        for call in fixture.python_calls()
                        if call["name"] == "finalize_engine_realization.py"
                    ],
                )

    def test_continuous_owner_monitor_rejects_intruding_host_pid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            try:
                result = fixture.run("--only", "yolo26_m", mode="owner_intrusion")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "GPU process ownership proof rejected during yolo26_m",
                    result.stderr,
                )
                self.assertIn("escapes container init ancestry", result.stderr)
                self.assertFalse(fixture.state.exists())
                self.assertIn(
                    "rollback",
                    [
                        call["args"][0]
                        for call in fixture.python_calls()
                        if call["name"] == "finalize_engine_realization.py"
                    ],
                )

                evidence = list(
                    (fixture.artifact_root / "logs").glob(
                        "yolo26_m_fake-yolo26_m.failure.log"
                    )
                )
                self.assertEqual(len(evidence), 1)
                self.assertEqual(evidence[0].stat().st_mode & 0o777, 0o600)
                evidence_text = evidence[0].read_text(encoding="utf-8")
                self.assertIn("gpu_owner_snapshot_before=", evidence_text)
                self.assertIn("ownership_proof=", evidence_text)
                self.assertIn("escapes container init ancestry", evidence_text)

                calls = fixture.docker_calls()
                failure_log_index = next(
                    index
                    for index, call in enumerate(calls)
                    if call[:1] == ["logs"] and "--tail" in call
                )
                removal_index = next(
                    index
                    for index, call in enumerate(calls)
                    if call[:2] == ["rm", "-f"]
                )
                self.assertLess(failure_log_index, removal_index)
            finally:
                fixture.cleanup_processes()

    def test_dynamic_same_container_child_is_accepted_without_top_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            try:
                result = fixture.run(
                    "--only", "yolo26_m", mode="same_container_late_child"
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(fixture.container_child.exists())
                child_pid = int(fixture.container_child.read_text(encoding="utf-8"))
                self.assertGreater(child_pid, 1)
                self.assertFalse(
                    any(call[:1] == ["top"] for call in fixture.docker_calls())
                )
                operations = [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ]
                self.assertEqual(operations, ["recover", "snapshot", "commit"])
                logs = list((fixture.artifact_root / "logs").glob("yolo26_m_*.log"))
                self.assertEqual(len(logs), 1)
                self.assertEqual(logs[0].stat().st_mode & 0o777, 0o600)
                log_text = logs[0].read_text(encoding="utf-8")
                self.assertIn("gpu_memory_total_mib=12288", log_text)
                self.assertIn("gpu_memory_guard_mib=11000", log_text)
                self.assertIn("gpu_memory_peak_mib=512", log_text)
                sample_count = int(
                    next(
                        line.split("=", 1)[1]
                        for line in log_text.splitlines()
                        if line.startswith("gpu_memory_sample_count=")
                    )
                )
                self.assertGreater(sample_count, 0)
                self.assertIn("gpu_memory_sampler_state=stopped", log_text)
                self.assertIn("gpu_memory_evidence_sha256=", log_text)
            finally:
                fixture.cleanup_processes()

    def test_wholebody49_s_memory_guard_persists_maximum_observed_before_rollback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            try:
                result = fixture.run(
                    "--only", "wholebody49_s_masks", mode="memory_guard"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "wholebody49_s_masks exceeded the 10000 MiB GPU-memory guard "
                    "(10441 MiB maximum observed at ",
                    result.stderr,
                )
                evidence = list(
                    (fixture.artifact_root / "logs").glob(
                        "wholebody49_s_masks_fake-wholebody49_s_masks.failure.log"
                    )
                )
                self.assertEqual(len(evidence), 1)
                self.assertEqual(evidence[0].stat().st_mode & 0o777, 0o600)
                evidence_text = evidence[0].read_text(encoding="utf-8")
                self.assertIn("gpu_memory_total_mib=12288", evidence_text)
                self.assertIn("gpu_memory_guard_mib=10000", evidence_text)
                self.assertIn("gpu_memory_peak_mib=10441", evidence_text)
                self.assertIn("gpu_memory_sampler_state=guard_breached", evidence_text)
                self.assertIn("gpu_memory_breach_utc=", evidence_text)
                operations = [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ]
                self.assertEqual(operations, ["recover", "snapshot", "rollback"])
                self.assertFalse(fixture.state.exists())
            finally:
                fixture.cleanup_processes()

    def test_transient_memory_breach_signals_wrapper_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            try:
                result = fixture.run(
                    "--only",
                    "wholebody49_s_masks",
                    mode="sampler_transient_breach",
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "exceeded the 10000 MiB GPU-memory guard "
                    "(10441 MiB maximum observed at ",
                    result.stderr,
                )
                operations = [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ]
                self.assertEqual(operations, ["recover", "snapshot", "rollback"])
                self.assertFalse(fixture.state.exists())
            finally:
                fixture.cleanup_processes()

    def test_sampler_death_at_container_exit_fails_closed_and_rolls_back(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            try:
                result = fixture.run("--only", "yolo26_m", mode="sampler_death")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("state=sampler_error", result.stderr)
                self.assertTrue(
                    "NVML guard sampler exited before container completion"
                    in result.stderr
                    or "NVML guard evidence was not clean at container exit"
                    in result.stderr,
                    result.stderr,
                )
                operations = [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ]
                self.assertEqual(operations, ["recover", "snapshot", "rollback"])
                self.assertFalse(fixture.state.exists())
            finally:
                fixture.cleanup_processes()

    def test_stable_vanished_owner_is_accepted_only_after_container_completion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            try:
                result = fixture.run(
                    "--only", "yolo26_m", mode="vanished_at_completion"
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertGreaterEqual(int(fixture.owner_state.read_text()), 8)
                operations = [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ]
                self.assertEqual(operations, ["recover", "snapshot", "commit"])
            finally:
                fixture.cleanup_processes()

    def test_init_identity_drift_persists_evidence_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            try:
                result = fixture.run("--only", "yolo26_m", mode="init_identity_drift")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "container init changed after identity snapshot", result.stderr
                )
                evidence = list(
                    (fixture.artifact_root / "logs").glob(
                        "yolo26_m_fake-yolo26_m.failure.log"
                    )
                )
                self.assertEqual(len(evidence), 1)
                self.assertEqual(evidence[0].stat().st_mode & 0o777, 0o600)
                operations = [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ]
                self.assertEqual(operations, ["recover", "snapshot", "rollback"])
                self.assertFalse(fixture.state.exists())
            finally:
                fixture.cleanup_processes()

    def test_owner_appearing_after_snapshot_blocks_launch_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run("--only", "yolo26_m", mode="owner_after_snapshot")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("appeared after transaction snapshot", result.stderr)
            self.assertFalse(any("--detach" in call for call in fixture.docker_calls()))
            operations = [
                call["args"][0]
                for call in fixture.python_calls()
                if call["name"] == "finalize_engine_realization.py"
            ]
            self.assertEqual(operations, ["recover", "snapshot", "rollback"])

    def test_unproven_container_removal_skips_host_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            try:
                result = fixture.run("--only", "yolo26_m", mode="cleanup_still_running")
                self.assertEqual(result.returncode, 96)
                self.assertIn(
                    "rollback skipped because builder containment is unproven",
                    result.stderr,
                )
                self.assertTrue(fixture.state.exists())
                operations = [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ]
                self.assertEqual(operations, ["recover", "snapshot"])
            finally:
                fixture.cleanup_processes()

    def test_fixture_cleanup_refuses_a_recycled_process_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            current = _process_identity(os.getpid())
            stale = {
                **current,
                "start_time_ticks": current["start_time_ticks"] + 1,
            }
            fixture.state.write_text(
                json.dumps(
                    {
                        "init_pid": os.getpid(),
                        "init_identity": stale,
                        "foreign_pid": None,
                        "foreign_identity": None,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(signal, "pidfd_send_signal") as send_signal:
                fixture.cleanup_processes()
            send_signal.assert_not_called()

    def test_sigterm_contains_builder_then_rolls_back_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            proc = subprocess.Popen(
                [str(WRAPPER), "--only", "yolo26_m"],
                cwd=REPO_ROOT,
                env=fixture.env("signal_wait"),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            while not fixture.owner_state.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            try:
                self.assertTrue(
                    fixture.owner_state.exists(),
                    "detached fake builder did not enter ownership monitoring",
                )
                proc.terminate()
                stdout, stderr = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 143, stdout + stderr)
                self.assertFalse(fixture.state.exists())
                operations = [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ]
                self.assertEqual(operations, ["recover", "snapshot", "rollback"])
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate(timeout=5)
                fixture.cleanup_processes()

    def test_preexisting_owner_prevents_detached_launch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run("--only", "yolo26_m", mode="preexisting_owner")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("GPU compute owner(s) are still active", result.stderr)
            self.assertFalse(any("--detach" in call for call in fixture.docker_calls()))

    def test_plan_blocks_stale_maintenance_container_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            fixture.state.write_text(
                json.dumps(
                    {
                        "name": "stale-maintenance",
                        "mounts": [],
                        "labels": {"noesis.ds9.role": "engine-maintenance"},
                    }
                ),
                encoding="utf-8",
            )
            payload = json.loads(fixture.state.read_text(encoding="utf-8"))
            payload["labels"]["noesis.ds9.artifact-root-sha256"] = (
                fixture.artifact_root_id
            )
            fixture.state.write_text(json.dumps(payload), encoding="utf-8")
            result = fixture.run("--only", "yolo26_m", "--plan")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("plan mode is read-only", result.stderr)
            self.assertTrue(fixture.state.exists())
            self.assertFalse(any(call[:1] == ["rm"] for call in fixture.docker_calls()))

    def test_actual_startup_removes_stale_container_before_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            fixture.state.write_text(
                json.dumps(
                    {
                        "name": "stale-maintenance",
                        "mounts": [],
                        "labels": {"noesis.ds9.role": "engine-maintenance"},
                    }
                ),
                encoding="utf-8",
            )
            payload = json.loads(fixture.state.read_text(encoding="utf-8"))
            payload["labels"]["noesis.ds9.artifact-root-sha256"] = (
                fixture.artifact_root_id
            )
            fixture.state.write_text(json.dumps(payload), encoding="utf-8")
            result = fixture.run("--only", "yolo26_m", mode="preexisting_owner")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(fixture.state.exists())
            self.assertTrue(
                any(call[:2] == ["rm", "-f"] for call in fixture.docker_calls())
            )

    def test_foreign_artifact_root_container_is_never_removed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            fixture.state.write_text(
                json.dumps(
                    {
                        "name": "foreign-maintenance",
                        "mounts": [],
                        "labels": {
                            "noesis.ds9.role": "engine-maintenance",
                            "noesis.ds9.artifact-root-sha256": "f" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = fixture.run("--only", "yolo26_m")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("foreign-root maintenance container", result.stderr)
            self.assertTrue(fixture.state.exists())
            self.assertFalse(any(call[:1] == ["rm"] for call in fixture.docker_calls()))

    def test_foreign_same_name_collision_is_never_removed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run("--only", "yolo26_m", mode="foreign_name_collision")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("container name is already owned", result.stderr)
            self.assertFalse(any(call[:1] == ["rm"] for call in fixture.docker_calls()))
            operations = [
                call["args"][0]
                for call in fixture.python_calls()
                if call["name"] == "finalize_engine_realization.py"
            ]
            self.assertEqual(operations, ["recover"])

    def test_recovery_blocker_prevents_any_build(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run(
                "--only", "yolo26_m", "--plan", mode="recovery_blocked"
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("transaction recovery/audit failed", result.stderr)
            self.assertFalse(
                any(call[:1] == ["run"] for call in fixture.docker_calls())
            )

    def test_real_v3dt_preflights_authoritative_runtime_common_before_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run(
                "--v3dt",
                extra_env={"NOESIS_DS9_TRT_TIMEOUT_SECONDS": "0"},
            )
            self.assertNotEqual(result.returncode, 0)
            validations = [
                call["args"]
                for call in fixture.python_calls()
                if call["name"] == "validate_asset_manifest.py"
            ]
            self.assertIn(
                [
                    "--artifact-root",
                    str(fixture.artifact_root),
                    "--check-files",
                    "--profile",
                    "runtime_common",
                    "--require-provenance",
                    "--require-realization",
                ],
                validations,
            )
            self.assertFalse(any("--detach" in call for call in fixture.docker_calls()))

    def test_success_holds_lock_through_finalize_and_confirms_removal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run("--only", "yolo26_m", mode="success")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(fixture.state.exists())
            finalizer_calls = [
                call
                for call in fixture.python_calls()
                if call["name"] == "finalize_engine_realization.py"
            ]
            self.assertEqual(
                [call["args"][0] for call in finalizer_calls],
                ["recover", "snapshot", "commit"],
            )
            self.assertIn("--apply", finalizer_calls[0]["args"])
            self.assertTrue(all(call["lock_held"] is True for call in finalizer_calls))
            self.assertFalse(
                any(
                    call["name"] == "reconcile_engine_provenance.py"
                    for call in fixture.python_calls()
                )
            )
            docker_calls = fixture.docker_calls()
            create = next(call for call in docker_calls if call[:1] == ["create"])
            start = next(call for call in docker_calls if call[:1] == ["start"])
            self.assertLess(docker_calls.index(create), docker_calls.index(start))
            self.assertIn("--transaction-manifest", create)
            self.assertIn("--expected-transaction-sha256", create)
            self.assertIn("noesis.ds9.role=engine-maintenance", create)
            self.assertIn(
                f"noesis.ds9.artifact-root-sha256={fixture.artifact_root_id}", create
            )
            transaction_mount = next(
                create[index + 1]
                for index, token in enumerate(create[:-1])
                if token == "--mount" and "engine_finalize" in create[index + 1]
            )
            self.assertIn("readonly", transaction_mount)
            commit_args = finalizer_calls[-1]["args"]
            for required in (
                "--gpu-guard-evidence",
                "--expected-gpu-guard-sha256",
                "--gpu-guard-container-id",
                "--gpu-guard-wrapper-pid",
                "--gpu-guard-wrapper-start-time-ticks",
            ):
                self.assertIn(required, commit_args)

    def test_manifest_without_passed_build_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run("--only", "yolo26_m", mode="manifest_no_build")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("lacks a build command", result.stderr)
            self.assertIn(
                "rollback",
                [
                    call["args"][0]
                    for call in fixture.python_calls()
                    if call["name"] == "finalize_engine_realization.py"
                ],
            )

    def test_failed_finalizer_commit_rolls_back_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run("--only", "yolo26_m", mode="finalizer_commit_failure")
            self.assertNotEqual(result.returncode, 0)
            operations = [
                call["args"][0]
                for call in fixture.python_calls()
                if call["name"] == "finalize_engine_realization.py"
            ]
            self.assertEqual(operations, ["recover", "snapshot", "commit", "rollback"])
            self.assertFalse(
                (
                    fixture.artifact_root / "models/engines/yolo26m_b3_fp16.engine"
                ).exists()
            )

    def test_invalid_timeout_and_guard_are_rejected(self) -> None:
        for key, value in (
            ("NOESIS_DS9_TRT_TIMEOUT_SECONDS", "0"),
            ("NOESIS_DS9_GPU_GUARD_MB", "not-a-number"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as raw:
                fixture = WrapperFixture(Path(raw))
                result = fixture.run(
                    "--only", "yolo26_m", "--plan", extra_env={key: value}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("positive integer", result.stderr)

    def test_watchdog_cannot_undercut_internal_build_and_load_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = WrapperFixture(Path(raw))
            result = fixture.run(
                "--only",
                "yolo26_m",
                "--plan",
                extra_env={"NOESIS_DS9_TRT_TIMEOUT_SECONDS": "2399"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be at least 2400s", result.stderr)

    def test_grouped_real_modes_end_with_realization_validation(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('if [[ "${MODE}" == "all" || "${MODE}" == "v3dt" ]]', source)
        self.assertIn("--require-provenance", source)
        self.assertIn("--require-realization", source)
        self.assertIn("(2 * prior_bytes)", source)

    def test_wrapper_is_valid_bash(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(WRAPPER)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
