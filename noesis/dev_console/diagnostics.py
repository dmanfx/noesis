from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from noesis.ds8_preflight import REPO_ROOT, resolve_config_path
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import build_effective_config


_SENSITIVE_ENV_TOKENS = ("SECRET", "TOKEN", "PASSWORD", "PASSWD", "API_KEY", "PRIVATE_KEY")
_DEFAULT_DS8_PORT_LABELS = {6008: "WebSocket", 8080: "REST", 8554: "RTSP mosaic"}


def _pids_from_owner(owner: Mapping[str, Any]) -> List[int]:
    text = f"{owner.get('users', '')} {owner.get('raw', '')}"
    pids: List[int] = []
    raw_pids = owner.get("pids")
    if isinstance(raw_pids, list):
        for raw_pid in raw_pids:
            try:
                pid = int(raw_pid)
            except (TypeError, ValueError):
                continue
            if pid not in pids:
                pids.append(pid)
    for match in re.finditer(r"\bpid=(\d+)\b", text):
        pid = int(match.group(1))
        if pid not in pids:
            pids.append(pid)
    return pids


def _port_owner_from_ss(port: int) -> Dict[str, Any]:
    if not shutil.which("ss"):
        return {}
    try:
        proc = subprocess.run(
            ["ss", "-ltnp"],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except Exception:
        return {}
    needle = f":{int(port)} "
    for line in proc.stdout.splitlines():
        if needle not in line:
            continue
        owner: Dict[str, Any] = {"raw": line.strip()}
        if "users:" in line:
            owner["users"] = line.split("users:", 1)[1].strip()
        owner["pids"] = _pids_from_owner(owner)
        return owner
    return {}


def port_status(host: str, port: int, *, label: str = "") -> Dict[str, Any]:
    busy = False
    error = ""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            busy = sock.connect_ex((host, int(port))) == 0
    except Exception as exc:
        error = str(exc)
    return {
        "label": label or str(port),
        "host": host,
        "port": int(port),
        "busy": busy,
        "error": error,
        "owner": _port_owner_from_ss(int(port)) if busy else {},
    }


def find_free_port(host: str, preferred: int) -> int:
    for port in range(int(preferred), int(preferred) + 80):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                return port
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _path_entry(label: str, raw: Any, base_yaml_path: Path) -> Optional[Dict[str, Any]]:
    if raw in (None, ""):
        return None
    text = str(raw)
    if "://" in text:
        return None
    path = resolve_config_path(base_yaml_path, text)
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "label": label,
        "path": str(path),
        "exists": exists,
        "size_bytes": None if stat is None else int(stat.st_size),
        "mtime": None if stat is None else float(stat.st_mtime),
        "mtime_text": None if stat is None else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
    }


def _append_path(entries: List[Dict[str, Any]], seen: Set[str], label: str, raw: Any, base_yaml_path: Path) -> None:
    entry = _path_entry(label, raw, base_yaml_path)
    if not entry:
        return
    key = entry["path"]
    if key in seen:
        return
    seen.add(key)
    entries.append(entry)


def artifact_audit(spec: LaunchSpec) -> Dict[str, Any]:
    cfg = build_effective_config(spec)
    base_path = spec.pipeline_path
    entries: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    _append_path(entries, seen, "pipeline", spec.pipeline_config, base_path)
    _append_path(entries, seen, "cameras", spec.cameras_config, base_path)
    depth_registration = cfg.get("depth_registration")
    if isinstance(depth_registration, Mapping):
        _append_path(entries, seen, "depth registration", depth_registration.get("path"), base_path)
    preprocess = cfg.get("preprocess")
    if isinstance(preprocess, Mapping):
        _append_path(entries, seen, "preprocess", preprocess.get("config-file"), base_path)
    models = cfg.get("models")
    if isinstance(models, Mapping):
        for name, model in models.items():
            if not isinstance(model, Mapping):
                continue
            _append_path(entries, seen, f"{name} config", model.get("config-file-path"), base_path)
            _append_path(entries, seen, f"{name} engine", model.get("engine"), base_path)
    tracker = cfg.get("tracker")
    if isinstance(tracker, Mapping):
        _append_path(entries, seen, "tracker config", tracker.get("config-file"), base_path)
        _append_path(entries, seen, "tracker low-level library", tracker.get("ll-lib-file"), base_path)
    analytics = cfg.get("analytics")
    if isinstance(analytics, Mapping):
        _append_path(entries, seen, "analytics config", analytics.get("config-file"), base_path)
        _append_path(entries, seen, "analytics stages", analytics.get("stages_config"), base_path)
        exclude = analytics.get("exclude")
        if isinstance(exclude, Mapping):
            _append_path(entries, seen, "analytics exclude", exclude.get("config-file"), base_path)
    sources = cfg.get("sources")
    if isinstance(sources, list):
        for idx, source in enumerate(sources):
            if not isinstance(source, Mapping):
                continue
            dewarper = source.get("dewarper")
            if isinstance(dewarper, Mapping):
                _append_path(entries, seen, f"source {idx} dewarper", dewarper.get("config-file"), base_path)

    missing = [entry for entry in entries if not entry["exists"]]
    return {
        "items": entries,
        "total": len(entries),
        "missing": len(missing),
        "ready": not missing,
    }


def gpu_snapshot() -> Dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "error": "nvidia-smi not found", "gpus": []}
    query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.5,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip() or proc.stdout.strip(), "gpus": []}
    gpus: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "utilization_gpu_pct": parts[2],
                "memory_used_mib": parts[3],
                "memory_total_mib": parts[4],
                "temperature_c": parts[5],
                "power_draw_w": parts[6],
            }
        )
    return {"available": bool(gpus), "error": "", "gpus": gpus}


def _safe_read_bytes(path: Path, *, max_bytes: int = 512 * 1024) -> bytes:
    try:
        return path.read_bytes()[:max_bytes]
    except Exception:
        return b""


def _redact_env_value(key: str, value: str) -> str:
    upper = key.upper()
    if any(token in upper for token in _SENSITIVE_ENV_TOKENS):
        return "<redacted>"
    if len(value) > 240:
        return f"{value[:240]}..."
    return value


def _process_cmdline_parts(pid: int) -> List[str]:
    raw = _safe_read_bytes(Path("/proc") / str(pid) / "cmdline")
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _process_command(pid: int) -> str:
    parts = _process_cmdline_parts(pid)
    if parts:
        return " ".join(parts)
    if shutil.which("ps"):
        try:
            proc = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            return proc.stdout.strip()
        except Exception:
            return ""
    return ""


def _cli_value(args: List[str], names: Set[str]) -> str:
    for idx, arg in enumerate(args):
        if arg in names and idx + 1 < len(args):
            return args[idx + 1]
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                return arg[len(prefix) :]
    return ""


def _cli_flag(args: List[str], names: Set[str]) -> bool:
    return any(arg in names for arg in args)


def _looks_like_ds8_command(command: str, args: Optional[List[str]] = None) -> bool:
    if args is None:
        try:
            tokens = shlex.split(command or "")
        except ValueError:
            tokens = (command or "").split()
    else:
        tokens = args
    if not tokens:
        return False
    executable = Path(tokens[0]).name.lower()
    python_launched = executable.startswith("python")
    for idx, token in enumerate(tokens[:-1]):
        if token == "-m" and tokens[idx + 1] == "noesis.ds8_runtime" and python_launched:
            return True
    if Path(tokens[0]).name == "ds8_runtime.py":
        return True
    if python_launched:
        return any(Path(token.strip("\"'")).name == "ds8_runtime.py" for token in tokens[1:])
    return False


def _ds8_launch_metadata(args: List[str], env: Mapping[str, str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    entrypoints = [token for token in args if Path(token).name == "ds8_runtime.py" or token == "noesis.ds8_runtime"]
    if entrypoints:
        metadata["entrypoint"] = entrypoints[0]
    pgie_profile = _cli_value(args, {"--pgie-profile", "--pgie_profile", "-pgie-profile", "-pgie_profile"}) or str(
        env.get("NOESIS_PGIE_PROFILE") or ""
    )
    tracking_mode = _cli_value(args, {"--tracking-mode"}) or str(env.get("NOESIS_TRACKING_MODE") or "")
    if _cli_flag(args, {"--v3dt"}):
        tracking_mode = "v3dt"
    fields = {
        "pipeline_config": _cli_value(args, {"--pipeline-config"}) or str(env.get("NOESIS_DS8_PIPELINE_CONFIG") or ""),
        "cameras_config": _cli_value(args, {"--cameras-config"}) or str(env.get("NOESIS_CAMERAS_CONFIG") or ""),
        "pgie_profile": pgie_profile,
        "size": _cli_value(args, {"--size"}),
        "tracking_mode": tracking_mode,
        "ws_host": _cli_value(args, {"--ws-host"}) or str(env.get("NOESIS_WS_HOST") or ""),
        "ws_port": _cli_value(args, {"--ws-port"}) or str(env.get("NOESIS_WS_PORT") or ""),
        "rest_host": _cli_value(args, {"--rest-host"}) or str(env.get("NOESIS_REST_HOST") or ""),
        "rest_port": _cli_value(args, {"--rest-port"}) or str(env.get("NOESIS_REST_PORT") or ""),
        "log_level": _cli_value(args, {"--log-level"}) or str(env.get("NOESIS_LOG_LEVEL") or ""),
        "depth_enable_seconds": _cli_value(args, {"--depth-enable-seconds"}) or str(env.get("NOESIS_DEPTH_ENABLE_SECONDS") or ""),
    }
    for key, value in fields.items():
        if value not in ("", None):
            metadata[key] = value
    if _cli_flag(args, {"--enable-rest"}):
        metadata["enable_rest"] = True
    if _cli_flag(args, {"--disable-rest"}):
        metadata["enable_rest"] = False
    return metadata


def _process_environ(pid: int) -> Dict[str, str]:
    raw = _safe_read_bytes(Path("/proc") / str(pid) / "environ")
    env: Dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key_raw, value_raw = item.split(b"=", 1)
        key = key_raw.decode("utf-8", errors="replace")
        value = value_raw.decode("utf-8", errors="replace")
        if key.startswith("NOESIS_") or key in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}:
            env[key] = _redact_env_value(key, value)
    return dict(sorted(env.items()))


def _process_stats(pid: int) -> Dict[str, Any]:
    if not shutil.which("ps"):
        return {}
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etimes=,pcpu=,pmem=,rss=,comm="],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return {}
    line = proc.stdout.strip()
    if proc.returncode != 0 or not line:
        return {}
    parts = line.split(None, 4)
    if len(parts) < 5:
        return {}
    return {
        "elapsed_s": int(float(parts[0])),
        "cpu_pct": parts[1],
        "mem_pct": parts[2],
        "rss_kib": int(float(parts[3])),
        "comm": parts[4],
    }


def _process_snapshot(pid: int, *, managed_pid: Optional[int] = None) -> Dict[str, Any]:
    args = _process_cmdline_parts(pid)
    command = " ".join(args) if args else _process_command(pid)
    cwd = ""
    try:
        cwd = os.readlink(Path("/proc") / str(pid) / "cwd")
    except Exception:
        cwd = ""
    stats = _process_stats(pid)
    env = _process_environ(pid)
    looks_like_ds8 = _looks_like_ds8_command(command, args)
    return {
        "pid": int(pid),
        "managed_by_console": managed_pid is not None and int(managed_pid) == int(pid),
        "looks_like_ds8": looks_like_ds8,
        "command": command,
        "cwd": cwd,
        "stats": stats,
        "env": env,
        "launch": _ds8_launch_metadata(args, env) if looks_like_ds8 else {},
    }


def _ss_listen_snapshot() -> List[str]:
    if not shutil.which("ss"):
        return []
    try:
        proc = subprocess.run(
            ["ss", "-ltnp"],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except Exception:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip() and not line.startswith("State ")]


def _local_endpoint_from_ss(line: str) -> Optional[Dict[str, Any]]:
    parts = line.split()
    if len(parts) < 4:
        return None
    local = parts[3]
    if ":" not in local:
        return None
    host, port_raw = local.rsplit(":", 1)
    try:
        port = int(port_raw)
    except ValueError:
        return None
    return {"host": host.strip("[]") or "", "port": port}


def _listening_ports_by_pid() -> Dict[int, List[Dict[str, Any]]]:
    by_pid: Dict[int, List[Dict[str, Any]]] = {}
    for line in _ss_listen_snapshot():
        endpoint = _local_endpoint_from_ss(line)
        if endpoint is None:
            continue
        owner: Dict[str, Any] = {"raw": line}
        if "users:" in line:
            owner["users"] = line.split("users:", 1)[1].strip()
        owner["pids"] = _pids_from_owner(owner)
        for pid in _pids_from_owner(owner):
            by_pid.setdefault(pid, []).append(
                {
                    "label": _DEFAULT_DS8_PORT_LABELS.get(int(endpoint["port"]), "listening"),
                    "host": endpoint["host"],
                    "port": int(endpoint["port"]),
                    "owner": owner,
                    "selected": False,
                    "source": "process-scan",
                }
            )
    return by_pid


def _discover_ds8_runtime_pids() -> List[int]:
    pids: List[int] = []
    proc_root = Path("/proc")
    try:
        children = list(proc_root.iterdir())
    except Exception:
        return pids
    for child in children:
        if not child.name.isdigit():
            continue
        pid = int(child.name)
        if _looks_like_ds8_command(_process_command(pid)):
            pids.append(pid)
    return sorted(set(pids))


def _append_process_port(entry: Dict[str, Any], port: Mapping[str, Any], *, selected: bool, source: str) -> None:
    port_value = int(port.get("port") or 0)
    host = str(port.get("host") or "")
    ports = entry.setdefault("ports", [])
    for existing in ports:
        if int(existing.get("port") or 0) == port_value and str(existing.get("host") or "") == host:
            if selected:
                existing["selected"] = True
                existing["source"] = source
                existing["label"] = port.get("label") or existing.get("label") or str(port_value)
            return
    ports.append(
        {
            "label": port.get("label") or str(port_value),
            "host": host,
            "port": port_value,
            "selected": bool(selected),
            "source": source,
        }
    )


def _launch_port_numbers(launch: Mapping[str, Any]) -> Set[int]:
    ports: Set[int] = set()
    for key in ("ws_port", "rest_port", "rtsp_port"):
        raw = launch.get(key)
        if raw in (None, ""):
            continue
        try:
            ports.add(int(raw))
        except (TypeError, ValueError):
            continue
    return ports


def process_inventory(
    ports: Iterable[Mapping[str, Any]],
    *,
    managed_pid: Optional[int] = None,
    include_discovered_ds8: bool = True,
) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for port in ports:
        owner = port.get("owner") if isinstance(port.get("owner"), Mapping) else {}
        for pid in _pids_from_owner(owner or {}):
            entry = grouped.setdefault(pid, {"pid": pid, "ports": []})
            _append_process_port(entry, port, selected=True, source="selected-port")
    listening_by_pid = _listening_ports_by_pid() if include_discovered_ds8 else {}
    if include_discovered_ds8:
        for pid in _discover_ds8_runtime_pids():
            grouped.setdefault(pid, {"pid": pid, "ports": []})
    processes: List[Dict[str, Any]] = []
    for pid, entry in grouped.items():
        snapshot = _process_snapshot(pid, managed_pid=managed_pid)
        selected_ports = {
            int(port.get("port") or 0)
            for port in entry.get("ports", [])
            if isinstance(port, Mapping) and port.get("selected")
        }
        meaningful_ports = set(_DEFAULT_DS8_PORT_LABELS) | selected_ports | _launch_port_numbers(snapshot.get("launch", {}))
        for port in listening_by_pid.get(pid, []):
            if int(port.get("port") or 0) in meaningful_ports:
                _append_process_port(entry, port, selected=False, source="process-scan")
        snapshot["ports"] = sorted(entry["ports"], key=lambda item: (not item.get("selected"), item["label"], item["port"]))
        processes.append(snapshot)
    return sorted(processes, key=lambda item: (not item.get("looks_like_ds8", False), int(item["pid"])))


def noesis_port_ownership(diagnostics: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Map busy DS8-owned port numbers to the runtime that owns them."""
    processes = [
        item
        for item in (diagnostics.get("processes") or [])
        if isinstance(item, Mapping) and item.get("looks_like_ds8")
    ]
    ds8_pids = {int(item.get("pid") or 0) for item in processes if item.get("pid")}
    ownership: Dict[int, Dict[str, Any]] = {}

    for process in processes:
        pid = int(process.get("pid") or 0)
        if not pid:
            continue
        for port in process.get("ports") or []:
            if not isinstance(port, Mapping) or port.get("label") == "Console":
                continue
            try:
                port_num = int(port.get("port") or 0)
            except (TypeError, ValueError):
                continue
            if port_num <= 0:
                continue
            ownership[port_num] = {
                "pid": pid,
                "managed_by_console": bool(process.get("managed_by_console")),
                "label": str(port.get("label") or port_num),
            }

    for item in diagnostics.get("ports") or []:
        if not isinstance(item, Mapping) or not item.get("busy") or item.get("label") == "Console":
            continue
        try:
            port_num = int(item.get("port") or 0)
        except (TypeError, ValueError):
            continue
        if port_num <= 0 or port_num in ownership:
            continue
        owner = item.get("owner") if isinstance(item.get("owner"), Mapping) else {}
        owner_pids = set(_pids_from_owner(owner))
        if not owner_pids or not owner_pids.issubset(ds8_pids):
            continue
        pid = next(iter(owner_pids))
        process = next((entry for entry in processes if int(entry.get("pid") or 0) == pid), {})
        ownership[port_num] = {
            "pid": pid,
            "managed_by_console": bool(process.get("managed_by_console")),
            "label": str(item.get("label") or port_num),
        }
    return ownership


def _process_port_number(process: Mapping[str, Any], label: str) -> Optional[int]:
    for port in process.get("ports") or []:
        if not isinstance(port, Mapping) or port.get("label") != label:
            continue
        try:
            return int(port.get("port"))
        except (TypeError, ValueError):
            return None
    return None


def _coerce_port(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def runtime_target_from_process(process: Mapping[str, Any]) -> Dict[str, Any]:
    launch = process.get("launch") if isinstance(process.get("launch"), Mapping) else {}
    ws_port = _coerce_port(launch.get("ws_port")) or _process_port_number(process, "WebSocket")
    rest_port = _coerce_port(launch.get("rest_port")) or _process_port_number(process, "REST")
    rtsp_port = _coerce_port(launch.get("rtsp_port")) or _process_port_number(process, "RTSP mosaic")
    return {
        "pid": process.get("pid"),
        "managed_by_console": bool(process.get("managed_by_console")),
        "looks_like_ds8": bool(process.get("looks_like_ds8")),
        "launch": dict(launch),
        "ports": list(process.get("ports") or []),
        "stats": dict(process.get("stats") or {}),
        "command": str(process.get("command") or ""),
        "ws_host": str(launch.get("ws_host") or "127.0.0.1"),
        "ws_port": ws_port,
        "rest_host": str(launch.get("rest_host") or "127.0.0.1"),
        "rest_port": rest_port,
        "rtsp_port": rtsp_port,
    }


def observed_runtime_target(
    processes: Iterable[Mapping[str, Any]],
    *,
    managed_pid: Optional[int] = None,
    spec: Optional[LaunchSpec] = None,
) -> Optional[Dict[str, Any]]:
    ds8 = [item for item in processes if item.get("looks_like_ds8")]
    if managed_pid is not None:
        managed = next((item for item in ds8 if int(item.get("pid") or 0) == int(managed_pid)), None)
        if managed is not None:
            return runtime_target_from_process(managed)
    external = [item for item in ds8 if not item.get("managed_by_console")]
    if not external:
        return None
    if spec is not None and len(external) > 1:
        for item in external:
            target = runtime_target_from_process(item)
            if target.get("ws_port") == int(spec.ws_port):
                return target
    ranked = sorted(
        external,
        key=lambda item: (
            -len(item.get("ports") or []),
            -int((item.get("stats") or {}).get("elapsed_s") or 0),
            int(item.get("pid") or 0),
        ),
    )
    return runtime_target_from_process(ranked[0])


def diagnostics_snapshot(spec: LaunchSpec, *, managed_pid: Optional[int] = None) -> Dict[str, Any]:
    ports = [
        port_status(spec.ws_host, int(spec.ws_port), label="WebSocket"),
        port_status(spec.rest_host, int(spec.rest_port), label="REST"),
        port_status("127.0.0.1", int(spec.rtsp_port), label="RTSP mosaic"),
        port_status("127.0.0.1", 9090, label="Console"),
    ]
    suggestions = {
        "ws_port": find_free_port(spec.ws_host, int(spec.ws_port)),
        "rest_port": find_free_port(spec.rest_host, int(spec.rest_port)),
        "rtsp_port": find_free_port("127.0.0.1", int(spec.rtsp_port)),
    }
    processes = process_inventory(ports, managed_pid=managed_pid)
    observed_runtime = observed_runtime_target(processes, managed_pid=managed_pid, spec=spec)
    return {
        "ports": ports,
        "suggested_ports": suggestions,
        "artifacts": artifact_audit(spec),
        "gpu": gpu_snapshot(),
        "processes": processes,
        "observed_runtime": observed_runtime,
        "ds8_runtimes": [
            {
                "pid": item.get("pid"),
                "managed_by_console": item.get("managed_by_console"),
                "ports": item.get("ports", []),
                "launch": item.get("launch", {}),
                "command": item.get("command", ""),
            }
            for item in processes
            if item.get("looks_like_ds8")
        ],
    }
