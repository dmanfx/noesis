#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
for _path in (str(DS9_ROOT), str(REPO_ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DS9_ROOT))

from noesis_core.servicemaker_shutdown import (  # noqa: E402
    synthetic_stub_lifecycle_evidence,
)
from scripts.internal_auth_client import (  # noqa: E402
    RequiredInternalAuth,
    add_auth_token_file_argument,
    configure_required_auth_environment,
    connect_required_websocket,
    load_required_internal_auth,
)
from scripts.zero_copy_boundary_diagnostics import (  # noqa: E402
    BoundaryGateTracker,
    extract_boundary_diagnostics,
)

_SYNTHETIC_BACKEND_LOG_MARKER = (
    '{"event":"pipeline_backend_selected","backend":"synthetic_stub",'
    '"native_runtime":false,"promotable":false}'
)
_SYNTHETIC_LIFECYCLE_LOG_MARKERS = (
    _SYNTHETIC_BACKEND_LOG_MARKER,
    "Orderly pipeline EOS accepted:",
    "EOS received on pipeline (reason=shutdown_requested)",
    "pyservicemaker wait() returned (pipeline stopped)",
    "Shutdown complete",
)


async def _collect_stats(
    ws_url: str,
    auth: RequiredInternalAuth,
    duration_s: float,
    startup_timeout_s: float,
    max_p99_ms: float,
    max_violations: int,
    depth_camera: str,
    proc: Optional[subprocess.Popen[str]],
) -> Dict[str, Any]:
    connect_deadline = time.time() + max(1.0, float(startup_timeout_s))
    ws = None
    last_err: Optional[str] = None
    while time.time() < connect_deadline:
        try:
            ws = await connect_required_websocket(ws_url, auth, max_size=None)
            break
        except Exception as exc:
            last_err = str(exc)
            await asyncio.sleep(0.25)

    if ws is None:
        return {"ok": False, "error": f"ws_connect_failed:{last_err or 'unknown'}", "samples": 0}

    samples = 0
    max_seen_violations = 0
    boundary_gate = BoundaryGateTracker()
    max_seen_ws_p99 = None
    max_seen_rest_p99 = None
    boundary_diagnostics: Dict[str, Any] | None = None
    errors_seen: list[str] = []
    depth_req_sent = 0
    depth_resp_seen = 0
    camera_hint: Optional[str] = None
    process_exit_code: Optional[int] = None

    start = time.monotonic()
    next_depth_req = start
    end = start + max(1.0, float(duration_s))

    try:
        while time.monotonic() < end:
            if proc is not None:
                process_exit_code = proc.poll()
                if process_exit_code is not None:
                    return {
                        "ok": False,
                        "error": "runtime_exited_early",
                        "process_exit_code": process_exit_code,
                        "samples": samples,
                    }

            now = time.monotonic()
            if now >= next_depth_req:
                req = {
                    "type": "get_ma_depth",
                    "camera": str(depth_camera or "").strip() or camera_hint or "camera_0",
                    "request_id": f"zero-copy-stats-smoke-{depth_req_sent + 1}",
                }
                try:
                    await ws.send(json.dumps(req, separators=(",", ":")))
                    depth_req_sent += 1
                except Exception:
                    pass
                next_depth_req = now + 1.0

            timeout_s = min(1.0, max(0.05, end - time.monotonic()))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                if "ConnectionClosed" in type(exc).__name__:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    try:
                        ws = await connect_required_websocket(
                            ws_url, auth, max_size=None
                        )
                        continue
                    except Exception as reconnect_exc:
                        return {
                            "ok": False,
                            "error": f"ws_reconnect_failed:{type(reconnect_exc).__name__}",
                            "samples": samples,
                        }
                return {"ok": False, "error": f"ws_recv_failed:{type(exc).__name__}", "samples": samples}

            if not isinstance(msg, str):
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue

            msg_type = str(payload.get("type") or "")
            if msg_type == "ma_depth_response":
                depth_resp_seen += 1
                continue
            if msg_type != "stats":
                continue

            stats = payload.get("payload") or {}
            pipe = stats.get("pipeline") or {}
            cameras = stats.get("cameras") or {}
            if camera_hint is None and isinstance(cameras, dict) and cameras:
                camera_hint = str(next(iter(cameras.keys())))

            samples += 1
            boundary_max_changed = boundary_gate.observe(pipe)
            try:
                max_seen_violations = max(max_seen_violations, int(pipe.get("zero_copy_violations", 0) or 0))
            except Exception:
                pass
            if boundary_max_changed:
                boundary_diagnostics = extract_boundary_diagnostics(
                    pipe,
                    allowed_p99_ms=float(max_p99_ms),
                )
            for channel, field in (
                ("ws", "boundary_cpu_serialization_ws_p99_ms"),
                ("rest", "boundary_cpu_serialization_rest_p99_ms"),
            ):
                try:
                    value = pipe.get(field)
                    if value is None:
                        continue
                    numeric = float(value)
                    if channel == "ws" and (
                        max_seen_ws_p99 is None or numeric > max_seen_ws_p99
                    ):
                        max_seen_ws_p99 = numeric
                    if channel == "rest" and (
                        max_seen_rest_p99 is None or numeric > max_seen_rest_p99
                    ):
                        max_seen_rest_p99 = numeric
                except Exception:
                    pass
            for err in list(pipe.get("errors") or []):
                s = str(err).strip()
                if s:
                    errors_seen.append(s)

    finally:
        try:
            await ws.close()
        except Exception:
            pass

    boundary_evidence = {
        **boundary_gate.evidence(),
        "max_boundary_ws_p99_ms": (
            float(max_seen_ws_p99) if max_seen_ws_p99 is not None else None
        ),
        "max_boundary_rest_p99_ms": (
            float(max_seen_rest_p99) if max_seen_rest_p99 is not None else None
        ),
        "boundary_diagnostics": boundary_diagnostics,
    }
    if samples < 1:
        return {"ok": False, "error": "no_stats_samples", "samples": 0}
    if errors_seen:
        return {
            "ok": False,
            "error": "pipeline_errors_present",
            "samples": samples,
            "pipeline_errors": errors_seen[:16],
            **boundary_evidence,
        }
    if max_seen_violations > int(max_violations):
        return {
            "ok": False,
            "error": "zero_copy_violations_exceeded",
            "samples": samples,
            "max_zero_copy_violations": int(max_seen_violations),
            "allowed_max_violations": int(max_violations),
            **boundary_evidence,
        }
    boundary_failure = boundary_gate.failure(allowed_p99_ms=float(max_p99_ms))
    if boundary_failure is not None:
        return {
            "ok": False,
            "error": boundary_failure,
            "samples": samples,
            "allowed_max_p99_ms": float(max_p99_ms),
            **boundary_evidence,
        }

    return {
        "ok": True,
        "samples": samples,
        "max_zero_copy_violations": int(max_seen_violations),
        **boundary_evidence,
        "ws_depth_requests": depth_req_sent,
        "ws_depth_responses": depth_resp_seen,
    }


def _spawn_runtime(
    ws_port: int,
    pipeline_config: Path,
    cameras_config: Path,
    log_path: Path,
    *,
    auth: RequiredInternalAuth,
    stub: bool,
    skip_cuda_preflight: bool,
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        str(DS9_ROOT / "noesis" / "ds9_runtime.py"),
        "--pipeline-config",
        str(pipeline_config),
        "--cameras-config",
        str(cameras_config),
        "--disable-rest",
        "--ws-port",
        str(int(ws_port)),
        "--log-level",
        "INFO",
    ]
    env = dict(os.environ)
    configure_required_auth_environment(env, auth)
    synthetic_state_dir: Optional[Path] = None
    if stub:
        synthetic_state_dir = _configure_synthetic_runtime_environment(env)
        env["NOESIS_DS9_STUB_PIPELINE"] = "1"
        env["NOESIS_MOSAIC_RTSP_ENABLED"] = "0"
        env["NOESIS_MOSAIC_WEBRTC_ENABLED"] = "0"
        cmd.extend(["--storage-base", str(synthetic_state_dir / "depth")])
    if skip_cuda_preflight:
        env["NOESIS_SKIP_CUDA_PREFLIGHT"] = "1"
    env.setdefault("NOESIS_WS_PORT_FALLBACK_TRIES", "32")
    env.setdefault("NOESIS_WS_BIND_RETRY_TRIES", "4")

    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    if synthetic_state_dir is not None:
        setattr(proc, "_noesis_synthetic_state_dir", synthetic_state_dir)
    return proc


def _configure_synthetic_runtime_environment(env: Dict[str, str]) -> Path:
    """Isolate every mutable stub-runtime state path from the operator home."""

    original_home = Path(env.get("HOME") or str(Path.home())).expanduser()
    camera_secrets = Path(
        env.get("NOESIS_CAMERA_SECRETS_FILE")
        or original_home / ".local/state/noesis/secrets/camera_sources.json"
    ).expanduser().resolve(strict=False)
    mapanything_key = Path(
        env.get("NOESIS_MAPANYTHING_API_KEY_FILE")
        or original_home / ".local/state/noesis/secrets/mapanything_rpc.key"
    ).expanduser().resolve(strict=False)
    python_user_base = Path(
        env.get("PYTHONUSERBASE") or original_home / ".local"
    ).expanduser().resolve(strict=False)
    state_dir = Path(tempfile.mkdtemp(prefix="noesis-ds9-synthetic-stub-"))
    state_dir.chmod(0o700)
    env.update(
        {
            "HOME": str(state_dir),
            "PYTHONUSERBASE": str(python_user_base),
            "XDG_STATE_HOME": str(state_dir / "xdg-state"),
            "NOESIS_CAMERA_SECRETS_FILE": str(camera_secrets),
            "NOESIS_MAPANYTHING_API_KEY_FILE": str(mapanything_key),
            "NOESIS_ANALYTICS_EXCLUDE_CONFIG": str(state_dir / "analytics-exclude.ini"),
            "NOESIS_IDENTITY_V2_STORE": str(state_dir / "identity-v2.sqlite3"),
            "NOESIS_WORLD_JOURNAL_PATH": str(state_dir / "world.jsonl"),
            "NOESIS_BUILD_DIR": str(state_dir / "build"),
            "NOESIS_HOUSEHOLD_ARCHIVE_STATE": "0",
        }
    )
    return state_dir


def _reserve_local_port() -> tuple[int, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return int(sock.getsockname()[1]), sock


def _terminate_process(
    proc: subprocess.Popen[str], timeout_s: float = 8.0
) -> Dict[str, Any]:
    if proc.poll() is not None:
        exit_code = proc.poll()
        return {
            "signal_requested": False,
            "forced_kill": False,
            "exit_code": exit_code,
            "graceful": exit_code == 0,
        }
    signal_requested = False
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        signal_requested = True
    except Exception:
        try:
            proc.terminate()
            signal_requested = True
        except Exception:
            pass
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        if proc.poll() is not None:
            exit_code = proc.poll()
            return {
                "signal_requested": signal_requested,
                "forced_kill": False,
                "exit_code": exit_code,
                "graceful": exit_code == 0,
            }
        time.sleep(0.1)
    forced_kill = False
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        forced_kill = True
    except Exception:
        try:
            proc.kill()
            forced_kill = True
        except Exception:
            pass
    try:
        proc.wait(timeout=2.0)
    except Exception:
        pass
    return {
        "signal_requested": signal_requested,
        "forced_kill": forced_kill,
        "exit_code": proc.poll(),
        "graceful": False,
    }


def _synthetic_lifecycle_receipt(
    log_path: Path,
    termination: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        log_text = ""
    missing_markers = [
        marker for marker in _SYNTHETIC_LIFECYCLE_LOG_MARKERS if marker not in log_text
    ]
    termination_payload = dict(termination or {})
    return {
        "ok": bool(termination_payload.get("graceful")) and not missing_markers,
        "termination": termination_payload,
        "missing_markers": missing_markers,
    }


def _tail(path: Path, lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    arr = text.splitlines()
    return "\n".join(arr[-max(1, int(lines)):])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-copy stats smoke test (WS contract + thresholds)")
    p.add_argument("--pipeline-config", type=Path, default=DS9_ROOT / "config" / "infer.yaml")
    p.add_argument("--cameras-config", type=Path, default=REPO_ROOT / "config" / "cameras.yaml")
    p.add_argument("--duration-s", type=float, default=60.0)
    p.add_argument("--stats-ws", default="ws://127.0.0.1:6008")
    p.add_argument("--startup-timeout", type=float, default=15.0)
    p.add_argument("--max-p99-ms", type=float, default=3.0)
    p.add_argument("--max-violations", type=int, default=0)
    p.add_argument(
        "--depth-camera",
        default="__zero_copy_probe__",
        help="Camera id used for WS depth-path traffic.",
    )
    p.add_argument("--no-spawn", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--skip-cuda-preflight", action="store_true")
    p.add_argument("--log-path", type=Path, default=None)
    add_auth_token_file_argument(p)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        auth = load_required_internal_auth(args.auth_token_file)
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"internal_auth_unavailable:{exc}"},
                separators=(",", ":"),
            )
        )
        return 1
    if bool(args.stub) and bool(args.no_spawn):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "synthetic_stub_requires_spawn",
                    "lifecycle_evidence": synthetic_stub_lifecycle_evidence(),
                    "test_backend_only": True,
                },
                separators=(",", ":"),
            )
        )
        return 1
    ws_url = str(args.stats_ws)
    parsed = urlparse(ws_url)
    ws_port = int(parsed.port or (443 if parsed.scheme == "wss" else 6008))
    ws_lock: Optional[socket.socket] = None

    if args.log_path is not None:
        log_path = Path(args.log_path)
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="zero_copy_stats_smoke_", suffix=".log", delete=False)
        log_path = Path(tmp.name)
        tmp.close()

    proc: Optional[subprocess.Popen[str]] = None
    if not args.no_spawn:
        try:
            ws_port, ws_lock = _reserve_local_port()
        except Exception:
            if ws_lock is not None:
                ws_lock.close()
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "port_reservation_failed",
                        "stats_ws": ws_url,
                        "log_path": str(log_path),
                    },
                    separators=(",", ":"),
                    sort_keys=False,
                )
            )
            return 1
        ws_url = f"{parsed.scheme}://{parsed.hostname or '127.0.0.1'}:{ws_port}"
        proc = _spawn_runtime(
            ws_port=ws_port,
            pipeline_config=Path(args.pipeline_config),
            cameras_config=Path(args.cameras_config),
            log_path=log_path,
            auth=auth,
            stub=bool(args.stub),
            skip_cuda_preflight=bool(args.skip_cuda_preflight),
        )
        if ws_lock is not None:
            ws_lock.close()

    started = time.monotonic()
    termination: Optional[Dict[str, Any]] = None
    try:
        result = asyncio.run(
            _collect_stats(
                ws_url=ws_url,
                auth=auth,
                duration_s=float(args.duration_s),
                startup_timeout_s=float(args.startup_timeout),
                max_p99_ms=float(args.max_p99_ms),
                max_violations=int(args.max_violations),
                depth_camera=str(args.depth_camera),
                proc=proc,
            )
        )
    finally:
        if proc is not None:
            termination = _terminate_process(proc)
            state_dir = getattr(proc, "_noesis_synthetic_state_dir", None)
            if state_dir is not None:
                shutil.rmtree(Path(state_dir), ignore_errors=True)

    synthetic_receipt: Optional[Dict[str, Any]] = None
    if bool(args.stub):
        synthetic_receipt = _synthetic_lifecycle_receipt(log_path, termination)
        if not bool(synthetic_receipt.get("ok")):
            result = dict(result)
            result["ok"] = False
            result.setdefault("error", "synthetic_stub_lifecycle_unproven")

    elapsed_s = max(0.0, time.monotonic() - started)
    payload = {
        "ok": bool(result.get("ok", False)),
        "duration_s": round(elapsed_s, 3),
        "stats_ws": ws_url,
        "log_path": str(log_path),
        **(
            {
                "lifecycle_evidence": synthetic_stub_lifecycle_evidence(),
                "synthetic_lifecycle": synthetic_receipt,
                "test_backend_only": True,
            }
            if bool(args.stub)
            else {}
        ),
        **result,
    }
    if not bool(result.get("ok", False)):
        tail = _tail(log_path)
        if tail:
            payload["runtime_tail"] = tail
        print(json.dumps(payload, separators=(",", ":"), sort_keys=False))
        return 1

    print(json.dumps(payload, separators=(",", ":"), sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
