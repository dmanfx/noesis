#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.internal_auth_client import (  # noqa: E402
    RequiredInternalAuth,
    add_auth_token_file_argument,
    build_required_auth_request,
    configure_required_auth_environment,
    connect_required_websocket,
    load_required_internal_auth,
)
from scripts.zero_copy_boundary_diagnostics import (  # noqa: E402
    BoundaryGateTracker,
    extract_boundary_diagnostics,
)


def _json_print(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=False))


def _parse_port_from_url(url: str, default_port: int) -> int:
    parsed = urllib.parse.urlparse(str(url))
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme == "https" or parsed.scheme == "wss":
        return 443
    return int(default_port)


def _replace_url_port(url: str, port: int) -> str:
    parsed = urllib.parse.urlparse(str(url))
    host = parsed.hostname or "127.0.0.1"
    netloc = f"{host}:{int(port)}"
    return urllib.parse.urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _reserve_local_port() -> tuple[int, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return int(sock.getsockname()[1]), sock


def _request_depth_refresh(
    rest_url: str,
    seconds: int,
    auth: RequiredInternalAuth,
    timeout_s: float = 2.0,
) -> Dict[str, Any]:
    parsed = urllib.parse.urlparse(rest_url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query["seconds"] = str(max(1, int(seconds)))
    final_url = urllib.parse.urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urllib.parse.urlencode(query),
            parsed.fragment,
        )
    )
    try:
        req = build_required_auth_request(final_url, auth, method="POST")
        with urllib.request.urlopen(req, timeout=max(0.1, float(timeout_s))) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(data)
        return {
            "ok": bool(payload.get("enabled", False)),
            "status": "ok",
            "response": payload,
            "url": final_url,
        }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": f"http_{exc.code}", "url": final_url}
    except Exception as exc:
        return {"ok": False, "status": f"error:{type(exc).__name__}", "url": final_url}


def _rest_refresh_contract_error(
    *, attempts: int, successes: int, last_status: str
) -> Optional[str]:
    if int(attempts) < 1:
        return "no_rest_refresh_attempts"
    if int(successes) < 1:
        return "no_successful_rest_refresh"
    if str(last_status) != "ok":
        return "rest_refresh_final_status_not_ok"
    return None


async def _collect_stats_and_drive(
    ws_url: str,
    auth: RequiredInternalAuth,
    duration_s: float,
    startup_timeout_s: float,
    max_p99_ms: float,
    max_violations: int,
    rest_url: str,
    depth_camera: str,
    rest_refresh_seconds: int,
    rest_refresh_interval_s: float,
    allow_network_denied: bool,
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
            if "Operation not permitted" in last_err:
                if allow_network_denied:
                    return {"ok": True, "skipped": "network_denied", "samples": 0}
                return {"ok": False, "error": "network_denied", "samples": 0}
            await asyncio.sleep(0.25)

    if ws is None:
        return {
            "ok": False,
            "error": f"ws_connect_failed:{last_err or 'unknown'}",
            "samples": 0,
        }

    start = time.monotonic()
    end = start + max(1.0, float(duration_s))
    next_ws_req = start
    next_rest_refresh = start

    samples = 0
    ws_depth_requests = 0
    ws_depth_responses = 0
    rest_refresh_attempts = 0
    rest_refresh_success = 0
    rest_refresh_last_status = "none"

    zero_copy_core_enabled_seen = False
    max_seen_violations = 0
    boundary_gate = BoundaryGateTracker()
    max_seen_ws_p99 = None
    max_seen_rest_p99 = None
    boundary_diagnostics: Dict[str, Any] | None = None
    errors_seen: list[str] = []
    camera_hint: Optional[str] = None
    process_exited = False
    process_exit_code: Optional[int] = None

    try:
        while time.monotonic() < end:
            now = time.monotonic()

            if proc is not None:
                process_exit_code = proc.poll()
                if process_exit_code is not None:
                    process_exited = True
                    break

            if now >= next_rest_refresh:
                rest_refresh_attempts += 1
                refresh_result = await asyncio.to_thread(
                    _request_depth_refresh,
                    rest_url,
                    int(rest_refresh_seconds),
                    auth,
                    2.0,
                )
                rest_refresh_last_status = str(refresh_result.get("status", "unknown"))
                if bool(refresh_result.get("ok", False)):
                    rest_refresh_success += 1
                next_rest_refresh = now + max(0.5, float(rest_refresh_interval_s))

            if now >= next_ws_req:
                request_camera = (
                    str(depth_camera or "").strip() or camera_hint or "camera_0"
                )
                request_payload = {
                    "type": "get_ma_depth",
                    "camera": request_camera,
                    "request_id": f"zero-copy-smoke-{ws_depth_requests + 1}",
                }
                try:
                    await ws.send(json.dumps(request_payload, separators=(",", ":")))
                    ws_depth_requests += 1
                except Exception:
                    pass
                next_ws_req = now + 1.0

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
                            "process_exited": process_exited,
                            "process_exit_code": process_exit_code,
                        }
                return {
                    "ok": False,
                    "error": f"ws_recv_failed:{type(exc).__name__}",
                    "samples": samples,
                    "process_exited": process_exited,
                    "process_exit_code": process_exit_code,
                }

            if not isinstance(msg, str):
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue

            msg_type = str(payload.get("type") or "")
            if msg_type == "ma_depth_response":
                ws_depth_responses += 1
                continue

            if msg_type != "stats":
                continue

            stats = payload.get("payload") or {}
            pipe = stats.get("pipeline") or {}
            cameras = stats.get("cameras") or {}
            if camera_hint is None and isinstance(cameras, dict) and cameras:
                try:
                    camera_hint = str(next(iter(cameras.keys())))
                except Exception:
                    camera_hint = None

            samples += 1
            if bool(pipe.get("zero_copy_core_enabled", False)):
                zero_copy_core_enabled_seen = True

            boundary_max_changed = boundary_gate.observe(pipe)

            try:
                max_seen_violations = max(
                    max_seen_violations, int(pipe.get("zero_copy_violations", 0) or 0)
                )
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
                text = str(err).strip()
                if text:
                    errors_seen.append(text)

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

    if process_exited:
        return {
            "ok": False,
            "error": "runtime_exited_early",
            "samples": samples,
            "process_exited": True,
            "process_exit_code": process_exit_code,
            "ws_depth_requests": ws_depth_requests,
            "ws_depth_responses": ws_depth_responses,
            "rest_refresh_attempts": rest_refresh_attempts,
            "rest_refresh_success": rest_refresh_success,
            "rest_refresh_last_status": rest_refresh_last_status,
            **boundary_evidence,
        }

    if samples < 1:
        return {
            "ok": False,
            "error": "no_stats_samples",
            "samples": 0,
            "ws_depth_requests": ws_depth_requests,
            "ws_depth_responses": ws_depth_responses,
            "rest_refresh_attempts": rest_refresh_attempts,
            "rest_refresh_success": rest_refresh_success,
            "rest_refresh_last_status": rest_refresh_last_status,
            **boundary_evidence,
        }

    if not zero_copy_core_enabled_seen:
        return {
            "ok": False,
            "error": "zero_copy_core_never_enabled",
            "samples": samples,
            "ws_depth_requests": ws_depth_requests,
            "ws_depth_responses": ws_depth_responses,
            "rest_refresh_attempts": rest_refresh_attempts,
            "rest_refresh_success": rest_refresh_success,
            "rest_refresh_last_status": rest_refresh_last_status,
            **boundary_evidence,
        }

    if errors_seen:
        return {
            "ok": False,
            "error": "pipeline_errors_present",
            "samples": samples,
            "pipeline_errors": errors_seen[:16],
            "ws_depth_requests": ws_depth_requests,
            "ws_depth_responses": ws_depth_responses,
            "rest_refresh_attempts": rest_refresh_attempts,
            "rest_refresh_success": rest_refresh_success,
            "rest_refresh_last_status": rest_refresh_last_status,
            **boundary_evidence,
        }

    rest_refresh_error = _rest_refresh_contract_error(
        attempts=rest_refresh_attempts,
        successes=rest_refresh_success,
        last_status=rest_refresh_last_status,
    )
    if rest_refresh_error is not None:
        return {
            "ok": False,
            "error": rest_refresh_error,
            "samples": samples,
            "ws_depth_requests": ws_depth_requests,
            "ws_depth_responses": ws_depth_responses,
            "rest_refresh_attempts": rest_refresh_attempts,
            "rest_refresh_success": rest_refresh_success,
            "rest_refresh_last_status": rest_refresh_last_status,
            **boundary_evidence,
        }

    if max_seen_violations > int(max_violations):
        return {
            "ok": False,
            "error": "zero_copy_violations_exceeded",
            "samples": samples,
            "max_zero_copy_violations": int(max_seen_violations),
            "allowed_zero_copy_violations": int(max_violations),
            "ws_depth_requests": ws_depth_requests,
            "ws_depth_responses": ws_depth_responses,
            "rest_refresh_attempts": rest_refresh_attempts,
            "rest_refresh_success": rest_refresh_success,
            "rest_refresh_last_status": rest_refresh_last_status,
            **boundary_evidence,
        }

    boundary_failure = boundary_gate.failure(allowed_p99_ms=float(max_p99_ms))
    if boundary_failure is not None:
        return {
            "ok": False,
            "error": boundary_failure,
            "samples": samples,
            "allowed_boundary_p99_ms": float(max_p99_ms),
            "ws_depth_requests": ws_depth_requests,
            "ws_depth_responses": ws_depth_responses,
            "rest_refresh_attempts": rest_refresh_attempts,
            "rest_refresh_success": rest_refresh_success,
            "rest_refresh_last_status": rest_refresh_last_status,
            **boundary_evidence,
        }

    return {
        "ok": True,
        "samples": samples,
        "max_zero_copy_violations": int(max_seen_violations),
        **boundary_evidence,
        "ws_depth_requests": ws_depth_requests,
        "ws_depth_responses": ws_depth_responses,
        "rest_refresh_attempts": rest_refresh_attempts,
        "rest_refresh_success": rest_refresh_success,
        "rest_refresh_last_status": rest_refresh_last_status,
    }


def _spawn_runtime(
    ws_port: int,
    rest_port: int,
    pipeline_config: Path,
    cameras_config: Path,
    *,
    auth: RequiredInternalAuth,
    stub: bool,
    skip_cuda_preflight: bool,
    log_path: Path,
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "noesis/ds8_runtime.py",
        "--pipeline-config",
        str(pipeline_config),
        "--cameras-config",
        str(cameras_config),
        "--enable-rest",
        "--ws-port",
        str(int(ws_port)),
        "--rest-port",
        str(int(rest_port)),
        "--log-level",
        "INFO",
    ]
    env = dict(os.environ)
    configure_required_auth_environment(env, auth)
    if stub:
        env["NOESIS_DS8_STUB_PIPELINE"] = "1"
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
    return proc


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _tail(path: Path, lines: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    arr = text.splitlines()
    return "\n".join(arr[-max(1, int(lines)) :])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-copy smoke gate (runtime + WS stats + boundary traffic)"
    )
    parser.add_argument(
        "--pipeline-config", type=Path, default=REPO_ROOT / "config" / "infer.yaml"
    )
    parser.add_argument(
        "--cameras-config", type=Path, default=REPO_ROOT / "config" / "cameras.yaml"
    )
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--stats-ws", default="ws://127.0.0.1:6008")
    parser.add_argument(
        "--rest-url", default="http://127.0.0.1:8080/api/v1/depth/refresh"
    )
    parser.add_argument(
        "--depth-camera",
        default="__zero_copy_probe__",
        help="Camera id used for WS depth-path traffic; defaults to a non-streaming probe id to keep payload small.",
    )
    parser.add_argument("--rest-refresh-seconds", type=int, default=20)
    parser.add_argument("--rest-refresh-interval-s", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--max-p99-ms", type=float, default=3.0)
    parser.add_argument("--max-violations", type=int, default=0)
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--skip-cuda-preflight", action="store_true")
    parser.add_argument("--allow-network-denied", action="store_true")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--tail-lines", type=int, default=40)
    add_auth_token_file_argument(parser)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        auth = load_required_internal_auth(args.auth_token_file)
    except Exception as exc:
        _json_print({"ok": False, "error": f"internal_auth_unavailable:{exc}"})
        return 1
    ws_url = str(args.stats_ws)
    rest_url = str(args.rest_url)

    if args.log_path is not None:
        log_path = Path(args.log_path)
    else:
        tmp = tempfile.NamedTemporaryFile(
            prefix="zero_copy_smoke_", suffix=".log", delete=False
        )
        log_path = Path(tmp.name)
        tmp.close()

    proc: Optional[subprocess.Popen[str]] = None
    ws_port = _parse_port_from_url(ws_url, 6008)
    rest_port = _parse_port_from_url(rest_url, 8080)
    ws_lock: Optional[socket.socket] = None
    rest_lock: Optional[socket.socket] = None

    if not args.no_spawn:
        try:
            ws_port, ws_lock = _reserve_local_port()
            rest_port, rest_lock = _reserve_local_port()
        except Exception:
            if ws_lock is not None:
                ws_lock.close()
            if rest_lock is not None:
                rest_lock.close()
            _json_print(
                {
                    "ok": False,
                    "error": "port_reservation_failed",
                    "stats_ws": ws_url,
                    "rest_url": rest_url,
                    "log_path": str(log_path),
                }
            )
            return 1

        ws_url = _replace_url_port(ws_url, ws_port)
        rest_url = _replace_url_port(rest_url, rest_port)
        proc = _spawn_runtime(
            ws_port=ws_port,
            rest_port=rest_port,
            pipeline_config=Path(args.pipeline_config),
            cameras_config=Path(args.cameras_config),
            auth=auth,
            stub=bool(args.stub),
            skip_cuda_preflight=bool(args.skip_cuda_preflight),
            log_path=log_path,
        )
        if ws_lock is not None:
            ws_lock.close()
        if rest_lock is not None:
            rest_lock.close()

    started = time.monotonic()
    try:
        result = asyncio.run(
            _collect_stats_and_drive(
                ws_url=ws_url,
                auth=auth,
                duration_s=float(args.duration_s),
                startup_timeout_s=float(args.startup_timeout),
                max_p99_ms=float(args.max_p99_ms),
                max_violations=int(args.max_violations),
                rest_url=rest_url,
                depth_camera=str(args.depth_camera),
                rest_refresh_seconds=int(args.rest_refresh_seconds),
                rest_refresh_interval_s=float(args.rest_refresh_interval_s),
                allow_network_denied=bool(args.allow_network_denied),
                proc=proc,
            )
        )
    finally:
        if proc is not None:
            _terminate_process(proc)

    elapsed_s = max(0.0, time.monotonic() - started)

    payload: Dict[str, Any] = {
        "ok": bool(result.get("ok", False)),
        "duration_s": round(elapsed_s, 3),
        "stats_ws": ws_url,
        "rest_url": rest_url,
        "log_path": str(log_path),
        **result,
    }

    if not bool(result.get("ok", False)):
        tail = _tail(log_path, int(args.tail_lines))
        if tail:
            payload["runtime_tail"] = tail
        _json_print(payload)
        return 1

    _json_print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
