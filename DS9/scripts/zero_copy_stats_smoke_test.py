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


async def _collect_stats(
    ws_url: str,
    duration_s: float,
    startup_timeout_s: float,
    max_p99_ms: float,
    max_violations: int,
    depth_camera: str,
    proc: Optional[subprocess.Popen[str]],
) -> Dict[str, Any]:
    import websockets

    connect_deadline = time.time() + max(1.0, float(startup_timeout_s))
    ws = None
    last_err: Optional[str] = None
    while time.time() < connect_deadline:
        try:
            ws = await websockets.connect(ws_url, max_size=None)
            break
        except Exception as exc:
            last_err = str(exc)
            await asyncio.sleep(0.25)

    if ws is None:
        return {"ok": False, "error": f"ws_connect_failed:{last_err or 'unknown'}", "samples": 0}

    samples = 0
    max_seen_violations = 0
    max_seen_p99 = None
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
                        ws = await websockets.connect(ws_url, max_size=None)
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
            try:
                max_seen_violations = max(max_seen_violations, int(pipe.get("zero_copy_violations", 0) or 0))
            except Exception:
                pass
            try:
                p99 = pipe.get("boundary_cpu_serialization_p99_ms")
                if p99 is not None:
                    p99_val = float(p99)
                    if max_seen_p99 is None or p99_val > float(max_seen_p99):
                        max_seen_p99 = p99_val
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

    if samples < 1:
        return {"ok": False, "error": "no_stats_samples", "samples": 0}
    if errors_seen:
        return {"ok": False, "error": "pipeline_errors_present", "samples": samples, "pipeline_errors": errors_seen[:16]}
    if max_seen_violations > int(max_violations):
        return {
            "ok": False,
            "error": "zero_copy_violations_exceeded",
            "samples": samples,
            "max_zero_copy_violations": int(max_seen_violations),
            "allowed_max_violations": int(max_violations),
        }
    if max_seen_p99 is not None and float(max_seen_p99) > float(max_p99_ms):
        return {
            "ok": False,
            "error": "boundary_p99_exceeded",
            "samples": samples,
            "max_boundary_p99_ms": float(max_seen_p99),
            "allowed_max_p99_ms": float(max_p99_ms),
        }

    return {
        "ok": True,
        "samples": samples,
        "max_zero_copy_violations": int(max_seen_violations),
        "max_boundary_p99_ms": float(max_seen_p99) if max_seen_p99 is not None else None,
        "ws_depth_requests": depth_req_sent,
        "ws_depth_responses": depth_resp_seen,
    }


def _spawn_runtime(
    ws_port: int,
    pipeline_config: Path,
    cameras_config: Path,
    log_path: Path,
    *,
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
    if stub:
        env["NOESIS_DS9_STUB_PIPELINE"] = "1"
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


def _reserve_local_port() -> tuple[int, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return int(sock.getsockname()[1]), sock


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
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
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
    return p.parse_args()


def main() -> int:
    args = _parse_args()
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
            stub=bool(args.stub),
            skip_cuda_preflight=bool(args.skip_cuda_preflight),
        )
        if ws_lock is not None:
            ws_lock.close()

    started = time.monotonic()
    try:
        result = asyncio.run(
            _collect_stats(
                ws_url=ws_url,
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
            _terminate_process(proc)

    elapsed_s = max(0.0, time.monotonic() - started)
    payload = {
        "ok": bool(result.get("ok", False)),
        "duration_s": round(elapsed_s, 3),
        "stats_ws": ws_url,
        "log_path": str(log_path),
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
