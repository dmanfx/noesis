#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
for _path in (str(DS9_ROOT), str(REPO_ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DS9_ROOT))


async def _collect_stats(ws_url: str, duration_s: float, startup_timeout_s: float) -> Dict[str, Any]:
    import websockets

    deadline = time.time() + max(1.0, startup_timeout_s)
    ws = None
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            ws = await websockets.connect(ws_url, max_size=None)
            break
        except Exception as exc:
            last_err = str(exc)
            if "Operation not permitted" in last_err:
                return {"ok": False, "error": "network_denied", "samples": 0}
            await asyncio.sleep(0.25)

    if ws is None:
        return {"ok": False, "error": f"ws_connect_failed:{last_err or 'unknown'}", "samples": 0}

    start = time.time()
    samples = 0
    seen_enabled = False
    max_violations = 0
    p99_samples: List[float] = []

    try:
        while (time.time() - start) < duration_s:
            timeout_s = max(0.2, min(1.0, duration_s - (time.time() - start)))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"ws_recv_failed:{exc}",
                    "samples": samples,
                }

            if not isinstance(msg, str):
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue
            if payload.get("type") != "stats":
                continue
            stats = payload.get("payload") or {}
            pipe = stats.get("pipeline") or {}
            samples += 1
            if bool(pipe.get("zero_copy_core_enabled", False)):
                seen_enabled = True
            try:
                max_violations = max(max_violations, int(pipe.get("zero_copy_violations", 0) or 0))
            except Exception:
                pass
            try:
                p99 = pipe.get("boundary_cpu_serialization_p99_ms")
                if p99 is not None:
                    p99_samples.append(float(p99))
            except Exception:
                pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

    return {
        "ok": True,
        "samples": samples,
        "zero_copy_core_enabled_seen": seen_enabled,
        "max_zero_copy_violations": max_violations,
        "max_boundary_p99_ms": (max(p99_samples) if p99_samples else None),
    }


def _spawn_runtime(
    ws_port: int,
    pipeline_config: Path,
    duration_s: float,
    stub: bool,
    skip_cuda_preflight: bool,
    log_path: Path,
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        str(DS9_ROOT / "noesis" / "ds9_runtime.py"),
        "--pipeline-config",
        str(pipeline_config),
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


def _tail(path: Path, lines: int = 60) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    arr = text.splitlines()
    return "\n".join(arr[-lines:])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-copy soak test via DS9 stats stream")
    p.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket endpoint")
    p.add_argument("--duration", type=float, default=30.0, help="Soak duration seconds")
    p.add_argument("--startup-timeout", type=float, default=10.0, help="WS connect timeout seconds")
    p.add_argument("--max-p99-ms", type=float, default=3.0, help="Boundary serialization p99 budget")
    p.add_argument("--max-violations", type=int, default=0, help="Allowed max zero_copy_violations")
    p.add_argument("--no-spawn", action="store_true", help="Do not spawn runtime; use existing WS")
    p.add_argument("--stub", action="store_true", help="Spawn runtime in stub pipeline mode")
    p.add_argument(
        "--skip-cuda-preflight",
        action="store_true",
        help="Set NOESIS_SKIP_CUDA_PREFLIGHT=1 for spawned runtime",
    )
    p.add_argument(
        "--allow-network-denied",
        action="store_true",
        help="Treat network sandbox denial as a skipped soak (exit 0).",
    )
    p.add_argument(
        "--pipeline-config",
        type=Path,
        default=DS9_ROOT / "config" / "infer.yaml",
        help="Pipeline config path when spawning runtime",
    )
    p.add_argument("--log-path", type=Path, default=None, help="Optional runtime log path")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ws_url = str(args.ws)
    parsed = urlparse(ws_url)
    ws_port = int(parsed.port or 6008)

    proc: Optional[subprocess.Popen[str]] = None
    if args.log_path is not None:
        log_path = args.log_path
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="zero_copy_soak_", suffix=".log", delete=False)
        log_path = Path(tmp.name)
        tmp.close()

    if not args.no_spawn:
        proc = _spawn_runtime(
            ws_port=ws_port,
            pipeline_config=args.pipeline_config,
            duration_s=float(args.duration),
            stub=bool(args.stub),
            skip_cuda_preflight=bool(args.skip_cuda_preflight),
            log_path=log_path,
        )

    try:
        result = asyncio.run(_collect_stats(ws_url, float(args.duration), float(args.startup_timeout)))
        if not bool(result.get("ok", False)):
            if result.get("error") == "network_denied" and bool(args.allow_network_denied):
                print("[SKIP] network denied in environment; soak not runnable here")
                return 0
            print(f"[FAIL] {result.get('error', 'stats_collection_failed')}")
            if proc is not None and proc.poll() is not None:
                print(f"[INFO] runtime exited rc={proc.returncode}")
            if log_path.exists():
                print("--- runtime tail ---")
                print(_tail(log_path))
            return 1

        samples = int(result.get("samples", 0) or 0)
        if samples < 1:
            print("[FAIL] no stats samples observed")
            if log_path.exists():
                print("--- runtime tail ---")
                print(_tail(log_path))
            return 1

        if not bool(result.get("zero_copy_core_enabled_seen", False)):
            print("[FAIL] zero_copy_core_enabled was never observed true")
            return 1

        max_viol = int(result.get("max_zero_copy_violations", 0) or 0)
        if max_viol > int(args.max_violations):
            print(f"[FAIL] max_zero_copy_violations={max_viol} exceeds limit={int(args.max_violations)}")
            return 1

        max_p99 = result.get("max_boundary_p99_ms")
        if max_p99 is not None and float(max_p99) > float(args.max_p99_ms):
            print(f"[FAIL] max_boundary_p99_ms={float(max_p99):.3f} exceeds limit={float(args.max_p99_ms):.3f}")
            return 1

        print(
            "[PASS] zero-copy soak gate",
            f"samples={samples}",
            f"max_zero_copy_violations={max_viol}",
            f"max_boundary_p99_ms={max_p99}",
        )
        return 0
    finally:
        if proc is not None:
            _terminate_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
