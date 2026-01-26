#!/usr/bin/env python3
"""Regression test for the SV3DT OOM-kill issue (host RAM leak).

This script spawns `noesis/ds8_runtime.py` and monitors RSS via /proc. It is meant
to be run headless and without any WebSocket clients connected.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        return -1
    return -1


def _spawn_runtime(args: argparse.Namespace) -> subprocess.Popen[str]:
    ws_port = int(args.ws_port) if args.ws_port > 0 else _pick_free_port()
    cmd = [
        sys.executable,
        "noesis/ds8_runtime.py",
        "--pipeline-config",
        args.pipeline_config,
        "--cameras-config",
        args.cameras_config,
        "--disable-rest",
        "--ws-port",
        str(ws_port),
        "--log-level",
        args.log_level,
    ]
    env = os.environ.copy()
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    # Keep signal clean; these do not fix SV3DT leaks but remove DS8-side noise sources.
    env.setdefault("NOESIS_V3DT_OBJ3D_CACHE", "0")
    env.setdefault("NOESIS_V3DT_META_EXTRACT", "0")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect runaway RSS growth (SV3DT OOM regression)")
    parser.add_argument("--pipeline-config", default="config/infer_v3dt.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument("--ws-port", type=int, default=0)
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--sample-s", type=float, default=1.0)
    parser.add_argument("--max-rss-mib", type=float, default=4096.0)
    parser.add_argument("--max-delta-mib", type=float, default=2048.0)
    parser.add_argument("--no-spawn", action="store_true", help="Reserved for future (not implemented)")
    args = parser.parse_args()

    if args.no_spawn:
        print("[FAIL] --no-spawn is not implemented for this script (it only validates spawned runtimes).")
        return 2

    cfg = Path(args.pipeline_config)
    if not cfg.exists():
        print(f"[FAIL] pipeline config not found: {cfg}")
        return 2

    proc: subprocess.Popen[str] | None = None
    try:
        proc = _spawn_runtime(args)

        # Drain startup output until the pipeline is running (or the process exits).
        start = time.time()
        ready = False
        while time.time() - start < 45.0:
            if proc.poll() is not None:
                break
            line = proc.stdout.readline() if proc.stdout is not None else ""
            if not line:
                continue
            if "Main Loop Running" in line:
                ready = True
                break

        if proc.poll() is not None:
            print(f"[FAIL] runtime exited during startup (code={proc.returncode})")
            return 1
        if not ready:
            print("[FAIL] runtime did not reach 'Main Loop Running' within 45s")
            return 1

        # Monitor RSS for the configured duration.
        max_rss_kb = int(float(args.max_rss_mib) * 1024.0)
        max_delta_kb = int(float(args.max_delta_mib) * 1024.0)
        sample_s = max(0.25, float(args.sample_s))
        duration_s = max(1.0, float(args.duration_s))

        rss0 = _rss_kb(proc.pid)
        if rss0 <= 0:
            print("[FAIL] unable to read initial RSS")
            return 1
        rss_max = rss0
        deadline = time.time() + duration_s

        while time.time() < deadline:
            if proc.poll() is not None:
                print(f"[FAIL] runtime exited early during monitoring (code={proc.returncode})")
                return 1
            rss = _rss_kb(proc.pid)
            if rss > rss_max:
                rss_max = rss
            delta = rss - rss0
            if rss >= max_rss_kb or delta >= max_delta_kb:
                print(
                    "[FAIL] runaway RSS detected "
                    f"(rss={rss/1024.0:.1f}MiB delta={delta/1024.0:.1f}MiB "
                    f"max_rss={args.max_rss_mib:.0f}MiB max_delta={args.max_delta_mib:.0f}MiB)"
                )
                return 1
            time.sleep(sample_s)

        print(
            "[PASS] RSS stayed bounded "
            f"(rss0={rss0/1024.0:.1f}MiB rss_max={rss_max/1024.0:.1f}MiB duration={duration_s:.0f}s)"
        )
        return 0
    finally:
        if proc is not None:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                proc.wait(timeout=8.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
