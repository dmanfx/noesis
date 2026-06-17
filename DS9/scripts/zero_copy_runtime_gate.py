#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
for _path in (str(DS9_ROOT), str(REPO_ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DS9_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-copy runtime gate (pre-soak)")
    p.add_argument("--pipeline-config", type=Path, default=DS9_ROOT / "config" / "infer.yaml")
    p.add_argument("--cameras-config", type=Path, default=REPO_ROOT / "config" / "cameras.yaml")
    p.add_argument("--duration-s", type=float, default=1200.0)
    p.add_argument("--stats-ws", default="ws://127.0.0.1:6008")
    p.add_argument("--rest-url", default="http://127.0.0.1:8080/api/v1/depth/refresh")
    p.add_argument(
        "--depth-camera",
        default="__zero_copy_probe__",
        help="Camera id used for WS depth-path traffic.",
    )
    p.add_argument(
        "--depth-burst-interval-s",
        type=float,
        default=15.0,
        help="Interval between depth refresh bursts via REST.",
    )
    p.add_argument("--depth-refresh-seconds", type=int, default=20)
    p.add_argument("--startup-timeout", type=float, default=25.0)
    p.add_argument("--max-p99-ms", type=float, default=3.0)
    p.add_argument("--max-violations", type=int, default=0)
    p.add_argument("--no-spawn", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--skip-cuda-preflight", action="store_true")
    p.add_argument("--allow-network-denied", action="store_true")
    p.add_argument("--log-path", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cmd = [
        sys.executable,
        str(DS9_ROOT / "scripts" / "zero_copy_smoke_test.py"),
        "--pipeline-config",
        str(args.pipeline_config),
        "--cameras-config",
        str(args.cameras_config),
        "--duration-s",
        str(float(args.duration_s)),
        "--stats-ws",
        str(args.stats_ws),
        "--rest-url",
        str(args.rest_url),
        "--depth-camera",
        str(args.depth_camera),
        "--rest-refresh-seconds",
        str(int(args.depth_refresh_seconds)),
        "--rest-refresh-interval-s",
        str(float(args.depth_burst_interval_s)),
        "--startup-timeout",
        str(float(args.startup_timeout)),
        "--max-p99-ms",
        str(float(args.max_p99_ms)),
        "--max-violations",
        str(int(args.max_violations)),
    ]

    if args.no_spawn:
        cmd.append("--no-spawn")
    if args.stub:
        cmd.append("--stub")
    if args.skip_cuda_preflight:
        cmd.append("--skip-cuda-preflight")
    if args.allow_network_denied:
        cmd.append("--allow-network-denied")
    if args.log_path is not None:
        cmd.extend(["--log-path", str(args.log_path)])

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.returncode != 0 and proc.stderr:
        print(proc.stderr.strip(), file=sys.stderr)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
