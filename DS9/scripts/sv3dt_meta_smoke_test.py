#!/usr/bin/env python3
"""Smoke test to verify SV3DT/MV3DT 3D meta appears in tracking telemetry."""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[FAIL] websockets package required: {exc}")
    sys.exit(1)

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent


EXPECTED_BBOX3D_KEYS = {
    "xCentre",
    "yCentre",
    "zCentre",
    "xLen",
    "yLen",
    "zLen",
    "xRot",
    "yRot",
    "zRot",
}


def _has_bbox3d(track: dict) -> bool:
    bbox3d = track.get("bbox3d")
    if not isinstance(bbox3d, dict):
        return False
    keys = set(bbox3d.keys())
    return EXPECTED_BBOX3D_KEYS.issubset(keys)


async def _collect_bbox3d(uri: str, duration: float = 12.0) -> bool:
    start = time.time()
    tracking_msgs = 0
    async with websockets.connect(uri) as ws:
        while time.time() - start < duration:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "tracking":
                continue
            tracking_msgs += 1
            tracks = payload.get("tracks") or []
            for track in tracks:
                if _has_bbox3d(track):
                    return True
    if tracking_msgs == 0:
        raise RuntimeError("No tracking messages observed (no people/tracks or telemetry hook not running).")
    return False


def _spawn_runtime(args: argparse.Namespace) -> subprocess.Popen[str]:
    ws_port = int(args.ws_port)
    cmd = [
        sys.executable,
        str(DS9_ROOT / "noesis" / "ds9_runtime.py"),
        "--disable-rest",
        "--ws-host",
        "127.0.0.1",
        "--ws-port",
        str(ws_port),
    ]
    if args.tracking_mode:
        cmd.extend(["--tracking-mode", args.tracking_mode])
    if args.pipeline_config:
        cmd.extend(["--pipeline-config", args.pipeline_config])
    if args.cameras_config:
        cmd.extend(["--cameras-config", args.cameras_config])
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.PIPE,  # keep runtime stdin open; it will self-terminate on EOF
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def _drain_stdout(proc: subprocess.Popen[str], sink: deque[str]) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        sink.append(line.rstrip("\n"))


async def _probe_ws(uri: str) -> None:
    async with websockets.connect(uri, open_timeout=2.0):
        return


def _wait_for_ws_ready(uri: str, timeout_s: float, proc: subprocess.Popen[str] | None = None) -> bool:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            asyncio.run(_probe_ws(uri))
            return True
        except Exception as exc:
            last_err = exc
            time.sleep(0.25)
    if last_err is not None:
        print(f"[FAIL] timed out waiting for WebSocket handshake (last_err={last_err}).")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="SV3DT/MV3DT bbox3d smoke test")
    parser.add_argument("--ws", default="", help="WebSocket URL (optional; auto-picked when spawning)")
    parser.add_argument("--ws-port", type=int, default=0, help="WebSocket port (used when spawning)")
    parser.add_argument("--pipeline-config", default=None)
    parser.add_argument("--cameras-config", default=None)
    parser.add_argument("--tracking-mode", choices=("v3dt",), default=None)
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn runtime")
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()

    if not args.no_spawn and args.pipeline_config is None:
        print(
            "[FAIL] DS9-native V3DT pipeline config is not staged; pass --pipeline-config "
            "for an explicit DS9 V3DT config or use --no-spawn against an external DS9 runtime."
        )
        return 1
    if args.cameras_config is None:
        args.cameras_config = str(REPO_ROOT / "config" / "cameras.yaml")

    proc = None
    log_tail: deque[str] = deque(maxlen=200)
    drain_thread: threading.Thread | None = None
    if args.ws_port <= 0 and not args.no_spawn:
        args.ws_port = _pick_free_port()
    if not args.ws:
        args.ws = f"ws://127.0.0.1:{int(args.ws_port) if args.ws_port else 6008}"
    if not args.no_spawn:
        proc = _spawn_runtime(args)
        drain_thread = threading.Thread(target=_drain_stdout, args=(proc, log_tail), daemon=True)
        drain_thread.start()
        ready = _wait_for_ws_ready(args.ws, timeout_s=120.0, proc=proc)
        if not ready:
            if proc.poll() is not None:
                print(f"[FAIL] runtime exited before WS became ready (code={proc.returncode}).")
            else:
                print("[FAIL] timed out waiting for WebSocket server to accept connections.")
            if log_tail:
                print("--- ds9_runtime tail ---")
                print("\n".join(log_tail))
            return 1

    try:
        ok = False
        last_err = None
        for _ in range(8):
            try:
                ok = asyncio.run(_collect_bbox3d(args.ws, duration=args.duration))
                if ok:
                    last_err = None
                    break
                last_err = RuntimeError("Tracking telemetry observed but bbox3d was missing from all tracks.")
            except Exception as exc:
                last_err = exc
            time.sleep(1.0)
        if not ok:
            hint = ""
            if "No tracking messages observed" in str(last_err):
                hint = (
                    " Ensure a person is visible to the camera(s), or pass an explicit "
                    "DS9 V3DT pipeline config once those assets are staged."
                )
            exit_note = ""
            if proc is not None and proc.poll() is not None:
                exit_note = f" runtime_exited(code={proc.returncode})."
            print(f"[FAIL] bbox3d not observed (last_err={last_err}).{exit_note}{hint}")
            if log_tail:
                print("--- ds9_runtime tail ---")
                print("\n".join(log_tail))
            return 1
        print("[PASS] bbox3d observed in tracking telemetry")
        return 0
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
