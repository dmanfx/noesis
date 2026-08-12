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
from collections.abc import Awaitable, Callable
from pathlib import Path

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
for _path in (str(DS9_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from noesis.v3dt_assets import V3DTAssetError, validate_v3dt_assets  # noqa: E402
from noesis_core.v3dt_validation import (  # noqa: E402
    V3DTBBoxTimeoutContract,
    V3DT_BBOX_DEFAULT_ATTEMPTS,
)
from scripts.internal_auth_client import (  # noqa: E402
    RequiredInternalAuth,
    add_auth_token_file_argument,
    configure_required_auth_environment,
    connect_required_websocket,
    load_required_internal_auth,
)


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


async def _collect_bbox3d(
    uri: str,
    auth: RequiredInternalAuth,
    duration: float = 12.0,
) -> bool:
    deadline = time.monotonic() + float(duration)
    tracking_msgs = 0
    async with connect_required_websocket(uri, auth, max_size=None) as ws:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=min(2.0, remaining))
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


async def _run_bbox3d_attempts(
    uri: str,
    auth: RequiredInternalAuth,
    contract: V3DTBBoxTimeoutContract,
    *,
    collect: Callable[..., Awaitable[bool]] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> tuple[bool, Exception | None]:
    """Run the exact bounded retry schedule used by the outer runner."""

    collector = _collect_bbox3d if collect is None else collect
    sleeper = asyncio.sleep if sleep is None else sleep
    last_error: Exception | None = None
    for attempt in range(contract.attempts):
        try:
            ok = await collector(
                uri,
                auth,
                duration=contract.duration_seconds,
            )
        except Exception as exc:
            last_error = exc
        else:
            if ok:
                return True, None
            last_error = RuntimeError(
                "Tracking telemetry observed but bbox3d was missing from all tracks."
            )
        if attempt + 1 < contract.attempts:
            await sleeper(contract.retry_delay_seconds)
    return False, last_error


def _spawn_runtime(
    args: argparse.Namespace,
    auth: RequiredInternalAuth,
) -> subprocess.Popen[str]:
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
    configure_required_auth_environment(env, auth)
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


async def _probe_ws(uri: str, auth: RequiredInternalAuth) -> None:
    async with connect_required_websocket(uri, auth, open_timeout=2.0):
        return


def _wait_for_ws_ready(
    uri: str,
    auth: RequiredInternalAuth,
    timeout_s: float,
    proc: subprocess.Popen[str] | None = None,
) -> bool:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            asyncio.run(_probe_ws(uri, auth))
            return True
        except Exception as exc:
            last_err = exc
            time.sleep(0.25)
    if last_err is not None:
        print(f"[FAIL] timed out waiting for WebSocket handshake (last_err={last_err}).")
    return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SV3DT/MV3DT bbox3d smoke test")
    parser.add_argument("--ws", default="", help="WebSocket URL (optional; auto-picked when spawning)")
    parser.add_argument("--ws-port", type=int, default=0, help="WebSocket port (used when spawning)")
    parser.add_argument(
        "--pipeline-config", default=str(DS9_ROOT / "config" / "infer_v3dt.yaml")
    )
    parser.add_argument(
        "--cameras-config", default=str(DS9_ROOT / "config" / "cameras_v3dt.yaml")
    )
    parser.add_argument("--tracking-mode", choices=("v3dt",), default="v3dt")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn runtime")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument(
        "--attempts",
        type=int,
        default=V3DT_BBOX_DEFAULT_ATTEMPTS,
        help="Bounded bbox3d collection attempts; retry timing is shared with the runner.",
    )
    add_auth_token_file_argument(parser)
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()

    try:
        timeout_contract = V3DTBBoxTimeoutContract(
            duration_seconds=args.duration,
            attempts=args.attempts,
        )
    except (TypeError, ValueError) as exc:
        print(f"[FAIL] invalid bbox3d timeout contract: {exc}")
        return 2

    try:
        auth = load_required_internal_auth(args.auth_token_file)
    except Exception as exc:
        print(f"[FAIL] required internal auth unavailable: {exc}")
        return 1

    if not args.no_spawn:
        try:
            validate_v3dt_assets(
                Path(args.pipeline_config),
                cameras_config=Path(args.cameras_config),
                require_engines=True,
            )
        except (OSError, V3DTAssetError) as exc:
            print(f"[FAIL] {exc}")
            return 1

    proc = None
    log_tail: deque[str] = deque(maxlen=200)
    drain_thread: threading.Thread | None = None
    if args.ws_port <= 0 and not args.no_spawn:
        args.ws_port = _pick_free_port()
    if not args.ws:
        args.ws = f"ws://127.0.0.1:{int(args.ws_port) if args.ws_port else 6008}"
    if not args.no_spawn:
        proc = _spawn_runtime(args, auth)
        drain_thread = threading.Thread(target=_drain_stdout, args=(proc, log_tail), daemon=True)
        drain_thread.start()
        ready = _wait_for_ws_ready(args.ws, auth, timeout_s=120.0, proc=proc)
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
        ok, last_err = asyncio.run(
            _run_bbox3d_attempts(
                args.ws,
                auth,
                timeout_contract,
            )
        )
        if not ok:
            hint = ""
            if "No tracking messages observed" in str(last_err):
                hint = (
                    " Ensure a person is visible to the DS9 V3DT camera sources."
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
                proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
