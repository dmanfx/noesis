#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ERROR_SIGNATURES = (
    "Fatal Python error",
    "Segmentation fault",
    "SIGSEGV",
    "Traceback",
    "CRITICAL",
    "ERROR",
)


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=False))


def _reserve_auto_port() -> tuple[int, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    return port, sock


def _reserve_requested_port(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock


def _scan_log(log_path: Path) -> tuple[list[str], list[str]]:
    found: set[str] = set()
    samples: list[str] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return [], []

    for line in lines:
        for sig in ERROR_SIGNATURES:
            if sig in line:
                found.add(sig)
                if len(samples) < 8:
                    samples.append(line.rstrip())

    if not samples and lines:
        tail = [ln.rstrip() for ln in lines[-8:] if ln.strip()]
        samples.extend(tail)

    return sorted(found), samples


def _stop_process(proc: subprocess.Popen[str], timeout_s: float = 5.0) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic 30s DS8 runtime smoke gate")
    parser.add_argument("--duration-s", type=float, default=30.0, help="Gate runtime duration in seconds")
    parser.add_argument("--ws-port", type=int, default=None, help="Requested websocket port")
    parser.add_argument("--log-path", type=Path, default=None, help="Path to runtime combined stdout/stderr log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.duration_s <= 0:
        _json_print(
            {
                "ok": False,
                "duration_s": args.duration_s,
                "ws_port": args.ws_port,
                "log_path": str(args.log_path) if args.log_path else "",
                "process_exit": None,
                "signatures_found": [],
                "notes": "duration must be > 0",
            }
        )
        return 2

    if args.log_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="ds8_runtime_gate_", suffix=".log", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        log_path = tmp_path
    else:
        log_path = args.log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)

    lock_sock: socket.socket | None = None
    selected_port: int | None = None

    try:
        if args.ws_port is None:
            selected_port, lock_sock = _reserve_auto_port()
        else:
            selected_port = args.ws_port
            lock_sock = _reserve_requested_port(selected_port)
    except OSError as exc:
        _json_print(
            {
                "ok": False,
                "duration_s": args.duration_s,
                "ws_port": args.ws_port,
                "log_path": str(log_path),
                "process_exit": None,
                "signatures_found": [],
                "notes": f"requested ws port unavailable: {exc}",
            }
        )
        return 2

    cmd = [
        "python3",
        "-X",
        "faulthandler",
        "noesis/ds8_runtime.py",
        "--pipeline-config",
        "config/infer.yaml",
        "--disable-rest",
        "--log-level",
        "INFO",
        "--ws-port",
        str(selected_port),
    ]

    proc: subprocess.Popen[str] | None = None
    final_exit: int | None = None
    notes = ""
    start = time.monotonic()

    try:
        with log_path.open("w", encoding="utf-8") as logf:
            # Release the socket lock just before spawn to minimize race while still validating availability.
            if lock_sock is not None:
                lock_sock.close()
                lock_sock = None

            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            deadline = start + args.duration_s
            while True:
                now = time.monotonic()
                if now >= deadline:
                    break
                code = proc.poll()
                if code is not None:
                    final_exit = code
                    notes = "process exited before duration"
                    break
                sleep_s = min(0.25, max(0.0, deadline - now))
                time.sleep(sleep_s)

            if final_exit is None:
                final_exit = proc.poll()

    except KeyboardInterrupt:
        notes = "interrupted"
    finally:
        if lock_sock is not None:
            lock_sock.close()
        if proc is not None:
            final_exit = _stop_process(proc)

    elapsed = max(0.0, time.monotonic() - start)
    signatures_found, samples = _scan_log(log_path)

    stayed_alive = bool(proc is not None and notes != "process exited before duration" and elapsed >= args.duration_s)
    ok = stayed_alive and not signatures_found

    if not notes:
        if not stayed_alive:
            notes = "process did not stay alive for full duration"
        elif signatures_found:
            notes = "error signatures found in log"
        else:
            notes = "passed"

    payload: dict[str, Any] = {
        "ok": ok,
        "duration_s": round(elapsed, 3),
        "ws_port": selected_port,
        "log_path": str(log_path),
        "process_exit": final_exit,
        "signatures_found": signatures_found,
        "notes": notes,
    }

    if samples:
        payload["samples"] = samples[:8]

    _json_print(payload)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
