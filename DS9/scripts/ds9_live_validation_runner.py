#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent

GATE_NAMES = (
    "rtsp",
    "webrtc",
    "reid",
    "bev",
    "bridge",
    "floorplan",
    "ma-depth",
    "zero-copy-stats",
    "zero-copy-rest",
)

NATIVE_CRASH_SIGNATURES = (
    "Fatal Python error",
    "Segmentation fault",
    "SIGSEGV",
    "malloc():",
    "double free",
    "corrupted size",
    "corrupted double-linked list",
    "terminate called without an active exception",
    "Aborted (core dumped)",
)

RTSP_ERROR_SIGNATURES = (
    "ERROR:",
    "Could not",
    "not-linked",
    "Internal data stream error",
    "streaming stopped",
    "Connection refused",
    "No route to host",
    "Service Unavailable",
)


@dataclass
class StepResult:
    name: str
    ok: bool
    returncode: int | None
    duration_s: float
    log_path: str
    command: str
    notes: str = ""
    timed_out: bool = False
    signatures: list[str] | None = None


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _cmd_text(cmd: Sequence[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def _normalize_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _find_signatures(text: str, signatures: Iterable[str]) -> list[str]:
    found = []
    for signature in signatures:
        if signature in text:
            found.append(signature)
    return found


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_command(
    *,
    name: str,
    cmd: Sequence[str],
    output_dir: Path,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    timeout_ok: bool = False,
    timeout_ok_note: str = "bounded timeout reached",
    fail_signatures: Sequence[str] = (),
) -> StepResult:
    start = time.monotonic()
    log_path = output_dir / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _cmd_text(cmd)
    text = ""
    timed_out = False
    returncode: int | None = None
    notes = ""

    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=REPO_ROOT,
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        returncode = proc.returncode
        text = proc.stdout or ""
        ok = returncode == 0
        notes = "exit_0" if ok else f"exit_{returncode}"
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        text = _normalize_text(exc.stdout) + _normalize_text(exc.stderr)
        ok = bool(timeout_ok)
        notes = timeout_ok_note if ok else f"timed out after {timeout_s:.1f}s"
    except FileNotFoundError as exc:
        returncode = None
        text = f"{type(exc).__name__}: {exc}\n"
        ok = False
        notes = "command not found"
    except Exception as exc:
        returncode = None
        text = f"{type(exc).__name__}: {exc}\n"
        ok = False
        notes = "command failed before completion"

    log_path.write_text(text, encoding="utf-8", errors="replace")

    signatures = _find_signatures(text, fail_signatures)
    if signatures:
        ok = False
        if notes:
            notes = f"{notes}; signatures={','.join(signatures)}"
        else:
            notes = f"signatures={','.join(signatures)}"

    return StepResult(
        name=name,
        ok=ok,
        returncode=returncode,
        duration_s=round(time.monotonic() - start, 3),
        log_path=_repo_rel(log_path),
        command=command,
        notes=notes,
        timed_out=timed_out,
        signatures=signatures,
    )


def _parse_host_port(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    return host, int(port)


def _wait_for_tcp(host: str, port: int, timeout_s: float, interval_s: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(interval_s)
    return False


def _runtime_env(extra_env: Sequence[str]) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "1")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "1")
    env.setdefault("NOESIS_REID_ENABLED", "1")
    env.setdefault("NOESIS_POSE_FEATURE_DEBUG", "1")
    for item in extra_env:
        if "=" not in item:
            raise SystemExit(f"--env values must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--env values must include a non-empty key, got: {item}")
        env[key] = value
    return env


def _spawn_runtime(args: argparse.Namespace, output_dir: Path) -> tuple[subprocess.Popen[None], Path]:
    runtime_log = output_dir / "runtime.log"
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    runtime_cmd = [
        sys.executable,
        "DS9/noesis/ds9_runtime.py",
        "--pipeline-config",
        str(args.pipeline_config),
        "--cameras-config",
        str(args.cameras_config),
        "--enable-rest",
    ]
    runtime_log.write_text(f"$ {_cmd_text(runtime_cmd)}\n\n", encoding="utf-8")
    log_fh = runtime_log.open("a", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            runtime_cmd,
            cwd=REPO_ROOT,
            env=_runtime_env(args.env),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log_fh.close()
    return proc, runtime_log


def _runtime_ready(args: argparse.Namespace, proc: subprocess.Popen[None] | None) -> tuple[bool, str]:
    checks = [
        ("ws", *_parse_host_port(args.ws, 80)),
        ("rest", *_parse_host_port(args.rest, 80)),
        ("rtsp", *_parse_host_port(args.rtsp_url, 554)),
    ]
    ready: set[str] = set()
    deadline = time.monotonic() + float(args.startup_timeout_s)
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            pending = [label for label, _host, _port in checks if label not in ready]
            return False, f"runtime exited before readiness (rc={proc.returncode}, pending={','.join(pending)})"
        for label, host, port in checks:
            if label in ready:
                continue
            if _wait_for_tcp(str(host), int(port), timeout_s=0.25, interval_s=0.05):
                ready.add(label)
        if len(ready) == len(checks):
            break
        time.sleep(0.25)
    missing = [f"{label} {host}:{port}" for label, host, port in checks if label not in ready]
    if missing:
        return False, "not ready: " + ", ".join(missing)
    return True, "ready"


def _shutdown_runtime(
    proc: subprocess.Popen[None] | None,
    runtime_log: Path | None,
    timeout_s: float,
) -> StepResult:
    start = time.monotonic()
    command = "SIGINT DS9 runtime"
    if proc is None:
        return StepResult(
            name="shutdown",
            ok=True,
            returncode=None,
            duration_s=0.0,
            log_path=_repo_rel(runtime_log) if runtime_log else "",
            command=command,
            notes="skipped: --no-spawn",
        )

    notes = ""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        except Exception as exc:
            notes = f"SIGINT failed: {exc}"
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            notes = f"{notes}; shutdown timeout, sent SIGTERM".strip("; ")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                notes = f"{notes}; SIGTERM timeout, sent SIGKILL"
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10.0)

    text = ""
    if runtime_log and runtime_log.exists():
        text = runtime_log.read_text(encoding="utf-8", errors="replace")
    signatures = _find_signatures(text, NATIVE_CRASH_SIGNATURES)
    wait_warning = "Wait thread did not terminate cleanly" in text
    if wait_warning:
        notes = f"{notes}; wait-thread warning observed".strip("; ")
    if not notes:
        notes = "exit_0" if proc.returncode == 0 else f"exit_{proc.returncode}"
    ok = proc.returncode == 0 and not signatures

    return StepResult(
        name="shutdown",
        ok=ok,
        returncode=proc.returncode,
        duration_s=round(time.monotonic() - start, 3),
        log_path=_repo_rel(runtime_log) if runtime_log else "",
        command=command,
        notes=notes,
        signatures=signatures,
    )


def _build_gates(args: argparse.Namespace) -> list[tuple[str, list[str], float | None, bool, Sequence[str]]]:
    py = sys.executable
    pcfg = str(args.pipeline_config)
    ccfg = str(args.cameras_config)
    rest_refresh_url = args.rest.rstrip("/") + "/api/v1/depth/refresh"
    return [
        (
            "rtsp",
            [
                "gst-launch-1.0",
                "-e",
                "rtspsrc",
                f"location={args.rtsp_url}",
                "latency=100",
                "!",
                "rtph264depay",
                "!",
                "h264parse",
                "!",
                "avdec_h264",
                "!",
                "fakesink",
                "sync=false",
            ],
            args.rtsp_duration_s,
            True,
            RTSP_ERROR_SIGNATURES,
        ),
        (
            "webrtc",
            [
                py,
                "scripts/webrtc_gateway_smoke_test.py",
                "--ws",
                args.ws,
                "--duration",
                str(args.webrtc_duration_s),
                "--pt",
                str(args.webrtc_pt),
                "--min-rtp",
                str(args.webrtc_min_rtp),
                "--min-decoded",
                str(args.webrtc_min_decoded),
            ],
            args.webrtc_duration_s + 25.0,
            False,
            (),
        ),
        (
            "reid",
            [
                py,
                "scripts/reid_stable_id_smoke_test.py",
                "--no-spawn",
                "--ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--duration",
                str(args.reid_duration_s),
            ],
            args.reid_duration_s + 25.0,
            False,
            (),
        ),
        (
            "bev",
            [
                py,
                "scripts/menon_bev_track_parity_smoke_test.py",
                "--no-spawn",
                "--ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--duration",
                str(args.bev_duration_s),
            ],
            args.bev_duration_s + 25.0,
            False,
            (),
        ),
        (
            "bridge",
            [
                py,
                "DS9/scripts/ds9_bridge_contract_smoke_test.py",
                "--ws",
                args.ws,
                "--duration",
                str(args.bridge_duration_s),
                "--require-embedding-track",
            ],
            args.bridge_duration_s + 30.0,
            False,
            (),
        ),
        (
            "floorplan",
            [
                py,
                "scripts/floorplan_rpc_smoke_test.py",
                "--no-spawn",
                "--ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--max-age-sec",
                str(args.floorplan_max_age_sec),
            ],
            45.0,
            False,
            (),
        ),
        (
            "ma-depth",
            [
                py,
                "scripts/ma_depth_rpc_smoke_test.py",
                "--no-spawn",
                "--ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--camera",
                args.ma_camera,
            ],
            90.0,
            False,
            (),
        ),
        (
            "zero-copy-stats",
            [
                py,
                "scripts/zero_copy_stats_smoke_test.py",
                "--no-spawn",
                "--stats-ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--duration-s",
                str(args.zero_copy_stats_duration_s),
            ],
            args.zero_copy_stats_duration_s + 30.0,
            False,
            (),
        ),
        (
            "zero-copy-rest",
            [
                py,
                "scripts/zero_copy_smoke_test.py",
                "--no-spawn",
                "--stats-ws",
                args.ws,
                "--rest-url",
                rest_refresh_url,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--duration-s",
                str(args.zero_copy_rest_duration_s),
            ],
            args.zero_copy_rest_duration_s + 40.0,
            False,
            (),
        ),
    ]


def _write_report(output_dir: Path, summary: Mapping[str, object]) -> Path:
    report_path = output_dir / "summary.md"
    lines = [
        "# DS9 Live Validation Summary",
        "",
        f"- ok: `{summary.get('ok')}`",
        f"- started_at_utc: `{summary.get('started_at_utc')}`",
        f"- output_dir: `{summary.get('output_dir')}`",
        f"- pipeline_config: `{summary.get('pipeline_config')}`",
        f"- cameras_config: `{summary.get('cameras_config')}`",
        f"- runtime_log: `{summary.get('runtime_log')}`",
        "",
        "| Step | OK | RC | Seconds | Notes | Log |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for result in summary.get("results", []):
        if not isinstance(result, Mapping):
            continue
        lines.append(
            "| {name} | `{ok}` | `{returncode}` | `{duration_s}` | {notes} | `{log_path}` |".format(
                name=result.get("name"),
                ok=result.get("ok"),
                returncode=result.get("returncode"),
                duration_s=result.get("duration_s"),
                notes=str(result.get("notes") or "").replace("|", "\\|"),
                log_path=result.get("log_path"),
            )
        )
    lines.append("")
    lines.append("Use the per-step logs above as the evidence bundle for doc updates.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DS9 live-RTSP production validation bundle.")
    parser.add_argument("--pipeline-config", type=Path, default=Path("DS9/config/infer.yaml"))
    parser.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--rest", default="http://127.0.0.1:8080")
    parser.add_argument("--rtsp-url", default="rtsp://127.0.0.1:8554/mosaic")
    parser.add_argument("--startup-timeout-s", type=float, default=90.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=35.0)
    parser.add_argument("--no-spawn", action="store_true", help="Attach to an already-running DS9 runtime.")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip", action="append", choices=GATE_NAMES, default=[], help="Skip a gate; may be repeated.")
    parser.add_argument("--env", action="append", default=[], help="Extra runtime env KEY=VALUE; may be repeated.")
    parser.add_argument("--rtsp-duration-s", type=float, default=20.0)
    parser.add_argument("--webrtc-duration-s", type=float, default=6.0)
    parser.add_argument("--webrtc-pt", type=int, default=103)
    parser.add_argument("--webrtc-min-rtp", type=int, default=10)
    parser.add_argument("--webrtc-min-decoded", type=int, default=1)
    parser.add_argument("--reid-duration-s", type=float, default=35.0)
    parser.add_argument("--bev-duration-s", type=float, default=45.0)
    parser.add_argument("--bridge-duration-s", type=float, default=75.0)
    parser.add_argument("--floorplan-max-age-sec", type=float, default=1200.0)
    parser.add_argument("--ma-camera", default="family-room")
    parser.add_argument("--zero-copy-stats-duration-s", type=float, default=60.0)
    parser.add_argument("--zero-copy-rest-duration-s", type=float, default=90.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (DS9_ROOT / "build" / "live_validation" / _timestamp())
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[StepResult] = []
    runtime_proc: subprocess.Popen[None] | None = None
    runtime_log: Path | None = None
    started_at = _timestamp()

    try:
        if not args.skip_preflight:
            result = _run_command(
                name="preflight",
                cmd=[sys.executable, "DS9/scripts/ds9_preflight.py", "--config", str(args.pipeline_config)],
                output_dir=output_dir,
                timeout_s=60.0,
            )
            results.append(result)
            if not result.ok:
                raise SystemExit(1)

        if args.no_spawn:
            runtime_log = None
        else:
            runtime_proc, runtime_log = _spawn_runtime(args, output_dir)

        ready, ready_note = _runtime_ready(args, runtime_proc)
        results.append(
            StepResult(
                name="runtime-ready",
                ok=ready,
                returncode=runtime_proc.returncode if runtime_proc and runtime_proc.poll() is not None else None,
                duration_s=0.0,
                log_path=_repo_rel(runtime_log) if runtime_log else "",
                command="tcp readiness checks for WS/REST/RTSP",
                notes=ready_note,
            )
        )
        if not ready:
            raise SystemExit(1)

        skips = set(args.skip or [])
        for name, cmd, timeout_s, timeout_ok, signatures in _build_gates(args):
            if name in skips:
                results.append(
                    StepResult(
                        name=name,
                        ok=True,
                        returncode=None,
                        duration_s=0.0,
                        log_path="",
                        command=_cmd_text(cmd),
                        notes="skipped",
                    )
                )
                continue
            result = _run_command(
                name=name,
                cmd=cmd,
                output_dir=output_dir,
                timeout_s=timeout_s,
                timeout_ok=timeout_ok,
                fail_signatures=signatures,
            )
            results.append(result)
            if args.fail_fast and not result.ok:
                break
    except SystemExit:
        pass
    finally:
        results.append(_shutdown_runtime(runtime_proc, runtime_log, args.shutdown_timeout_s))

    summary = {
        "ok": all(result.ok for result in results),
        "started_at_utc": started_at,
        "output_dir": _repo_rel(output_dir),
        "pipeline_config": str(args.pipeline_config),
        "cameras_config": str(args.cameras_config),
        "ws": args.ws,
        "rest": args.rest,
        "rtsp_url": args.rtsp_url,
        "runtime_log": _repo_rel(runtime_log) if runtime_log else "",
        "results": [asdict(result) for result in results],
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    report_path = _write_report(output_dir, summary)
    print(json.dumps({"ok": summary["ok"], "summary": _repo_rel(summary_path), "report": _repo_rel(report_path)}, indent=2))
    return 0 if bool(summary["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
