#!/usr/bin/env python3
"""Capture and validate private guided-walk alignment evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.validation.alignment_walk import (  # noqa: E402
    AlignmentWalkCapture,
    append_waypoint_marker,
    create_run_directory,
    default_output_root,
    verify_run,
    write_alignment_report,
    write_waypoint_calibration_report,
)
from scripts.internal_auth_client import (  # noqa: E402
    add_auth_token_file_argument,
    connect_required_websocket,
    load_required_internal_auth,
)


def _default_run_id() -> str:
    return time.strftime("alignment_walk_%Y%m%dT%H%M%SZ", time.gmtime())


async def _capture(args: argparse.Namespace, capture: AlignmentWalkCapture) -> None:
    auth = load_required_internal_auth(args.auth_token_file)
    duration_s = max(1.0, float(args.duration))
    async with connect_required_websocket(
        str(args.ws),
        auth,
        max_size=None,
        open_timeout=float(args.connect_timeout),
        close_timeout=5.0,
        ping_interval=20.0,
        ping_timeout=20.0,
    ) as websocket:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            timeout_s = min(2.0, max(0.05, deadline - time.monotonic()))
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                continue
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, Mapping):
                continue
            capture.process_message(
                message,
                received_at_us=time.time_ns() // 1_000,
                received_monotonic_ns=time.monotonic_ns(),
            )


def _capture_command(args: argparse.Namespace) -> int:
    os.umask(0o077)
    run_id = str(args.run_id or _default_run_id())
    run_dir = create_run_directory(args.output_root, run_id)
    capture = AlignmentWalkCapture(run_dir=run_dir, run_id=run_id, ws_uri=str(args.ws))
    try:
        if args.waypoints:
            payload = json.loads(Path(args.waypoints).read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise RuntimeError("waypoint manifest must be a JSON object")
            capture.copy_waypoints(payload)
        asyncio.run(_capture(args, capture))
    except KeyboardInterrupt:
        capture.finish(status="complete")
    except BaseException as exc:
        capture.finish(status="failed", error=str(exc))
        raise
    else:
        capture.finish(status="complete")
    result = verify_run(run_dir)
    print(json.dumps({"run_dir": str(run_dir), **result}, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _mark_command(args: argparse.Namespace) -> int:
    os.umask(0o077)
    marker = append_waypoint_marker(
        args.run_dir,
        waypoint_id=str(args.waypoint),
        phase=str(args.phase),
        actor=args.actor,
    )
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    os.umask(0o077)
    result = verify_run(args.run_dir, require_complete=not args.allow_incomplete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _report_command(args: argparse.Namespace) -> int:
    os.umask(0o077)
    result = write_alignment_report(
        args.run_dir,
        room_zones_path=args.room_zones,
        window_before_s=float(args.window_before),
        window_after_s=float(args.window_after),
        ambiguity_margin=float(args.ambiguity_margin),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"pass", "warning"} else 1


def _waypoint_calibration_command(args: argparse.Namespace) -> int:
    os.umask(0o077)
    result = write_waypoint_calibration_report(
        args.run_dir,
        window_before_s=float(args.window_before),
        window_after_s=float(args.window_after),
        ambiguity_margin=float(args.ambiguity_margin),
        good_error_m=float(args.good_error_m),
        fail_error_m=float(args.fail_error_m),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"pass", "warning"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture sanitized live alignment evidence.")
    capture.add_argument("--ws", default="ws://127.0.0.1:6008", help="Authenticated Noesis WebSocket endpoint.")
    capture.add_argument("--duration", type=float, default=900.0, help="Capture duration in seconds.")
    capture.add_argument("--connect-timeout", type=float, default=10.0)
    capture.add_argument("--run-id", default="")
    capture.add_argument("--output-root", type=Path, default=default_output_root())
    capture.add_argument("--waypoints", type=Path, default=None)
    add_auth_token_file_argument(capture)
    capture.set_defaults(handler=_capture_command)

    marker = subparsers.add_parser("mark", help="Durably timestamp an operator waypoint.")
    marker.add_argument("--run-dir", type=Path, required=True)
    marker.add_argument("--waypoint", required=True)
    marker.add_argument("--phase", choices=("arrived", "leave"), default="arrived")
    marker.add_argument("--actor", default=None, help="Optional run-local actor alias, not a person identity.")
    marker.set_defaults(handler=_mark_command)

    verify = subparsers.add_parser("verify", help="Verify privacy, sequence, and artifact digests.")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--allow-incomplete", action="store_true")
    verify.set_defaults(handler=_verify_command)

    report = subparsers.add_parser("report", help="Build room-containment and waypoint-assignment reports.")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--room-zones", type=Path, required=True)
    report.add_argument("--window-before", type=float, default=2.0)
    report.add_argument("--window-after", type=float, default=3.0)
    report.add_argument("--ambiguity-margin", type=float, default=0.12)
    report.set_defaults(handler=_report_command)

    waypoint_calibration = subparsers.add_parser(
        "waypoint-calibration",
        help=(
            "Build exact-frame fit/holdout waypoint evidence and an advisory similarity."
        ),
    )
    waypoint_calibration.add_argument("--run-dir", type=Path, required=True)
    waypoint_calibration.add_argument("--window-before", type=float, default=2.0)
    waypoint_calibration.add_argument("--window-after", type=float, default=3.0)
    waypoint_calibration.add_argument("--ambiguity-margin", type=float, default=0.12)
    waypoint_calibration.add_argument("--good-error-m", type=float, default=0.5)
    waypoint_calibration.add_argument("--fail-error-m", type=float, default=1.0)
    waypoint_calibration.set_defaults(handler=_waypoint_calibration_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"alignment walk command failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
