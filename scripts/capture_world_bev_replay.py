#!/usr/bin/env python3
"""Bounded native baseline replay with exact tracking/world/BEV review capture.

Uses the selected native environment, models and replay renderer. Temporarily
stops the appliance supervisor and named Noesis service, then restores their
previous activation in finally.
Replay identity/world databases are fresh, not copies of household state.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_STARTUP_BUDGET_S = 60.
COHORT_COMPLETION_GRACE_S = 1.
sys.path.insert(0, str(ROOT))

from scripts.internal_auth_client import connect_required_websocket, load_required_internal_auth


def validated_sources(sources, base_config=None):
    """Check actual decoded raster before stopping the selected runtime.

    Source intrinsics precede the streammux resize. A higher-resolution file
    cannot silently reuse the lower-resolution camera's dewarper intrinsics.
    Prepare a full-FoV, uniformly scaled derivative explicitly when needed.
    """
    config = yaml.safe_load((base_config or ROOT / "DS9/config/infer.yaml").read_text())
    cameras = ("living-room", "kitchen", "family-room")
    normalized, evidence = [], []
    seen = set()
    for value in sources:
        camera, separator, filename = value.partition("=")
        if not separator or camera not in cameras or camera in seen:
            raise ValueError("provide each canonical camera exactly once as camera=video")
        seen.add(camera)
        video = Path(filename).resolve(strict=True)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height", "-of", "json", str(video)], text=True, timeout=15))
        stream = probe["streams"][0]
        actual = (int(stream["width"]), int(stream["height"]))
        index = cameras.index(camera)
        mask = config.get("dewarper_validity_masks", {}).get("sources", {}).get(str(index), {})
        expected = (mask.get("source-width"), mask.get("source-height"))
        if config["sources"][index].get("dewarper", {}).get("enable") and actual != expected:
            raise ValueError(f"{camera}: video raster {actual} differs from calibrated dewarper input {expected}; prepare a full-FoV scaled derivative before replay")
        normalized.append(f"{camera}={video}")
        evidence.append({"camera": camera, "source": str(video), "raster": actual,
                         "calibrated_input_raster": expected})
    if seen != set(cameras):
        raise ValueError("replay requires all three canonical cameras")
    return normalized, evidence


async def capture(process, output, auth, duration):
    counts = Counter()
    started = time.monotonic()
    connected = False
    first_tracking_at = None
    last_tracking_at = None
    startup_budget = CAPTURE_STARTUP_BUDGET_S
    startup_deadline = started + startup_budget
    collection_deadline = startup_deadline + duration
    pending_cohorts = {}
    def deadline():
        if first_tracking_at is None:
            return startup_deadline
        # Finish an already received canonical triplet at the window edge.
        # This does not extend the absolute collection budget or drop any
        # admitted cohort. A stalled producer still fails within one second.
        grace = COHORT_COMPLETION_GRACE_S if pending_cohorts else 0.
        return min(first_tracking_at + duration + grace, collection_deadline)
    stop_reason = None
    latest_cohorts = {}
    exact_world = exact_bev = mismatches = 0
    bytes_written = 0
    with output.open("x", encoding="utf-8") as handle:
        while time.monotonic() < deadline() and process.poll() is None:
            try:
                async with connect_required_websocket(
                    "ws://127.0.0.1:6008", auth, compression=None,
                    max_size=16 * 1024 * 1024, open_timeout=1, close_timeout=2,
                ) as ws:
                    connected = True
                    await ws.send(json.dumps({"type": "set_vis_toggle", "toggle_name": "localization_details_enabled", "enabled": True}))
                    for camera in ("living-room", "kitchen", "family-room"):
                        await ws.send(json.dumps({"type": "get_floorplan", "camera": camera,
                                                  "scene_prior_only": True, "request_id": "replay-" + camera}))
                    while time.monotonic() < deadline() and process.poll() is None:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=min(1., max(0., deadline()-time.monotonic())))
                        except asyncio.TimeoutError:
                            continue
                        if first_tracking_at is None and time.monotonic() >= startup_deadline:
                            stop_reason = "startup_timeout"
                            break
                        if not isinstance(raw, str):
                            continue
                        row = json.loads(raw)
                        kind = row.get("type")
                        if kind in {"tracking", "world_snapshot", "bev-frame"}:
                            cohort = row.get("cohort")
                            fields = ("source_id", "frame_id", "observed_at_us", "tracking_publication_sequence")
                            key = tuple(cohort.get(field) for field in fields) if isinstance(cohort, dict) else ()
                            valid = len(key) == 4 and all(type(v) is int and v >= 0 for v in key) and 0 <= key[0] < 3
                            if kind == "tracking":
                                if valid:
                                    last_tracking_at = time.monotonic()
                                    if first_tracking_at is None: first_tracking_at = last_tracking_at
                                    latest_cohorts[key[0]] = key
                                    pending_cohorts[key] = {"world_snapshot", "bev-frame"}
                                    if len(pending_cohorts) > 64:
                                        raise RuntimeError("review capture pending cohort bound exceeded")
                                else: mismatches += 1
                            elif valid and latest_cohorts.get(key[0]) == key:
                                if kind == "world_snapshot": exact_world += 1
                                else: exact_bev += 1
                                pending = pending_cohorts.get(key)
                                if pending is not None:
                                    pending.discard(kind)
                                    if not pending:
                                        del pending_cohorts[key]
                            else: mismatches += 1
                        if kind not in {"tracking", "world_snapshot", "world_event", "bev-frame", "bev-status", "floorplan_response", "stats", "error"}:
                            continue
                        encoded = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                        bytes_written += len(encoded.encode("utf-8"))
                        if bytes_written > 64 * 1024 * 1024:
                            raise RuntimeError("review capture byte bound exceeded")
                        handle.write(encoded)
                        counts[kind] += 1
                        if sum(counts.values()) > 100_000:
                            raise RuntimeError("review capture message bound exceeded")
                    break
            except (OSError, TimeoutError):
                if connected:
                    stop_reason = "socket_closed"
                    break
                await asyncio.sleep(.25)
            except Exception as exc:
                # An orderly runtime EOS closes the socket. Retain what was
                # actually observed; the report does not claim complete input.
                if type(exc).__name__.startswith("ConnectionClosed"):
                    stop_reason = "socket_closed"
                    break
                raise
    stopped = time.monotonic()
    if stop_reason is None:
        if process.poll() is not None: stop_reason = "child_exit"
        elif first_tracking_at is None: stop_reason = "startup_timeout"
        elif pending_cohorts: stop_reason = "cohort_drain_timeout"
        elif stopped-first_tracking_at >= duration: stop_reason = "evidence_window_complete"
        else: stop_reason = "capture_deadline"
    return {"messages": dict(counts), "elapsed_s": stopped - started, "connected": connected,
        "bytes": bytes_written, "byte_semantics": "serialized JSON, not decoded video",
        "stop_reason": stop_reason, "requested_evidence_s": duration,
        "startup_budget_s": startup_budget, "collection_budget_s": startup_budget+duration,
        "cohort_completion_grace_s": COHORT_COMPLETION_GRACE_S, "pending_cohorts": len(pending_cohorts),
        "first_tracking_elapsed_s": None if first_tracking_at is None else first_tracking_at-started,
        "first_tracking_evidence_s": 0. if first_tracking_at is None else stopped-first_tracking_at,
        "tracking_evidence_span_s": 0. if last_tracking_at is None else last_tracking_at-first_tracking_at,
        "exact_world_cohorts": exact_world, "exact_bev_cohorts": exact_bev, "cohort_mismatches": mismatches}


def capture_completed(report, *, world_quality=False):
    counts = report.get("messages", {})
    operations = report.get("operation_receipts", [])
    log = report.get("runtime_log", {})
    return bool(type(report.get("replay_exit_code")) is int and report["replay_exit_code"] == 0 and report.get("connected") is True
        and all(counts.get(kind, 0) > 0 for kind in ("tracking", "world_snapshot", "bev-frame"))
        and report.get("exact_world_cohorts", 0) > 0 and report.get("exact_bev_cohorts", 0) > 0
        and counts.get("tracking") == counts.get("world_snapshot") == counts.get("bev-frame")
        and counts.get("tracking") == report.get("exact_world_cohorts") == report.get("exact_bev_cohorts")
        and report.get("pending_cohorts", 0) == 0
        and report.get("cohort_mismatches") == 0 and not counts.get("error", 0)
        and not report.get("failure") and not log.get("failure")
        and report.get("stop_reason") == "evidence_window_complete"
        and report.get("requested_evidence_s", 0) > 0
        and report.get("first_tracking_evidence_s", 0) >= report["requested_evidence_s"]
        and not any(s in {"TERM", "KILL"} for s in report.get("child_shutdown_signals", []))
        and report.get("cleanup_errors") == [] and operations
        and all(row.get("status") == "returned" for row in operations)
        and report.get("service_restored") == "active"
        and (not world_quality or log.get("actor_close_acknowledged") is True and log.get("shutdown_complete") is True))


def capture_exit_code(report, *, world_quality=False):
    return 0 if capture_completed(report, world_quality=world_quality) else 1


def runtime_log_outcome(path):
    """Bounded owned log only; returned close acknowledgement is not thread death."""
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("owned runtime log bound exceeded")
        with path.open("rb") as handle:
            data = handle.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024: raise ValueError("owned runtime log bound exceeded")
        text = data.decode("utf-8", errors="replace")
        ack = re.search(r"Canonical World actor close acknowledged: completed=1 target=\d+ thread=\d+", text)
        shutdown = text.rfind("Shutdown complete")
        failure = bool(re.search(r"\b(?:ERROR|CRITICAL)\b|Traceback|Runtime finalization was not proven|Native pipeline teardown failed", text))
        return {"actor_close_acknowledged": ack is not None, "shutdown_complete": shutdown >= 0
            and (ack is None or shutdown > ack.start()), "failure": failure}
    except (OSError, ValueError) as exc:
        return {"failure": str(exc), "actor_close_acknowledged": False, "shutdown_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--model-size", choices=("m", "l"), default="m")
    parser.add_argument("--base-config", type=Path, default=ROOT / "DS9/config/infer.yaml")
    parser.add_argument("--pgie-profile", choices=("yolo26", "yolo26-watch"), default="yolo26")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--tracking-mode", choices=("baseline", "mv3dt"), default="baseline")
    parser.add_argument("--native-artifact-receipt", type=Path)
    parser.add_argument("--native-artifact-receipt-sha256")
    args = parser.parse_args()
    args.base_config = args.base_config.resolve(strict=True)
    args.code_root = args.code_root.resolve(strict=True)
    if args.code_root != ROOT.resolve():
        parser.error("--code-root must match the executing isolated code checkout")
    original_authority = args.original_root or os.environ.get("NOESIS_XC3D_ORIGINAL_ROOT")
    if original_authority is None:
        if not (ROOT / ".noesis-team.json").is_file():
            parser.error("isolated replay requires --original-root or NOESIS_XC3D_ORIGINAL_ROOT")
        original_authority = ROOT
    args.original_root = Path(original_authority).resolve(strict=True)
    if not 1 <= args.duration <= 600:
        parser.error("duration must be between 1 and 600 seconds")
    plan = replay_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if (plan["physical_admission"]["status"] == "unavailable" or
            plan.get("artifact_selection", {}).get("status") == "unavailable"):
        print(json.dumps(plan, indent=2))
        return 78
    args.source, raster_evidence = validated_sources(args.source, args.base_config)
    out = args.output.resolve()
    unit = "noesis-appliance.service"
    target = "menon-appliance.target"  # Supervisor only; no Menon implementation work.
    target_state = subprocess.check_output(
        ["systemctl", "--user", "show", target, "-p", "ActiveState", "--value"], text=True
    ).strip()
    target_job = subprocess.check_output(
        ["systemctl", "--user", "show", target, "-p", "Job", "--value"], text=True
    ).strip()
    target_was_active = target_state in {"active", "activating", "reloading"} or target_job not in {"", "0"}
    pid = int(subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "MainPID", "--value"], text=True))
    if pid <= 0:
        raise RuntimeError("No running Noesis service from which to establish native environment")
    if Path(f"/proc/{pid}/cwd").resolve() != args.original_root:
        raise RuntimeError("managed baseline owner does not belong to ORIGINAL root")
    if target_job not in {"", "0"}:
        raise RuntimeError("pending target job requires root reconciliation before replay")
    command_before = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "DS9/noesis/ds9_runtime.py" not in command_before:
        raise RuntimeError("Service main process is not the canonical DS9 runtime")
    env = merge_replay_environment(os.environ, Path(f"/proc/{pid}/environ").read_bytes(), args)
    pin = None
    if _replay_candidate(args):
        from noesis.v3dt_assets import candidate_analytics_pin, preflight_candidate_analytics
        pin = candidate_analytics_pin(args.code_root, args.native_artifact_receipt,
                                      args.native_artifact_receipt_sha256)
        preflight_candidate_analytics(pin, args.code_root, load=True)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    # Preserve SDK/model/secrets paths while separating replay persistence.
    for name, filename in (("NOESIS_WORLD_JOURNAL_PATH", "world.sqlite3"),
                           ("NOESIS_IDENTITY_V2_STORE", "identity.sqlite3")):
        path = out / filename
        sqlite3.connect(path).close()
        path.chmod(0o600)
        env[name] = str(path)
    env["NOESIS_DS9_RUNTIME_ROOT"] = str(out / "runtime")
    env["NOESIS_XC3D_ORIGINAL_ROOT"] = str(args.original_root)
    env["NOESIS_IDENTITY_V2_EVIDENCE_PATH"] = str(out / "identity-evidence.jsonl")
    python = str(pin.python) if pin is not None else str(Path(env["NOESIS_DS91_NATIVE_ROOT"]) / "venv/bin/python")
    config = out / "infer-replay.yaml"
    render = [python, "DS9/scripts/render_recorded_replay_config.py", "--output", str(config), "--base-config", str(args.base_config), "--original-root", str(args.original_root)]
    for source in args.source:
        render += ["--source", source]
    subprocess.run(render, cwd=ROOT, env=env, check=True)
    auth = load_required_internal_auth(env["NOESIS_INTERNAL_AUTH_TOKEN_FILE"], env=env)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = {"git_revision": revision, "sources": args.source, "model_size": args.model_size,
                "tracking_mode": args.tracking_mode, "base_config": str(args.base_config),
                "pgie_profile": args.pgie_profile, "service_pid_before": pid,
                "service_command_before": command_before, "duration_limit_s": args.duration,
                "new_identity_store": True, "new_world_store": True}
    metadata["source_rasters"] = raster_evidence
    metadata["object_depth_environment"] = replay_object_depth_environment(env)
    metadata["source_code_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in (
            "DS9/noesis/pipelines/hooks.py", "noesis/calibration/manager.py",
            "noesis/telemetry/person_ground_state.py", "noesis/telemetry/bev.py",
            "noesis_core/world_service.py", "noesis_core/world/resolver.py",
            "noesis_core/world/fusion.py", "noesis_core/world/support_surfaces.py",
            "noesis_core/contracts/support_surface.py",
        )
    }
    replay_config = yaml.safe_load(config.read_text())
    metadata["calibration_artifact_sha256"] = {}
    for section in ("depth_registration", "support_surfaces", "scene_priors"):
        value = replay_config.get(section, {}).get("path")
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            metadata["calibration_artifact_sha256"][section] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    command = [python, *replay_child_command(args, config)]
    if pin is not None:
        preflight_candidate_analytics(pin, args.code_root, load=False)
    def consume(process):
        return asyncio.run(capture(process, out / "cohorts.ndjson", auth, args.duration))
    report = run_owned_replay(pid=pid, target_was_active=target_was_active,
                             command=command, code_root=args.code_root,
                             original_root=args.original_root, env=env, output=out, consume=consume,
                             world_quality=bool(replay_config.get("xc3d_world_quality", {}).get("enabled")))
    print(json.dumps(report, indent=2))
    return capture_exit_code(report, world_quality=bool(replay_config.get("xc3d_world_quality", {}).get("enabled")))


def run_owned_replay(*, pid, target_was_active, command, code_root, original_root, env, output, consume, world_quality=False):
    """One root-leased process; finally restores the ORIGINAL managed unit/target.

    Every uncertain stop/start is intent-recorded before issue. Cleanup failure
    cannot skip the remaining original restoration attempts; receipt reports all.
    """
    unit, target = "noesis-appliance.service", "menon-appliance.target"
    process = None
    issued = []
    errors = []
    report = {"original_root": str(original_root), "code_root": str(code_root)}
    target_attempted = service_attempted = False
    def receipt():
        report.update(operation_receipts=issued, cleanup_errors=errors)
        (output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    def operate(verb, name, *flags):
        row = {"verb": verb, "unit": name, "status": "intent"}
        issued.append(row)
        receipt()
        subprocess.run(["systemctl", "--user", verb, *flags, name], check=True, timeout=90)
        row["status"] = "returned"
        receipt()
    def current_pid():
        return int(subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "MainPID", "--value"], text=True, timeout=10))
    try:
        if current_pid() != pid:
            raise RuntimeError("managed_owner_pid_drift_before_stop")
        target_attempted = True
        operate("stop", target)
        if current_pid() not in {0, pid}:
            raise RuntimeError("managed_owner_pid_drift_after_target_stop")
        service_attempted = True
        operate("stop", unit)
        if current_pid() != 0:
            raise RuntimeError("managed_owner_not_stopped")
        with (output / "runtime.log").open("x") as log:
            process = subprocess.Popen(command, cwd=code_root, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report.update(consume(process))
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ":" + str(exc)
        report.setdefault("stop_reason", "capture_error")
        raise
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    for signal_name, timeout in (("send_signal", 30), ("terminate", 15), ("kill", 10)):
                        report.setdefault("child_shutdown_signals", []).append(
                            {"send_signal": "SIGINT", "terminate": "TERM", "kill": "KILL"}[signal_name])
                        getattr(process, signal_name)(signal.SIGINT) if signal_name == "send_signal" else getattr(process, signal_name)()
                        try:
                            process.wait(timeout=timeout)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                report["replay_exit_code"] = process.poll()
            except BaseException as exc:
                errors.append("child_shutdown:" + str(exc))
        # A still-running child owns the ports. Do not start a competing baseline.
        child_stopped = process is None or process.poll() is not None
        if not child_stopped:
            errors.append("child_owner_uncertain_baseline_restore_blocked")
        if service_attempted and child_stopped:
            for verb in ("reset-failed", "start"):
                try:
                    operate(verb, unit)
                except BaseException as exc:
                    errors.append("original_service_" + verb + ":" + str(exc))
        if target_attempted and target_was_active and child_stopped:
            try:
                operate("start", target, "--no-block")
            except BaseException as exc:
                errors.append("original_target_start:" + str(exc))
        if service_attempted and child_stopped:
            try:
                report["service_restored"] = subprocess.check_output(
                    ["systemctl", "--user", "is-active", unit], text=True, timeout=10).strip()
            except BaseException as exc:
                errors.append("original_service_readback:" + str(exc))
        report["runtime_log"] = runtime_log_outcome(output / "runtime.log")
        report.update(operation_receipts=issued, cleanup_errors=errors)
        report["completion_status"] = "complete" if capture_completed(report, world_quality=world_quality) else (
            "failed" if report.get("failure") or report.get("replay_exit_code") not in (None, 0)
            or errors or report["runtime_log"].get("failure") else "partial")
        receipt()
        if errors and "failure" not in report:
            raise RuntimeError("replay restoration incomplete: " + ";".join(errors))
    return report


def _replay_candidate(args, payload=None):
    from noesis.v3dt_assets import xc3d_candidate_enabled, xc3d_world_quality_enabled
    payload = yaml.safe_load(args.base_config.read_text()) if payload is None else payload
    if not isinstance(payload, dict):
        raise ValueError("pipeline config must be a mapping")
    world_quality = xc3d_world_quality_enabled(payload)
    candidate = xc3d_candidate_enabled(payload)
    if world_quality and args.tracking_mode != "baseline":
        raise ValueError("world-quality replay requires baseline NvDCF")
    if candidate and not world_quality and args.tracking_mode != "mv3dt":
        raise ValueError("physical candidate replay requires mv3dt")
    if args.tracking_mode == "mv3dt" and not candidate:
        raise ValueError("mv3dt replay requires the explicit isolated M960 candidate")
    if candidate and (args.model_size != "m" or args.pgie_profile != "yolo26" or not 1 <= args.duration <= 90):
        raise ValueError("XC3D replay preserves M960/yolo26 and <=90s evidence")
    return candidate


def replay_child_command(args, config):
    """The same actual child arguments for the read-only plan and execution."""
    command = ["DS9/scripts/run_canonical_runtime_host.py", "run",
               "--pipeline-config", str(config), "--model-size", args.model_size,
               "--pgie-profile", args.pgie_profile, "--tracking-mode", args.tracking_mode,
               "--code-root", str(args.code_root), "--original-root", str(args.original_root)]
    if _replay_candidate(args):
        receipt = getattr(args, "native_artifact_receipt", None)
        receipt_hash = getattr(args, "native_artifact_receipt_sha256", None)
        if receipt is not None and receipt_hash is not None:
            command += ["--native-artifact-receipt", str(receipt),
                        "--native-artifact-receipt-sha256", receipt_hash]
    return command


def replay_plan(args):
    """Read-only source contract; no subprocess, SDK, stores, services or secrets."""
    sys.path.insert(0, str(args.code_root / "DS9"))
    from noesis.v3dt_assets import xc3d_admission_refusal, candidate_artifact_plan, candidate_analytics_pin
    payload = yaml.safe_load(args.base_config.read_text())
    candidate = _replay_candidate(args, payload)
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError("fresh absolute private task output required")
    parsed = {}
    for value in args.source:
        camera, separator, raw = value.partition("=")
        path = Path(raw)
        if not separator or camera not in {"living-room", "kitchen", "family-room"} or camera in parsed or not path.is_absolute() or not path.is_file():
            raise ValueError("exact three absolute original retained source paths required")
        parsed[camera] = str(path.resolve())
    if len(parsed) != 3:
        raise ValueError("all three retained sources required")
    result = {"code_root": str(args.code_root), "original_root": str(args.original_root),
            "base_config": str(args.base_config), "sources": parsed, "tracking_mode": args.tracking_mode,
            "physical_admission": xc3d_admission_refusal(payload),
            "raster_validation": "required_ffprobe_before_any_stop", "source_clock": "unqualified_derivative",
            "fresh_stores": ["world.sqlite3", "identity.sqlite3"], "stores_created": False,
            "selected_interpreter": "running_baseline_env:NOESIS_DS91_NATIVE_ROOT/venv/bin/python",
            "asset_data_config_secret_authority": str(args.original_root),
            "would_stop": ["menon-appliance.target", "noesis-appliance.service"],
            "finally_restore": ["ORIGINAL:noesis-appliance.service", "ORIGINAL:menon-appliance.target(previous state)"],
            "service_operations_issued": [], "sdk_initialized": False,
            "startup_budget_s": CAPTURE_STARTUP_BUDGET_S,
            "collection_budget_s": CAPTURE_STARTUP_BUDGET_S + args.duration,
            "evidence_s": args.duration, "hard_runtime_s": 180,
            "message_limit": 100000, "byte_limit": 64 * 1024 * 1024,
            "candidate_command": replay_child_command(args, args.output / "infer-replay.yaml")}
    if candidate:
        receipt = getattr(args, "native_artifact_receipt", None)
        receipt_hash = getattr(args, "native_artifact_receipt_sha256", None)
        result["artifact_selection"] = candidate_artifact_plan(args.code_root, receipt, receipt_hash)
        if result["artifact_selection"]["status"] != "unavailable":
            result["selected_interpreter"] = str(candidate_analytics_pin(args.code_root, receipt, receipt_hash).python)
    return result


def replay_object_depth_environment(env):
    """Report only selected raw child overrides, never infer observed cadence.

    An unset override delegates to the source-bound hooks.py default reader.
    Do not dump the process environment: it also contains private credentials.
    """
    names = (
        "NOESIS_OBJECT_DEPTH_MAX_HZ_PER_TRACK",
        "NOESIS_OBJECT_DEPTH_MAX_OBJECTS_PER_FRAME",
        "NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS",
        "NOESIS_OBJECT_DEPTH_CACHE_MAX_AGE_MS",
        "NOESIS_OBJECT_DEPTH_CACHE_MAX_BBOX_SHIFT",
        "NOESIS_OBJECT_DEPTH_CACHE_MAX_TRACKS",
        "NOESIS_OBJECT_DEPTH_MAX_ANCHOR_SAMPLES",
        "NOESIS_OBJECT_DEPTH_NATIVE_MAX_STAT_SAMPLES",
    )
    return {
        "semantics": "raw merged replay-child overrides, not observed sample timing",
        "unset_semantics": "null means unset; defaults/validation belong to the source-bound reader",
        "defaults_source": "DS9/noesis/pipelines/hooks.py",
        "overrides": {name: env.get(name) for name in names},
    }


def merge_replay_environment(caller, baseline_environ, args):
    """Explicit pin wins AFTER original baseline environment, never via caller env."""
    env = dict(caller)
    original_environment = {}
    for item in baseline_environ.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            original_environment[key.decode()] = value.decode()
    env.update(original_environment)
    if _replay_candidate(args):
        from noesis.v3dt_assets import candidate_analytics_pin, candidate_native_environment
        pin = candidate_analytics_pin(args.code_root, args.native_artifact_receipt,
                                      args.native_artifact_receipt_sha256)
        from DS9.scripts.run_canonical_runtime_host import candidate_asset_environment
        env = candidate_asset_environment(env, original_root=args.original_root,
            artifact_root=original_environment.get("NOESIS_DS9_ARTIFACT_ROOT", ""), sdk_root=pin.sdk_root)
        env = candidate_native_environment(pin, env)
    return env


if __name__ == "__main__":
    raise SystemExit(main())
