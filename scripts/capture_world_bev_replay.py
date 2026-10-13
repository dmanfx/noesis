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
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
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
    with output.open("x", encoding="utf-8") as handle:
        while time.monotonic() - started < duration + 120 and process.poll() is None:
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
                    while time.monotonic() - started < duration + 120 and process.poll() is None:
                        if first_tracking_at is not None and time.monotonic() - first_tracking_at >= duration:
                            break
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1)
                        except asyncio.TimeoutError:
                            continue
                        if not isinstance(raw, str):
                            continue
                        row = json.loads(raw)
                        kind = row.get("type")
                        if kind == "tracking" and first_tracking_at is None:
                            first_tracking_at = time.monotonic()
                        if kind not in {"tracking", "world_snapshot", "world_event", "bev-frame", "bev-status", "floorplan_response", "stats", "error"}:
                            continue
                        handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                        counts[kind] += 1
                        if sum(counts.values()) > 100_000:
                            raise RuntimeError("review capture message bound exceeded")
                    break
            except (OSError, TimeoutError):
                if connected:
                    break
                await asyncio.sleep(.25)
            except Exception as exc:
                # An orderly runtime EOS closes the socket. Retain what was
                # actually observed; the report does not claim complete input.
                if type(exc).__name__.startswith("ConnectionClosed"):
                    break
                raise
    return {"messages": dict(counts), "elapsed_s": time.monotonic() - started, "connected": connected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--model-size", choices=("m", "l"), default="m")
    parser.add_argument("--base-config", type=Path, default=ROOT / "DS9/config/infer.yaml")
    parser.add_argument("--pgie-profile", choices=("yolo26", "yolo26-watch"), default="yolo26")
    args = parser.parse_args()
    args.base_config = args.base_config.resolve(strict=True)
    if not 1 <= args.duration <= 600:
        parser.error("duration must be between 1 and 600 seconds")
    args.source, raster_evidence = validated_sources(args.source, args.base_config)
    out = args.output.resolve()
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
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
    command_before = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "DS9/noesis/ds9_runtime.py" not in command_before:
        raise RuntimeError("Service main process is not the canonical DS9 runtime")
    env = dict(os.environ)
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode()
    # Preserve SDK/model/secrets paths while separating replay persistence.
    for name, filename in (("NOESIS_WORLD_JOURNAL_PATH", "world.sqlite3"),
                           ("NOESIS_IDENTITY_V2_STORE", "identity.sqlite3")):
        path = out / filename
        sqlite3.connect(path).close()
        path.chmod(0o600)
        env[name] = str(path)
    env["NOESIS_DS9_RUNTIME_ROOT"] = str(out / "runtime")
    env["NOESIS_IDENTITY_V2_EVIDENCE_PATH"] = str(out / "identity-evidence.jsonl")
    python = str(Path(env["NOESIS_DS91_NATIVE_ROOT"]) / "venv/bin/python")
    config = out / "infer-replay.yaml"
    render = [python, "DS9/scripts/render_recorded_replay_config.py", "--output", str(config), "--base-config", str(args.base_config)]
    for source in args.source:
        render += ["--source", source]
    subprocess.run(render, cwd=ROOT, env=env, check=True)
    auth = load_required_internal_auth(env["NOESIS_INTERNAL_AUTH_TOKEN_FILE"], env=env)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = {"git_revision": revision, "sources": args.source, "model_size": args.model_size,
                "tracking_mode": "baseline", "base_config": str(args.base_config),
                "pgie_profile": args.pgie_profile, "service_pid_before": pid,
                "service_command_before": command_before, "duration_limit_s": args.duration,
                "new_identity_store": True, "new_world_store": True}
    metadata["source_rasters"] = raster_evidence
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
    process = None
    stopped = False
    report = {}
    try:
        current = int(subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "MainPID", "--value"], text=True))
        if current != pid:
            raise RuntimeError("Service changed concurrently; not stopping a different owner")
        stopped = True
        # The target's Upholds would otherwise restart Noesis into the replay
        # ports. Preserve its state and stop it before the service, as the
        # existing reconstruction/replay lease does.
        # A target can still be inactive while its queued start job waits for
        # readiness dependencies. Stop unconditionally to cancel that job too.
        subprocess.run(["systemctl", "--user", "stop", target], check=True, timeout=60)
        subprocess.run(["systemctl", "--user", "stop", unit], check=True, timeout=90)
        with (out / "runtime.log").open("x") as log:
            process = subprocess.Popen(
                [python, "DS9/scripts/run_canonical_runtime_host.py", "run",
                 "--pipeline-config", str(config), "--model-size", args.model_size,
                 "--pgie-profile", args.pgie_profile],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            report = asyncio.run(capture(process, out / "cohorts.ndjson", auth, args.duration))
    finally:
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            report["replay_exit_code"] = process.returncode
        if stopped:
            subprocess.run(["systemctl", "--user", "reset-failed", unit], check=True)
            try:
                subprocess.run(["systemctl", "--user", "start", unit], check=True, timeout=90)
                if target_was_active:
                    # Gateway readiness has its own longer startup budget.
                    subprocess.run(["systemctl", "--user", "start", "--no-block", target], check=True, timeout=15)
                report["service_restored"] = subprocess.check_output(
                    ["systemctl", "--user", "is-active", unit], text=True).strip()
            finally:
                (out / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report.get("messages", {}).get("tracking", 0) and report.get("messages", {}).get("bev-frame", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
