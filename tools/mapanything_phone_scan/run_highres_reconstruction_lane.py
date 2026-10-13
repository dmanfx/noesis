#!/usr/bin/env python3
"""Run an opt-in, source-preserving DA3/PCF resolution experiment.

The browser defaults, scan state, baseline outputs and runtime configuration
are never written. GPU stages use the documented appliance pause/restore lease.
Fusion follows after runtime restoration; subsequent presentation tools consume
the new explicit paths. A failed stage remains available for diagnosis.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Depth-Anything-3" / "src"))

from tools.mapanything_phone_scan.da3_inference import DA3PhoneScanSettings
from tools.mapanything_phone_scan.run_mapanything_prior_variants import _preprocessing_options
from tools.mapanything_phone_scan.windowed_da3_inference import run_adaptive_da3_phone_scan

TARGET = "menon-appliance.target"
RUNTIME = "noesis-appliance.service"


def utc():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def active(unit):
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit],
                          check=False).returncode == 0


def control(action):
    # This host's target stop need not wait for/stop every already-running
    # native member. Include the exact GPU owner explicitly, without changing
    # unit files or relying on kill/restart-on-failure behavior.
    subprocess.run(["systemctl", "--user", action, TARGET, RUNTIME], check=True, timeout=300)


def restore(root):
    path = root / "runtime_lease.json"
    lease = json.loads(path.read_text())
    if lease["appliance_was_active"] and lease["pause_requested"] and not lease.get("restore_passed"):
        try:
            control("start")
            if not active(RUNTIME):
                raise RuntimeError("Noesis was not active after appliance restoration")
            lease.update(restore_passed=True, restored_at=utc(), restore_error=None)
        except BaseException as exc:
            lease.update(restore_passed=False, restore_error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save(path, lease)
    return lease


def run_stage(root, name, command, state, *, state_path=None):
    state_path = root / "lane_manifest.json" if state_path is None else state_path
    row = {"name": name, "command": command, "started_at": utc(), "status": "running"}
    log_path = root / f"{name}.log"
    if log_path.exists():
        log_path = root / f"{name}_attempt_{len(state['stages']) + 1}.log"
    if log_path.exists():
        raise ValueError("stage log already exists; refusing to erase prior diagnostics")
    row["log"] = str(log_path)
    state["stages"].append(row)
    save(state_path, state)
    started = time.monotonic()
    print(f"[{name}] starting", flush=True)
    process = None
    try:
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        row["status"] = "complete"
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        row.update(finished_at=utc(), elapsed_seconds=time.monotonic() - started)
        save(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metric-engine", type=Path)
    parser.add_argument("--target-revision", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--camera-id", default="living-room")
    parser.add_argument("--process-res", type=int, default=756)
    parser.add_argument("--da3-window", type=int, default=64)
    parser.add_argument("--da3-overlap", type=int, default=32)
    parser.add_argument("--ma-window", type=int, default=32)
    parser.add_argument("--ma-overlap", type=int, default=16)
    parser.add_argument("--restore-runtime", action="store_true")
    parser.add_argument("--run-da3-stage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.restore_runtime:
        print(json.dumps(restore(root), indent=2))
        return 0
    if args.run_da3_stage:
        run_inputs = json.loads((root / "da3/run_inputs.json").read_text())
        settings_data = run_inputs["settings"]
        settings_data["metric_engine_path"] = Path(settings_data["metric_engine_path"])
        scan = Path(run_inputs["scan"])
        prepared_path = Path(run_inputs["prepared_manifest"])
        if binding(prepared_path)["sha256"] != run_inputs["prepared_sha256"]:
            raise RuntimeError("DA3 stage prepared binding changed")
        run_adaptive_da3_phone_scan(scan, root / "da3", json.loads(prepared_path.read_text()),
            DA3PhoneScanSettings(**settings_data),
            lambda fraction, message: print(f"DA3 {fraction:.3f}: {message}", flush=True))
        return 0
    _preprocessing_options(args.process_res)
    if any(value is None for value in (args.scan_dir, args.metric_engine, args.target_revision, args.calibration)):
        parser.error("--scan-dir, --metric-engine, --target-revision and --calibration are required for a new run")
    if os.environ.get("NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE") != "1":
        raise RuntimeError("explicit NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE=1 lease is required")
    scan = args.scan_dir.resolve()
    engine = args.metric_engine.resolve()
    target, calibration = args.target_revision.resolve(), args.calibration.resolve()
    prepared_path = scan / "prepared_frames_manifest.json"
    prepared = json.loads(prepared_path.read_text())
    if (not 2 <= len(prepared["frames"]) <= 256 or not engine.is_file()
            or not (target / "room_points.npz").is_file() or not calibration.is_file()):
        raise ValueError("missing validated metric engine or bounded prepared cohort")
    for window, overlap in ((args.da3_window, args.da3_overlap), (args.ma_window, args.ma_overlap)):
        if not 16 <= window <= 80 or not 4 <= overlap < window:
            raise ValueError("invalid bounded inference window")
    if root.exists() or root.is_relative_to(scan) or scan.is_relative_to(root):
        raise ValueError("use a fresh output root outside the input scan")
    root.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "triton-cache"):
        (root / name).mkdir()
    os.environ["TMPDIR"] = str(root / "tmp")
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton-cache")
    settings = DA3PhoneScanSettings(process_res=args.process_res,
        metric_engine_path=engine, max_joint_views=args.da3_window,
        window_overlap_views=args.da3_overlap)
    state = {"schema": "noesis.offline_highres_lane.v1", "created_at": utc(),
        "review_only": True, "accepted_for_canonical_use": False,
        "scan": str(scan), "prepared_manifest": binding(prepared_path),
        "scan_state": binding(scan / "scan_state.json"),
        "static_target_revision": str(target), "calibration": binding(calibration),
        "view_count": len(prepared["frames"]), "settings": asdict(settings),
        "mapanything": {"process_res": args.process_res, "max_joint_views": args.ma_window,
                         "window_overlap_views": args.ma_overlap}, "stages": [], "status": "running"}
    save(root / "lane_manifest.json", state)
    da3 = root / "da3"
    da3.mkdir()
    save(da3 / "run_inputs.json", {"scan": str(scan), "stage": "da3",
        "prepared_manifest": str(prepared_path), "prepared_sha256": binding(prepared_path)["sha256"],
        "view_count": len(prepared["frames"]), "settings": asdict(settings), "review_only": True})
    lease = {"configured": True, "appliance_target": TARGET,
        "appliance_was_active": active(TARGET), "noesis_was_active": active(RUNTIME),
        "pause_requested": False, "appliance_paused": False, "restore_passed": None,
        "created_at": utc()}
    if lease["noesis_was_active"] and not lease["appliance_was_active"]:
        raise RuntimeError("Noesis is active outside the documented appliance target")
    save(root / "runtime_lease.json", lease)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        try:
            if lease["appliance_was_active"]:
                lease["pause_requested"] = True
                save(root / "runtime_lease.json", lease)
                print("Pausing appliance for exclusive reconstruction GPU use", flush=True)
                control("stop")
                lease.update(appliance_paused=True, paused_at=utc())
                save(root / "runtime_lease.json", lease)
            if active(RUNTIME):
                raise RuntimeError("Noesis remained active after GPU handoff")
            run_stage(root, "da3", [sys.executable, str(Path(__file__).resolve()),
                "--output-root", str(root), "--run-da3-stage"], state)
            def alignment_command(source, destination):
                return [sys.executable, "-m", "tools.mapanything_phone_scan.diagnose_alignment",
                    "--scan-dir", str(scan), "--camera-id", args.camera_id,
                    "--target-revision", str(target), "--calibration-path", str(calibration),
                    "--source-raw-root", str(source / "raw"),
                    "--source-output-manifest", str(source / "scan_outputs_manifest.json"),
                    "--output-dir", str(destination), "--save-snapshot"]
            run_stage(root, "da3_alignment", alignment_command(da3, root / "alignment_da3"), state)
            run_stage(root, "conditioned_mapanything", [sys.executable,
                str(REPO / "tools/mapanything_phone_scan/run_mapanything_prior_variants.py"), str(scan),
                "--da3-raw", str(da3 / "raw"), "--variants", "da3_pose_sparse_depth",
                "--world-from-da3", str(root / "alignment_da3/phone_ma_to_noesis_world.json"),
                "--target-revision", str(target), "--calibration", str(calibration), "--camera", args.camera_id,
                "--output-root", str(root / "pcf"), "--process-res", str(args.process_res),
                "--max-joint-views", str(args.ma_window), "--window-overlap-views", str(args.ma_overlap)], state)
        finally:
            print("Restoring appliance after GPU stages", flush=True)
            restore(root)
        run_stage(root, "consensus", [sys.executable,
            str(REPO / "tools/mapanything_phone_scan/build_consensus_fusion.py"), str(scan),
            "--mapanything-raw", str(root / "pcf/mapanything_da3_pose_sparse_depth/raw"),
            "--da3-raw", str(da3 / "raw"), "--pose-carrier", "da3",
            "--output-dir", str(root / "pcf/prior_conditioned_consensus_da3_carrier")], state)
        run_stage(root, "consensus_alignment", alignment_command(
            root / "pcf/prior_conditioned_consensus_da3_carrier", root / "alignment_consensus"), state)
        if binding(prepared_path) != state["prepared_manifest"]:
            raise RuntimeError("input prepared manifest changed during the experiment")
        if binding(scan / "scan_state.json") != state["scan_state"] or binding(calibration) != state["calibration"]:
            raise RuntimeError("input scan state or calibration changed during the experiment")
        state.update(status="complete", finished_at=utc())
        return 0
    except BaseException as exc:
        state.update(status="failed", finished_at=utc(), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["runtime_lease"] = json.loads((root / "runtime_lease.json").read_text())
        save(root / "lane_manifest.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
