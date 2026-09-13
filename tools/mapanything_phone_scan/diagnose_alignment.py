"""Replay an explicit room fit without updating its saved scan or live world.

Each invocation requires a new output directory. Rejected fits retain their
measurements, and optional snapshots support further CPU-only experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from tools.mapanything_phone_scan.alignment import (
    NoesisAlignmentError,
    NoesisAlignmentSettings,
    run_noesis_alignment,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--target-revision", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-raw-root", type=Path)
    parser.add_argument("--source-output-manifest", type=Path)
    parser.add_argument("--save-snapshot", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.source_raw_root) != bool(args.source_output_manifest):
        parser.error("Supply both explicit source paths when comparing another reconstruction")
    scan_dir, output_dir = args.scan_dir.resolve(), args.output_dir.resolve()
    try:
        output_dir.relative_to(scan_dir)
    except ValueError:
        pass
    else:
        parser.error("Diagnostic output must be outside the saved scan directory")
    if output_dir.exists():
        parser.error("Use a new output directory to preserve previous experiments")
    state = json.loads((scan_dir / "scan_state.json").read_text(encoding="utf-8"))
    source_raw_root = args.source_raw_root.resolve() if args.source_raw_root else None
    source_output_manifest = args.source_output_manifest.resolve() if args.source_output_manifest else None
    outputs = (
        json.loads(source_output_manifest.read_text(encoding="utf-8"))
        if source_output_manifest else state["outputs"]
    )
    settings = NoesisAlignmentSettings(
        camera_id=args.camera_id,
        target_revision=args.target_revision.resolve(),
        calibration_path=args.calibration_path.resolve(),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary: dict[str, Any] = {
        "schema": "noesis.phone_scan.alignment_diagnostic_run.v1",
        "scan_dir": str(scan_dir), "scan_id": state.get("id"),
        "working_directory": str(Path.cwd()),
        "camera_id": args.camera_id, "target_revision": str(settings.target_revision),
        "command": [sys.executable, "-m", "tools.mapanything_phone_scan.diagnose_alignment", *(argv if argv is not None else sys.argv[1:])],
        "updates_scan_state": False, "promotes_live_world": False,
    }

    def save_diagnostics(report: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        (output_dir / "diagnostic_report.json").write_text(
            json.dumps(report, default=_json_default, indent=2) + "\n", encoding="utf-8",
        )
        if args.save_snapshot:
            np.savez_compressed(output_dir / "diagnostic_inputs.npz", **arrays)

    def progress(fraction: float, message: str) -> None:
        print(f"{time.monotonic() - started:.1f}s {fraction:.0%} {message}", flush=True)

    try:
        result = run_noesis_alignment(
            scan_dir, output_dir, outputs, settings, progress,
            source_raw_root=source_raw_root,
            source_output_manifest=source_output_manifest,
            diagnostic_callback=save_diagnostics,
        )
        summary.update(status="passed", quality_gate=result["quality_gate"])
        exit_code = 0
    except NoesisAlignmentError as exc:
        summary.update(status="rejected", error=str(exc))
        exit_code = 2
    except Exception as exc:
        summary.update(status="error", error=f"{type(exc).__name__}: {exc}")
        exit_code = 1
    summary["elapsed_seconds"] = time.monotonic() - started
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, default=_json_default, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, default=_json_default), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
