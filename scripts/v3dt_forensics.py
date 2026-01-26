#!/usr/bin/env python3
"""V3DT/SV3DT forensics CLI: snapshot, analyze, and panel outputs."""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import time
from pathlib import Path
from typing import Optional

from noesis.diagnostics.v3dt_forensics import (
    analyze_tracking_log,
    build_snapshot,
    render_panel_html,
    render_report_markdown,
    render_snapshot_markdown,
)

_DEFAULT_SCALE_SWEEP = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]


def _default_pipeline() -> Path:
    candidates = [
        Path("build/effective_pipeline_yolo11_seg.yaml"),
        Path("build/effective_pipeline_rfdetr_seg.yaml"),
        Path("config/infer_v3dt_medium.yaml"),
        Path("config/infer_v3dt_sv3dt.yaml"),
        Path("config/infer.yaml"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _pick_latest(pattern: str) -> Optional[Path]:
    paths = list(Path("diagnostics").glob(pattern))
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]


def _write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def _parse_scale_sweep(raw: str) -> Optional[list[float]]:
    value = (raw or "").strip()
    if not value:
        return None
    if value.lower() in ("auto", "default"):
        return list(_DEFAULT_SCALE_SWEEP)
    tokens = value.replace(",", " ").replace(";", " ").split()
    scales = []
    seen = set()
    for token in tokens:
        try:
            scale = float(token)
        except Exception:
            continue
        if not (scale > 0.0) or scale in seen:
            continue
        seen.add(scale)
        scales.append(scale)
    return scales or None


def _snapshot_cmd(args: argparse.Namespace) -> int:
    snapshot = build_snapshot(
        pipeline_path=Path(args.pipeline_config),
        cameras_path=Path(args.cameras_config),
        calibration_path=Path(args.calibration),
        caminfo_dir=Path(args.caminfo_dir),
        alignment_path=Path(args.alignment),
        tracker_config_path=Path(args.tracker_config) if args.tracker_config else None,
    )
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = Path(args.output_dir)
    json_path = out_dir / f"v3dt_snapshot_{ts}.json"
    md_path = out_dir / f"v3dt_snapshot_{ts}.md"
    _write_output(json_path, json.dumps(snapshot, indent=2, ensure_ascii=True))
    _write_output(md_path, render_snapshot_markdown(snapshot))
    print(f"Wrote snapshot: {json_path}")
    print(f"Wrote snapshot summary: {md_path}")
    return 0


def _analyze_cmd(args: argparse.Namespace) -> int:
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[FAIL] log file not found: {log_path}")
        return 1
    snapshot = None
    if args.snapshot:
        snap_path = Path(args.snapshot)
        if not snap_path.exists():
            print(f"[FAIL] snapshot file not found: {snap_path}")
            return 1
        snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
    scale_sweep = _parse_scale_sweep(args.scale_sweep)
    report = analyze_tracking_log(log_path, snapshot=snapshot, scale_sweep=scale_sweep)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = Path(args.output_dir)
    json_path = out_dir / f"v3dt_report_{ts}.json"
    md_path = out_dir / f"v3dt_report_{ts}.md"
    _write_output(json_path, json.dumps(report, indent=2, ensure_ascii=True))
    _write_output(md_path, render_report_markdown(report))
    print(f"Wrote report: {json_path}")
    print(f"Wrote report summary: {md_path}")
    return 0


def _panel_cmd(args: argparse.Namespace) -> int:
    snapshot_path = Path(args.snapshot) if args.snapshot else _pick_latest("v3dt_snapshot_*.json")
    report_path = Path(args.report) if args.report else _pick_latest("v3dt_report_*.json")
    if snapshot_path is None or report_path is None:
        print("[FAIL] Could not locate snapshot/report JSON. Provide --snapshot and --report.")
        return 1
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    html = render_panel_html(snapshot, report)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = Path(args.output_dir)
    out_path = out_dir / f"v3dt_panel_{ts}.html"
    _write_output(out_path, html)
    print(f"Wrote panel: {out_path}")
    return 0


def _serve_cmd(args: argparse.Namespace) -> int:
    root = Path(args.dir).resolve()
    if not root.exists():
        print(f"[FAIL] directory not found: {root}")
        return 1
    os.chdir(root)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {root} at http://{args.host}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Stopping server")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="V3DT forensics: snapshot, analyze, panel")
    sub = parser.add_subparsers(dest="command")

    snap = sub.add_parser("snapshot", help="Capture calibration + camInfo snapshot")
    snap.add_argument("--pipeline-config", default=str(_default_pipeline()))
    snap.add_argument("--cameras-config", default="config/cameras.yaml")
    snap.add_argument("--calibration", default="config/camera_calibration.json")
    snap.add_argument("--alignment", default="config/ply_alignment.json")
    snap.add_argument("--caminfo-dir", default="config/v3dt")
    snap.add_argument("--tracker-config", default="")
    snap.add_argument("--output-dir", default="diagnostics")

    analyze = sub.add_parser("analyze", help="Analyze V3DT tracking NDJSON")
    analyze.add_argument("--log", required=True, help="NDJSON log from NOESIS_V3DT_DIAG_LOG")
    analyze.add_argument("--snapshot", default="", help="Snapshot JSON to enable reprojection checks")
    analyze.add_argument(
        "--scale-sweep",
        default="",
        help="Comma-separated world scale factors (e.g. 0.01,0.1,1,10) or 'auto'",
    )
    analyze.add_argument("--output-dir", default="diagnostics")

    panel = sub.add_parser("panel", help="Generate HTML panel from snapshot + report")
    panel.add_argument("--snapshot", default="", help="Snapshot JSON (defaults to latest)")
    panel.add_argument("--report", default="", help="Report JSON (defaults to latest)")
    panel.add_argument("--output-dir", default="diagnostics")

    serve = sub.add_parser("serve", help="Serve diagnostics directory")
    serve.add_argument("--dir", default="diagnostics")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8777)

    args = parser.parse_args()
    if args.command == "snapshot":
        return _snapshot_cmd(args)
    if args.command == "analyze":
        return _analyze_cmd(args)
    if args.command == "panel":
        return _panel_cmd(args)
    if args.command == "serve":
        return _serve_cmd(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
