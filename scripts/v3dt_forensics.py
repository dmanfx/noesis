#!/usr/bin/env python3
"""V3DT/SV3DT forensics CLI: snapshot, analyze, and panel outputs."""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit

from noesis.diagnostics.v3dt_forensics import (
    analyze_tracking_log,
    build_snapshot,
    render_panel_html,
    render_report_markdown,
    render_snapshot_markdown,
)
from noesis_core.private_paths import (
    PrivatePathError,
    atomic_write_private_file,
    ensure_private_directory,
    read_private_file,
    validate_private_file,
)

_DEFAULT_SCALE_SWEEP = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_LOG_BYTES = 512 * 1024 * 1024
_MAX_PANEL_BYTES = 16 * 1024 * 1024
_PANEL_NAME_RE = re.compile(r"^v3dt_panel_[A-Za-z0-9._-]{1,96}\.html$")


def _default_diagnostics_dir() -> Path:
    configured = str(os.environ.get("NOESIS_V3DT_DIAG_DIR", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state" / "noesis" / "diagnostics"


def _default_pipeline() -> Path:
    candidates = [
        Path("build/effective_pipeline_yolo11_seg.yaml"),
        Path("build/effective_pipeline_rfdetr_seg.yaml"),
        Path("DS9/config/infer_v3dt.yaml"),
        Path("DS9/config/infer.yaml"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _pick_latest(directory: Path, pattern: str) -> Optional[Path]:
    root = ensure_private_directory(directory, label="V3DT diagnostics")
    paths = [
        validate_private_file(path, label="V3DT diagnostic artifact")
        for path in root.glob(pattern)
    ]
    if not paths:
        return None
    paths.sort(key=lambda p: p.lstat().st_mtime_ns, reverse=True)
    return paths[0]


def _write_output(path: Path, content: str) -> None:
    atomic_write_private_file(
        path,
        content.encode("utf-8"),
        label="V3DT diagnostic artifact",
    )


def _read_private_json(path: Path, *, label: str) -> object:
    payload = read_private_file(path, label=label, max_bytes=_MAX_JSON_BYTES)
    return json.loads(payload.decode("utf-8"))


def _validated_log(path: Path) -> Path:
    validated = validate_private_file(path, label="V3DT tracking log")
    if validated.lstat().st_size > _MAX_LOG_BYTES:
        raise PrivatePathError(
            f"V3DT tracking log exceeds the {_MAX_LOG_BYTES}-byte analysis limit"
        )
    return validated

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
    out_dir = ensure_private_directory(args.output_dir, label="V3DT diagnostics")
    json_path = out_dir / f"v3dt_snapshot_{ts}.json"
    md_path = out_dir / f"v3dt_snapshot_{ts}.md"
    _write_output(json_path, json.dumps(snapshot, indent=2, ensure_ascii=True))
    _write_output(md_path, render_snapshot_markdown(snapshot))
    print(f"Wrote snapshot: {json_path}")
    print(f"Wrote snapshot summary: {md_path}")
    return 0


def _analyze_cmd(args: argparse.Namespace) -> int:
    log_path = Path(args.log).expanduser()
    if not log_path.exists():
        print(f"[FAIL] log file not found: {log_path}")
        return 1
    log_path = _validated_log(log_path)
    snapshot = None
    if args.snapshot:
        snap_path = Path(args.snapshot).expanduser()
        if not snap_path.exists():
            print(f"[FAIL] snapshot file not found: {snap_path}")
            return 1
        snapshot = _read_private_json(snap_path, label="V3DT snapshot")
    scale_sweep = _parse_scale_sweep(args.scale_sweep)
    report = analyze_tracking_log(log_path, snapshot=snapshot, scale_sweep=scale_sweep)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = ensure_private_directory(args.output_dir, label="V3DT diagnostics")
    json_path = out_dir / f"v3dt_report_{ts}.json"
    md_path = out_dir / f"v3dt_report_{ts}.md"
    _write_output(json_path, json.dumps(report, indent=2, ensure_ascii=True))
    _write_output(md_path, render_report_markdown(report))
    print(f"Wrote report: {json_path}")
    print(f"Wrote report summary: {md_path}")
    return 0


def _panel_cmd(args: argparse.Namespace) -> int:
    out_dir = ensure_private_directory(args.output_dir, label="V3DT diagnostics")
    snapshot_path = (
        Path(args.snapshot).expanduser()
        if args.snapshot
        else _pick_latest(out_dir, "v3dt_snapshot_*.json")
    )
    report_path = (
        Path(args.report).expanduser()
        if args.report
        else _pick_latest(out_dir, "v3dt_report_*.json")
    )
    if snapshot_path is None or report_path is None:
        print("[FAIL] Could not locate snapshot/report JSON. Provide --snapshot and --report.")
        return 1
    snapshot = _read_private_json(snapshot_path, label="V3DT snapshot")
    report = _read_private_json(report_path, label="V3DT report")
    html = render_panel_html(snapshot, report)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_path = out_dir / f"v3dt_panel_{ts}.html"
    _write_output(out_path, html)
    print(f"Wrote panel: {out_path}")
    return 0


def _serve_cmd(args: argparse.Namespace) -> int:
    if args.host not in {"127.0.0.1", "localhost"}:
        print("[FAIL] V3DT panels may only bind to the local loopback interface")
        return 1
    root = ensure_private_directory(args.dir, label="V3DT diagnostics")

    class _PrivatePanelHandler(http.server.BaseHTTPRequestHandler):
        def _send_panel(self, *, include_body: bool) -> None:
            requested = unquote(urlsplit(self.path).path).lstrip("/")
            if not _PANEL_NAME_RE.fullmatch(requested):
                self.send_error(404)
                return
            try:
                payload = read_private_file(
                    root / requested,
                    label="V3DT panel",
                    max_bytes=_MAX_PANEL_BYTES,
                )
            except PrivatePathError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if include_body:
                self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            self._send_panel(include_body=True)

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            self._send_panel(include_body=False)

    with socketserver.TCPServer((args.host, args.port), _PrivatePanelHandler) as httpd:
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
    snap.add_argument("--output-dir", default=str(_default_diagnostics_dir()))

    analyze = sub.add_parser("analyze", help="Analyze V3DT tracking NDJSON")
    analyze.add_argument("--log", required=True, help="NDJSON log from NOESIS_V3DT_DIAG_LOG")
    analyze.add_argument("--snapshot", default="", help="Snapshot JSON to enable reprojection checks")
    analyze.add_argument(
        "--scale-sweep",
        default="",
        help="Comma-separated world scale factors (e.g. 0.01,0.1,1,10) or 'auto'",
    )
    analyze.add_argument("--output-dir", default=str(_default_diagnostics_dir()))

    panel = sub.add_parser("panel", help="Generate HTML panel from snapshot + report")
    panel.add_argument("--snapshot", default="", help="Snapshot JSON (defaults to latest)")
    panel.add_argument("--report", default="", help="Report JSON (defaults to latest)")
    panel.add_argument("--output-dir", default=str(_default_diagnostics_dir()))

    serve = sub.add_parser("serve", help="Serve diagnostics directory")
    serve.add_argument("--dir", default=str(_default_diagnostics_dir()))
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
