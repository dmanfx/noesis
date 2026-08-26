#!/usr/bin/env python3
"""Validate saved or live DS9.1 WebSocket telemetry through the shared toolbox."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.validation.artifacts import ArtifactIndex  # noqa: E402
from noesis.validation.core import SourceMetadata  # noqa: E402
from noesis.validation.reports import write_markdown  # noqa: E402
from noesis.validation.telemetry import (  # noqa: E402
    build_telemetry_report,
    build_track_audit,
    extract_telemetry_samples,
    read_ndjson,
    write_ndjson,
)


async def _capture_ws(uri: str, *, duration_s: float) -> list[dict[str, Any]]:
    try:
        import websockets  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"websockets package required for --ws capture: {exc}") from exc

    messages: list[dict[str, Any]] = []
    async with websockets.connect(uri, max_size=None) as ws:
        end_at = time.time() + max(1.0, float(duration_s))
        while time.time() < end_at:
            timeout = max(0.1, end_at - time.time())
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(2.0, timeout))
            except asyncio.TimeoutError:
                continue
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                messages.append(payload)
    return messages


def _source_metadata(args: argparse.Namespace) -> SourceMetadata:
    return SourceMetadata(
        repo="Noesis_Devel",
        pipeline_config=str(args.pipeline_config) if args.pipeline_config else None,
        cameras_config=str(args.cameras_config) if args.cameras_config else None,
        menon_available=None,
        menon_revision=None,
    )


def _load_messages(args: argparse.Namespace) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for path in args.input:
        rows.extend(read_ndjson(path))
    if args.ws:
        rows.extend(asyncio.run(_capture_ws(str(args.ws), duration_s=float(args.duration))))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DS9.1 tracking/BEV telemetry from NDJSON or live WebSocket.")
    parser.add_argument("--input", action="append", default=[], help="NDJSON file with one WebSocket JSON message per line.")
    parser.add_argument("--ws", default="", help="Live WebSocket URL to capture.")
    parser.add_argument("--duration", type=float, default=20.0, help="Live WebSocket capture duration seconds.")
    parser.add_argument("--output-dir", default="diagnostics/validation", help="Output directory for reports.")
    parser.add_argument("--run-id", default="", help="Override report run id.")
    parser.add_argument("--pipeline-config", default="DS9/config/infer.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument("--expected-world-frame", default="backend_world_m")
    parser.add_argument("--expected-bev-frame-mode", default="world")
    args = parser.parse_args()

    if not args.input and not args.ws:
        parser.error("provide --input and/or --ws")

    messages = _load_messages(args)
    run_id = str(args.run_id or f"telemetry_{int(time.time())}")
    output_dir = Path(args.output_dir) / run_id
    artifacts = ArtifactIndex(output_dir)
    telemetry_copy = write_ndjson(messages, artifacts.path("telemetry/messages.ndjson"))
    artifacts.add("telemetry_messages_ndjson", telemetry_copy)

    report = build_telemetry_report(
        messages,
        run_id=run_id,
        source=_source_metadata(args),
        expected_world_frame=str(args.expected_world_frame),
        expected_bev_frame_mode=str(args.expected_bev_frame_mode),
    )
    samples = extract_telemetry_samples(messages)
    audit_path = artifacts.path("tracking/track_audit.json")
    audit_path.write_text(json.dumps(build_track_audit(samples.track_samples), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.add("track_audit_json", audit_path)
    index_path = artifacts.path("visual/index.json")
    artifacts.add("visual_index", index_path)
    artifacts.write("visual/index.json")
    report.artifacts = artifacts.artifacts

    json_path = report.write_json(output_dir / "validation_report.json")
    md_path = write_markdown(report, output_dir / "validation_report.md")
    print(
        json.dumps(
            {
                "status": report.status.value,
                "level": report.level.value,
                "json": str(json_path),
                "markdown": str(md_path),
                "message_count": len(messages),
            },
            indent=2,
        )
    )
    return 1 if report.status.value == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
