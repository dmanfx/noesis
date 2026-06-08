#!/usr/bin/env python3
"""Validate a captured Menon placement trace through the shared toolbox."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.validation.artifacts import ArtifactIndex  # noqa: E402
from noesis.validation.menon_trace import (  # noqa: E402
    build_menon_trace_report,
    load_menon_trace,
    parse_bev_menon_trails,
    parse_menon_avatars,
    parse_menon_placements,
    parse_menon_trails,
    parse_transform_audits,
)
from noesis.validation.reports import write_markdown  # noqa: E402


def _seq(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else []


def _git_revision(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=path,
            check=True,
            text=True,
            capture_output=True,
        )
        revision = proc.stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=path,
            text=True,
            capture_output=True,
        ).returncode != 0
        return f"{revision}-dirty" if dirty else revision
    except Exception:
        return None


def resolve_menon_root(value: str | None) -> Path | None:
    raw = value or os.environ.get("MENON_ROOT") or ""
    if not raw.strip():
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _with_source_metadata(payload: Mapping[str, Any], *, menon_root: Path | None) -> dict[str, Any]:
    trace = dict(payload)
    source = dict(payload.get("source") if isinstance(payload.get("source"), Mapping) else {})
    if menon_root is not None:
        source["menon_available"] = True
        revision = _git_revision(menon_root)
        if revision and not source.get("menon_revision"):
            source["menon_revision"] = revision
    elif "menon_available" not in source:
        source["menon_available"] = False
    trace["source"] = source
    return trace


def _index_external_artifacts(index: ArtifactIndex, payload: Mapping[str, Any]) -> None:
    raw_artifacts = payload.get("artifacts")
    if isinstance(raw_artifacts, Mapping):
        for key, value in raw_artifacts.items():
            index.artifacts[f"trace_{key}"] = value
    for idx, item in enumerate(_seq(payload.get("screenshots"))):
        if isinstance(item, Mapping):
            name = item.get("id") or item.get("name") or f"screenshot_{idx:03d}"
            index.artifacts[f"trace_screenshot_{name}"] = dict(item)
        elif isinstance(item, str):
            index.artifacts[f"trace_screenshot_{idx:03d}"] = item


def run_trace_report(
    trace_path: str | Path,
    *,
    output_dir: str | Path = "diagnostics/validation",
    run_id: str = "",
    menon_root: str | None = None,
    require_menon_root: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    payload = load_menon_trace(trace_path)
    resolved_menon_root = resolve_menon_root(menon_root)
    run_name = str(run_id or payload.get("run_id") or f"menon_trace_{int(time.time())}")
    run_dir = Path(output_dir) / run_name
    artifacts = ArtifactIndex(run_dir)

    trace_copy = artifacts.path("menon/trace.json")
    trace_copy.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.add("menon_trace_json", trace_copy)

    audit = {
        "placement_count": len(parse_menon_placements(payload)),
        "trail_count": len(parse_menon_trails(payload)),
        "bev_menon_trail_count": len(parse_bev_menon_trails(payload)),
        "avatar_sample_count": len(parse_menon_avatars(payload)),
        "transform_audit_count": len(parse_transform_audits(payload)),
        "menon_root_required": bool(require_menon_root),
        "menon_root_available": resolved_menon_root is not None,
    }
    audit_path = artifacts.path("menon/trace_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.add("menon_trace_audit_json", audit_path)
    _index_external_artifacts(artifacts, payload)
    index_path = artifacts.path("visual/index.json")
    artifacts.add("visual_index", index_path)
    artifacts.write("visual/index.json")

    trace_for_report = _with_source_metadata(payload, menon_root=resolved_menon_root)
    report = build_menon_trace_report(
        trace_for_report,
        run_id=run_name,
        menon_root=str(resolved_menon_root) if resolved_menon_root else None,
        require_menon_root=bool(require_menon_root),
    )
    report.artifacts = artifacts.artifacts
    json_path = report.write_json(run_dir / "validation_report.json")
    md_path = write_markdown(report, run_dir / "validation_report.md")
    return json_path, md_path, report.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Noesis-to-Menon placement trace.")
    parser.add_argument("--trace", required=True, help="JSON trace exported from Menon/debug tooling.")
    parser.add_argument("--output-dir", default="diagnostics/validation", help="Output directory for reports.")
    parser.add_argument("--run-id", default="", help="Override report run id.")
    parser.add_argument("--menon-root", default="", help="Menon checkout root. Falls back to MENON_ROOT when omitted.")
    parser.add_argument(
        "--require-menon-root",
        action="store_true",
        help="Mark Menon-specific evidence blocked when no valid Menon checkout is available.",
    )
    args = parser.parse_args()

    json_path, md_path, payload = run_trace_report(
        args.trace,
        output_dir=args.output_dir,
        run_id=args.run_id,
        menon_root=args.menon_root,
        require_menon_root=args.require_menon_root,
    )
    print(
        json.dumps(
            {
                "status": payload["summary"]["status"],
                "level": payload["summary"]["level"],
                "json": str(json_path),
                "markdown": str(md_path),
                "trace_audit": payload["artifacts"].get("menon_trace_audit_json"),
            },
            indent=2,
        )
    )
    return 1 if payload["summary"]["status"] in ("fail", "blocked") else 0


if __name__ == "__main__":
    sys.exit(main())
