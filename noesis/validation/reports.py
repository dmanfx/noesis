from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import ValidationReport


def render_markdown(report: ValidationReport) -> str:
    payload = report.to_dict()
    summary: dict[str, Any] = payload["summary"]
    lines = [
        f"# Validation Report: {payload['run_id']}",
        "",
        f"- Status: {summary['status']}",
        f"- Level: {summary['level']}",
        f"- Checks: {summary['check_count']}",
        f"- Pass: {summary['pass_count']}",
        f"- Warning: {summary['warning_count']}",
        f"- Fail: {summary['failure_count']}",
        f"- Blocked: {summary['blocked_count']}",
        "",
        "## Checks",
        "",
        "| ID | Domain | Status | Level | Detail |",
        "|---|---|---|---|---|",
    ]
    for check in payload.get("checks", []):
        detail = str(check.get("detail", "")).replace("|", "\\|")
        lines.append(
            f"| {check.get('id', '')} | {check.get('domain', '')} | "
            f"{check.get('status', '')} | {check.get('level', '')} | {detail} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown(report: ValidationReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(report), encoding="utf-8")
    return target
