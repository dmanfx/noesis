#!/usr/bin/env python3
"""Validate AGENTS/docs consistency for Noesis governance docs."""

from __future__ import annotations

import sys
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_AGENTS_FILES = [
    Path("AGENTS.md"),
    Path("docs/AGENTS.md"),
    Path("noesis/AGENTS.md"),
    Path("plans/AGENTS.md"),
    Path("plans/DS8/v3dt/AGENTS.md"),
]

REQUIRED_POLICY_HEADER = "## Policy precedence"

REQUIRED_REFERENCES = [
    Path("plans/DS8/ds8_master_work_orders.md"),
    Path("plans/DS8/ds8_design_decisions.md"),
    Path("docs/DS8_README_FOR_AGENTS.md"),
    Path("docs/DS8_testing_guide.md"),
    Path("docs/DS8_api_contracts_ws.md"),
    Path("docs/DS8_api_contracts_rest.md"),
    Path("docs/DS8_metadata_contracts.md"),
    Path("docs/DS8_Baselines.md"),
    Path("docs/CODEBASE_DESCRIPTION.md"),
    Path("docs/DS8_MIGRATION_KNOWLEDGE_BASE.md"),
    Path("docs/DS8_v3dt_forensics.md"),
    Path("plans/DS8/v3dt/integration_plan.md"),
    Path("plans/DS8/v3dt/research_notes.md"),
    Path("plans/DS8/v3dt/work_order.md"),
    Path("config/infer_v3dt_baseline.yaml"),
    Path("config/cameras_v3dt_baseline.yaml"),
    Path("config/v3dt/nvtracker_v3dt_baseline.yml"),
]

V3DT_RECOVERY_FILES = [
    Path("plans/DS8/v3dt/AGENTS.md"),
    Path("plans/DS8/v3dt/README.md"),
]

V3DT_REQUIRED_CALIBRATION_FLAG = "--calibration"

BANNED_PATTERNS_IN_ACTIVE_DOCS = [
    re.compile(r"\bds7\b", re.IGNORECASE),
    re.compile(r"\bmain\.py\b", re.IGNORECASE),
]


def _read_text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _exists(path: Path) -> bool:
    return (REPO_ROOT / path).exists()


def main() -> int:
    failures: list[str] = []

    for agents_file in ACTIVE_AGENTS_FILES:
        if not _exists(agents_file):
            failures.append(f"Missing AGENTS file: {agents_file}")
            continue
        text = _read_text(agents_file)
        if REQUIRED_POLICY_HEADER not in text:
            failures.append(
                f"Missing '{REQUIRED_POLICY_HEADER}' in: {agents_file}"
            )

    for ref in REQUIRED_REFERENCES:
        if not _exists(ref):
            failures.append(f"Missing referenced file: {ref}")

    for v3dt_file in V3DT_RECOVERY_FILES:
        if not _exists(v3dt_file):
            failures.append(f"Missing V3DT doc for recovery check: {v3dt_file}")
            continue
        text = _read_text(v3dt_file)
        if V3DT_REQUIRED_CALIBRATION_FLAG not in text:
            failures.append(
                f"Missing calibration recovery flag in: {v3dt_file}"
            )

    docs_root = REPO_ROOT / "docs"
    if docs_root.exists():
        for path in sorted(docs_root.rglob("*.md")):
            try:
                rel = path.relative_to(REPO_ROOT)
            except Exception:
                rel = path
            if str(rel).startswith("docs/history/"):
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in BANNED_PATTERNS_IN_ACTIVE_DOCS:
                if pattern.search(text):
                    failures.append(
                        f"Banned reference '{pattern.pattern}' found in active doc: {rel}"
                    )

    if failures:
        print("AGENTS/docs consistency check FAILED:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("AGENTS/docs consistency check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
