#!/usr/bin/env python3
"""Check that active documentation describes the native DS9.1 application."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_AGENTS_FILES = (
    Path("AGENTS.md"),
    Path("docs/AGENTS.md"),
    Path("DS9/AGENTS.md"),
    Path("noesis/AGENTS.md"),
    Path("plans/AGENTS.md"),
    Path("plans/household_identity/AGENTS.md"),
    Path("plans/noesis_menon_validation/AGENTS.md"),
)

REQUIRED_REFERENCES = (
    Path("README.md"),
    Path("docs/README.md"),
    Path("docs/CODEBASE_DESCRIPTION.md"),
    Path("docs/runtime_baseline.md"),
    Path("docs/testing_guide.md"),
    Path("docs/api_contracts_ws.md"),
    Path("docs/api_contracts_rest.md"),
    Path("docs/metadata_contracts.md"),
    Path("docs/architecture_decisions.md"),
    Path("docs/upgrade_history.md"),
    Path("docs/flow_diagram_high_level.md"),
    Path("docs/flow_diagram_low_level.md"),
    Path("docs/history/README.md"),
    Path("DS9/README.md"),
    Path("DS9/PIPELINE_GRAPH.md"),
    Path("DS9/docs/runtime_host_boundary.md"),
    Path("DS9/docs/validation_runbook.md"),
    Path("DS9/docs/deepstream_9_1_agent_skills.md"),
    Path(".agents/skills/deepstream-dev/SKILL.md"),
    Path(".agents/skills/deepstream-profile-pipeline/SKILL.md"),
    Path(".agents/skills/deepstream-run-mv3dt/SKILL.md"),
    Path("DS9/docs/runtime_ownership.yaml"),
    Path("DS9/asset_manifest.yaml"),
    Path("DS9/scripts/run_canonical_runtime_host.py"),
    Path("DS9/scripts/native_noesis_wait_ready.py"),
    Path("DS9/scripts/zero_copy_stats_smoke_test.py"),
    Path("DS9/scripts/run_canonical_engine_maintenance_host.sh"),
    Path("scripts/webrtc_gateway_smoke_test.py"),
    Path("DS9/tests/test_world_snapshot_runtime.py"),
    Path("tests/test_active_floorplan_registry.py"),
    Path("plans/archive/README.md"),
    Path("plans/ds91_native_host_only_migration.md"),
)

CURRENT_DOC_ROOTS = (
    Path("docs"),
    Path("DS9/docs"),
    Path("plans/household_identity"),
    Path("plans/noesis_menon_validation"),
)

CURRENT_STANDALONE_DOCS = (
    Path("README.md"),
    Path("AGENTS.md"),
    Path("DS9/README.md"),
    Path("DS9/PIPELINE_GRAPH.md"),
    Path("DS9/DS9_REBUILD_AND_SMOKE_GATES.md"),
    Path("DS9/AGENTS.md"),
    Path("noesis/AGENTS.md"),
    Path("plans/AGENTS.md"),
    Path("plans/ds91_native_host_only_migration.md"),
    Path("contracts/README.md"),
    Path("oai2-fe/README.md"),
    Path("reid/README.md"),
    Path("testpipelines/roomform/README.md"),
    Path("tools/mapanything_phone_scan/README.md"),
    Path("utils/onnx2trt/README.md"),
    Path(".agents/skills/deepstream-dev/SKILL.md"),
    Path(".agents/skills/deepstream-profile-pipeline/SKILL.md"),
    Path(".agents/skills/deepstream-run-mv3dt/SKILL.md"),
    Path("DS9/csrc/nvdsroiexclude/README.md"),
    Path("DS9/gst-plugins/noesiseos/README.md"),
    Path("DS9/gst-plugins/noesisforceidr/README.md"),
    Path("DS9/config/v3dt/living_family_phone_optimized/README.md"),
    Path("DS9/config/v3dt/living_kitchen_tracking_candidate/README.md"),
    Path("DS9/config/v3dt/living_room_optimized/README.md"),
)

ARCHIVE_PREFIXES = (
    "docs/history/",
    "plans/archive/",
)

STALE_OPERATIONAL_PATTERNS = (
    re.compile(r"plans/DS8/"),
    re.compile(r"docs/DS8_[A-Za-z0-9_.-]+"),
    re.compile(r"/opt/nvidia/deepstream/deepstream-9\.0"),
    re.compile(r"TensorRT\s+10\.14\.1\.48", re.IGNORECASE),
    re.compile(r"CUDA\s+13\.1\b", re.IGNORECASE),
    re.compile(r"\bcanonical DS8\b", re.IGNORECASE),
    re.compile(r"\bDS8 is canonical\b", re.IGNORECASE),
    re.compile(r"python(?:3)?\s+noesis/ds8_runtime\.py", re.IGNORECASE),
    re.compile(r"VITE_REST_(?:URL|PORT)[^\n]*8082", re.IGNORECASE),
)

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _active_markdown_files() -> list[Path]:
    found = set(CURRENT_STANDALONE_DOCS)
    for root in CURRENT_DOC_ROOTS:
        absolute = REPO_ROOT / root
        if not absolute.exists():
            continue
        for path in absolute.rglob("*.md"):
            rel = path.relative_to(REPO_ROOT)
            if any(rel.as_posix().startswith(prefix) for prefix in ARCHIVE_PREFIXES):
                continue
            found.add(rel)
    return sorted(path for path in found if (REPO_ROOT / path).is_file())


def _local_link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target or target.startswith(("http://", "https://", "mailto:", "/")):
        return None
    if any(marker in target for marker in ("<", ">", "*")):
        return None
    return (REPO_ROOT / source.parent / target).resolve()


def main() -> int:
    failures: list[str] = []

    for agents_file in ACTIVE_AGENTS_FILES:
        absolute = REPO_ROOT / agents_file
        if not absolute.is_file():
            failures.append(f"Missing AGENTS file: {agents_file}")
            continue
        if "## Policy precedence" not in absolute.read_text(encoding="utf-8"):
            failures.append(f"Missing policy precedence section: {agents_file}")

    for ref in REQUIRED_REFERENCES:
        if not (REPO_ROOT / ref).exists():
            failures.append(f"Missing current reference: {ref}")

    for rel in _active_markdown_files():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for pattern in STALE_OPERATIONAL_PATTERNS:
            if pattern.search(text):
                failures.append(
                    f"Stale operational reference {pattern.pattern!r}: {rel}"
                )
        for raw_target in MARKDOWN_LINK.findall(text):
            target = _local_link_target(rel, raw_target)
            if target is not None and not target.exists():
                failures.append(f"Broken local link in {rel}: {raw_target}")

    for diagram in (
        Path("README.md"),
        Path("docs/flow_diagram_high_level.md"),
        Path("docs/flow_diagram_low_level.md"),
        Path("DS9/PIPELINE_GRAPH.md"),
    ):
        if "```mermaid" not in (REPO_ROOT / diagram).read_text(encoding="utf-8"):
            failures.append(f"Missing Mermaid diagram: {diagram}")

    if failures:
        print("AGENTS/docs consistency check FAILED:")
        for item in failures:
            print(f"- {item}")
        return 1

    print(
        "AGENTS/docs consistency check PASSED "
        f"({len(_active_markdown_files())} active Markdown files)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
