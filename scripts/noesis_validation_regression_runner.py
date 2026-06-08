#!/usr/bin/env python3
"""Run registered Noesis/Menon validation fixtures as a regression suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.validation.artifacts import ArtifactIndex  # noqa: E402
from noesis.validation.core import CheckStatus  # noqa: E402
from noesis.validation.fixtures import FixtureRegistry, FixtureRegistryEntry  # noqa: E402
from noesis.validation.golden import compare_artifact_expectations  # noqa: E402
from noesis.validation.menon_browser import browser_snapshot_to_menon_trace  # noqa: E402
from noesis.validation.reports import write_markdown  # noqa: E402
from noesis.validation.telemetry import build_telemetry_report, build_track_audit, extract_telemetry_samples, read_ndjson, write_ndjson  # noqa: E402
from scripts.noesis_validation_menon_trace_report import run_trace_report  # noqa: E402
from scripts.noesis_validation_runner import _load_json, _run_fixture  # noqa: E402


STATUS_RANK = {
    "pass": 0,
    "skipped": 1,
    "warning": 2,
    "blocked": 3,
    "fail": 4,
}

FAILURE_DIAGNOSTICS = {
    "calibration_failure": "Inspect intrinsics, extrinsics, dewarped scope, and active calibration bundle inputs.",
    "transform_failure": "Inspect matrix convention, frame declarations, scale, axis handedness, and round-trip evidence.",
    "projection_failure": "Inspect pixel/world/BEV/Menon projection paths, footpoints, and reprojection overlays.",
    "scene_failure": "Inspect generated room geometry, mesh topology, depth/plane agreement, and asset scale.",
    "temporal_failure": "Inspect smoothing ownership, timestamps, identity continuity, occlusion handling, and path stability.",
    "semantic_failure": "Inspect room labels, doorway logic, object support, collisions, and zone consistency.",
    "sync_failure": "Inspect Noesis and Menon timestamps, WebSocket buffering, replay clocks, and render/update lag.",
    "model_failure": "Inspect detector, pose, depth, reconstruction model outputs, confidence, and hallucination evidence.",
    "data_quality_failure": "Inspect source frame quality, low light, glare, motion blur, occlusion, and fixture masks.",
    "infrastructure_failure": "Inspect missing fixtures, unavailable runtimes, Menon checkout, tool dependencies, and artifact paths.",
    "regression_failure": "Inspect expectation thresholds, golden artifacts, changed outputs, and recent code changes.",
}


def _case_kind(entry: FixtureRegistryEntry) -> str:
    tiers = set(entry.tiers)
    if "menon-browser" in tiers or "menon_browser" in entry.path.stem or "browser_snapshot" in entry.path.stem:
        return "menon_browser"
    if "menon-trace" in tiers or entry.path.name.endswith("_menon_trace.json") or "menon_trace" in entry.path.stem:
        return "menon_trace"
    if entry.path.suffix == ".ndjson" or "runtime-contract" in tiers:
        return "telemetry"
    return "fixture"


def _expected_failures(entry: FixtureRegistryEntry, summary: dict[str, Any]) -> list[str]:
    expected = entry.expected if isinstance(entry.expected, dict) else {}
    failures: list[str] = []
    expected_status = expected.get("status")
    if isinstance(expected_status, str) and expected_status in STATUS_RANK:
        if STATUS_RANK[str(summary["status"])] > STATUS_RANK[expected_status]:
            failures.append(f"status {summary['status']} is worse than expected {expected_status}")
    for summary_key, expected_key in (
        ("check_count", "min_check_count"),
        ("failure_count", "max_failure_count"),
        ("warning_count", "max_warning_count"),
        ("blocked_count", "max_blocked_count"),
    ):
        raw = expected.get(expected_key)
        if raw is None:
            continue
        try:
            value = int(summary.get(summary_key, 0))
            threshold = int(raw)
        except Exception:
            continue
        if expected_key.startswith("min_") and value < threshold:
            failures.append(f"{summary_key} {value} is below expected minimum {threshold}")
        if expected_key.startswith("max_") and value > threshold:
            failures.append(f"{summary_key} {value} is above expected maximum {threshold}")
    return failures


def _failure_categories(payload: dict[str, Any], regression_failures: list[str]) -> tuple[dict[str, int], str | None, str | None]:
    counts: dict[str, int] = defaultdict(int)
    checks = payload.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            status = str(check.get("status") or "")
            if status not in {"warning", "fail", "blocked"}:
                continue
            failure_type = check.get("failure_type")
            category = str(failure_type or f"{status}_without_failure_type")
            counts[category] += 1
    if regression_failures:
        counts["regression_failure"] += len(regression_failures)
    if not counts:
        return {}, None, None
    ordered = dict(sorted(counts.items()))
    dominant = max(ordered, key=lambda key: (ordered[key], key))
    return ordered, dominant, FAILURE_DIAGNOSTICS.get(dominant, "Inspect the failed checks and attached validation artifacts.")


def _write_telemetry_case(entry: FixtureRegistryEntry, *, output_dir: Path) -> dict[str, Any]:
    messages = read_ndjson(entry.path)
    artifacts = ArtifactIndex(output_dir)
    telemetry_copy = write_ndjson(messages, artifacts.path("telemetry/messages.ndjson"))
    artifacts.add("telemetry_messages_ndjson", telemetry_copy)
    samples = extract_telemetry_samples(messages)
    audit_path = artifacts.path("tracking/track_audit.json")
    audit_path.write_text(json.dumps(build_track_audit(samples.track_samples), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.add("track_audit_json", audit_path)
    index_path = artifacts.path("visual/index.json")
    artifacts.add("visual_index", index_path)
    artifacts.write("visual/index.json")
    report = build_telemetry_report(messages, run_id=entry.fixture_id)
    report.artifacts = artifacts.artifacts
    json_path = report.write_json(output_dir / "validation_report.json")
    md_path = write_markdown(report, output_dir / "validation_report.md")
    payload = report.to_dict()
    payload["_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return payload


def _write_fixture_case(entry: FixtureRegistryEntry, *, output_dir: Path) -> dict[str, Any]:
    payload = _load_json(entry.path)
    report = _run_fixture(payload, run_id=entry.fixture_id, artifact_dir=output_dir, fixture_dir=entry.path.parent)
    json_path = report.write_json(output_dir / "validation_report.json")
    md_path = write_markdown(report, output_dir / "validation_report.md")
    report_payload = report.to_dict()
    report_payload["_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report_payload


def _write_menon_browser_case(entry: FixtureRegistryEntry, *, output_dir: Path) -> dict[str, Any]:
    snapshot = _load_json(entry.path)
    source_dir = output_dir / "browser"
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_copy = source_dir / "browser_snapshot.json"
    snapshot_copy.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trace = browser_snapshot_to_menon_trace(
        snapshot,
        run_id=entry.fixture_id,
        page_url=str(snapshot.get("href") or ""),
        raw_snapshot_path="browser/browser_snapshot.json",
    )
    trace_input = source_dir / "trace_input.json"
    trace_input.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_path, md_path, payload = run_trace_report(trace_input, output_dir=output_dir.parent, run_id=entry.fixture_id)
    payload["_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return payload


def run_regression_suite(
    registry_path: str | Path,
    *,
    output_dir: str | Path = "diagnostics/validation",
    run_id: str = "",
    fixture_ids: list[str] | None = None,
    require_menon_root: bool = False,
    menon_root: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    registry = FixtureRegistry.load(registry_path)
    selected_ids = fixture_ids or list(registry.entries)
    suite_id = str(run_id or f"regression_{int(time.time())}")
    suite_dir = Path(output_dir) / suite_id
    case_dir = suite_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for fixture_id in selected_ids:
        entry = registry.entries[fixture_id]
        kind = _case_kind(entry)
        target = case_dir / entry.fixture_id
        if kind == "menon_trace":
            json_path, md_path, payload = run_trace_report(
                entry.path,
                output_dir=case_dir,
                run_id=entry.fixture_id,
                menon_root=menon_root,
                require_menon_root=require_menon_root,
            )
            payload["_paths"] = {"json": str(json_path), "markdown": str(md_path)}
        elif kind == "menon_browser":
            payload = _write_menon_browser_case(entry, output_dir=target)
        elif kind == "telemetry":
            payload = _write_telemetry_case(entry, output_dir=target)
        else:
            payload = _write_fixture_case(entry, output_dir=target)
        summary = dict(payload["summary"])
        regression_failures = _expected_failures(entry, summary)
        expected = entry.expected if isinstance(entry.expected, dict) else {}
        artifact_expectations = expected.get("artifacts")
        artifact_comparisons: list[dict[str, Any]] = []
        if isinstance(artifact_expectations, list):
            artifact_failures, artifact_comparisons = compare_artifact_expectations(
                payload,
                run_dir=target,
                expectations=[item for item in artifact_expectations if isinstance(item, dict)],
                registry_dir=registry.path.parent,
                diff_dir=target / "regression_diffs",
            )
            regression_failures.extend(artifact_failures)
        failure_categories, dominant_failure_category, suggested_diagnostic_focus = _failure_categories(payload, regression_failures)
        cases.append(
            {
                "fixture_id": entry.fixture_id,
                "kind": kind,
                "status": summary["status"],
                "regression_status": "fail" if regression_failures else "pass",
                "regression_failures": regression_failures,
                "failure_categories": failure_categories,
                "dominant_failure_category": dominant_failure_category,
                "suggested_diagnostic_focus": suggested_diagnostic_focus,
                "artifact_comparisons": artifact_comparisons,
                "level": summary["level"],
                "check_count": summary["check_count"],
                "failure_count": summary["failure_count"],
                "warning_count": summary["warning_count"],
                "blocked_count": summary["blocked_count"],
                "json": payload["_paths"]["json"],
                "markdown": payload["_paths"]["markdown"],
            }
        )
    reported_status = max((case["status"] for case in cases), key=lambda status: STATUS_RANK[str(status)]) if cases else CheckStatus.SKIPPED.value
    regression_failure_count = sum(1 for case in cases if case["regression_status"] == "fail")
    suite_status = "fail" if regression_failure_count else reported_status
    aggregate_failure_categories: dict[str, int] = defaultdict(int)
    for case in cases:
        for category, count in dict(case.get("failure_categories") or {}).items():
            aggregate_failure_categories[str(category)] += int(count)
    dominant_failure_category = None
    suggested_diagnostic_focus = None
    if aggregate_failure_categories:
        dominant_failure_category = max(aggregate_failure_categories, key=lambda key: (aggregate_failure_categories[key], key))
        suggested_diagnostic_focus = FAILURE_DIAGNOSTICS.get(dominant_failure_category, "Inspect the failed checks and attached validation artifacts.")
    summary_payload = {
        "schema_version": 1,
        "run_id": suite_id,
        "registry": str(registry.path),
        "status": suite_status,
        "reported_status": reported_status,
        "failure_categories": dict(sorted(aggregate_failure_categories.items())),
        "dominant_failure_category": dominant_failure_category,
        "suggested_diagnostic_focus": suggested_diagnostic_focus,
        "case_count": len(cases),
        "regression_failure_count": regression_failure_count,
        "failure_count": sum(1 for case in cases if case["status"] == "fail"),
        "blocked_count": sum(1 for case in cases if case["status"] == "blocked"),
        "warning_count": sum(1 for case in cases if case["status"] == "warning"),
        "pass_count": sum(1 for case in cases if case["status"] == "pass"),
        "cases": cases,
    }
    summary_path = suite_dir / "regression_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path, summary_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run registered Noesis/Menon validation fixtures as a regression suite.")
    parser.add_argument("--fixture-registry", default="plans/noesis_menon_validation/fixture_registry.json")
    parser.add_argument("--fixture-id", action="append", default=[], help="Limit run to one or more fixture ids.")
    parser.add_argument("--output-dir", default="diagnostics/validation")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--menon-root", default="")
    parser.add_argument("--require-menon-root", action="store_true")
    args = parser.parse_args()
    summary_path, payload = run_regression_suite(
        args.fixture_registry,
        output_dir=args.output_dir,
        run_id=args.run_id,
        fixture_ids=list(args.fixture_id),
        require_menon_root=bool(args.require_menon_root),
        menon_root=args.menon_root,
    )
    print(json.dumps({"status": payload["status"], "summary": str(summary_path), "case_count": payload["case_count"]}, indent=2))
    return 1 if payload["status"] in ("fail", "blocked") else 0


if __name__ == "__main__":
    sys.exit(main())
