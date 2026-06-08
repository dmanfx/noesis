#!/usr/bin/env python3
"""Capture Menon browser debug state and validate it as a Noesis/Menon trace."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.validation.menon_browser import browser_snapshot_to_menon_trace  # noqa: E402
from scripts.noesis_validation_menon_trace_report import run_trace_report  # noqa: E402


BROWSER_SNAPSHOT_JS = r"""
() => {
  const clone = (value) => {
    if (value === undefined) return null;
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (err) {
      return { __clone_error: String(err && err.message ? err.message : err) };
    }
  };
  const call = (fn) => {
    try {
      if (typeof fn !== 'function') return null;
      return clone(fn());
    } catch (err) {
      return { __call_error: String(err && err.message ? err.message : err) };
    }
  };
  const ws = window.webSocketClient || {};
  const menonTrackDebugFn = ws.getMenonTrackDebugSnapshot || window.getMenonTrackDebugSnapshot;
  const menonCameraReprojectionDebugFn =
    ws.getMenonCameraReprojectionDebug ||
    ws.getCameraReprojectionDebug ||
    window.getMenonCameraReprojectionDebug ||
    window.getCameraReprojectionDebug ||
    window.menonCameraReprojectionDebug;
  return {
    schema_version: 1,
    capturedAtMs: Date.now(),
    href: window.location && window.location.href,
    trackingProjectionMode: window.trackingProjectionMode || null,
    rawActiveTracksByCamera: clone(window.__menonRawActiveTracksByCamera || {}),
    trackingRealtimeByCamera: clone(window.__trackingRealtimeByCamera || {}),
    trackingStatsByCamera: clone(window.__trackingStatsByCamera || {}),
    latestRawTrackInfos: clone(window.latestRawTrackInfos || []),
    latestRawTrackingPaths: clone(window.latestRawTrackingPaths || []),
    latestTrackInfos: clone(window.latestTrackInfos || []),
    latestTrackingPaths: clone(window.latestTrackingPaths || []),
    recentReprojectionTrackInfos: clone(window.__reprojectionRecentTrackInfos || []),
    menonTrackDebug: call(() => menonTrackDebugFn && menonTrackDebugFn(60000)),
    reprojectionDebugInfo: call(() => ws.getReprojectionDebugInfo && ws.getReprojectionDebugInfo()),
    reprojectionCompareInfo: call(() => ws.getReprojectionCompareInfo && ws.getReprojectionCompareInfo()),
    cameraReprojections: clone(window.__menonCameraReprojections || window.cameraReprojections || []),
    menonCameraReprojectionDebug: call(() => menonCameraReprojectionDebugFn && menonCameraReprojectionDebugFn()),
    projectionShellAlignmentDebug: call(() => ws.getProjectionShellAlignmentForTrackingDebug && ws.getProjectionShellAlignmentForTrackingDebug()),
    virtualTwinAlignmentDebug: call(() => ws.getVirtualTwinAlignmentForTrackingDebug && ws.getVirtualTwinAlignmentForTrackingDebug()),
    windowReprojectionDebugInfo: call(() => window.reprojectionDebugInfo && window.reprojectionDebugInfo()),
    windowReprojectionCompareReport: call(() => window.reprojectionCompareReport && window.reprojectionCompareReport()),
    windowReprojectionCameraDebug: call(() => window.reprojectionCameraDebug && window.reprojectionCameraDebug()),
    windowReprojectionPoseDebug: call(() => window.reprojectionPoseDebug && window.reprojectionPoseDebug()),
    windowReprojectionPoseParityDebug: call(() => window.reprojectionPoseParityDebug && window.reprojectionPoseParityDebug()),
    calibrationData: call(() => window.calibrationManager && window.calibrationManager.getCalibrationData && window.calibrationManager.getCalibrationData()),
    coordinateTransformStatus: {
      isTransformationSet: call(() => window.coordinateTransform && window.coordinateTransform.getIsTransformationSet && window.coordinateTransform.getIsTransformationSet()),
      unitScales: call(() => window.coordinateTransform && window.coordinateTransform.getUnitScales && window.coordinateTransform.getUnitScales())
    }
  };
}
"""


async def capture_browser_snapshot(
    *,
    url: str,
    duration_s: float,
    screenshot_path: Path | None,
    headless: bool,
    timeout_ms: int,
) -> dict[str, Any]:
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Python Playwright is required for browser capture. Install it in this environment and run `python -m playwright install chromium`."
        ) from exc

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if duration_s > 0:
                await page.wait_for_timeout(int(duration_s * 1000))
            snapshot = await page.evaluate(BROWSER_SNAPSHOT_JS)
            if screenshot_path is not None:
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(screenshot_path), full_page=True)
            if not isinstance(snapshot, Mapping):
                raise RuntimeError("Menon browser snapshot did not serialize as a JSON object.")
            return dict(snapshot)
        finally:
            await browser.close()


def write_trace_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    output_path: str | Path,
    run_id: str = "",
    page_url: str = "",
    screenshot_path: str | Path | None = None,
    raw_snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_snapshot_path) if raw_snapshot_path is not None else output.with_name("browser_snapshot.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        evidence_path = str(raw_path.relative_to(output.parent))
    except Exception:
        evidence_path = str(raw_path)
    trace = browser_snapshot_to_menon_trace(
        snapshot,
        run_id=run_id,
        page_url=page_url,
        screenshot_path=screenshot_path,
        raw_snapshot_path=evidence_path,
    )
    output.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return trace


def _resolve_screenshot_path(raw: str | None, output_path: Path) -> Path | None:
    if raw is None:
        return None
    if raw.strip():
        return Path(raw)
    return output_path.with_name("menon_browser_capture.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Menon browser debug state and validate Noesis-to-Menon placement.")
    parser.add_argument("--url", default="http://127.0.0.1:5173", help="Menon browser URL to capture.")
    parser.add_argument("--output", required=True, help="Trace JSON output path.")
    parser.add_argument("--run-id", default="", help="Run id for the trace and optional validation report.")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds to wait after the page loads before capture.")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Page navigation timeout in milliseconds.")
    parser.add_argument("--screenshot", nargs="?", const="", default=None, help="Optional screenshot path. Omit value to place it beside --output.")
    parser.add_argument("--headed", action="store_true", help="Show the browser window instead of running headless.")
    parser.add_argument("--validate", action="store_true", help="Run the shared Menon trace report after capture.")
    parser.add_argument("--report-output-dir", default="diagnostics/validation", help="Output directory for --validate reports.")
    args = parser.parse_args()

    output_path = Path(args.output)
    screenshot_path = _resolve_screenshot_path(args.screenshot, output_path)
    try:
        snapshot = asyncio.run(
            capture_browser_snapshot(
                url=args.url,
                duration_s=float(args.duration),
                screenshot_path=screenshot_path,
                headless=not args.headed,
                timeout_ms=int(args.timeout_ms),
            )
        )
        trace = write_trace_from_snapshot(
            snapshot,
            output_path=output_path,
            run_id=args.run_id,
            page_url=args.url,
            screenshot_path=screenshot_path,
        )
        payload: dict[str, Any] = {
            "trace": str(output_path),
            "browser_snapshot": str(output_path.with_name("browser_snapshot.json")),
            "placement_count": len(trace.get("placements") or []),
            "has_world_to_menon_transform": "world_to_menon_col_major" in trace,
        }
        if screenshot_path is not None:
            payload["screenshot"] = str(screenshot_path)
        if args.validate:
            json_path, md_path, report = run_trace_report(
                output_path,
                output_dir=args.report_output_dir,
                run_id=args.run_id or trace.get("run_id") or "",
            )
            payload["validation"] = {
                "status": report["summary"]["status"],
                "level": report["summary"]["level"],
                "json": str(json_path),
                "markdown": str(md_path),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 1 if report["summary"]["status"] in ("fail", "blocked") else 0
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Menon browser capture failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
