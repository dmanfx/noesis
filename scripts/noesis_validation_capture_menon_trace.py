#!/usr/bin/env python3
"""Capture Menon browser debug state and validate it as a Noesis/Menon trace."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.validation.menon_browser import browser_snapshot_to_menon_trace  # noqa: E402
from noesis_core.private_paths import (  # noqa: E402
    PrivatePathError,
    atomic_write_private_file,
    ensure_private_directory,
    read_private_file,
    validate_private_file,
)
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
    authState: {
      authenticated: window.menonAuth?.state?.authenticated === true,
      role: String(window.menonAuth?.state?.role || window.menonAuth?.role || '')
    },
    rawActiveTracksByCamera: clone(window.__menonRawActiveTracksByCamera || {}),
    trackingRealtimeByCamera: clone(window.__trackingRealtimeByCamera || {}),
    trackingStatsByCamera: clone(window.__trackingStatsByCamera || {}),
    latestRawTrackInfos: clone(window.latestRawTrackInfos || []),
    latestRawTrackingPaths: clone(window.latestRawTrackingPaths || []),
    latestTrackInfos: clone(window.latestTrackInfos || []),
    latestTrackingPaths: clone(window.latestTrackingPaths || []),
    recentReprojectionTrackInfos: clone(window.__reprojectionRecentTrackInfos || []),
    menonTrackDebug: call(() => menonTrackDebugFn && menonTrackDebugFn(60000)),
    canonicalWorldState: call(() => window.getCanonicalWorldState && window.getCanonicalWorldState()),
    canonicalWorldPresentation: clone(window.__canonicalWorldPresentation || null),
    promotedSceneCohort: call(() => window.getPromotedSceneCohortState && window.getPromotedSceneCohortState()),
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


def _origin(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(str(value))
    except Exception:
        return None
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if scheme not in {"http", "https"} or not hostname or parsed.username or parsed.password:
        return None
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError:
        return None
    return scheme, hostname, port


def validate_browser_auth_proof(
    snapshot: Mapping[str, Any],
    session_payload: Mapping[str, Any],
    *,
    requested_url: str,
    final_url: str,
    checked_at_ms: int,
) -> dict[str, Any]:
    requested_origin = _origin(requested_url)
    final_origin = _origin(final_url)
    snapshot_origin = _origin(str(snapshot.get("href") or ""))
    if requested_origin is None or final_origin != requested_origin or snapshot_origin != final_origin:
        raise RuntimeError("Menon browser capture changed origin or did not report its exact final page URL.")
    if str(snapshot.get("href") or "") != final_url:
        raise RuntimeError("Menon browser snapshot URL does not match the final Playwright page URL.")

    auth_state = snapshot.get("authState")
    page_role = str(auth_state.get("role") or "").strip() if isinstance(auth_state, Mapping) else ""
    session = session_payload.get("session") if isinstance(session_payload, Mapping) else None
    user = session.get("user") if isinstance(session, Mapping) else None
    session_role = str(user.get("role") or "").strip() if isinstance(user, Mapping) else ""
    session_id = str(session.get("id") or "").strip() if isinstance(session, Mapping) else ""
    expires_at = str(session.get("expiresAt") or "").strip() if isinstance(session, Mapping) else ""
    try:
        expiry_ms = int(datetime.fromisoformat(expires_at.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp() * 1000)
    except Exception as exc:
        raise RuntimeError("Menon auth-session proof returned an invalid expiry.") from exc
    if (
        session_payload.get("authenticated") is not True
        or not isinstance(auth_state, Mapping)
        or auth_state.get("authenticated") is not True
        or session_role not in {"owner", "operator"}
        or page_role != session_role
        or not session_id
        or expiry_ms <= checked_at_ms
    ):
        raise RuntimeError(
            "Menon browser capture requires a current authenticated owner/operator session proven by /api/auth/session."
        )
    origin_host = f"[{final_origin[1]}]" if ":" in final_origin[1] else final_origin[1]
    return {
        "authenticated": True,
        "role": session_role,
        "sessionId": session_id,
        "expiresAt": expires_at,
        "checkedAtMs": checked_at_ms,
        "origin": f"{final_origin[0]}://{origin_host}:{final_origin[2]}",
    }


async def capture_browser_snapshot(
    *,
    url: str,
    duration_s: float,
    screenshot_path: Path | None,
    headless: bool,
    timeout_ms: int,
    storage_state_path: Path | None = None,
) -> dict[str, Any]:
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Python Playwright is required for browser capture. Install it in this environment and run `python -m playwright install chromium`."
        ) from exc

    storage_state: dict[str, Any] | None = None
    if storage_state_path is not None:
        try:
            encoded_state = read_private_file(
                storage_state_path,
                label="Playwright storage state",
                max_bytes=1024 * 1024,
            )
            parsed_state = json.loads(encoded_state.decode("utf-8"))
        except (PrivatePathError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Playwright storage state changed or became invalid before browser launch.") from exc
        if not isinstance(parsed_state, Mapping):
            raise RuntimeError("Playwright storage state must contain a JSON object.")
        storage_state = dict(parsed_state)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        try:
            context = await browser.new_context(
                viewport={"width": 1440, "height": 1000},
                storage_state=storage_state,
            )
            try:
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                if duration_s > 0:
                    await page.wait_for_timeout(int(duration_s * 1000))
                snapshot = await page.evaluate(BROWSER_SNAPSHOT_JS)
                if not isinstance(snapshot, Mapping):
                    raise RuntimeError("Menon browser snapshot did not serialize as a JSON object.")
                final_url = page.url
                auth_endpoint = urljoin(final_url, "/api/auth/session")
                auth_response = await context.request.get(
                    auth_endpoint,
                    headers={"Accept": "application/json"},
                    timeout=timeout_ms,
                    fail_on_status_code=False,
                )
                if not auth_response.ok:
                    raise RuntimeError(
                        f"Menon auth-session proof failed with HTTP {auth_response.status}."
                    )
                try:
                    session_payload = await auth_response.json()
                except Exception as exc:
                    raise RuntimeError("Menon auth-session proof returned invalid JSON.") from exc
                if not isinstance(session_payload, Mapping):
                    raise RuntimeError("Menon auth-session proof returned an invalid payload.")
                checked_at_ms = int(time.time() * 1000)
                auth_proof = validate_browser_auth_proof(
                    snapshot,
                    session_payload,
                    requested_url=url,
                    final_url=final_url,
                    checked_at_ms=checked_at_ms,
                )
                snapshot = dict(snapshot)
                snapshot["authSessionProof"] = auth_proof
                if screenshot_path is not None:
                    ensure_private_directory(screenshot_path.parent, label="Menon browser evidence")
                    if screenshot_path.exists() or screenshot_path.is_symlink():
                        validate_private_file(screenshot_path, label="Menon browser screenshot")
                    await page.screenshot(path=str(screenshot_path), full_page=True)
                    os.chmod(screenshot_path, 0o600)
                    validate_private_file(screenshot_path, label="Menon browser screenshot")
                return snapshot
            finally:
                await context.close()
        finally:
            await browser.close()


def validate_storage_state_path(raw_path: str | Path) -> Path:
    try:
        path = validate_private_file(
            Path(raw_path).expanduser(),
            label="Playwright storage state",
            allowed_modes=(0o600,),
        )
        encoded = read_private_file(
            path,
            label="Playwright storage state",
            max_bytes=1024 * 1024,
        )
    except PrivatePathError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("Playwright storage state must be valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Playwright storage state must contain a JSON object.")
    return path


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
    ensure_private_directory(output.parent, label="Menon browser evidence")
    raw_path = Path(raw_snapshot_path) if raw_snapshot_path is not None else output.with_name("browser_snapshot.json")
    ensure_private_directory(raw_path.parent, label="Menon raw browser evidence")
    atomic_write_private_file(
        raw_path,
        (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        label="Menon raw browser snapshot",
    )
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
    atomic_write_private_file(
        output,
        (json.dumps(trace, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        label="Menon browser trace",
    )
    return trace


def _resolve_screenshot_path(raw: str | None, output_path: Path) -> Path | None:
    if raw is None:
        return None
    if raw.strip():
        return Path(raw)
    return output_path.with_name("menon_browser_capture.png")


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description="Capture Menon browser debug state and validate Noesis-to-Menon placement.")
    parser.add_argument("--url", default="http://127.0.0.1:5175", help="Authenticated Menon gateway URL to capture.")
    parser.add_argument(
        "--storage-state",
        required=True,
        help="Owner-only (0600) Playwright storage-state JSON captured after Menon login.",
    )
    parser.add_argument("--output", required=True, help="Trace JSON output path.")
    parser.add_argument("--run-id", default="", help="Run id for the trace and optional validation report.")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds to wait after the page loads before capture.")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Page navigation timeout in milliseconds.")
    parser.add_argument("--screenshot", nargs="?", const="", default=None, help="Optional screenshot path. Omit value to place it beside --output.")
    parser.add_argument("--headed", action="store_true", help="Show the browser window instead of running headless.")
    parser.add_argument("--validate", action="store_true", help="Run the shared Menon trace report after capture.")
    parser.add_argument(
        "--report-output-dir",
        default="",
        help="Private output directory for --validate reports (default: reports beside --output).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    screenshot_path = _resolve_screenshot_path(args.screenshot, output_path)
    try:
        ensure_private_directory(output_path.parent, label="Menon browser evidence")
        if screenshot_path is not None:
            ensure_private_directory(screenshot_path.parent, label="Menon browser evidence")
        storage_state_path = validate_storage_state_path(args.storage_state)
        snapshot = asyncio.run(
            capture_browser_snapshot(
                url=args.url,
                duration_s=float(args.duration),
                screenshot_path=screenshot_path,
                headless=not args.headed,
                timeout_ms=int(args.timeout_ms),
                storage_state_path=storage_state_path,
            )
        )
        trace = write_trace_from_snapshot(
            snapshot,
            output_path=output_path,
            run_id=args.run_id,
            page_url=str(snapshot.get("href") or ""),
            screenshot_path=screenshot_path,
        )
        if trace.get("browser", {}).get("projection_mode") != "canonical_world_snapshot":
            raise RuntimeError("Menon did not expose the canonical world-snapshot presentation path.")
        if trace.get("browser", {}).get("canonical_admission") != "accepted":
            raise RuntimeError(
                "Menon canonical browser evidence was rejected: "
                f"{trace.get('browser', {}).get('canonical_admission') or 'unknown'}."
            )
        if "world_to_menon_col_major" not in trace:
            raise RuntimeError("Menon did not expose the authored backend-world to scene transform.")
        if not trace.get("placements"):
            raise RuntimeError("Menon exposed no current canonical entity placement evidence.")
        if not isinstance(trace.get("production_geometry"), Mapping):
            raise RuntimeError(
                "Menon exposed no bounded production renderer geometry evidence."
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
            report_output_dir = ensure_private_directory(
                Path(args.report_output_dir) if args.report_output_dir else output_path.parent / "reports",
                label="Menon validation reports",
            )
            json_path, md_path, report = run_trace_report(
                output_path,
                output_dir=report_output_dir,
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
