#!/usr/bin/env python3
"""Noesis-owned native readiness probe.

Replaces selector-file lookup for `wait-ready noesis`. Checks REST capabilities,
deployment health identity from env (not a selector file), and WebSocket health
v2. Does not speak Docker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

REQUIRED_CAPABILITIES = ("tracking_observations", "global_world")
REQUIRED_HEALTH_KEYS = (
    "NOESIS_DEPLOYMENT_ID",
    "NOESIS_HEALTH_SELECTOR_SHA256",
    "NOESIS_STATE_RELEASE_ID",
    "NOESIS_SOFTWARE_REVISION",
)


class NativeReadinessError(RuntimeError):
    """Fail-closed native Noesis readiness error."""


def _fail(message: str) -> None:
    raise NativeReadinessError(message)


def _load_token(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        _fail("NOESIS_INTERNAL_AUTH_TOKEN_FILE must be set")
    try:
        token = open(text, encoding="utf-8").read().strip()
    except OSError as exc:
        _fail(f"internal auth token is unreadable: {exc}")
    if not token:
        _fail("internal auth token is empty")
    return token


def _expected_identity(env: dict[str, str]) -> dict[str, str]:
    identity = {name: str(env.get(name) or "").strip() for name in REQUIRED_HEALTH_KEYS}
    if not all(identity.values()):
        _fail(
            "native readiness requires NOESIS_DEPLOYMENT_ID, "
            "NOESIS_HEALTH_SELECTOR_SHA256, NOESIS_STATE_RELEASE_ID, and "
            "NOESIS_SOFTWARE_REVISION"
        )
    return identity


def _request_json(url: str, token: str, *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Connection": "close",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if int(response.status) != 200:
                _fail(f"{url} returned HTTP {response.status}")
            raw = response.read(1_048_576)
    except urllib.error.HTTPError as exc:
        _fail(f"{url} returned HTTP {exc.code}")
    except urllib.error.URLError as exc:
        _fail(f"{url} is not reachable: {exc.reason}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{url} returned invalid JSON: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{url} returned a non-object JSON payload")
    return payload


def _capability_status(payload: dict[str, Any], name: str) -> str:
    rows = payload.get("capabilities")
    if not isinstance(rows, list):
        _fail("capability health payload is missing capabilities")
    for row in rows:
        if isinstance(row, dict) and str(row.get("capability") or "") == name:
            return str(row.get("status") or "")
    return ""


def validate_capabilities(payload: dict[str, Any]) -> dict[str, str]:
    if payload.get("contract") != "noesis.capability.health":
        _fail("capability health contract is invalid")
    instance_id = str(payload.get("instance_id") or "").strip()
    run_id = str(payload.get("run_id") or "").strip()
    if not instance_id or not run_id:
        _fail("capability health identity is invalid")
    for name in REQUIRED_CAPABILITIES:
        if _capability_status(payload, name) != "healthy":
            _fail(f"required capability is not healthy: {name}")
    return {"instance_id": instance_id, "run_id": run_id}


def validate_deployment(
    payload: dict[str, Any],
    *,
    identity: dict[str, str],
    producer: dict[str, str],
) -> dict[str, Any]:
    if (
        payload.get("contract") != "noesis.appliance.deployment_health"
        or int(payload.get("contract_version") or 0) != 1
        or payload.get("ready") is not True
        or payload.get("runtime_family") != "ds9"
        or not str(payload.get("runtime_variant") or "").startswith("ds9:")
    ):
        _fail("deployment health contract or backend identity is invalid")
    if (
        str(payload.get("deployment_id") or "") != identity["NOESIS_DEPLOYMENT_ID"]
        or str(payload.get("selector_sha256") or "")
        != identity["NOESIS_HEALTH_SELECTOR_SHA256"]
        or str(payload.get("state_release_id") or "")
        != identity["NOESIS_STATE_RELEASE_ID"]
        or str(payload.get("software_revision") or "")
        != identity["NOESIS_SOFTWARE_REVISION"]
        or str(payload.get("instance_id") or "") != producer["instance_id"]
        or str(payload.get("run_id") or "") != producer["run_id"]
    ):
        _fail("deployment health identity does not match native runtime identity")
    generated_at_us = payload.get("generated_at_us")
    if not isinstance(generated_at_us, int) or generated_at_us < 1:
        _fail("deployment health generated_at_us is invalid")
    return payload


def probe_websocket(
    *,
    token: str,
    rest_payload: dict[str, Any],
    timeout_s: float,
) -> None:
    from websockets.sync.client import connect

    host = str(os.environ.get("NOESIS_WS_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("NOESIS_WS_PORT") or "6008")
    with connect(
        f"ws://{host}:{port}/healthz",
        additional_headers={"Authorization": f"Bearer {token}"},
        compression=None,
        proxy=None,
        open_timeout=timeout_s,
        close_timeout=timeout_s,
        max_size=4096,
    ) as websocket:
        raw = websocket.recv(timeout=timeout_s)
    if not isinstance(raw, str):
        _fail("WebSocket health response was not text")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(f"WebSocket health response was invalid JSON: {exc}")
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "health"
        or payload.get("contract") != "noesis.ws.health"
        or int(payload.get("contract_version") or 0) != 2
    ):
        _fail("WebSocket health contract is invalid")
    for key in (
        "deployment_id",
        "selector_sha256",
        "state_release_id",
        "runtime_family",
        "runtime_variant",
        "instance_id",
        "run_id",
        "boot_id",
        "software_revision",
    ):
        if payload.get(key) != rest_payload.get(key):
            _fail("REST and WebSocket producer identities differ")


def probe_once(*, token: str, identity: dict[str, str], timeout_s: float) -> dict[str, Any]:
    rest_host = str(os.environ.get("NOESIS_REST_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    rest_port = int(os.environ.get("NOESIS_REST_PORT") or "8080")
    base = f"http://{rest_host}:{rest_port}"
    capabilities = validate_capabilities(
        _request_json(f"{base}/api/v1/health/capabilities", token, timeout_s=timeout_s)
    )
    first = validate_deployment(
        _request_json(f"{base}/api/v1/health/deployment", token, timeout_s=timeout_s),
        identity=identity,
        producer=capabilities,
    )
    probe_websocket(token=token, rest_payload=first, timeout_s=min(timeout_s, 5.0))
    time.sleep(0.02)
    second = validate_deployment(
        _request_json(f"{base}/api/v1/health/deployment", token, timeout_s=timeout_s),
        identity=identity,
        producer=capabilities,
    )
    if (
        second["instance_id"] != first["instance_id"]
        or second["run_id"] != first["run_id"]
        or second["boot_id"] != first["boot_id"]
        or int(second["generated_at_us"]) <= int(first["generated_at_us"])
    ):
        _fail("deployment health did not advance with a stable producer identity")
    return {
        "ok": True,
        "backend": "native_host",
        "runtime_family": second["runtime_family"],
        "runtime_variant": second["runtime_variant"],
        "deployment_id": second["deployment_id"],
        "instance_id": second["instance_id"],
        "run_id": second["run_id"],
    }


def wait_ready(*, timeout_ms: int) -> dict[str, Any]:
    identity = _expected_identity(dict(os.environ))
    token = _load_token(str(os.environ.get("NOESIS_INTERNAL_AUTH_TOKEN_FILE") or ""))
    deadline = time.monotonic() + max(timeout_ms, 1) / 1000.0
    last_error = "native Noesis readiness did not succeed"
    while time.monotonic() < deadline:
        remaining = max(deadline - time.monotonic(), 0.2)
        try:
            return probe_once(
                token=token,
                identity=identity,
                timeout_s=min(remaining, 5.0),
            )
        except NativeReadinessError as exc:
            last_error = str(exc)
            time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))
    _fail(last_error)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native Noesis readiness probe")
    parser.add_argument("--timeout-ms", type=int, default=240000)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-ms", type=int, default=10000)
    parser.add_argument("--failure-threshold", type=int, default=3)
    parser.add_argument("--ws-every-cycles", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.watch:
        print(
            "native Noesis wait-ready does not implement appliance watch mode",
            file=sys.stderr,
        )
        return 2
    try:
        payload = wait_ready(timeout_ms=int(args.timeout_ms))
    except NativeReadinessError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
