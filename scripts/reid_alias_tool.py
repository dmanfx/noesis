#!/usr/bin/env python3
"""CLI helper for the DS8 ReID alias REST API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional
from urllib import error, request


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
API_ENV = "NOESIS_API_URL"


def _resolve_base_url(args: argparse.Namespace) -> str:
    url = str(getattr(args, "url", None) or os.environ.get(API_ENV) or DEFAULT_BASE_URL)
    host = getattr(args, "host", None)
    port = getattr(args, "port", None)
    if host or port:
        host = host or "127.0.0.1"
        port = port or 8080
        url = f"http://{host}:{port}"
    return url.rstrip("/")


def _request_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=10) as resp:
            body = resp.read()
    except error.HTTPError as exc:
        body = exc.read()
        msg = body.decode("utf-8", errors="replace") if body else str(exc)
        raise RuntimeError(f"{exc.code} {exc.reason}: {msg}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc

    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON response: {exc}") from exc


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_list(args: argparse.Namespace) -> int:
    url = _resolve_base_url(args) + "/api/v1/reid/aliases"
    _print_json(_request_json("GET", url))
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    url = _resolve_base_url(args) + f"/api/v1/reid/aliases/history?limit={int(args.limit)}"
    _print_json(_request_json("GET", url))
    return 0


def _cmd_suggest(args: argparse.Namespace) -> int:
    url = _resolve_base_url(args) + "/api/v1/reid/aliases/suggest"
    payload: Dict[str, Any] = {
        "limit": int(args.limit),
        "require_inactive": bool(args.require_inactive),
    }
    if args.min_sim is not None:
        payload["min_sim"] = float(args.min_sim)
    _print_json(_request_json("POST", url, payload))
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    url = _resolve_base_url(args) + "/api/v1/reid/aliases/merge"
    payload: Dict[str, Any] = {
        "a": int(args.a),
        "b": int(args.b),
        "append_embeddings": bool(args.append_embeddings),
        "force": bool(args.force),
    }
    if args.canonical is not None:
        payload["canonical"] = int(args.canonical)
    _print_json(_request_json("POST", url, payload))
    return 0


def _load_pairs(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _cmd_merge_batch(args: argparse.Namespace) -> int:
    url = _resolve_base_url(args) + "/api/v1/reid/aliases/merge-batch"
    payload: Dict[str, Any] = {"force": bool(args.force)}
    raw = _load_pairs(args.file)
    if isinstance(raw, dict) and "pairs" in raw:
        payload["pairs"] = raw.get("pairs")
        if "force" in raw and not args.force:
            payload["force"] = bool(raw.get("force"))
    else:
        payload["pairs"] = raw
    _print_json(_request_json("POST", url, payload))
    return 0


def _cmd_unset(args: argparse.Namespace) -> int:
    url = _resolve_base_url(args) + "/api/v1/reid/aliases/unset"
    payload = {"src": int(args.src)}
    _print_json(_request_json("POST", url, payload))
    return 0


def _cmd_clear(args: argparse.Namespace) -> int:
    url = _resolve_base_url(args) + "/api/v1/reid/aliases/clear"
    _print_json(_request_json("POST", url, {}))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReID alias REST helper")
    parser.add_argument("--url", default=None, help=f"Base URL (default: ${API_ENV} or {DEFAULT_BASE_URL})")
    parser.add_argument("--host", default=None, help="Override host for base URL")
    parser.add_argument("--port", type=int, default=None, help="Override port for base URL")

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List current aliases")
    list_parser.set_defaults(func=_cmd_list)

    suggest_parser = subparsers.add_parser("suggest", help="Suggest alias merges")
    suggest_parser.add_argument("--min-sim", type=float, default=None, help="Minimum similarity")
    suggest_parser.add_argument("--limit", type=int, default=20, help="Max candidates")
    suggest_parser.add_argument(
        "--require-inactive",
        type=int,
        default=1,
        choices=[0, 1],
        help="Require both IDs inactive (1/0)",
    )
    suggest_parser.set_defaults(func=_cmd_suggest)

    merge_parser = subparsers.add_parser("merge", help="Merge two IDs")
    merge_parser.add_argument("--a", type=int, required=True, help="Source ID (A)")
    merge_parser.add_argument("--b", type=int, required=True, help="Source ID (B)")
    merge_parser.add_argument("--canonical", type=int, default=None, help="Explicit canonical ID")
    merge_parser.add_argument(
        "--append-embeddings",
        type=int,
        default=1,
        choices=[0, 1],
        help="Append embeddings to canonical gallery (1/0)",
    )
    merge_parser.add_argument(
        "--force",
        type=int,
        default=0,
        choices=[0, 1],
        help="Force merge even if guardrails would block (1/0)",
    )
    merge_parser.set_defaults(func=_cmd_merge)

    batch_parser = subparsers.add_parser("merge-batch", help="Apply batch merges from JSON")
    batch_parser.add_argument("--file", required=True, help="JSON file containing merge pairs")
    batch_parser.add_argument(
        "--force",
        type=int,
        default=0,
        choices=[0, 1],
        help="Force merges (1/0)",
    )
    batch_parser.set_defaults(func=_cmd_merge_batch)

    unset_parser = subparsers.add_parser("unset", help="Remove an alias mapping")
    unset_parser.add_argument("--src", type=int, required=True, help="Alias source to remove")
    unset_parser.set_defaults(func=_cmd_unset)

    clear_parser = subparsers.add_parser("clear", help="Clear all aliases")
    clear_parser.set_defaults(func=_cmd_clear)

    history_parser = subparsers.add_parser("history", help="Show alias history")
    history_parser.add_argument("--limit", type=int, default=100, help="Max history entries")
    history_parser.set_defaults(func=_cmd_history)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
