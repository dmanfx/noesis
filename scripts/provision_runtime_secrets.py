#!/usr/bin/env python3
"""Provision owner-only runtime credentials without echoing their values."""

from __future__ import annotations

import argparse
import getpass
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.private_paths import atomic_write_private_file  # noqa: E402
from noesis_core.runtime_secrets import (  # noqa: E402
    camera_secrets_path,
    mapanything_api_key_path,
    materialize_pipeline_config,
)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(decoded, Mapping):
        raise RuntimeError(f"configuration must be a mapping: {path}")
    return decoded


def _camera_names(path: Path) -> list[str]:
    decoded = _load_yaml(path)
    cameras = decoded.get("cameras")
    if not isinstance(cameras, Mapping) or not cameras:
        raise RuntimeError("camera labels config has no cameras mapping")
    ordered: list[tuple[int, str]] = []
    for raw_index, row in cameras.items():
        if not isinstance(row, Mapping):
            raise RuntimeError("camera labels entries must be mappings")
        name = str(row.get("name") or "").strip()
        if not name:
            raise RuntimeError("camera labels entries must have names")
        ordered.append((int(raw_index), name))
    ordered.sort()
    expected = list(range(len(ordered)))
    if [index for index, _name in ordered] != expected:
        raise RuntimeError("camera label indexes must be contiguous from zero")
    return [name for _index, name in ordered]


def _camera_uris(
    path: Path,
    names: list[str],
    *,
    interactive: bool,
) -> dict[str, str]:
    decoded = _load_yaml(path)
    sources = decoded.get("sources")
    if not isinstance(sources, list) or len(sources) != len(names):
        raise RuntimeError("pipeline source count does not match camera labels")
    result: dict[str, str] = {}
    for index, (source, name) in enumerate(zip(sources, names, strict=True)):
        if not isinstance(source, Mapping):
            raise RuntimeError(f"pipeline source {index} must be a mapping")
        ref = str(source.get("uri_secret") or "").strip()
        uri = str(source.get("uri") or "").strip()
        if ref:
            if ref != name:
                raise RuntimeError(
                    f"pipeline source {index} secret reference does not match camera label"
                )
            if not interactive:
                raise RuntimeError(
                    "pipeline uses camera secret references; rerun with --interactive-camera-input"
                )
            uri = getpass.getpass(f"RTSP URI for {ref}: ").strip()
        parsed = urlsplit(uri)
        if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise RuntimeError(f"pipeline source {index} is not a valid RTSP source")
        result[name] = uri
    materialize_pipeline_config(
        {"sources": [{"uri_secret": name} for name in names]},
        camera_registry=result,
    )
    return result


def provision(args: argparse.Namespace) -> None:
    mapanything_path = mapanything_api_key_path(args.mapanything_key_file)
    if args.rotate_mapanything_key_only:
        if not mapanything_path.exists() or mapanything_path.is_symlink():
            raise RuntimeError("MapAnything RPC key destination is not an existing safe file")
        atomic_write_private_file(
            mapanything_path,
            secrets.token_urlsafe(48).encode("ascii"),
            label="MapAnything RPC key",
        )
        print("Rotated the MapAnything RPC key in owner-only state.")
        return
    if args.replace_camera_secrets and not args.interactive_camera_input:
        raise RuntimeError(
            "--replace-camera-secrets requires --interactive-camera-input"
        )

    names = _camera_names(args.cameras_config)
    sources = _camera_uris(
        args.pipeline_config,
        names,
        interactive=bool(args.interactive_camera_input),
    )
    camera_payload = json.dumps(
        {"version": 1, "sources": sources}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    camera_path = camera_secrets_path(args.camera_secrets_file)
    if (camera_path.exists() or camera_path.is_symlink()) and not args.replace_camera_secrets:
        raise RuntimeError("camera source secret destination already exists")
    if args.replace_camera_secrets:
        atomic_write_private_file(
            camera_path,
            camera_payload,
            label="camera source secrets",
        )
        print(f"Replaced {len(sources)} camera source references in owner-only state.")
        return
    if mapanything_path.exists() or mapanything_path.is_symlink():
        raise RuntimeError("MapAnything RPC key destination already exists")

    atomic_write_private_file(
        camera_path,
        camera_payload,
        label="camera source secrets",
    )
    try:
        key_payload = secrets.token_urlsafe(48).encode("ascii")
        atomic_write_private_file(
            mapanything_path,
            key_payload,
            label="MapAnything RPC key",
        )
    except Exception:
        camera_path.unlink(missing_ok=True)
        raise

    print(f"Provisioned {len(sources)} camera source references in owner-only state.")
    print("Provisioned a rotated MapAnything RPC key in owner-only state.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate camera URIs and rotate the MapAnything RPC key without printing values."
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=REPO_ROOT / "DS9" / "config" / "infer.yaml",
    )
    parser.add_argument(
        "--cameras-config",
        type=Path,
        default=REPO_ROOT / "config" / "cameras.yaml",
    )
    parser.add_argument("--camera-secrets-file", type=Path)
    parser.add_argument("--mapanything-key-file", type=Path)
    parser.add_argument(
        "--interactive-camera-input",
        action="store_true",
        help="Prompt without echo for each URI when the pipeline already uses uri_secret references.",
    )
    parser.add_argument(
        "--replace-camera-secrets",
        action="store_true",
        help="Atomically replace an existing safe camera registry; requires interactive input.",
    )
    parser.add_argument(
        "--rotate-mapanything-key-only",
        action="store_true",
        help="Atomically rotate only the existing MapAnything RPC key.",
    )
    return parser


if __name__ == "__main__":
    provision(build_parser().parse_args())
