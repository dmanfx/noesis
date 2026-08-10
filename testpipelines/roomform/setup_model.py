#!/usr/bin/env python3
"""Download and verify Roomform's accuracy-first released checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


MODEL_NAME = "patch-graph-joint-rgb-55m-offset-head-r2.pt"
MODEL_URL = f"https://huggingface.co/jchiu/roomform/resolve/main/{MODEL_NAME}"
MODEL_SHA256 = "9a1255fb73eba8b0d34f9a09960f20369c7d2457c2e50cb69de87e8429867fb5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == MODEL_SHA256:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    urllib.request.urlretrieve(MODEL_URL, partial)
    actual = sha256(partial)
    if actual != MODEL_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Roomform checkpoint SHA-256 mismatch: expected {MODEL_SHA256}, "
            f"received {actual}"
        )
    os.replace(partial, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("ROOMFORM_MODEL_DIR", "checkpoints/roomform")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = download_checkpoint(args.model_dir.expanduser().resolve() / MODEL_NAME)
    print(
        json.dumps(
            {
                "checkpoint": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
