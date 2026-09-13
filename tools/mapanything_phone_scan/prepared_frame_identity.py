"""Shared identity for prepared RGB frames and calibrated VIO rows."""

from __future__ import annotations

import re
from typing import Any, Mapping


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def prepared_frame_identity(index: int, image_sha256: str) -> str:
    """Return the collision-resistant identity used across capture and PCF."""
    try:
        index = int(index)
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared frame index must be an integer") from exc
    digest = str(image_sha256 or "").strip().lower()
    if index < 0 or not _SHA256_RE.fullmatch(digest):
        raise ValueError("prepared frame identity needs a nonnegative index and SHA-256")
    return f"prepared:{index}:{digest}"


def prepared_frame_identity_from_row(index: int, row: Mapping[str, Any]) -> str:
    """Build an identity from a prepared manifest row's declared image hash."""
    if not isinstance(row, Mapping):
        raise ValueError("prepared frame row must be an object")
    return prepared_frame_identity(index, str(row.get("sha256") or ""))


__all__ = ["prepared_frame_identity", "prepared_frame_identity_from_row"]
