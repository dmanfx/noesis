"""Strict structural registration dispatch and shared evidence utilities.

Camera-marker world-to-scene registration has been retired. Camera extrinsics,
target frame bindings and canonical person localization remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class PresentationRegistrationError(ValueError):
    """The declared scene presentation cannot be admitted."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PresentationRegistrationError(f"expected JSON object: {path}")
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(payload) != expected:
        raise PresentationRegistrationError(f"{label} has missing or unknown fields")


def _sha(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PresentationRegistrationError(f"{label} is not a SHA-256 digest")


def _matrix(values: Any, *, label: str) -> np.ndarray:
    if not isinstance(values, (tuple, list)) or len(values) != 16:
        raise PresentationRegistrationError(f"{label} requires 16 column-major values")
    result = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
    if not np.isfinite(result).all() or not np.allclose(result[3], (0, 0, 0, 1), atol=1e-9):
        raise PresentationRegistrationError(f"{label} is not a finite affine matrix")
    return result


def load_registration(path: str | Path, *, authored_obj_path: str | Path) -> dict[str, Any]:
    """Admit only structural mappings; camera-marker registration is retired."""
    source = Path(path)
    if not source.is_file() or not 0 < source.stat().st_size <= 512 * 1024:
        raise PresentationRegistrationError("scene presentation artifact is missing or oversized")
    artifact = _read_json(source)
    if artifact.get("contract") not in (
        "noesis.scene_presentation.structural_registration",
        "noesis.scene_presentation.owner_selected_structural_registration",
    ):
        raise PresentationRegistrationError("inherited camera-marker registration is retired; a structural registration is required")
    from noesis.calibration.structural_presentation import load_structural_presentation
    return load_structural_presentation(source, authored_obj_path=authored_obj_path)
