"""Strict JSON decoding for authority, evidence, and private-state inputs."""

from __future__ import annotations

import json
import math
from typing import Any, Literal


StrictJSONReason = Literal[
    "duplicate_key",
    "invalid_utf8",
    "nonfinite_number",
    "invalid_json",
]


class StrictJSONError(ValueError):
    """Raised when authority JSON is ambiguous or outside strict JSON semantics."""

    def __init__(self, message: str, *, reason: StrictJSONReason) -> None:
        super().__init__(message)
        self.reason: StrictJSONReason = reason


def strict_json_loads(payload: bytes | str, *, label: str) -> Any:
    """Decode unique-key, finite UTF-8 JSON without exposing input values.

    Duplicate keys are rejected recursively through ``object_pairs_hook``.
    Python's decoder otherwise accepts JavaScript non-finite constants and can
    turn a syntactically finite exponent such as ``1e309`` into infinity, so
    both paths are rejected here as well.
    """

    description = " ".join(str(label or "JSON document").split()) or "JSON document"
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise StrictJSONError(
                f"{description} is not valid UTF-8 JSON",
                reason="invalid_utf8",
            ) from exc
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError("strict_json_loads payload must be bytes or str")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StrictJSONError(
                    f"{description} contains a duplicate JSON object key",
                    reason="duplicate_key",
                )
            result[key] = value
        return result

    def finite_float(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            raise StrictJSONError(
                f"{description} contains a non-finite JSON number",
                reason="nonfinite_number",
            )
        return value

    def reject_constant(_raw: str) -> Any:
        raise StrictJSONError(
            f"{description} contains a non-finite JSON number",
            reason="nonfinite_number",
        )

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except StrictJSONError:
        raise
    except (ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise StrictJSONError(
            f"{description} is invalid JSON",
            reason="invalid_json",
        ) from exc


__all__ = ["StrictJSONError", "StrictJSONReason", "strict_json_loads"]
