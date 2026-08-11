"""Read-only, biometric-free legacy migration review projection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from noesis_core.private_paths import PrivatePathError, read_private_file
from noesis_core.strict_json import strict_json_loads


class MigrationReviewError(RuntimeError):
    pass


_FORBIDDEN_KEYS = {
    "anchor_vectors",
    "embedding",
    "embeddings",
    "raw_embedding",
    "vector",
    "vectors",
}
_PRIVATE_PATH_VALUE = re.compile(
    r"(?:^|[\s\"'(])(?:/|~/|file://|[A-Za-z]:[\\/])"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _is_forbidden_biometric_key(key: object) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _FORBIDDEN_KEYS or normalized.endswith(
        ("_embedding", "_embeddings", "_vector", "_vectors")
    )


def _assert_biometric_free(value: Any, *, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_biometric_key(key):
                raise MigrationReviewError(
                    f"migration review contains forbidden biometric field {path}.{key}"
                )
            _assert_biometric_free(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_biometric_free(child, path=f"{path}[{index}]")


def _safe_text(value: Any, *, field: str, maximum: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > maximum:
        raise MigrationReviewError(f"migration review {field} is too long")
    if _PRIVATE_PATH_VALUE.search(text):
        return "[private path redacted]"
    return text


def _bounded_list(value: Any, *, field: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise MigrationReviewError(f"migration review {field} must be a list")
    if len(value) > maximum:
        raise MigrationReviewError(f"migration review {field} exceeds its item bound")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MigrationReviewError(f"migration review {field} must be an integer") from exc
    if parsed < 0 or parsed > 1_000_000:
        raise MigrationReviewError(f"migration review {field} is outside its bound")
    return parsed


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise MigrationReviewError(f"migration review {field} must be a boolean")
    return value


def _project_resident(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MigrationReviewError(f"migration review {field} must be an object")
    reasons = _bounded_list(
        value.get("blocked_reasons", []),
        field=f"{field}.blocked_reasons",
        maximum=64,
    )
    projected = {
        "resident_uuid": _safe_text(
            value.get("resident_uuid"), field=f"{field}.resident_uuid", maximum=160
        ),
        "display_name": _safe_text(
            value.get("display_name"), field=f"{field}.display_name", maximum=160
        ),
        "normalized_name": _safe_text(
            value.get("normalized_name"),
            field=f"{field}.normalized_name",
            maximum=160,
        ),
        "compatibility_sid": _nonnegative_int(
            value.get("compatibility_sid", 0), field=f"{field}.compatibility_sid"
        ),
        "recorded_embedding_count": _nonnegative_int(
            value.get("recorded_embedding_count", 0),
            field=f"{field}.recorded_embedding_count",
        ),
        "actual_embedding_count": _nonnegative_int(
            value.get("actual_embedding_count", 0),
            field=f"{field}.actual_embedding_count",
        ),
        "anchor_count": _nonnegative_int(
            value.get("anchor_count", 0), field=f"{field}.anchor_count"
        ),
        "blocked_reasons": [
            _safe_text(item, field=f"{field}.blocked_reasons", maximum=160)
            for item in reasons
        ],
        "gallery_compatible": _boolean(
            value.get("gallery_compatible", False),
            field=f"{field}.gallery_compatible",
        ),
    }
    return projected


def _project_duplicate_group(value: Any, *, index: int) -> dict[str, Any]:
    field = f"duplicate_normalized_name_groups[{index}]"
    if not isinstance(value, Mapping):
        raise MigrationReviewError(f"migration review {field} must be an object")
    residents = _bounded_list(
        value.get("residents", []), field=f"{field}.residents", maximum=256
    )
    projected = []
    for resident_index, resident in enumerate(residents):
        if not isinstance(resident, Mapping):
            raise MigrationReviewError(
                f"migration review {field}.residents[{resident_index}] must be an object"
            )
        projected.append(
            {
                "resident_uuid": _safe_text(
                    resident.get("resident_uuid"),
                    field=f"{field}.residents[{resident_index}].resident_uuid",
                    maximum=160,
                ),
                "display_name": _safe_text(
                    resident.get("display_name"),
                    field=f"{field}.residents[{resident_index}].display_name",
                    maximum=160,
                ),
                "compatibility_sid": _nonnegative_int(
                    resident.get("compatibility_sid", 0),
                    field=f"{field}.residents[{resident_index}].compatibility_sid",
                ),
                "anchor_count": _nonnegative_int(
                    resident.get("anchor_count", 0),
                    field=f"{field}.residents[{resident_index}].anchor_count",
                ),
            }
        )
    return {
        "normalized_name": _safe_text(
            value.get("normalized_name"), field=f"{field}.normalized_name", maximum=160
        ),
        "residents": projected,
    }


def _project_issue(value: Any, *, index: int) -> dict[str, Any]:
    field = f"issues[{index}]"
    if not isinstance(value, Mapping):
        raise MigrationReviewError(f"migration review {field} must be an object")
    severity = _safe_text(value.get("severity"), field=f"{field}.severity", maximum=32)
    if severity not in {"info", "warning", "error", "fatal"}:
        raise MigrationReviewError(f"migration review {field}.severity is invalid")
    return {
        "code": _safe_text(value.get("code"), field=f"{field}.code", maximum=160),
        "severity": severity,
        "subject": _safe_text(
            value.get("subject"), field=f"{field}.subject", maximum=240
        ),
        "detail": _safe_text(value.get("detail"), field=f"{field}.detail"),
    }


def load_migration_review(path: str | Path) -> dict[str, Any]:
    try:
        payload = read_private_file(
            Path(path).expanduser(),
            label="identity migration review",
            max_bytes=4 * 1024 * 1024,
        )
        raw = strict_json_loads(payload, label="identity migration review")
    except PrivatePathError as exc:
        raise MigrationReviewError(str(exc)) from exc
    except Exception as exc:
        raise MigrationReviewError(f"migration review is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise MigrationReviewError("migration review root must be an object")
    if raw.get("contract") != "noesis.identity.legacy_migration_review":
        raise MigrationReviewError("migration review contract is invalid")
    if int(raw.get("contract_version", 0) or 0) != 1:
        raise MigrationReviewError("migration review contract version is invalid")
    disposition = str(raw.get("apply_disposition") or "")
    if disposition not in {"blocked", "partial", "full"}:
        raise MigrationReviewError("migration review apply disposition is invalid")
    residents = _bounded_list(raw.get("residents"), field="residents", maximum=256)
    issues = _bounded_list(raw.get("issues"), field="issues", maximum=512)
    duplicate_groups = _bounded_list(
        raw.get("duplicate_normalized_name_groups"),
        field="duplicate_normalized_name_groups",
        maximum=128,
    )
    _assert_biometric_free(raw)
    # Project only fields the owner UI needs. Source paths and operator assertions
    # remain in the private CLI report, not in the browser response.
    projected = {
        "contract": "noesis.identity.legacy_migration_review",
        "contract_version": 1,
        "source_digest": _safe_text(
            raw.get("source_digest"), field="source_digest", maximum=64
        ),
        "migration_key": _safe_text(
            raw.get("migration_key"), field="migration_key", maximum=240
        ),
        "apply_disposition": disposition,
        "full_migration_safe": _boolean(
            raw.get("full_migration_safe", False), field="full_migration_safe"
        ),
        "requires_accept_partial": _boolean(
            raw.get("requires_accept_partial", False),
            field="requires_accept_partial",
        ),
        "blocked_resident_count": _nonnegative_int(
            raw.get("blocked_resident_count", 0), field="blocked_resident_count"
        ),
        "migratable_resident_count": _nonnegative_int(
            raw.get("migratable_resident_count", 0), field="migratable_resident_count"
        ),
        "residents_without_importable_anchors": [
            _safe_text(value, field="residents_without_importable_anchors", maximum=160)
            for value in _bounded_list(
                raw.get("residents_without_importable_anchors", []),
                field="residents_without_importable_anchors",
                maximum=256,
            )
        ],
        "duplicate_normalized_name_groups": [
            _project_duplicate_group(value, index=index)
            for index, value in enumerate(duplicate_groups)
        ],
        "residents": [
            _project_resident(value, field=f"residents[{index}]")
            for index, value in enumerate(residents)
        ],
        "issues": [
            _project_issue(value, index=index) for index, value in enumerate(issues)
        ],
        "operator_action": _safe_text(
            raw.get("operator_action"), field="operator_action", maximum=500
        ),
        "apply_available_from_api": False,
    }
    if _SHA256.fullmatch(projected["source_digest"]) is None:
        raise MigrationReviewError("migration review source_digest is invalid")
    if not projected["migration_key"]:
        raise MigrationReviewError("migration review migration_key is empty")
    return projected


__all__ = ["MigrationReviewError", "load_migration_review"]
