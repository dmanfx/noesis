from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


MAX_ZONE_LABEL_LENGTH = 160

_StatusResolution = tuple[
    Literal["absent", "unique", "ambiguous", "invalid"],
    str | None,
]


def is_exact_zone_label(value: Any) -> bool:
    """Return whether *value* is a bounded, non-padded zone identifier."""
    return (
        isinstance(value, str)
        and 1 <= len(value) <= MAX_ZONE_LABEL_LENGTH
        and value == value.strip()
    )


def _mapping_membership_state(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"1", "true", "in", "inside", "present"}:
            return True
        if normalized in {"0", "false", "out", "outside", "absent"}:
            return False
    return None


def _resolve_status_labels(raw: Any) -> _StatusResolution:
    if raw is None:
        return ("absent", None)

    labels: list[Any]
    if isinstance(raw, Mapping):
        labels = []
        for label, raw_state in raw.items():
            state = _mapping_membership_state(raw_state)
            if state is None:
                return ("invalid", None)
            if state:
                labels.append(label)
    elif isinstance(raw, str):
        labels = [raw]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        labels = list(raw)
    else:
        return ("invalid", None)

    if not labels:
        return ("absent", None)
    if any(not is_exact_zone_label(label) for label in labels):
        return ("invalid", None)

    unique_labels = set(labels)
    if len(unique_labels) == 1:
        return ("unique", next(iter(unique_labels)))
    return ("ambiguous", None)


def resolve_authoritative_analytics_zone(
    analytics_meta: Mapping[str, Any] | None,
) -> str | None:
    """Resolve one exact per-object nvdsanalytics polygon-membership label.

    Overcrowding membership is the household's canonical room designation.
    ROI filtering remains a compatibility source only when no nonempty
    overcrowding membership exists. Invalid or ambiguous primary evidence
    fails closed instead of falling through to a different designation.
    """
    if not isinstance(analytics_meta, Mapping):
        return None

    oc_state, oc_label = _resolve_status_labels(analytics_meta.get("ocStatus"))
    if oc_state == "unique":
        return oc_label
    if oc_state != "absent":
        return None

    roi_state, roi_label = _resolve_status_labels(analytics_meta.get("roiStatus"))
    return roi_label if roi_state == "unique" else None
