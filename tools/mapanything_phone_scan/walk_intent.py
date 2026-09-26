"""Capture purpose, not a claim about trajectory or calibration accuracy.

Older recordings intentionally have no inferred intent. Their original files and
calibration bindings remain valid evidence; a later review label is separate from
what the recorder actually declared.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


WALK_INTENT_SCHEMA = "roomwalk.walk_intent.v1"
WALK_MODES = frozenset({"reconstruction", "path_refinement"})
SCAN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$")
INTENT_FIELDS = frozenset({"schema", "mode", "target_scan_id", "carry_protocol", "accuracy_target_m", "target_reference"})
REFERENCE_FIELDS = frozenset({"kind", "prior_id", "manifest_sha256", "camera_id", "frame_binding_sha256"})
REFERENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def validate_target_reference(value: Any) -> dict[str, str]:
    """Validate an immutable selection identity, not client-supplied authority."""
    if not isinstance(value, Mapping) or set(value) != REFERENCE_FIELDS:
        raise ValueError("PCF reference must contain its exact revision, camera and digests")
    if value.get("kind") != "scene_prior_pcf":
        raise ValueError("Unsupported path reference kind")
    for field, pattern in (("prior_id", REFERENCE_ID), ("camera_id", CAMERA_ID),
                           ("manifest_sha256", SHA256), ("frame_binding_sha256", SHA256)):
        if not isinstance(value.get(field), str) or pattern.fullmatch(value[field]) is None:
            raise ValueError(f"Invalid PCF reference {field}")
    return dict(value)


def validate_walk_intent(value: Any) -> dict[str, Any]:
    """Normalize a small purpose declaration without granting spatial authority."""
    if not isinstance(value, Mapping) or set(value) - INTENT_FIELDS:
        raise ValueError("Walk intent must contain only supported capture-purpose fields")
    if value.get("schema", WALK_INTENT_SCHEMA) != WALK_INTENT_SCHEMA:
        raise ValueError("Unsupported walk-intent schema")
    mode = value.get("mode")
    if not isinstance(mode, str) or mode not in WALK_MODES:
        raise ValueError("Choose reconstruction or path_refinement")
    target = value.get("target_scan_id")
    if target is not None and (not isinstance(target, str) or not SCAN_ID.fullmatch(target)):
        raise ValueError("Choose a valid existing reconstruction")
    if mode == "path_refinement" and target is None:
        raise ValueError("A path-refinement walk must name its reference reconstruction")
    protocol = "close_body" if mode == "path_refinement" else "coverage"
    if value.get("carry_protocol", protocol) != protocol:
        raise ValueError("Carry protocol does not match the walk mode")
    accuracy = value.get("accuracy_target_m", 0.1)
    if (isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
            or accuracy != 0.1):
        raise ValueError("The reference accuracy target is 0.1 metres, not a measured result")
    result = {"schema": WALK_INTENT_SCHEMA, "mode": mode, "target_scan_id": target,
              "carry_protocol": protocol, "accuracy_target_m": 0.1}
    if "target_reference" in value:
        if mode != "path_refinement":
            raise ValueError("A PCF path reference is only valid for path refinement")
        result["target_reference"] = validate_target_reference(value["target_reference"])
    return result


def effective_walk_intent(state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read explicit intent only; never invent close-body evidence for old walks."""
    value = state.get("walk_intent")
    return validate_walk_intent(value) if value is not None else None
