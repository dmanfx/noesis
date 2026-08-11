"""Restart-time model selection for the DS9 manual full-frame depth lane."""

from __future__ import annotations

import copy
from typing import Any, Mapping


MANUAL_DEPTH_MODELS = ("mapanything", "da3metric-large")


def normalize_manual_depth_model(value: Any) -> str:
    model = str(value or "").strip().lower()
    if not model:
        return "mapanything"
    if model not in MANUAL_DEPTH_MODELS:
        expected = ", ".join(MANUAL_DEPTH_MODELS)
        raise ValueError(
            f"unsupported manual depth model {value!r}; expected {expected}"
        )
    return model


def build_manual_depth_overlay(
    base_cfg: Mapping[str, Any], manual_depth_model: str
) -> dict[str, Any]:
    model = normalize_manual_depth_model(manual_depth_model)
    if model == "mapanything":
        return {}
    models_cfg = base_cfg.get("models") if isinstance(base_cfg, Mapping) else None
    mapanything_reference = (
        (models_cfg or {}).get("mapanything")
        if isinstance(models_cfg, Mapping)
        else None
    )
    if not isinstance(mapanything_reference, Mapping) or not mapanything_reference:
        raise ValueError(
            "DA3Metric-Large selection requires the calibrated "
            "models.mapanything reference profile in the base pipeline config"
        )
    return {
        "depth_registration": {
            # Baseline DAv2 registration was calibrated against this lane.
            "mapanything_reference": copy.deepcopy(dict(mapanything_reference)),
        },
        "models": {
            "mapanything": {
                "backend": "da3metric-large",
                "config-file-path": (
                    "DS9/pipelines/config_infer_secondary_da3metric_large.ini"
                ),
                "engine": (
                    "DS9/models/engines/"
                    "da3metric_large_294x518_b3_fp16.engine"
                ),
                "network_mode": "fp16",
                "metric_focal_denominator": 300.0,
            }
        },
    }
