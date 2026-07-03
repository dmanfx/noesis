from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from noesis.ds8_preflight import (
    derive_osd_policy_from_ini,
    mask_output_available,
    parse_nvinfer_ini,
    resolve_config_path,
    validate_metadata_compatibility,
)


def analyze_pipeline(
    pipeline_path: Path,
    *,
    tracking_mode: str = "baseline",
    strict_baseline: bool = False,
) -> Dict[str, Any]:
    path = Path(pipeline_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"pipeline YAML must be a mapping: {path}")

    pgie = ((cfg.get("models") or {}).get("pgie") or {}) if isinstance(cfg.get("models"), Mapping) else {}
    pgie_ini_path = resolve_config_path(path, pgie.get("config-file-path", ""))
    props: Dict[str, str] = {}
    osd_policy = {"process-mode": 0, "display-mask": 0, "display-bbox": 1, "display-text": 1}
    mask_available = False
    if pgie_ini_path.exists():
        props = parse_nvinfer_ini(pgie_ini_path)
        osd_policy = derive_osd_policy_from_ini(props)
        mask_available = mask_output_available(props)

    sources = cfg.get("sources") if isinstance(cfg.get("sources"), list) else []
    validations = [
        result.to_dict()
        for result in validate_metadata_compatibility(
            cfg,
            path,
            tracking_mode=tracking_mode,
            strict_baseline=strict_baseline,
        )
    ]
    model_summary: Dict[str, Any] = {}
    models = cfg.get("models") if isinstance(cfg.get("models"), Mapping) else {}
    for name, model in models.items():
        if isinstance(model, Mapping):
            model_summary[str(name)] = {
                "enabled": bool(model.get("enable", True)),
                "config": model.get("config-file-path"),
                "engine": model.get("engine"),
                "gie_id": model.get("gie_id"),
                "batch_size": model.get("batch_size"),
            }

    return {
        "pipeline": str(path),
        "source_count": len(sources),
        "sources": sources,
        "pgie_ini": str(pgie_ini_path) if pgie_ini_path else "",
        "pgie_properties": props,
        "mask_available": mask_available,
        "osd_policy": osd_policy,
        "models": model_summary,
        "validations": validations,
        "summary_json": json.dumps(
            {
                "source_count": len(sources),
                "mask_available": mask_available,
                "osd_policy": osd_policy,
            },
            sort_keys=True,
        ),
    }
