from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def nvinfer_props_from_infer_yaml(yaml_path: str | Path) -> Dict[str, Any]:
    """Extract minimal nvinfer properties from DS8 infer.yaml (PGIE section).

    Returns keys compatible with Gst nvinfer element set_property calls.
    """
    base_dir = Path(yaml_path).parent.resolve()
    repo_root = base_dir.parent

    def _resolve(path_val: str | Path) -> str:
        if not path_val:
            return ""
        p = Path(path_val)
        if p.is_absolute():
            return str(p)
        text = str(path_val)
        if text.startswith(("config/", "models/", "pipelines/")):
            return str((repo_root / text).resolve())
        return str((base_dir / p).resolve())

    cfg = _load_yaml(yaml_path)
    models = cfg.get("models", {}) or {}
    pgie = models.get("pgie", {}) or {}
    props: Dict[str, Any] = {}

    engine = pgie.get("engine")
    if isinstance(engine, str) and engine:
        props["model-engine-file"] = _resolve(engine)

    cfg_path = pgie.get("config-file-path") or pgie.get("config-file")
    if isinstance(cfg_path, str) and cfg_path:
        props["config-file-path"] = _resolve(cfg_path)

    bs = pgie.get("batch_size")
    if isinstance(bs, int) and bs > 0:
        props["batch-size"] = bs
    else:
        try:
            if int(bs) > 0:
                props["batch-size"] = int(bs)
        except Exception:
            pass

    gie_id = pgie.get("gie_id")
    try:
        if gie_id is not None:
            val = int(gie_id)
            if val > 0:
                props["unique-id"] = val
    except Exception:
        pass

    nm = (pgie.get("network_mode") or "").strip().lower()
    if nm:
        # 0=FP32, 1=INT8, 2=FP16
        mapping = {"fp32": 0, "int8": 1, "fp16": 2}
        props["network-mode"] = mapping.get(nm, 0)

    if bool(pgie.get("attach_tensor_meta") or False):
        props["output-tensor-meta"] = True

    return props


def tracker_config_from_yaml(default_path: str | Path = "config/nvtracker.yaml") -> Optional[str]:
    p = Path(default_path)
    return str(p.resolve()) if p.exists() else None


def analytics_config_from_yaml(default_path: str | Path = "config/nvdsanalytics.yaml") -> Optional[str]:
    p = Path(default_path)
    return str(p.resolve()) if p.exists() else None
