"""DS9-owned helpers for materializing runtime configuration.

This module deliberately has no dependency on a historical runtime or preflight
layer.  It is safe to import before Service Maker or native extensions are
loaded, which keeps DS9 configuration policy testable on non-DS9 hosts.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any, Dict, Mapping


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent


def resolve_pipeline_cfg_path(yaml_path: Path, raw: str) -> Path:
    """Resolve a DS9 pipeline reference using the same scope as the launcher."""

    value = str(raw or "").strip()
    if not value:
        return Path("")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if value.startswith("DS9/"):
        return (REPO_ROOT / candidate).resolve()
    if value.startswith(("config/", "models/", "pipelines/", "build/")):
        try:
            yaml_path.expanduser().resolve().relative_to(DS9_ROOT.resolve())
        except ValueError:
            scope = REPO_ROOT
        else:
            scope = DS9_ROOT
        return (scope / candidate).resolve()
    return (yaml_path.parent.resolve() / candidate).resolve()


def parse_ini_section(path: Path, section: str = "property") -> Dict[str, str]:
    if not path.is_file():
        return {}
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    if not parser.has_section(section):
        return {}
    return {key.strip(): str(value).strip() for key, value in parser.items(section)}


def mask_output_available(ini_props: Mapping[str, Any]) -> bool:
    output_mask = str(ini_props.get("output-instance-mask", "") or "").strip().lower()
    if output_mask in {"0", "false", "no", "off"}:
        return False

    parser_name = str(ini_props.get("parse-bbox-instance-mask-func-name", "") or "").strip()
    network_type = str(ini_props.get("network-type", "") or "").strip()
    blob_names = str(ini_props.get("output-blob-names", "") or "").strip().lower()

    if output_mask in {"1", "true", "yes", "on"} and parser_name:
        return True
    if output_mask in {"1", "true", "yes", "on"} and network_type == "3":
        return True
    if not output_mask and parser_name and network_type == "3":
        return True
    if output_mask in {"1", "true", "yes", "on"} and "mask" in blob_names:
        return True
    return False


def derive_osd_policy_from_ini(ini_props: Mapping[str, Any]) -> Dict[str, int]:
    if mask_output_available(ini_props):
        return {"process-mode": 1, "display-mask": 1, "display-bbox": 1, "display-text": 1}
    return {"process-mode": 1, "display-mask": 0, "display-bbox": 1, "display-text": 1}


def apply_osd_from_pgie_ini(
    pipeline_cfg: Mapping[str, Any],
    pipeline_path: Path,
) -> Dict[str, Any]:
    """Return a shallow config copy with OSD policy derived from the PGIE INI."""

    merged = dict(pipeline_cfg)
    models = pipeline_cfg.get("models") if isinstance(pipeline_cfg, Mapping) else {}
    pgie = (models or {}).get("pgie") if isinstance(models, Mapping) else {}
    ini_raw = str((pgie or {}).get("config-file-path", "") if isinstance(pgie, Mapping) else "")
    ini_path = (
        resolve_pipeline_cfg_path(pipeline_path, ini_raw)
        if ini_raw.strip()
        else Path("")
    )
    ini_props = parse_ini_section(ini_path) if ini_raw.strip() else {}
    merged["osd"] = derive_osd_policy_from_ini(ini_props)
    return merged
