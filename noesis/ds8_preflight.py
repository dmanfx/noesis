from __future__ import annotations

import configparser
import ctypes.util
import json
import os
import socket
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class Severity(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    INFO = "info"


@dataclass(frozen=True)
class ValidationResult:
    severity: Severity
    code: str
    message: str
    fix_hint: str = ""
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


def has_blocking(results: Iterable[ValidationResult]) -> bool:
    return any(result.severity == Severity.BLOCK for result in results)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _result(
    severity: Severity,
    code: str,
    message: str,
    fix_hint: str = "",
    evidence: str = "",
) -> ValidationResult:
    return ValidationResult(
        severity=severity,
        code=code,
        message=message,
        fix_hint=fix_hint,
        evidence=evidence,
    )


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _known_repo_relative(candidate: Path) -> bool:
    return bool(candidate.parts) and candidate.parts[0] in {
        "build",
        "config",
        "data",
        "models",
        "native",
        "pipelines",
        "scripts",
        "services",
    }


def resolve_config_path(base_yaml_path: Path, raw: Any) -> Path:
    text = str(raw or "").strip()
    if not text:
        return Path("")
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if _known_repo_relative(candidate):
        return (REPO_ROOT / candidate).resolve()
    base_candidate = (base_yaml_path.parent / candidate).resolve()
    if base_candidate.exists():
        return base_candidate
    return (REPO_ROOT / candidate).resolve()


def parse_nvinfer_ini(path: Path) -> Dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    props: Dict[str, str] = {}
    if parser.has_section("property"):
        props.update({key.strip(): str(value).strip() for key, value in parser.items("property")})
    for section in parser.sections():
        if section.startswith("class-attrs"):
            for key, value in parser.items(section):
                props.setdefault(f"{section}.{key.strip()}", str(value).strip())
    return props


def mask_output_available(props: Mapping[str, Any]) -> bool:
    output_mask = str(props.get("output-instance-mask", "")).strip().lower()
    if output_mask in {"0", "false", "no", "off"}:
        return False

    mask_parser = str(props.get("parse-bbox-instance-mask-func-name", "")).strip()
    network_type = str(props.get("network-type", "")).strip()
    blob_names = str(props.get("output-blob-names", "")).strip().lower()

    if output_mask in {"1", "true", "yes", "on"} and mask_parser:
        return True
    if output_mask in {"1", "true", "yes", "on"} and network_type == "3":
        return True
    if not output_mask and mask_parser and network_type == "3":
        return True
    if output_mask in {"1", "true", "yes", "on"} and "mask" in blob_names:
        return True
    return False


def derive_osd_policy_from_ini(props: Mapping[str, Any]) -> Dict[str, int]:
    if mask_output_available(props):
        return {"process-mode": 0, "display-mask": 1, "display-bbox": 0, "display-text": 1}
    return {"process-mode": 0, "display-mask": 0, "display-bbox": 1, "display-text": 1}


def _validate_path(
    results: List[ValidationResult],
    *,
    code: str,
    label: str,
    path: Path,
    required: bool = True,
) -> None:
    if path and path.exists():
        results.append(_result(Severity.INFO, f"{code}.ok", f"{label} exists", evidence=str(path)))
        return
    severity = Severity.BLOCK if required else Severity.WARN
    results.append(
        _result(
            severity,
            f"{code}.missing",
            f"{label} is missing: {path}",
            "Materialize the selected profile or correct the referenced path.",
            str(path),
        )
    )


def _model_cfg(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    models = cfg.get("models")
    if not isinstance(models, Mapping):
        return {}
    model = models.get(name)
    return model if isinstance(model, Mapping) else {}


def _validate_secondary_model(
    results: List[ValidationResult],
    *,
    cfg: Mapping[str, Any],
    base_yaml_path: Path,
    name: str,
    expected_gie_id: Optional[int] = None,
) -> None:
    model = _model_cfg(cfg, name)
    if not model or not _as_bool(model.get("enable"), default=False):
        results.append(_result(Severity.INFO, f"model.{name}.disabled", f"{name} model is disabled"))
        return
    raw_ini = model.get("config-file-path")
    if raw_ini:
        _validate_path(
            results,
            code=f"model.{name}.config",
            label=f"{name} nvinfer config",
            path=resolve_config_path(base_yaml_path, raw_ini),
        )
    raw_engine = model.get("engine")
    if raw_engine:
        _validate_path(
            results,
            code=f"model.{name}.engine",
            label=f"{name} TensorRT engine",
            path=resolve_config_path(base_yaml_path, raw_engine),
        )
    if expected_gie_id is not None and "gie_id" in model:
        try:
            actual = int(model.get("gie_id"))
        except Exception:
            actual = -1
        if actual != expected_gie_id:
            results.append(
                _result(
                    Severity.WARN,
                    f"model.{name}.gie_id",
                    f"{name} gie_id is {actual}, expected {expected_gie_id}",
                    "Keep model gie IDs aligned with DS8 metadata hooks.",
                )
            )


def validate_metadata_compatibility(
    effective_cfg: Mapping[str, Any],
    pipeline_path: Path,
    *,
    tracking_mode: str = "baseline",
    strict_baseline: bool = False,
) -> List[ValidationResult]:
    results: List[ValidationResult] = []
    pgie = _model_cfg(effective_cfg, "pgie")
    pgie_ini_raw = pgie.get("config-file-path") if isinstance(pgie, Mapping) else None
    if not pgie_ini_raw:
        return [
            _result(
                Severity.BLOCK,
                "metadata.pgie_config.missing",
                "models.pgie.config-file-path is not set",
                "Choose a preset or PGIE profile with an explicit nvinfer config.",
            )
        ]

    pgie_ini = resolve_config_path(pipeline_path, pgie_ini_raw)
    if not pgie_ini.exists():
        return [
            _result(
                Severity.BLOCK,
                "metadata.pgie_config.missing",
                f"PGIE config does not exist: {pgie_ini}",
                "Materialize the selected model profile or fix models.pgie.config-file-path.",
                str(pgie_ini),
            )
        ]

    props = parse_nvinfer_ini(pgie_ini)
    mask_available = mask_output_available(props)
    osd_policy = derive_osd_policy_from_ini(props)
    results.append(
        _result(
            Severity.INFO,
            "metadata.pgie.inspect",
            "PGIE metadata contract inspected",
            evidence=json.dumps(
                {
                    "config": str(pgie_ini),
                    "network_type": props.get("network-type"),
                    "mask_available": mask_available,
                    "osd_policy": osd_policy,
                },
                sort_keys=True,
            ),
        )
    )

    mode = str(tracking_mode or "baseline").strip().lower()
    if mode == "baseline" and not mask_available:
        severity = Severity.BLOCK if strict_baseline else Severity.WARN
        results.append(
            _result(
                severity,
                "metadata.object_depth.missing_mask",
                "Selected PGIE does not emit instance masks; object-depth anchors will report missing_mask.",
                "Use a segmentation PGIE for strict depth-fused baselines, or launch with strict baseline disabled.",
                str(pgie_ini),
            )
        )

    current_osd = effective_cfg.get("osd")
    if isinstance(current_osd, Mapping):
        current_mask = int(current_osd.get("display-mask", -1))
        current_bbox = int(current_osd.get("display-bbox", -1))
        if current_mask != osd_policy["display-mask"] or current_bbox != osd_policy["display-bbox"]:
            results.append(
                _result(
                    Severity.WARN,
                    "metadata.osd.policy_mismatch",
                    "Configured OSD display policy does not match the PGIE metadata contract.",
                    "Let the console materialize the launch YAML so detect uses bbox and seg uses masks.",
                    json.dumps({"configured": dict(current_osd), "expected": osd_policy}, sort_keys=True),
                )
            )
    else:
        results.append(
            _result(
                Severity.INFO,
                "metadata.osd.policy_derived",
                "OSD policy can be derived from the selected PGIE.",
                evidence=json.dumps(osd_policy, sort_keys=True),
            )
        )
    return results


def validate_depth_registration(
    cfg: Mapping[str, Any],
    pipeline_path: Path,
    *,
    tracking_mode: str,
) -> List[ValidationResult]:
    if str(tracking_mode or "baseline").strip().lower() != "baseline":
        return [_result(Severity.INFO, "depth_registration.not_required", "Depth registration is not required for v3dt mode")]

    depth_cfg = cfg.get("depth_registration")
    raw_path = None
    if isinstance(depth_cfg, Mapping):
        raw_path = depth_cfg.get("path")
    raw_path = raw_path or "config/depth_registration.json"
    path = resolve_config_path(pipeline_path, raw_path)
    if not path.exists():
        return [
            _result(
                Severity.BLOCK,
                "depth_registration.missing",
                f"Depth registration artifact is missing: {path}",
                "Run scripts/build_depth_registration.py before launching baseline DS8.",
                str(path),
            )
        ]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            _result(
                Severity.BLOCK,
                "depth_registration.invalid",
                f"Depth registration artifact is not valid JSON: {exc}",
                "Rebuild the depth registration artifact.",
                str(path),
            )
        ]
    if not isinstance(payload, Mapping):
        return [
            _result(
                Severity.BLOCK,
                "depth_registration.invalid",
                "Depth registration artifact must be a JSON object.",
                "Rebuild the depth registration artifact.",
                str(path),
            )
        ]
    return [_result(Severity.INFO, "depth_registration.ok", "Depth registration artifact is readable", evidence=str(path))]


def validate_ports(host: str, ports: Iterable[int]) -> List[ValidationResult]:
    results: List[ValidationResult] = []
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                conflict = sock.connect_ex((host, int(port))) == 0
        except Exception as exc:
            results.append(
                _result(
                    Severity.WARN,
                    f"port.{port}.unknown",
                    f"Could not probe {host}:{port}: {exc}",
                    "Inspect the port manually before launch if startup fails.",
                )
            )
            continue
        if conflict:
            results.append(
                _result(
                    Severity.BLOCK,
                    f"port.{port}.busy",
                    f"{host}:{port} is already accepting TCP connections.",
                    "Stop the conflicting process or choose a different port.",
                )
            )
        else:
            results.append(_result(Severity.INFO, f"port.{port}.free", f"{host}:{port} is free"))
    return results


def validate_cuda_preflight() -> List[ValidationResult]:
    if _as_bool(os.environ.get("NOESIS_SKIP_CUDA_PREFLIGHT"), default=False):
        return [_result(Severity.INFO, "cuda.preflight.skipped", "CUDA preflight skipped by NOESIS_SKIP_CUDA_PREFLIGHT")]

    cudart = ctypes.util.find_library("cudart")
    if not cudart:
        return [
            _result(
                Severity.WARN,
                "cuda.preflight.cudart_missing",
                "CUDA runtime library was not found in the current linker path.",
                "Use the DS8 runtime environment before launching GPU validation.",
            )
        ]
    return [_result(Severity.INFO, "cuda.preflight.ok", "CUDA runtime library is discoverable", evidence=str(cudart))]


def run_preflight(
    *,
    pipeline_path: Path,
    tracking_mode: str = "baseline",
    strict_baseline: bool = False,
    include_cuda: bool = False,
) -> List[ValidationResult]:
    path = Path(pipeline_path)
    results: List[ValidationResult] = []
    if not path.exists():
        return [
            _result(
                Severity.BLOCK,
                "pipeline_config.missing",
                f"Pipeline config does not exist: {path}",
                "Choose an existing config/infer*.yaml file.",
                str(path),
            )
        ]
    try:
        cfg = _load_yaml(path)
    except Exception as exc:
        return [
            _result(
                Severity.BLOCK,
                "pipeline_config.invalid",
                f"Pipeline config could not be parsed: {exc}",
                "Fix the YAML syntax before launch.",
                str(path),
            )
        ]
    if not isinstance(cfg, Mapping):
        return [
            _result(
                Severity.BLOCK,
                "pipeline_config.schema",
                "Pipeline config must be a YAML mapping.",
                "Use the DS8 config/infer.yaml schema.",
                str(path),
            )
        ]

    results.append(_result(Severity.INFO, "pipeline_config.ok", "Pipeline YAML is readable", evidence=str(path)))
    sources = cfg.get("sources")
    if isinstance(sources, list) and sources:
        results.append(_result(Severity.INFO, "sources.ok", f"{len(sources)} source(s) configured"))
    else:
        results.append(
            _result(
                Severity.BLOCK,
                "sources.missing",
                "No DS8 sources are configured.",
                "Add at least one source under the pipeline YAML sources list.",
            )
        )

    pgie = _model_cfg(cfg, "pgie")
    if pgie.get("engine"):
        _validate_path(
            results,
            code="model.pgie.engine",
            label="PGIE TensorRT engine",
            path=resolve_config_path(path, pgie.get("engine")),
        )
    results.extend(
        validate_metadata_compatibility(
            cfg,
            path,
            tracking_mode=tracking_mode,
            strict_baseline=strict_baseline,
        )
    )
    results.extend(validate_depth_registration(cfg, path, tracking_mode=tracking_mode))
    _validate_secondary_model(results, cfg=cfg, base_yaml_path=path, name="reid", expected_gie_id=3)
    _validate_secondary_model(results, cfg=cfg, base_yaml_path=path, name="pose", expected_gie_id=4)
    _validate_secondary_model(results, cfg=cfg, base_yaml_path=path, name="depth_tracking", expected_gie_id=5)
    _validate_secondary_model(results, cfg=cfg, base_yaml_path=path, name="mapanything", expected_gie_id=2)
    if include_cuda:
        results.extend(validate_cuda_preflight())
    return results
