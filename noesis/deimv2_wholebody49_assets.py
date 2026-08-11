from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

WHOLEBODY49_SIZES = ("s", "x")


@dataclass(frozen=True)
class Wholebody49Variant:
    size: str
    family: str
    mode: str
    template_name: str
    onnx_name: str
    engine_name: str
    output_names: str
    network_type: int

    @property
    def has_instance_masks(self) -> bool:
        return self.mode == "masks"


_VARIANTS: Dict[str, Wholebody49Variant] = {
    "s": Wholebody49Variant(
        size="s",
        family="dinov3_s",
        mode="masks",
        template_name="config_infer_primary_deimv2_wholebody49_masks.template.ini",
        onnx_name="deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx",
        engine_name="deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine",
        output_names="label_xyxy_score;masks",
        network_type=3,
    ),
    "x": Wholebody49Variant(
        size="x",
        family="dinov3_x",
        mode="boxes",
        template_name="config_infer_primary_deimv2_wholebody49_boxes.template.ini",
        onnx_name="deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx",
        engine_name="deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine",
        output_names="label_xyxy_score",
        network_type=0,
    ),
}


def _model_dir() -> Path:
    return Path(os.environ.get("NOESIS_MODEL_DIR", REPO_ROOT / "models")).expanduser().resolve()


def _onnx_dir() -> Path:
    return Path(os.environ.get("NOESIS_ONNX_DIR", _model_dir() / "onnx")).expanduser().resolve()


def _engine_dir() -> Path:
    return Path(os.environ.get("NOESIS_ENGINE_DIR", _model_dir() / "engines")).expanduser().resolve()


def _pipeline_dir() -> Path:
    return Path(os.environ.get("NOESIS_PIPELINE_DIR", REPO_ROOT / "pipelines")).expanduser().resolve()


def _build_dir() -> Path:
    return Path(os.environ.get("NOESIS_BUILD_DIR", REPO_ROOT / "build")).expanduser().resolve()


def resolve_wholebody49_assets(size: str) -> Dict[str, Path | str | int | bool]:
    size_norm = str(size or "").strip().lower()
    if size_norm not in _VARIANTS:
        choices = "/".join(WHOLEBODY49_SIZES)
        raise ValueError(f"Wholebody49 size must be one of {choices} (got: {size})")
    variant = _VARIANTS[size_norm]
    model_dir = _model_dir()
    pipeline_dir = _pipeline_dir()
    return {
        "size": variant.size,
        "family": variant.family,
        "mode": variant.mode,
        "has_instance_masks": variant.has_instance_masks,
        "network_type": variant.network_type,
        "output_names": variant.output_names,
        "template": (pipeline_dir / variant.template_name).resolve(),
        "preproc_template": (pipeline_dir / "config_preproc.ini").resolve(),
        "onnx": (_onnx_dir() / variant.onnx_name).resolve(),
        "engine": (_engine_dir() / variant.engine_name).resolve(),
        "labels": (model_dir / "deimv2_wholebody49" / "classes.txt").resolve(),
        "parser": (
            pipeline_dir
            / "nvdsinfer_deimv2_wholebody49"
            / "libnvdsinfer_deimv2_wholebody49.so"
        ).resolve(),
        "default_output": (_build_dir() / f"config_infer_primary_wholebody49_{size_norm}.ini").resolve(),
    }


def _replace_required(pattern: str, replacement: str, text: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(f"Wholebody49 template missing required field: {label}")
    return updated


def _require_property(props: Mapping[str, str], key: str, expected: str, *, label: str) -> None:
    actual = str(props.get(key, "") or "").strip()
    if actual != expected:
        raise ValueError(f"{label} requires {key}={expected} (got {actual or '<unset>'})")


def validate_wholebody49_preprocess_properties(
    props: Mapping[str, str], *, batch_size: int = 3
) -> None:
    """Validate the fixed tensor contract consumed by both DS8 and DS9."""

    label = "Wholebody49 preprocess"
    _require_property(props, "tensor-name", "images", label=label)
    _require_property(
        props,
        "network-input-shape",
        f"{int(batch_size)};3;640;640",
        label=label,
    )
    _require_property(props, "maintain-aspect-ratio", "0", label=label)
    _require_property(props, "symmetric-padding", "0", label=label)


def validate_wholebody49_pgie_properties(
    props: Mapping[str, str], *, batch_size: int = 3
) -> str:
    """Validate parser, output tensor, mask, and detector semantics.

    Returns the selected output mode (``masks`` or ``boxes``).  This is the
    shared product contract; runtime adapters remain responsible for resolving
    and loading their own SDK-specific parser and TensorRT artifacts.
    """

    label = "Wholebody49 PGIE"
    for key, expected in (
        ("gie-unique-id", "1"),
        ("batch-size", str(int(batch_size))),
        ("network-mode", "2"),
        ("input-tensor-from-meta", "1"),
        ("num-detected-classes", "49"),
        ("operate-on-class-ids", "0"),
        ("output-tensor-meta", "0"),
    ):
        _require_property(props, key, expected, label=label)

    network_type = str(props.get("network-type", "") or "").strip()
    if network_type == "3":
        _require_property(
            props,
            "parse-bbox-instance-mask-func-name",
            "NvDsInferParseDeimv2Wholebody49",
            label=label,
        )
        _require_property(props, "output-instance-mask", "1", label=label)
        _require_property(props, "output-blob-names", "label_xyxy_score;masks", label=label)
        return "masks"
    if network_type == "0":
        _require_property(
            props,
            "parse-bbox-func-name",
            "NvDsInferParseDeimv2Wholebody49Boxes",
            label=label,
        )
        _require_property(props, "output-blob-names", "label_xyxy_score", label=label)
        if str(props.get("output-instance-mask", "") or "").strip() not in ("", "0"):
            raise ValueError(f"{label} boxes mode forbids output-instance-mask=1")
        return "boxes"
    raise ValueError(f"{label} network-type must be 0 or 3 (got {network_type or '<unset>'})")


def materialize_wholebody49_pgie_ini(
    *,
    size: str,
    output_path: Path | None = None,
    batch_size: int = 3,
    include_model_source: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    if int(batch_size) <= 0:
        raise ValueError(f"Wholebody49 batch_size must be positive (got: {batch_size})")
    assets = resolve_wholebody49_assets(size)
    template_path = Path(assets["template"])
    if not template_path.exists():
        raise FileNotFoundError(f"Wholebody49 PGIE template missing: {template_path}")

    text = template_path.read_text(encoding="utf-8")
    if include_model_source:
        text = text.replace("@ONNX_PATH@", str(Path(assets["onnx"]).resolve()))
    else:
        text = re.sub(r"(?m)^\s*onnx-file\s*=.*(?:\n|$)", "", text)
    replacements = {
        "@ENGINE_PATH@": str(Path(assets["engine"]).resolve()),
        "@LABELS_PATH@": str(Path(assets["labels"]).resolve()),
        "@CUSTOM_LIB@": str(Path(assets["parser"]).resolve()),
        "@BATCH_SIZE@": str(int(batch_size)),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)

    resolved_output = (output_path or Path(assets["default_output"])).resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(text, encoding="utf-8")
    if logger is not None:
        logger.info(
            "Wholebody49 PGIE config materialized: %s (size=%s mode=%s)",
            resolved_output,
            assets["size"],
            assets["mode"],
        )
    return resolved_output


def materialize_wholebody49_preproc_config(
    *,
    output_path: Path,
    batch_size: int,
    src_ids: Iterable[int | str],
    logger: logging.Logger | None = None,
) -> Path:
    src_ids_tuple = tuple(src_ids)
    if not src_ids_tuple:
        raise ValueError("Wholebody49 src_ids must not be empty")
    if int(batch_size) <= 0:
        raise ValueError(f"Wholebody49 batch_size must be positive (got: {batch_size})")

    template_path = Path(resolve_wholebody49_assets("s")["preproc_template"])
    if not template_path.exists():
        raise FileNotFoundError(f"Wholebody49 preprocess template missing: {template_path}")

    text = template_path.read_text(encoding="utf-8")
    src_ids_text = ";".join(str(x) for x in src_ids_tuple)
    replacements = {
        "network-input-shape": f"{int(batch_size)};3;640;640",
        "processing-width": "640",
        "processing-height": "640",
        "tensor-name": "images",
        "maintain-aspect-ratio": "0",
        "symmetric-padding": "0",
        "src-ids": src_ids_text,
    }
    for key, value in replacements.items():
        text = _replace_required(rf"(?m)^{re.escape(key)}=.*$", f"{key}={value}", text, key)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if logger is not None:
        logger.info("Wholebody49 preprocess config materialized: %s", output_path)
    return output_path


def materialize_wholebody49_configs(
    *,
    size: str,
    batch_size: int,
    src_ids: Iterable[int | str],
    include_model_source: bool = True,
    logger: logging.Logger | None = None,
) -> Dict[str, Path | str | int | bool]:
    assets = resolve_wholebody49_assets(size)
    size_norm = str(assets["size"])
    pgie_config = materialize_wholebody49_pgie_ini(
        size=size_norm,
        output_path=Path(assets["default_output"]),
        batch_size=batch_size,
        include_model_source=include_model_source,
        logger=logger,
    )
    preprocess_config = materialize_wholebody49_preproc_config(
        output_path=_build_dir() / f"config_preproc_wholebody49_{size_norm}_b{int(batch_size)}.ini",
        batch_size=batch_size,
        src_ids=src_ids,
        logger=logger,
    )
    return {
        **assets,
        "pgie_config": pgie_config,
        "preprocess_config": preprocess_config,
    }
