from __future__ import annotations

import configparser
import hashlib
from pathlib import Path
from typing import Any, Mapping


REID_SWIN_PROFILE_NAME = "reid_sgie"
REID_SWIN_SOURCE_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/tao/"
    "reidentificationnet_transformer/versions/deployable_v1.0/files/"
    "swin_tiny_market1501_aicity156_featuredim256.onnx"
)
REID_SWIN_ONNX_NAME = "reid_swin_tiny_market1501_aicity156_featuredim256.onnx"
REID_SWIN_ENGINE_NAME = "reid_swin_tiny_aicity156_dyn_b16_fp16.engine"
REID_SWIN_CONFIG_NAME = "config_infer_secondary_reid_swin.ini"
REID_SWIN_ONNX_SHA256 = "37b008d887495d5fbad31c72aadbf381c5456cb619e0e2a9c246a6d48a2932ff"
REID_SWIN_ONNX_SIZE_BYTES = 113_893_579
REID_SWIN_INPUT_NAME = "input"
REID_SWIN_INPUT_SHAPE = (-1, 3, 256, 128)
REID_SWIN_OUTPUT_LAYER = "fc_pred"
REID_SWIN_EMBEDDING_DIM = 256
REID_SWIN_BATCH_SIZE = 16
REID_SWIN_GIE_ID = 3


def _require_property(
    props: Mapping[str, Any], key: str, expected: str, *, label: str
) -> None:
    actual = str(props.get(key, "") or "").strip()
    if actual != expected:
        raise ValueError(
            f"{label} requires {key}={expected} (got {actual or '<unset>'})"
        )


def _require_filename(raw: Any, expected: str, *, label: str) -> None:
    actual = Path(str(raw or "").strip()).name
    if actual != expected:
        raise ValueError(
            f"{label} requires artifact {expected} (got {actual or '<unset>'})"
        )


def load_reid_swin_nvinfer_properties(path: Path) -> Mapping[str, str]:
    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#",),
        strict=False,
    )
    read = parser.read(path, encoding="utf-8")
    if not read or "property" not in parser:
        raise ValueError(f"ReID Swin nvinfer config has no [property] section: {path}")
    return parser["property"]


def validate_reid_swin_nvinfer_properties(props: Mapping[str, Any]) -> None:
    """Validate the exact TAO Swin-Tiny SGIE and preprocessing contract.

    This model emits a raw 256-dimensional tensor. It intentionally has no
    custom parser: the native metadata bridge selects ``fc_pred`` directly.
    """

    label = "TAO Swin-Tiny ReID SGIE"
    expected = {
        "batch-size": str(REID_SWIN_BATCH_SIZE),
        "network-mode": "2",
        "gie-unique-id": str(REID_SWIN_GIE_ID),
        "process-mode": "2",
        "network-type": "1",
        "model-color-format": "0",
        "net-scale-factor": "0.01735207",
        "offsets": "123.675;116.28;103.53",
        "infer-dims": "3;256;128",
        "maintain-aspect-ratio": "0",
        "operate-on-gie-id": "1",
        "operate-on-class-ids": "0",
        "input-object-min-width": "24",
        "input-object-min-height": "48",
        "secondary-reinfer-interval": "12",
        "output-tensor-meta": "1",
        "classifier-async-mode": "0",
    }
    for key, value in expected.items():
        _require_property(props, key, value, label=label)

    _require_filename(props.get("onnx-file"), REID_SWIN_ONNX_NAME, label=label)
    _require_filename(
        props.get("model-engine-file"), REID_SWIN_ENGINE_NAME, label=label
    )
    forbidden = sorted(
        key
        for key in props
        if str(key).startswith("parse-")
        or str(key) in {"custom-lib-path", "output-blob-names"}
    )
    if forbidden:
        raise ValueError(
            f"{label} consumes raw {REID_SWIN_OUTPUT_LAYER} tensor metadata; "
            f"parser properties are forbidden: {', '.join(forbidden)}"
        )


def validate_reid_swin_model_config(model_cfg: Mapping[str, Any]) -> None:
    """Validate the SDK-neutral YAML contract consumed by DS8 and DS9."""

    label = "models.reid TAO Swin-Tiny profile"
    expected: Mapping[str, Any] = {
        "enable": True,
        "name": REID_SWIN_PROFILE_NAME,
        "batch_size": REID_SWIN_BATCH_SIZE,
        "gie_id": REID_SWIN_GIE_ID,
        "layer": REID_SWIN_OUTPUT_LAYER,
        "embedding_dim": REID_SWIN_EMBEDDING_DIM,
        "attach_tensor_meta": True,
    }
    for key, value in expected.items():
        actual = model_cfg.get(key)
        if actual != value:
            raise ValueError(
                f"{label} requires {key}={value!r} (got {actual!r})"
            )
    _require_filename(
        model_cfg.get("config-file-path"), REID_SWIN_CONFIG_NAME, label=label
    )
    _require_filename(model_cfg.get("engine"), REID_SWIN_ENGINE_NAME, label=label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_reid_swin_onnx_source(path: Path) -> None:
    """Require the provenance-locked NVIDIA TAO deployable ONNX bytes."""

    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"TAO Swin-Tiny ReID ONNX source is missing or empty: {path}")
    actual_size = path.stat().st_size
    if actual_size != REID_SWIN_ONNX_SIZE_BYTES:
        raise ValueError(
            "TAO Swin-Tiny ReID ONNX size mismatch: "
            f"expected {REID_SWIN_ONNX_SIZE_BYTES}, got {actual_size} ({path})"
        )
    actual_digest = sha256_file(path)
    if actual_digest != REID_SWIN_ONNX_SHA256:
        raise ValueError(
            "TAO Swin-Tiny ReID ONNX SHA-256 mismatch: "
            f"expected {REID_SWIN_ONNX_SHA256}, got {actual_digest} ({path})"
        )
