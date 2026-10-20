from __future__ import annotations

import configparser
import hashlib
import io
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


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
REID_SWIN_OBJECT_ORIGINS = (1, 4)


def reid_swin_replay_every_frame(config: Mapping[str, Any] | None) -> bool:
    """Narrow recorded experiment, never a normal/live cadence override."""
    config = {} if config is None else config
    value = config.get("xc3d_swin_replay_every_frame", False)
    if type(value) is not bool:
        raise ValueError("xc3d_swin_replay_every_frame must be boolean")
    if not value:
        return False
    if (config.get("recorded_replay") != {"realtime": True, "preserve_frames": True}
            or any(type(v) is not bool for v in config.get("recorded_replay", {}).values())
            or config.get("xc3d_world_quality") != {
                "contract": "noesis.xc3d.world_quality.v1", "enabled": True}
            or type(config["xc3d_world_quality"]["enabled"]) is not bool
            or config.get("xc3d_candidate") is not None):
        raise ValueError("Swin every-frame requires explicit finite world-quality replay")
    mux = config.get("streammux", {})
    if not isinstance(mux, Mapping) or type(mux.get("live-source")) is not int or mux["live-source"] != 0:
        raise ValueError("Swin every-frame requires recorded live-source=0")
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("Swin every-frame requires three registered file sources")
    for source, camera in zip(sources, ("living-room", "kitchen", "family-room")):
        if (not isinstance(source, Mapping) or source.get("camera_id") != camera
                or source.get("file-loop") is not False or source.get("uri_secret")):
            raise ValueError("Swin every-frame source binding changed")
        uri = urlsplit(str(source.get("uri", "")))
        if (uri.scheme != "file" or uri.netloc or not uri.path.startswith("/")
                or uri.query or uri.fragment):
            raise ValueError("Swin every-frame requires finite local file URIs")
    return True


def _swin_config_sections(path: Path) -> dict[str, dict[str, str]]:
    if path.stat().st_size > 65536:
        raise ValueError("Swin source INI exceeds bounded configuration size")
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    parser.read_string(path.read_text(encoding="utf-8"))
    if parser.defaults():
        raise ValueError("Swin source INI defaults are not canonical")
    return {section: dict(parser[section]) for section in parser.sections()}


def _canonical_swin_ini() -> Path:
    return Path(__file__).resolve().parents[1] / "DS9/pipelines" / REID_SWIN_CONFIG_NAME


def _swin_artifact_path(raw: str, owner: Path) -> Path:
    path = Path(raw)
    repo = Path(__file__).resolve().parents[1]
    if not path.is_absolute() and raw.startswith("DS9/"):
        path = repo / path
    elif not path.is_absolute():
        path = owner.parent / path
    path = path.resolve()
    # The canonical INI uses virtual DS9/models paths. Resolve that same
    # selected external model root before serializing an absolute task INI;
    # an absolute nonexistent checkout path would bypass preflight's aliases.
    try:
        relative = path.relative_to(repo / "DS9/models")
    except ValueError:
        return path
    model_override = os.environ.get("NOESIS_MODEL_DIR", "").strip()
    artifact_override = os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "").strip()
    model_root = (Path(model_override).expanduser() if model_override else
                  Path(artifact_override).expanduser() / "models" if artifact_override else
                  repo / "DS9/models")
    if relative.parts and relative.parts[0] in ("onnx", "engines"):
        directory = relative.parts[0]
        variable = "NOESIS_ONNX_DIR" if directory == "onnx" else "NOESIS_ENGINE_DIR"
        root = Path(os.environ.get(variable, str(model_root / directory))).expanduser()
        return (root / Path(*relative.parts[1:])).resolve()
    return (model_root / relative).resolve()


def validate_reid_swin_source_config(
    path: Path, *, pipeline_config: Mapping[str, Any] | None = None
) -> Mapping[str, str]:
    """Validate canonical12 or its exact, explicitly authorized replay0 derivative."""
    if not reid_swin_replay_every_frame(pipeline_config):
        props = load_reid_swin_nvinfer_properties(path)
        validate_reid_swin_nvinfer_properties(props)
        return props
    canonical = _canonical_swin_ini()
    expected = _swin_config_sections(canonical)
    validate_reid_swin_nvinfer_properties(expected.get("property", {}))
    model = pipeline_config.get("models", {}).get("reid", {})
    validate_reid_swin_model_config(model)
    if _swin_artifact_path(str(model["engine"]), canonical) != _swin_artifact_path(
            expected["property"]["model-engine-file"], canonical):
        raise ValueError("Swin replay YAML engine binding changed")
    actual = _swin_config_sections(Path(path))
    _require_filename(path, REID_SWIN_CONFIG_NAME, label="Swin replay source")
    if set(actual) != set(expected) or "property" not in actual:
        raise ValueError("Swin replay source sections changed")
    normalized = {section: dict(props) for section, props in actual.items()}
    _require_property(normalized["property"], "secondary-reinfer-interval", "0", label="Swin replay")
    normalized["property"]["secondary-reinfer-interval"] = "12"
    for key in ("onnx-file", "model-engine-file"):
        if _swin_artifact_path(actual["property"].get(key, ""), Path(path)) != _swin_artifact_path(
                expected["property"][key], canonical):
            raise ValueError("Swin replay artifact binding changed: " + key)
        normalized["property"][key] = expected["property"][key]
    if normalized != expected:
        raise ValueError("Swin replay changed properties beyond certified cadence")
    return actual["property"]


def render_reid_swin_replay_ini(pipeline_config: Mapping[str, Any]) -> str:
    """Return task-local source text; caller owns its finite replay output path."""
    if not reid_swin_replay_every_frame(pipeline_config):
        raise ValueError("Swin replay renderer requires explicit every-frame opt-in")
    canonical = _canonical_swin_ini()
    sections = _swin_config_sections(canonical)
    validate_reid_swin_nvinfer_properties(sections["property"])
    sections["property"]["secondary-reinfer-interval"] = "0"
    for key in ("onnx-file", "model-engine-file"):
        sections["property"][key] = str(_swin_artifact_path(sections["property"][key], canonical))
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_dict(sections)
    result = io.StringIO()
    parser.write(result)
    return result.getvalue()


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
        # SDK supports one ID or all, not an ID list. The native single-pass
        # tracker-output guard validates exactly REID_SWIN_OBJECT_ORIGINS;
        # proposal-only metadata never crosses that boundary.
        "operate-on-gie-id": "-1",
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
    """Validate the SDK-neutral YAML contract consumed by DS9.1."""

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
