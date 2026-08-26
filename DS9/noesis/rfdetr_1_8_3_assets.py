"""Runtime materialization for the reviewed RF-DETR 1.8.3 model matrix."""

from __future__ import annotations

import configparser
import hashlib
import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


DS9_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = DS9_ROOT / "config" / "rfdetr_1_8_3_models.json"
RELEASE_VERSION = "1.8.3"
MATRIX_SCHEMA = "noesis.ds9.rfdetr-model-matrix.v1"
SOURCE_PROVENANCE_SCHEMA = "noesis.ds9.rfdetr-artifact-provenance.v1"
RUNTIME_ENGINE_RECEIPT_SCHEMA = (
    "noesis.ds9.rfdetr-runtime-engine-provenance.v1"
)
RUNTIME_INPUT_CONTRACT = "rgb01_imagenet"
RUNTIME_ADAPTER_REVISION = "sub_div_float32_v1"
RUNTIME_ADAPTER_SPEC = {
    "input_contract": RUNTIME_INPUT_CONTRACT,
    "adapter_revision": RUNTIME_ADAPTER_REVISION,
    "input_dtype": "float32",
    "input_range": [0.0, 1.0],
    "operation": "(input - mean) / std",
    "channel_order": "RGB",
    "layout": "NCHW",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "nodes": [
        "NoesisInputNormalize/Sub",
        "NoesisInputNormalize/DivStd",
    ],
}
_RUNTIME_ENGINE_PROFILE_CONTRACTS = {
    "fp32_no_tf32": {
        "precision": "fp32",
        "fp16_enabled": False,
        "tf32_enabled": False,
        "trtexec_precision_args": ("--noTF32",),
        "network_mode": "0",
    },
    "fp16_tf32": {
        "precision": "fp16",
        "fp16_enabled": True,
        "tf32_enabled": True,
        "trtexec_precision_args": ("--fp16",),
        "network_mode": "2",
    },
}
RFDETR_DETECTION_SIZES = ("n", "s", "m", "l")
RFDETR_SEGMENTATION_SIZES = ("n", "s", "m", "l", "x", "2x")
RFDETR_KEYPOINT_VARIANTS = ("preview",)
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_MAX_PARSER_SOURCE_BYTES = 32 * 1024 * 1024
_SIZE_TO_VARIANT = {
    "detection": {
        "n": "nano",
        "s": "small",
        "m": "medium",
        "l": "large",
    },
    "segmentation": {
        "n": "nano",
        "s": "small",
        "m": "medium",
        "l": "large",
        "x": "xlarge",
        "2x": "2xlarge",
    },
    "keypoint": {
        "preview": "preview",
    },
}
_PARSER_CONTRACTS = {
    "detection": {
        "id": "parser.rfdetr_detect",
        "role": "RF-DETR detect-only output parser",
        "output": "DS9/pipelines/nvdsinfer_rfdetr/libnvdsinfer_rfdetr.so",
        "source": "DS9/pipelines/nvdsinfer_rfdetr/nvdsinfer_rfdetr.cpp",
        "builder": "DS9/scripts/build_custom_parsers.sh",
        "profiles": ("rfdetr", "full"),
    },
    "segmentation": {
        "id": "parser.rfdetr_seg",
        "role": "RF-DETR segmentation output parser",
        "output": "DS9/pipelines/nvdsinfer_rfdetr_seg/libnvdsinfer_rfdetr_seg.so",
        "source": "DS9/pipelines/nvdsinfer_rfdetr_seg/nvdsinfer_rfdetr_seg.cpp",
        "builder": "DS9/scripts/build_custom_parsers.sh",
        "profiles": ("rfdetr_seg", "full"),
    },
    "keypoint": {
        "id": "parser.rfdetr_keypoint",
        "role": "RF-DETR 1.8.3 keypoint-preview person bbox output parser",
        "output": (
            "DS9/pipelines/nvdsinfer_rfdetr_keypoint/"
            "libnvdsinfer_rfdetr_keypoint.so"
        ),
        "source": (
            "DS9/pipelines/nvdsinfer_rfdetr_keypoint/"
            "nvdsinfer_rfdetr_keypoint.cpp"
        ),
        "builder": "DS9/scripts/build_all_parsers_ds9.sh",
        "profiles": ("rfdetr_keypoint", "full"),
    },
}


class RFDETRAssetAttestationError(ValueError):
    """Raised before config generation when a selected runtime asset drifts."""


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _model_dir() -> Path:
    return _absolute(
        Path(os.environ.get("NOESIS_MODEL_DIR", DS9_ROOT / "models"))
    )


def _onnx_dir() -> Path:
    return _absolute(
        Path(os.environ.get("NOESIS_ONNX_DIR", _model_dir() / "onnx"))
    )


def _engine_dir() -> Path:
    return _absolute(
        Path(os.environ.get("NOESIS_ENGINE_DIR", _model_dir() / "engines"))
    )


def _pipeline_dir() -> Path:
    return _absolute(
        Path(os.environ.get("NOESIS_PIPELINE_DIR", DS9_ROOT / "pipelines"))
    )


def _build_dir() -> Path:
    return _absolute(
        Path(os.environ.get("NOESIS_BUILD_DIR", DS9_ROOT / "build"))
    )


def _safe_token(raw: object, label: str) -> str:
    value = str(raw or "")
    if (
        not _SAFE_TOKEN.fullmatch(value)
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ValueError(f"unsafe RF-DETR {label}: {value!r}")
    return value


def _required_sha256(raw: object, label: str) -> str:
    value = str(raw or "")
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid RF-DETR {label}: {value!r}")
    return value


def _load_matrix() -> dict[str, Any]:
    try:
        payload = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read RF-DETR model matrix: {MATRIX_PATH}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MATRIX_SCHEMA:
        raise ValueError(f"unexpected RF-DETR model matrix schema: {MATRIX_PATH}")
    release = payload.get("release")
    if not isinstance(release, dict) or release.get("version") != RELEASE_VERSION:
        raise ValueError(f"unexpected RF-DETR release in model matrix: {MATRIX_PATH}")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"RF-DETR model matrix has no model rows: {MATRIX_PATH}")
    return payload


def _model_row(family: str, size: str) -> dict[str, Any]:
    family_norm = str(family or "").strip().lower()
    size_norm = str(size or "").strip().lower()
    variants = _SIZE_TO_VARIANT.get(family_norm)
    if variants is None:
        raise ValueError(
            "RF-DETR runtime family must be detection, segmentation, "
            f"or keypoint (got: {family})"
        )
    if size_norm not in variants:
        choices = "/".join(variants)
        raise ValueError(
            f"RF-DETR {family_norm} size must be one of {choices} (got: {size})"
        )
    variant = variants[size_norm]
    rows = [
        row
        for row in _load_matrix()["models"]
        if isinstance(row, dict)
        and row.get("family") == family_norm
        and row.get("variant") == variant
    ]
    if len(rows) != 1:
        raise ValueError(
            f"RF-DETR matrix must contain exactly one {family_norm}/{variant} row"
        )
    row = dict(rows[0])
    if row.get("enabled") is not True:
        raise ValueError(
            f"RF-DETR runtime model is disabled: {row.get('id', variant)}"
        )
    return row


def _runtime_contract(row: Mapping[str, Any]) -> dict[str, str]:
    runtime = row.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError(
            f"RF-DETR model {row.get('id')} has no reviewed runtime artifact contract"
        )
    input_contract = str(runtime.get("input_contract") or "").strip()
    if input_contract != RUNTIME_INPUT_CONTRACT:
        raise ValueError(
            f"RF-DETR model {row.get('id')} runtime input contract must be "
            f"{RUNTIME_INPUT_CONTRACT} (got: {input_contract or '<unset>'})"
        )
    adapter_revision = str(runtime.get("adapter_revision") or "").strip()
    if adapter_revision != RUNTIME_ADAPTER_REVISION:
        raise ValueError(
            f"RF-DETR model {row.get('id')} runtime adapter revision must be "
            f"{RUNTIME_ADAPTER_REVISION} (got: {adapter_revision or '<unset>'})"
        )
    onnx_filename = _safe_token(runtime.get("onnx_filename"), "runtime ONNX filename")
    engine_profile = _safe_token(
        runtime.get("engine_profile"), "runtime engine profile"
    )
    if engine_profile not in _RUNTIME_ENGINE_PROFILE_CONTRACTS:
        choices = "/".join(_RUNTIME_ENGINE_PROFILE_CONTRACTS)
        raise ValueError(
            f"RF-DETR model {row.get('id')} runtime engine profile must be "
            f"one of {choices} (got: {engine_profile})"
        )
    engine_filename = _safe_token(
        runtime.get("engine_filename"), "runtime engine filename"
    )
    digests = {
        key: _required_sha256(
            runtime.get(key), f"runtime {key.replace('_', ' ')}"
        )
        for key in (
            "onnx_sha256",
            "onnx_receipt_sha256",
            "engine_sha256",
            "engine_receipt_sha256",
        )
    }
    return {
        "input_contract": input_contract,
        "adapter_revision": adapter_revision,
        "onnx_filename": onnx_filename,
        "engine_profile": engine_profile,
        "engine_filename": engine_filename,
        **digests,
    }


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_uid),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _require_no_symlink_chain(path: Path, *, root: Path, label: str) -> Path:
    candidate = _absolute(path)
    root = _absolute(root)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RFDETRAssetAttestationError(
            f"{label} escapes its reviewed root: {candidate}"
        ) from exc
    current = root
    for part in (None, *relative.parts):
        if part is not None:
            current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            raise RFDETRAssetAttestationError(
                f"{label} is missing: {current}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise RFDETRAssetAttestationError(
                f"{label} contains a symlink: {current}"
            )
    return candidate


def _read_stable_regular(
    path: Path,
    *,
    root: Path,
    label: str,
    capture: bool = False,
    maximum_bytes: int | None = None,
) -> tuple[str, int, bytes | None]:
    candidate = _require_no_symlink_chain(path, root=root, label=label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise RFDETRAssetAttestationError(
            f"{label} cannot be opened safely: {candidate}"
        ) from exc
    chunks: list[bytes] | None = [] if capture else None
    digest = hashlib.sha256()
    total = 0
    before_identity: tuple[int, ...] | None = None
    try:
        try:
            before = os.fstat(descriptor)
            before_identity = _identity(before)
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise RFDETRAssetAttestationError(
                    f"{label} must be a nonempty regular file: {candidate}"
                )
            if maximum_bytes is not None and before.st_size > maximum_bytes:
                raise RFDETRAssetAttestationError(
                    f"{label} exceeds its startup metadata bound: {candidate}"
                )
            while block := os.read(descriptor, 4 * 1024 * 1024):
                total += len(block)
                digest.update(block)
                if chunks is not None:
                    chunks.append(block)
            if (
                total != before.st_size
                or _identity(os.fstat(descriptor)) != before_identity
            ):
                raise RFDETRAssetAttestationError(
                    f"{label} changed while it was read: {candidate}"
                )
        except OSError as exc:
            raise RFDETRAssetAttestationError(
                f"{label} could not be read safely: {candidate}"
            ) from exc
    finally:
        os.close(descriptor)
    assert before_identity is not None
    try:
        path_identity = _identity(candidate.lstat())
    except OSError as exc:
        raise RFDETRAssetAttestationError(
            f"{label} disappeared during attestation: {candidate}"
        ) from exc
    if path_identity != before_identity:
        raise RFDETRAssetAttestationError(
            f"{label} path identity changed during attestation: {candidate}"
        )
    return digest.hexdigest(), total, (b"".join(chunks) if chunks is not None else None)


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise RFDETRAssetAttestationError(
                f"RF-DETR receipt contains duplicate key {key!r}"
            )
        payload[key] = value
    return payload


def _load_strict_json(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_pairs,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RFDETRAssetAttestationError(
            f"{label} is not strict JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise RFDETRAssetAttestationError(f"{label} root must be an object")
    return value


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    explicit: set[Any] = set()
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=False)
        try:
            duplicate = key in explicit
        except TypeError as exc:
            raise RFDETRAssetAttestationError(
                "RF-DETR asset manifest contains a non-scalar key"
            ) from exc
        if duplicate:
            raise RFDETRAssetAttestationError(
                f"RF-DETR asset manifest contains duplicate key {key!r}"
            )
        explicit.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_strict_manifest(payload: bytes) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")
        loader = _UniqueKeySafeLoader(text)
        try:
            value = loader.get_single_data()
        finally:
            loader.dispose()
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RFDETRAssetAttestationError(
            "RF-DETR asset manifest is not strict YAML"
        ) from exc
    if not isinstance(value, Mapping):
        raise RFDETRAssetAttestationError(
            "RF-DETR asset manifest root must be a mapping"
        )
    return value


def _relative_to_root(path: Path, root: Path, label: str) -> str:
    try:
        return _absolute(path).relative_to(_absolute(root)).as_posix()
    except ValueError as exc:
        raise RFDETRAssetAttestationError(
            f"{label} escapes the RF-DETR artifact root: {path}"
        ) from exc


def _json_digest(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RFDETRAssetAttestationError(
            "RF-DETR engine receipt cannot be canonically encoded"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha(value: object, label: str) -> str:
    digest = str(value or "")
    if _SHA256.fullmatch(digest) is None:
        raise RFDETRAssetAttestationError(
            f"{label} is not a lowercase SHA-256 digest"
        )
    return digest


def _attest_parser(
    *, family: str, parser_path: Path
) -> dict[str, str | int]:
    contract = _PARSER_CONTRACTS[family]
    repo_root = _absolute(DS9_ROOT.parent)
    manifest_path = _absolute(DS9_ROOT / "asset_manifest.yaml")
    _manifest_sha, _manifest_size, raw_manifest = _read_stable_regular(
        manifest_path,
        root=repo_root,
        label="RF-DETR DS9 asset manifest",
        capture=True,
        maximum_bytes=_MAX_METADATA_BYTES,
    )
    assert raw_manifest is not None
    manifest = _load_strict_manifest(raw_manifest)
    target = manifest.get("target")
    deepstream = target.get("deepstream") if isinstance(target, Mapping) else None
    if (
        manifest.get("schema_version") != 3
        or manifest.get("manifest_id") != "noesis-ds9-artifacts"
        or manifest.get("schema") != "DS9/docs/asset_manifest.schema.json"
        or not isinstance(target, Mapping)
        or target.get("cuda") != "13.2"
        or target.get("tensorrt") != "10.16.0.72"
        or not isinstance(deepstream, Mapping)
        or deepstream.get("major") != 9
        or deepstream.get("version") != "9.1"
    ):
        raise RFDETRAssetAttestationError(
            "RF-DETR parser manifest authority drifted"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RFDETRAssetAttestationError(
            "RF-DETR parser manifest artifacts must be a list"
        )
    records: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise RFDETRAssetAttestationError(
                "RF-DETR parser manifest contains a malformed artifact"
            )
        artifact_id = str(item.get("id") or "")
        if not artifact_id or artifact_id in records:
            raise RFDETRAssetAttestationError(
                "RF-DETR parser manifest contains a missing or duplicate artifact ID"
            )
        records[artifact_id] = item
    artifact = records.get(str(contract["id"]))
    compatibility = artifact.get("compatibility") if artifact else None
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("kind") != "nvinfer_parser"
        or artifact.get("role") != contract["role"]
        or artifact.get("output") != contract["output"]
        or artifact.get("sources") != [contract["source"]]
        or artifact.get("builder") != contract["builder"]
        or tuple(artifact.get("required_profiles") or ())
        != contract["profiles"]
        or artifact.get("state") not in {"staged_unverified", "validated"}
        or compatibility
        != {
            "deepstream_major": 9,
            "cuda": "13.2",
            "tensorrt": "10.16.0.72",
        }
    ):
        raise RFDETRAssetAttestationError(
            f"RF-DETR parser manifest contract drifted for {contract['id']}"
        )
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RFDETRAssetAttestationError(
            f"RF-DETR parser provenance is missing for {contract['id']}"
        )
    source_expected = _require_sha(
        provenance.get("source_sha256"),
        f"{contract['id']} source provenance",
    )
    output_expected = _require_sha(
        provenance.get("output_sha256"),
        f"{contract['id']} binary provenance",
    )
    if (
        _UTC_TIMESTAMP.fullmatch(str(provenance.get("built_at_utc") or ""))
        is None
        or not str(provenance.get("build_host") or "").strip()
        or not str(provenance.get("command") or "").strip()
    ):
        raise RFDETRAssetAttestationError(
            f"RF-DETR parser provenance metadata is malformed for {contract['id']}"
        )
    expected_parser = _absolute(repo_root / str(contract["output"]))
    if _absolute(parser_path) != expected_parser:
        raise RFDETRAssetAttestationError(
            f"RF-DETR selected parser path drifted for {contract['id']}"
        )
    source_path = _absolute(repo_root / str(contract["source"]))
    source_raw_sha, source_size, source_payload = _read_stable_regular(
        source_path,
        root=repo_root,
        label=f"{contract['id']} source",
        capture=True,
        maximum_bytes=_MAX_PARSER_SOURCE_BYTES,
    )
    output_sha, output_size, _ = _read_stable_regular(
        expected_parser,
        root=repo_root,
        label=f"{contract['id']} binary",
    )
    assert source_payload is not None
    source_digest = hashlib.sha256()
    source_digest.update(str(contract["source"]).encode("utf-8"))
    source_digest.update(b"\0")
    source_digest.update(source_payload)
    source_digest.update(b"\0")
    source_sha = source_digest.hexdigest()
    if source_sha != source_expected or output_sha != output_expected:
        raise RFDETRAssetAttestationError(
            f"RF-DETR parser bytes do not match manifest provenance for {contract['id']}"
        )
    return {
        "parser_source_sha256": source_sha,
        "parser_source_raw_sha256": source_raw_sha,
        "parser_source_size_bytes": source_size,
        "parser_sha256": output_sha,
        "parser_size_bytes": output_size,
    }


def attest_rfdetr_1_8_3_assets(
    *,
    family: str,
    size: str,
    resolved: Mapping[str, Path | str | int | bool] | None = None,
) -> dict[str, str | int]:
    """Attest the exact selected RF-DETR runtime chain before config writes."""

    row = _model_row(family, size)
    runtime = _runtime_contract(row)
    matrix = _load_matrix()
    release = matrix["release"]
    assets = (
        resolve_rfdetr_1_8_3_assets(family, size)
        if resolved is None
        else resolved
    )
    if (
        assets.get("id") != row["id"]
        or assets.get("family") != row["family"]
        or assets.get("variant") != row["variant"]
        or assets.get("engine_profile") != runtime["engine_profile"]
        or assets.get("adapter_revision") != runtime["adapter_revision"]
    ):
        raise RFDETRAssetAttestationError(
            f"RF-DETR resolved asset selection drifted for {row['id']}"
        )

    model_root = _model_dir()
    artifact_root = model_root.parent
    provenance_root = (
        model_root / "provenance" / "rfdetr" / RELEASE_VERSION
    )
    onnx_path = _absolute(
        _onnx_dir()
        / "rfdetr"
        / RELEASE_VERSION
        / "runtime"
        / runtime["onnx_filename"]
    )
    engine_path = _absolute(
        _engine_dir()
        / "rfdetr"
        / RELEASE_VERSION
        / "runtime"
        / runtime["engine_profile"]
        / runtime["engine_filename"]
    )
    onnx_receipt_path = _absolute(
        provenance_root
        / (
            f"{row['id']}.runtime_onnx."
            f"{runtime['adapter_revision']}.json"
        )
    )
    engine_receipt_path = _absolute(
        provenance_root
        / "runtime"
        / runtime["engine_profile"]
        / f"{row['id']}.engine.json"
    )
    if (
        _absolute(Path(assets["onnx"])) != onnx_path
        or _absolute(Path(assets["engine"])) != engine_path
    ):
        raise RFDETRAssetAttestationError(
            f"RF-DETR runtime artifact path drifted for {row['id']}"
        )

    onnx_sha, onnx_size, _ = _read_stable_regular(
        onnx_path,
        root=artifact_root,
        label=f"{row['id']} runtime ONNX",
    )
    if onnx_sha != runtime["onnx_sha256"]:
        raise RFDETRAssetAttestationError(
            f"{row['id']} runtime ONNX differs from the model matrix"
        )
    onnx_receipt_sha, _onnx_receipt_size, raw_onnx_receipt = (
        _read_stable_regular(
            onnx_receipt_path,
            root=artifact_root,
            label=f"{row['id']} runtime ONNX receipt",
            capture=True,
            maximum_bytes=_MAX_METADATA_BYTES,
        )
    )
    if onnx_receipt_sha != runtime["onnx_receipt_sha256"]:
        raise RFDETRAssetAttestationError(
            f"{row['id']} runtime ONNX receipt differs from the model matrix"
        )
    assert raw_onnx_receipt is not None
    onnx_receipt = _load_strict_json(
        raw_onnx_receipt, f"{row['id']} runtime ONNX receipt"
    )
    tensor_contract = onnx_receipt.get("tensor_contract")
    source = onnx_receipt.get("source")
    if (
        onnx_receipt.get("schema") != SOURCE_PROVENANCE_SCHEMA
        or onnx_receipt.get("artifact_kind") != "runtime_onnx"
        or onnx_receipt.get("model_id") != row["id"]
        or onnx_receipt.get("family") != row["family"]
        or onnx_receipt.get("variant") != row["variant"]
        or onnx_receipt.get("license") != row["license"]
        or onnx_receipt.get("release") != release
        or onnx_receipt.get("filename") != onnx_path.name
        or onnx_receipt.get("size_bytes") != onnx_size
        or onnx_receipt.get("sha256") != onnx_sha
        or onnx_receipt.get("adapter") != RUNTIME_ADAPTER_SPEC
        or not isinstance(tensor_contract, Mapping)
        or tensor_contract.get("input")
        != {
            "input": [
                int(release["batch_size"]),
                3,
                int(row["resolution"]),
                int(row["resolution"]),
            ]
        }
        or tensor_contract.get("outputs") != row["outputs"]
        or tensor_contract.get("runtime_adapter") != RUNTIME_ADAPTER_SPEC
        or tensor_contract.get("opset_imports", {}).get("ai.onnx")
        != release["onnx_opset"]
        or not isinstance(
            tensor_contract.get("normalized_input_consumer_count"), int
        )
        or isinstance(
            tensor_contract.get("normalized_input_consumer_count"), bool
        )
        or tensor_contract.get("normalized_input_consumer_count", 0) <= 0
        or not isinstance(source, Mapping)
        or source.get("filename") != row["onnx_filename"]
        or source.get("receipt") != f"{row['id']}.onnx.json"
        or not isinstance(source.get("size_bytes"), int)
        or source.get("size_bytes", 0) <= 0
        or _SHA256.fullmatch(str(source.get("sha256") or "")) is None
        or _SHA256.fullmatch(str(source.get("receipt_sha256") or "")) is None
    ):
        raise RFDETRAssetAttestationError(
            f"{row['id']} runtime ONNX receipt contract drifted"
        )

    engine_sha, engine_size, _ = _read_stable_regular(
        engine_path,
        root=artifact_root,
        label=f"{row['id']} runtime engine",
    )
    if engine_sha != runtime["engine_sha256"]:
        raise RFDETRAssetAttestationError(
            f"{row['id']} runtime engine differs from the model matrix"
        )
    engine_receipt_sha, _engine_receipt_size, raw_engine_receipt = (
        _read_stable_regular(
            engine_receipt_path,
            root=artifact_root,
            label=f"{row['id']} runtime engine receipt",
            capture=True,
            maximum_bytes=_MAX_METADATA_BYTES,
        )
    )
    if engine_receipt_sha != runtime["engine_receipt_sha256"]:
        raise RFDETRAssetAttestationError(
            f"{row['id']} runtime engine receipt differs from the model matrix"
        )
    assert raw_engine_receipt is not None
    engine_receipt = _load_strict_json(
        raw_engine_receipt, f"{row['id']} runtime engine receipt"
    )
    receipt_digest = engine_receipt.get("receipt_sha256")
    receipt_without_digest = dict(engine_receipt)
    receipt_without_digest.pop("receipt_sha256", None)
    build = engine_receipt.get("build_contract")
    profile_contract = _RUNTIME_ENGINE_PROFILE_CONTRACTS[
        runtime["engine_profile"]
    ]
    engine_record = engine_receipt.get("engine")
    engine_source = engine_receipt.get("source")
    input_contract = engine_receipt.get("runtime_input_contract")
    expected_onnx_relative = _relative_to_root(
        onnx_path, artifact_root, f"{row['id']} runtime ONNX"
    )
    expected_onnx_receipt_relative = _relative_to_root(
        onnx_receipt_path,
        artifact_root,
        f"{row['id']} runtime ONNX receipt",
    )
    expected_engine_relative = _relative_to_root(
        engine_path, artifact_root, f"{row['id']} runtime engine"
    )
    if (
        engine_receipt.get("schema") != RUNTIME_ENGINE_RECEIPT_SCHEMA
        or engine_receipt.get("promotion_status") != "unpromoted"
        or engine_receipt.get("runtime_selected") is not False
        or engine_receipt.get("artifact_role") != "runtime_input_engine"
        or engine_receipt.get("model_id") != row["id"]
        or engine_receipt.get("family") != row["family"]
        or engine_receipt.get("variant") != row["variant"]
        or engine_receipt.get("release") != release
        or _require_sha(receipt_digest, f"{row['id']} engine receipt digest")
        != _json_digest(receipt_without_digest)
        or not isinstance(build, Mapping)
        or build.get("profile") != runtime["engine_profile"]
        or build.get("precision") != profile_contract["precision"]
        or build.get("fp16_enabled") is not profile_contract["fp16_enabled"]
        or build.get("tf32_enabled") is not profile_contract["tf32_enabled"]
        or build.get("batch") != {"mode": "static", "size": 3}
        or not isinstance(build.get("trtexec_precision_args"), list)
        or tuple(build["trtexec_precision_args"])
        != profile_contract["trtexec_precision_args"]
        or not isinstance(engine_record, Mapping)
        or engine_record.get("path") != expected_engine_relative
        or engine_record.get("size_bytes") != engine_size
        or engine_record.get("sha256") != engine_sha
        or not isinstance(engine_source, Mapping)
        or engine_source.get("runtime_onnx") != expected_onnx_relative
        or engine_source.get("runtime_onnx_size_bytes") != onnx_size
        or engine_source.get("runtime_onnx_sha256") != onnx_sha
        or engine_source.get("runtime_onnx_receipt")
        != expected_onnx_receipt_relative
        or engine_source.get("runtime_onnx_receipt_sha256")
        != onnx_receipt_sha
        or engine_source.get("tensor_contract") != tensor_contract
        or not isinstance(input_contract, Mapping)
        or input_contract.get("input_contract") != RUNTIME_INPUT_CONTRACT
        or input_contract.get("adapter_revision")
        != RUNTIME_ADAPTER_REVISION
        or input_contract.get("adapter") != RUNTIME_ADAPTER_SPEC
    ):
        raise RFDETRAssetAttestationError(
            f"{row['id']} runtime engine receipt contract drifted"
        )
    return {
        "onnx_sha256": onnx_sha,
        "onnx_size_bytes": onnx_size,
        "onnx_receipt_sha256": onnx_receipt_sha,
        "engine_sha256": engine_sha,
        "engine_size_bytes": engine_size,
        "engine_receipt_sha256": engine_receipt_sha,
        **_attest_parser(
            family=str(row["family"]),
            parser_path=Path(assets["parser"]),
        ),
    }


def resolve_rfdetr_1_8_3_assets(
    family: str, size: str
) -> dict[str, Path | str | int | bool]:
    family_norm = str(family or "").strip().lower()
    size_norm = str(size or "").strip().lower()
    row = _model_row(family_norm, size_norm)
    runtime = _runtime_contract(row)
    outputs = row.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"RF-DETR model {row.get('id')} has no output contract")
    det_shape = outputs.get("dets")
    if (
        not isinstance(det_shape, list)
        or len(det_shape) != 3
        or int(det_shape[0]) != 3
        or int(det_shape[2]) != 4
    ):
        raise ValueError(f"RF-DETR model {row.get('id')} has invalid dets shape")
    query_count = int(det_shape[1])
    labels_shape = outputs.get("labels")
    if (
        not isinstance(labels_shape, list)
        or len(labels_shape) != 3
        or int(labels_shape[0]) != 3
        or int(labels_shape[1]) != query_count
    ):
        raise ValueError(f"RF-DETR model {row.get('id')} has invalid labels shape")
    resolution = int(row.get("resolution", 0))
    if resolution <= 0 or query_count <= 0:
        raise ValueError(f"RF-DETR model {row.get('id')} has invalid dimensions")

    versioned_onnx = _onnx_dir() / "rfdetr" / RELEASE_VERSION
    versioned_engine = _engine_dir() / "rfdetr" / RELEASE_VERSION
    pipeline_dir = _pipeline_dir()
    is_segmentation = family_norm == "segmentation"
    is_keypoint = family_norm == "keypoint"
    if is_keypoint:
        keypoint_shape = outputs.get("keypoints")
        if keypoint_shape != [3, query_count, 34, 8]:
            raise ValueError(
                f"RF-DETR model {row.get('id')} has invalid keypoints shape"
            )
        if labels_shape != [3, query_count, 2]:
            raise ValueError(
                f"RF-DETR model {row.get('id')} keypoint labels must be [3,Q,2]"
            )
        parser_dir = "nvdsinfer_rfdetr_keypoint"
        parser_name = "libnvdsinfer_rfdetr_keypoint.so"
        template_name = "config_infer_primary_rfdetr_keypoint.template.ini"
    elif is_segmentation:
        parser_dir = "nvdsinfer_rfdetr_seg"
        parser_name = "libnvdsinfer_rfdetr_seg.so"
        template_name = "config_infer_primary_rfdetr_seg.template.ini"
    else:
        parser_dir = "nvdsinfer_rfdetr"
        parser_name = "libnvdsinfer_rfdetr.so"
        template_name = "config_infer_primary_rfdetr.template.ini"
    return {
        "id": str(row["id"]),
        "family": family_norm,
        "variant": str(row["variant"]),
        "size": size_norm,
        "batch_size": int(det_shape[0]),
        "resolution": resolution,
        "query_count": query_count,
        "has_instance_masks": is_segmentation,
        "has_keypoints": is_keypoint,
        "uses_external_preprocess": not is_keypoint,
        "input_contract": runtime["input_contract"],
        "adapter_revision": runtime["adapter_revision"],
        "template": _absolute(pipeline_dir / template_name),
        "preproc_template": (
            _absolute(
                pipeline_dir
                / "config_preproc_rfdetr_1_8_3.template.ini"
            )
        ),
        "parent_onnx": _absolute(
            versioned_onnx
            / _safe_token(row.get("onnx_filename"), "ONNX filename")
        ),
        "onnx": _absolute(
            versioned_onnx / "runtime" / runtime["onnx_filename"]
        ),
        "engine_profile": runtime["engine_profile"],
        "engine": _absolute(
            versioned_engine
            / "runtime"
            / runtime["engine_profile"]
            / runtime["engine_filename"]
        ),
        "labels": _absolute(_model_dir() / "coco_labels.txt"),
        "parser": _absolute(pipeline_dir / parser_dir / parser_name),
        "default_output": _absolute(
            _build_dir()
            / f"config_infer_primary_rfdetr_1_8_3_{family_norm}_{size_norm}.ini"
        ),
        "default_preprocess_output": _absolute(
            _build_dir()
            / f"config_preproc_rfdetr_1_8_3_{family_norm}_{size_norm}_b3.ini"
        ),
    }


def _replace_required(
    pattern: str, replacement: str, text: str, label: str
) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(f"RF-DETR template missing required field: {label}")
    return updated


def _properties(path: Path) -> Mapping[str, str]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    if not parser.has_section("property"):
        raise ValueError(f"RF-DETR config lacks [property]: {path}")
    return parser["property"]


def _class_attrs_all(path: Path) -> Mapping[str, str]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    if not parser.has_section("class-attrs-all"):
        raise ValueError(f"RF-DETR config lacks [class-attrs-all]: {path}")
    return parser["class-attrs-all"]


def _require_property(
    props: Mapping[str, str], key: str, expected: str, *, label: str
) -> None:
    actual = str(props.get(key, "") or "").strip()
    if actual != expected:
        raise ValueError(
            f"{label} requires {key}={expected} (got {actual or '<unset>'})"
        )


def validate_rfdetr_preprocess_properties(
    props: Mapping[str, str], *, resolution: int, batch_size: int = 3
) -> None:
    label = "RF-DETR 1.8.3 preprocess"
    for key, expected in (
        ("network-input-shape", f"{int(batch_size)};3;{int(resolution)};{int(resolution)}"),
        ("processing-width", str(int(resolution))),
        ("processing-height", str(int(resolution))),
        ("tensor-name", "input"),
        ("network-color-format", "0"),
        ("tensor-data-type", "0"),
        ("maintain-aspect-ratio", "0"),
        ("symmetric-padding", "0"),
    ):
        _require_property(props, key, expected, label=label)


def validate_rfdetr_pgie_properties(
    props: Mapping[str, str],
    *,
    family: str,
    engine_profile: str,
    query_count: int,
    batch_size: int = 3,
) -> None:
    family_norm = str(family or "").strip().lower()
    if family_norm not in _SIZE_TO_VARIANT:
        raise ValueError(f"invalid RF-DETR PGIE family: {family}")
    profile_norm = str(engine_profile or "").strip()
    try:
        profile_contract = _RUNTIME_ENGINE_PROFILE_CONTRACTS[profile_norm]
    except KeyError as exc:
        choices = "/".join(_RUNTIME_ENGINE_PROFILE_CONTRACTS)
        raise ValueError(
            "RF-DETR PGIE engine profile must be one of "
            f"{choices} (got: {engine_profile or '<unset>'})"
        ) from exc
    label = f"RF-DETR 1.8.3 {family_norm} PGIE"
    input_tensor_from_meta = "0" if family_norm == "keypoint" else "1"
    output_tensor_meta = "1" if family_norm == "keypoint" else "0"
    for key, expected in (
        ("batch-size", str(int(batch_size))),
        ("gie-unique-id", "1"),
        ("network-mode", str(profile_contract["network_mode"])),
        ("input-tensor-from-meta", input_tensor_from_meta),
        ("maintain-aspect-ratio", "0"),
        ("symmetric-padding", "0"),
        ("cluster-mode", "4"),
        ("num-detected-classes", "1"),
        ("operate-on-class-ids", "0"),
        ("output-tensor-meta", output_tensor_meta),
    ):
        _require_property(props, key, expected, label=label)
    if str(props.get("onnx-file", "") or "").strip():
        raise ValueError(f"{label} runtime config must be engine-only")
    if family_norm == "segmentation":
        for key, expected in (
            ("network-type", "3"),
            ("parse-bbox-instance-mask-func-name", "NvDsInferParseRFDETRSeg"),
            ("output-instance-mask", "1"),
            ("output-blob-names", "dets;labels;masks"),
        ):
            _require_property(props, key, expected, label=label)
    elif family_norm == "keypoint":
        for key, expected in (
            ("network-type", "0"),
            ("parse-bbox-func-name", "NvDsInferParseRFDETRKeypoint"),
            ("output-blob-names", "dets;labels;keypoints"),
            ("disable-output-host-copy", "0"),
            ("net-scale-factor", "0.00392156862745098"),
            ("infer-dims", "3;576;576"),
            ("model-color-format", "0"),
            ("scaling-filter", "1"),
        ):
            _require_property(props, key, expected, label=label)
    else:
        for key, expected in (
            ("interval", "1"),
            ("network-type", "0"),
            ("parse-bbox-func-name", "NvDsInferParseRFDETR"),
            ("output-blob-names", "dets;labels"),
        ):
            _require_property(props, key, expected, label=label)
    attrs = str(props.get("model-engine-file", "") or "").strip()
    if not attrs:
        raise ValueError(f"{label} requires model-engine-file")


def validate_rfdetr_class_attrs(
    attrs: Mapping[str, str],
    *,
    family: str,
    query_count: int,
    expected_threshold: float | None = None,
) -> float:
    family_norm = str(family or "").strip().lower()
    if family_norm not in _SIZE_TO_VARIANT:
        raise ValueError(f"invalid RF-DETR class-attrs family: {family}")
    label = f"RF-DETR 1.8.3 {family_norm} [class-attrs-all]"
    topk = str(attrs.get("topk", "") or "").strip()
    if topk != str(int(query_count)):
        raise ValueError(
            f"{label} requires topk={int(query_count)} "
            f"(got {topk or '<unset>'})"
        )
    threshold_raw = str(
        attrs.get("pre-cluster-threshold", "") or ""
    ).strip()
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} requires a numeric pre-cluster-threshold "
            f"(got {threshold_raw or '<unset>'})"
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"{label} pre-cluster-threshold must be in [0,1] "
            f"(got {threshold_raw})"
        )
    if (
        expected_threshold is not None
        and abs(threshold - float(expected_threshold)) > 1e-9
    ):
        raise ValueError(
            f"{label} requires pre-cluster-threshold="
            f"{float(expected_threshold):g} (got {threshold_raw})"
        )
    return threshold


def materialize_rfdetr_1_8_3_configs(
    *,
    family: str,
    size: str,
    batch_size: int,
    src_ids: Iterable[int | str],
    include_model_source: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Path | str | int | bool]:
    requested_batch_size = int(batch_size)
    src_ids_tuple = tuple(src_ids)
    if not src_ids_tuple:
        raise ValueError("RF-DETR src_ids must not be empty")
    assets = resolve_rfdetr_1_8_3_assets(family, size)
    release_batch_size = int(assets["batch_size"])
    if requested_batch_size != release_batch_size:
        raise ValueError(
            "RF-DETR 1.8.3 requires static "
            f"batch_size={release_batch_size} (got: {batch_size})"
        )
    attest_rfdetr_1_8_3_assets(
        family=family,
        size=size,
        resolved=assets,
    )

    template = Path(assets["template"])
    if not template.is_file():
        raise FileNotFoundError(f"RF-DETR PGIE template missing: {template}")
    pgie_text = template.read_text(encoding="utf-8")
    if include_model_source:
        pgie_text = pgie_text.replace("@ONNX_PATH@", str(assets["onnx"]))
    else:
        pgie_text = re.sub(
            r"(?m)^\s*onnx-file\s*=.*(?:\n|$)", "", pgie_text, count=1
        )
    for token, value in {
        "@ENGINE_PATH@": str(assets["engine"]),
        "@LABELS_PATH@": str(assets["labels"]),
        "@CUSTOM_LIB@": str(assets["parser"]),
        "@BATCH_SIZE@": str(requested_batch_size),
        "@NETWORK_MODE@": str(
            _RUNTIME_ENGINE_PROFILE_CONTRACTS[
                str(assets["engine_profile"])
            ]["network_mode"]
        ),
        "@TOPK@": str(int(assets["query_count"])),
    }.items():
        if token not in pgie_text:
            raise ValueError(f"RF-DETR PGIE template missing token: {token}")
        pgie_text = pgie_text.replace(token, value)
    pgie_output = Path(assets["default_output"])
    pgie_output.parent.mkdir(parents=True, exist_ok=True)
    pgie_output.write_text(pgie_text, encoding="utf-8")

    preproc_template = Path(assets["preproc_template"])
    if not preproc_template.is_file():
        raise FileNotFoundError(
            f"RF-DETR preprocess template missing: {preproc_template}"
        )
    preprocess_lib = (
        Path(
            os.environ.get(
                "NOESIS_DEEPSTREAM_HOME",
                "/opt/nvidia/deepstream/deepstream-9.1",
            )
        ).expanduser()
        / "lib"
        / "gst-plugins"
        / "libcustom2d_preprocess.so"
    ).resolve()
    preproc_text = preproc_template.read_text(encoding="utf-8")
    for token, value in {
        "@BATCH_SIZE@": str(requested_batch_size),
        "@RESOLUTION@": str(int(assets["resolution"])),
        "@PREPROCESS_LIB@": str(preprocess_lib),
        "@SRC_IDS@": ";".join(str(value) for value in src_ids_tuple),
    }.items():
        if token not in preproc_text:
            raise ValueError(f"RF-DETR preprocess template missing token: {token}")
        preproc_text = preproc_text.replace(token, value)
    preprocess_output = Path(assets["default_preprocess_output"])
    preprocess_output.parent.mkdir(parents=True, exist_ok=True)
    preprocess_output.write_text(preproc_text, encoding="utf-8")

    validate_rfdetr_pgie_properties(
        _properties(pgie_output),
        family=str(assets["family"]),
        engine_profile=str(assets["engine_profile"]),
        query_count=int(assets["query_count"]),
        batch_size=requested_batch_size,
    )
    validate_rfdetr_class_attrs(
        _class_attrs_all(pgie_output),
        family=str(assets["family"]),
        query_count=int(assets["query_count"]),
        expected_threshold=(
            0.4 if str(assets["family"]) == "keypoint" else None
        ),
    )
    validate_rfdetr_preprocess_properties(
        _properties(preprocess_output),
        resolution=int(assets["resolution"]),
        batch_size=requested_batch_size,
    )
    if logger is not None:
        logger.info(
            "RF-DETR 1.8.3 configs materialized: family=%s size=%s "
            "engine_profile=%s pgie=%s preprocess=%s",
            assets["family"],
            assets["size"],
            assets["engine_profile"],
            pgie_output,
            preprocess_output,
        )
    return {
        **assets,
        "pgie_config": pgie_output,
        "preprocess_config": preprocess_output,
    }


__all__ = [
    "RELEASE_VERSION",
    "RFDETR_DETECTION_SIZES",
    "RFDETR_KEYPOINT_VARIANTS",
    "RFDETR_SEGMENTATION_SIZES",
    "RFDETRAssetAttestationError",
    "attest_rfdetr_1_8_3_assets",
    "materialize_rfdetr_1_8_3_configs",
    "resolve_rfdetr_1_8_3_assets",
    "validate_rfdetr_class_attrs",
    "validate_rfdetr_pgie_properties",
    "validate_rfdetr_preprocess_properties",
]
