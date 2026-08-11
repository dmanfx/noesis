#!/usr/bin/env python3
"""Compare official RF-DETR PyTorch export outputs with immutable ONNX evidence.

This is a diagnostic, owner-private validation lane.  It consumes one already
prepared static-B3 media case, runs the exact official module export path in
CUDA FP32, and publishes only numerical comparison metadata.  It does not
train, fine-tune, export, rebuild, promote, or select a runtime artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
MATRIX_PATH = DS9_ROOT / "config" / "rfdetr_1_8_3_models.json"
MEDIA_VALIDATOR_PATH = (
    DS9_ROOT / "scripts" / "validate_rfdetr_1_8_3_media.py"
)

MATRIX_SCHEMA = "noesis.ds9.rfdetr-model-matrix.v1"
RUN_SCHEMA = "noesis.ds9.rfdetr-media-validation-run.v1"
PROVENANCE_SCHEMA = "noesis.ds9.rfdetr-artifact-provenance.v1"
REPORT_SCHEMA = "noesis.ds9.rfdetr-pytorch-onnx-parity.v1"
RELEASE_VERSION = "1.8.3"
RELEASE_SOURCE = "https://github.com/roboflow/rf-detr"
DEFAULT_MODELS = (
    "detect_medium",
    "seg_small",
    "seg_medium",
    "keypoint_preview",
)
DEFAULT_CASE = "f0180"
DEFAULT_ATOL = 1.0e-4
DEFAULT_RTOL = 1.0e-3
MAX_JSON_BYTES = 8 * 1024 * 1024
HASH_CHUNK_BYTES = 4 * 1024 * 1024

SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
CUDA_DEVICE_RE = re.compile(r"^cuda:(0|[1-9][0-9]*)$")

OUTPUT_SOURCE_KEYS = {
    "dets": "pred_boxes",
    "labels": "pred_logits",
    "masks": "pred_masks",
    "keypoints": "pred_keypoints",
}
FAMILY_OUTPUTS = {
    "detection": ("dets", "labels"),
    "segmentation": ("dets", "labels", "masks"),
    "keypoint": ("dets", "labels", "keypoints"),
}


class ParityError(RuntimeError):
    """Raised when a parity authority or tensor contract fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_report_id() -> str:
    return "pytorch-parity-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    ).lower()


def _safe_token(raw: object, label: str) -> str:
    value = str(raw or "")
    if (
        SAFE_TOKEN.fullmatch(value) is None
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ParityError(f"unsafe {label}: {value!r}")
    return value


def _safe_filename(raw: object, label: str) -> str:
    value = str(raw or "")
    if (
        SAFE_FILENAME.fullmatch(value) is None
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ParityError(f"unsafe {label}: {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_CHUNK_BYTES):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ParityError(f"{label} must be a nonempty regular file: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ParityError(f"{label} exceeds {MAX_JSON_BYTES} bytes: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ParityError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ParityError(f"{label} root must be an object: {path}")
    return payload


def _explicit_artifact_root(raw: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ParityError(
            "--artifact-root or NOESIS_DS9_ARTIFACT_ROOT is required"
        )
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ParityError("artifact root must be absolute")
    root = candidate.resolve()
    if (
        root == Path("/")
        or root == REPO_ROOT.resolve()
        or root == DS9_ROOT.resolve()
        or not root.is_dir()
    ):
        raise ParityError(f"unsafe or unavailable artifact root: {root}")
    try:
        root.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return root
    raise ParityError("artifact root must remain outside the checkout")


def _relative_parts(raw: object, label: str) -> tuple[str, ...]:
    value = str(raw or "")
    candidate = Path(value)
    if (
        not value
        or candidate.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ParityError(f"unsafe relative {label}: {value!r}")
    return tuple(candidate.parts)


def _member(base: Path, raw: object, label: str) -> Path:
    parts = _relative_parts(raw, label)
    current = base
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ParityError(f"{label} traverses a symbolic link: {current}")
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ParityError(f"{label} escaped its authority root: {raw!r}") from exc
    return current


def _relative(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ParityError(f"{label} is outside the artifact root: {path}") from exc


def _validate_shape(raw: object, label: str) -> list[int]:
    if (
        not isinstance(raw, list)
        or not raw
        or any(type(value) is not int or value <= 0 for value in raw)
    ):
        raise ParityError(f"{label} must be a positive static shape")
    return [int(value) for value in raw]


def _load_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    matrix = _read_json(path, "RF-DETR model matrix")
    release = matrix.get("release")
    models = matrix.get("models")
    if (
        matrix.get("schema") != MATRIX_SCHEMA
        or not isinstance(release, dict)
        or release.get("package") != "rfdetr"
        or release.get("version") != RELEASE_VERSION
        or release.get("git_tag") != RELEASE_VERSION
        or release.get("source") != RELEASE_SOURCE
        or not isinstance(release.get("git_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", release["git_commit"]) is None
        or release.get("onnx_opset") != 17
        or release.get("batch_size") != 3
        or release.get("dynamic_batch") is not False
        or not isinstance(models, list)
        or not models
    ):
        raise ParityError("matrix is not the reviewed RF-DETR 1.8.3 contract")

    seen: set[str] = set()
    for row in models:
        if not isinstance(row, dict):
            raise ParityError("matrix contains a non-object model row")
        model_id = _safe_token(row.get("id"), "model ID")
        if model_id in seen:
            raise ParityError(f"duplicate model ID: {model_id}")
        seen.add(model_id)
        family = str(row.get("family") or "")
        expected_names = FAMILY_OUTPUTS.get(family)
        outputs = row.get("outputs")
        if (
            expected_names is None
            or row.get("package") not in {"rfdetr", "rfdetr_plus"}
            or not isinstance(row.get("class_name"), str)
            or not isinstance(row.get("resolution"), int)
            or row["resolution"] <= 0
            or not isinstance(outputs, dict)
            or tuple(outputs) != expected_names
        ):
            raise ParityError(f"invalid model contract for {model_id}")
        _safe_filename(row.get("checkpoint_filename"), "checkpoint filename")
        _safe_filename(row.get("onnx_filename"), "ONNX filename")
        if (
            not isinstance(row.get("checkpoint_size_bytes"), int)
            or row["checkpoint_size_bytes"] <= 0
            or MD5_RE.fullmatch(str(row.get("checkpoint_md5") or "")) is None
        ):
            raise ParityError(f"invalid checkpoint contract for {model_id}")
        expected_input = [3, 3, row["resolution"], row["resolution"]]
        for name, shape in outputs.items():
            normalized = _validate_shape(shape, f"{model_id} {name} shape")
            if normalized[0] != expected_input[0]:
                raise ParityError(f"{model_id} batch contract drifted")
    return matrix


def _split_models(raw: Sequence[str] | None) -> list[str]:
    if not raw:
        return list(DEFAULT_MODELS)
    values = [
        part.strip()
        for item in raw
        for part in str(item).split(",")
        if part.strip()
    ]
    return values


def _select_models(
    matrix: Mapping[str, Any], requested: Sequence[str] | None
) -> list[dict[str, Any]]:
    ids = _split_models(requested)
    if not ids:
        raise ParityError("at least one model must be selected")
    if len(set(ids)) != len(ids):
        raise ParityError("duplicate model selection")
    by_id = {str(row["id"]): row for row in matrix["models"]}
    unknown = [model_id for model_id in ids if model_id not in by_id]
    if unknown:
        raise ParityError("unknown model selection: " + ", ".join(unknown))
    disabled = [model_id for model_id in ids if not by_id[model_id].get("enabled")]
    if disabled:
        raise ParityError(
            "disabled or licensed model selection is not authorized: "
            + ", ".join(disabled)
        )
    unsupported = [
        model_id for model_id in ids if by_id[model_id].get("package") != "rfdetr"
    ]
    if unsupported:
        raise ParityError(
            "this verifier only accepts the pinned open rfdetr package: "
            + ", ".join(unsupported)
        )
    return [dict(by_id[model_id]) for model_id in ids]


def _validate_manifest_digest(payload: Mapping[str, Any], label: str) -> str:
    recorded = payload.get("manifest_sha256")
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
        raise ParityError(f"{label} lacks a valid manifest digest")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if _json_digest(unsigned) != recorded:
        raise ParityError(f"{label} manifest digest mismatch")
    return recorded


def _validate_private_run(run_dir: Path, artifact_root: Path) -> dict[str, Any]:
    canonical_root = (
        artifact_root
        / "models"
        / "validation"
        / "rfdetr"
        / RELEASE_VERSION
    )
    resolved = run_dir.expanduser().resolve()
    try:
        resolved.relative_to(canonical_root.resolve())
    except ValueError as exc:
        raise ParityError(
            f"run directory is outside the RF-DETR validation root: {resolved}"
        ) from exc
    if run_dir.is_symlink() or not resolved.is_dir():
        raise ParityError(f"run directory is unavailable or unsafe: {run_dir}")
    info = resolved.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ParityError("source media run must be caller-owned and mode 0700")
    run_path = resolved / "run.json"
    _require_regular_file(run_path, "source run manifest")
    run_info = run_path.stat()
    if run_info.st_uid != os.getuid() or stat.S_IMODE(run_info.st_mode) != 0o600:
        raise ParityError("source run manifest must be caller-owned and mode 0600")
    run = _read_json(run_path, "source run manifest")
    if (
        run.get("schema") != RUN_SCHEMA
        or run.get("promotion_status") != "unpromoted"
        or run.get("runtime_selected") is not False
        or not isinstance(run.get("models"), list)
        or not isinstance(run.get("cases"), list)
    ):
        raise ParityError("source run is not an unpromoted RF-DETR media run")
    _validate_manifest_digest(run, "source run")
    return run


def _model_from_run(
    run: Mapping[str, Any],
    model: Mapping[str, Any],
    case_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_models = [
        row
        for row in run["models"]
        if isinstance(row, dict) and row.get("id") == model["id"]
    ]
    if len(run_models) != 1:
        raise ParityError(f"source run has no unique {model['id']} record")
    row = run_models[0]
    expected_input = [3, 3, int(model["resolution"]), int(model["resolution"])]
    expected_outputs = {
        name: [int(value) for value in shape]
        for name, shape in model["outputs"].items()
    }
    if (
        row.get("family") != model["family"]
        or row.get("variant") != model["variant"]
        or row.get("resolution") != model["resolution"]
        or row.get("expected_input") != expected_input
        or row.get("expected_outputs") != expected_outputs
        or not isinstance(row.get("cases"), list)
    ):
        raise ParityError(f"{model['id']} source-run tensor contract drifted")
    cases = [
        case
        for case in row["cases"]
        if isinstance(case, dict) and case.get("id") == case_id
    ]
    if len(cases) != 1:
        raise ParityError(
            f"{model['id']} source run has no unique case {case_id}"
        )
    return row, cases[0]


def _validate_tensor_record(
    run_dir: Path,
    record: object,
    expected_shape: Sequence[int],
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise ParityError(f"{label} record must be an object")
    shape = [int(value) for value in expected_shape]
    expected_bytes = math.prod(shape) * 4
    if (
        record.get("dtype") != "float32"
        or record.get("shape") != shape
        or record.get("finite") is not True
        or record.get("size_bytes") != expected_bytes
        or not isinstance(record.get("sha256"), str)
        or SHA256_RE.fullmatch(record["sha256"]) is None
    ):
        raise ParityError(f"{label} tensor record drifted")
    path = _member(run_dir, record.get("path"), f"{label} path")
    _require_regular_file(path, label)
    if path.stat().st_size != expected_bytes or _sha256(path) != record["sha256"]:
        raise ParityError(f"{label} tensor bytes changed")
    return path


def _validate_model_authority(
    artifact_root: Path,
    run_model: Mapping[str, Any],
    model: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    model_id = str(model["id"])
    onnx_row = run_model.get("onnx")
    if not isinstance(onnx_row, dict):
        raise ParityError(f"{model_id} source run lacks ONNX authority")
    expected_onnx_relative = (
        Path("models")
        / "onnx"
        / "rfdetr"
        / RELEASE_VERSION
        / _safe_filename(model["onnx_filename"], "ONNX filename")
    ).as_posix()
    expected_receipt_relative = (
        Path("models")
        / "provenance"
        / "rfdetr"
        / RELEASE_VERSION
        / f"{_safe_token(model_id, 'model ID')}.onnx.json"
    ).as_posix()
    if (
        onnx_row.get("path") != expected_onnx_relative
        or onnx_row.get("receipt") != expected_receipt_relative
        or not isinstance(onnx_row.get("size_bytes"), int)
        or not isinstance(onnx_row.get("sha256"), str)
        or SHA256_RE.fullmatch(onnx_row["sha256"]) is None
        or not isinstance(onnx_row.get("receipt_sha256"), str)
        or SHA256_RE.fullmatch(onnx_row["receipt_sha256"]) is None
    ):
        raise ParityError(f"{model_id} source-run ONNX authority drifted")

    onnx_path = _member(artifact_root, onnx_row["path"], f"{model_id} ONNX")
    receipt_path = _member(
        artifact_root, onnx_row["receipt"], f"{model_id} ONNX receipt"
    )
    _require_regular_file(onnx_path, f"{model_id} ONNX")
    _require_regular_file(receipt_path, f"{model_id} ONNX receipt")
    if (
        onnx_path.stat().st_size != onnx_row["size_bytes"]
        or _sha256(onnx_path) != onnx_row["sha256"]
        or _sha256(receipt_path) != onnx_row["receipt_sha256"]
    ):
        raise ParityError(f"{model_id} immutable ONNX evidence changed")
    receipt = _read_json(receipt_path, f"{model_id} ONNX receipt")

    expected_input = {"input": [3, 3, model["resolution"], model["resolution"]]}
    expected_outputs = {
        name: [int(value) for value in shape]
        for name, shape in model["outputs"].items()
    }
    checkpoint_row = receipt.get("checkpoint")
    if (
        receipt.get("schema") != PROVENANCE_SCHEMA
        or receipt.get("artifact_kind") != "onnx"
        or receipt.get("model_id") != model_id
        or receipt.get("family") != model["family"]
        or receipt.get("variant") != model["variant"]
        or receipt.get("release") != dict(release)
        or receipt.get("filename") != model["onnx_filename"]
        or receipt.get("size_bytes") != onnx_row["size_bytes"]
        or receipt.get("sha256") != onnx_row["sha256"]
        or not isinstance(receipt.get("tensor_contract"), dict)
        or receipt["tensor_contract"].get("input") != expected_input
        or receipt["tensor_contract"].get("outputs") != expected_outputs
        or not isinstance(checkpoint_row, dict)
        or checkpoint_row.get("filename") != model["checkpoint_filename"]
        or checkpoint_row.get("md5") != model["checkpoint_md5"]
        or not isinstance(checkpoint_row.get("sha256"), str)
        or SHA256_RE.fullmatch(checkpoint_row["sha256"]) is None
    ):
        raise ParityError(f"{model_id} ONNX receipt contract drifted")

    checkpoint_relative = (
        Path("models")
        / "checkpoints"
        / "rfdetr"
        / RELEASE_VERSION
        / _safe_filename(model["checkpoint_filename"], "checkpoint filename")
    ).as_posix()
    checkpoint_path = _member(
        artifact_root, checkpoint_relative, f"{model_id} checkpoint"
    )
    _require_regular_file(checkpoint_path, f"{model_id} checkpoint")
    if checkpoint_path.stat().st_size != model["checkpoint_size_bytes"]:
        raise ParityError(f"{model_id} checkpoint size changed")
    checkpoint_md5, checkpoint_sha256 = _hashes(checkpoint_path)
    if (
        checkpoint_md5 != model["checkpoint_md5"]
        or checkpoint_sha256 != checkpoint_row["sha256"]
    ):
        raise ParityError(f"{model_id} checkpoint hash changed")

    checkpoint_receipt_relative = (
        Path("models")
        / "provenance"
        / "rfdetr"
        / RELEASE_VERSION
        / f"{_safe_token(model_id, 'model ID')}.checkpoint.json"
    ).as_posix()
    checkpoint_receipt_path = _member(
        artifact_root,
        checkpoint_receipt_relative,
        f"{model_id} checkpoint receipt",
    )
    checkpoint_receipt = _read_json(
        checkpoint_receipt_path, f"{model_id} checkpoint receipt"
    )
    if (
        checkpoint_receipt.get("schema") != PROVENANCE_SCHEMA
        or checkpoint_receipt.get("artifact_kind") != "checkpoint"
        or checkpoint_receipt.get("model_id") != model_id
        or checkpoint_receipt.get("release") != dict(release)
        or checkpoint_receipt.get("filename") != model["checkpoint_filename"]
        or checkpoint_receipt.get("size_bytes")
        != model["checkpoint_size_bytes"]
        or checkpoint_receipt.get("md5") != checkpoint_md5
        or checkpoint_receipt.get("sha256") != checkpoint_sha256
        or checkpoint_receipt.get("source_url") != model["checkpoint_url"]
    ):
        raise ParityError(f"{model_id} checkpoint receipt contract drifted")

    return {
        "onnx_path": onnx_path,
        "onnx_sha256": onnx_row["sha256"],
        "onnx_size_bytes": onnx_row["size_bytes"],
        "onnx_receipt_path": receipt_path,
        "onnx_receipt_sha256": onnx_row["receipt_sha256"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_md5": checkpoint_md5,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_receipt_path": checkpoint_receipt_path,
        "checkpoint_receipt_sha256": _sha256(checkpoint_receipt_path),
    }


def _installed_source_environment(release: Mapping[str, Any]) -> dict[str, str]:
    try:
        version = importlib.metadata.version("rfdetr")
        module = importlib.import_module("rfdetr")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ParityError("the pinned RF-DETR package is unavailable") from exc
    if version != release["version"]:
        raise ParityError(
            f"RF-DETR version mismatch: expected {release['version']}, found {version}"
        )
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    checkout = next(
        (
            parent
            for parent in (module_file.parent, *module_file.parents)
            if (parent / ".git").is_dir()
        ),
        None,
    )
    if checkout is None:
        raise ParityError("RF-DETR must be imported from a Git checkout")

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", "-C", str(checkout), *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ParityError(f"cannot verify RF-DETR source: {exc}") from exc

    head = git("rev-parse", "HEAD")
    commit = head.stdout.strip()
    if head.returncode != 0 or commit != release["git_commit"]:
        raise ParityError(
            "RF-DETR source commit mismatch: "
            f"expected {release['git_commit']}, found {commit or 'unavailable'}"
        )
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = git(*arguments)
        if result.returncode != 0:
            raise ParityError(
                f"RF-DETR source checkout has tracked changes: {checkout}"
            )
    return {
        "rfdetr": version,
        "rfdetr_source_commit": commit,
        "rfdetr_source_checkout": str(checkout),
    }


def _media_validator_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "noesis_rfdetr_media_validator_for_pytorch_parity",
        MEDIA_VALIDATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise ParityError(
            f"cannot load RF-DETR media validator: {MEDIA_VALIDATOR_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_float32(path: Path, shape: Sequence[int]) -> Any:
    import numpy as np

    if sys.byteorder != "little":
        raise ParityError("RF-DETR tensor evidence requires a little-endian host")
    expected = math.prod(shape)
    values = np.fromfile(path, dtype="<f4")
    if values.size != expected:
        raise ParityError(f"tensor element count drifted: {path}")
    values = values.reshape(tuple(shape))
    if not np.isfinite(values).all():
        raise ParityError(f"tensor contains non-finite values: {path}")
    return values


def _map_export_outputs(
    raw: object,
    expected_names: Sequence[str],
) -> dict[str, tuple[str, Any]]:
    names = tuple(str(name) for name in expected_names)
    if names not in set(FAMILY_OUTPUTS.values()):
        raise ParityError(f"unsupported output-name contract: {names}")
    if not isinstance(raw, (tuple, list)) or len(raw) != len(names):
        raise ParityError(
            "official exported module did not return its exact output tuple"
        )
    return {
        name: (OUTPUT_SOURCE_KEYS[name], value)
        for name, value in zip(names, raw, strict=True)
    }


def _compare_tensors(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import numpy as np

    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if (
        ref.shape != cand.shape
        or ref.dtype != np.float32
        or cand.dtype != np.float32
        or not np.isfinite(ref).all()
        or not np.isfinite(cand).all()
        or not math.isfinite(atol)
        or not math.isfinite(rtol)
        or atol < 0.0
        or rtol < 0.0
    ):
        raise ParityError("tensor comparison contract drifted")
    absolute = np.abs(cand.astype(np.float64) - ref.astype(np.float64))
    tolerance = float(atol) + float(rtol) * np.abs(ref.astype(np.float64))
    within = absolute <= tolerance
    denominator = np.maximum(np.abs(ref.astype(np.float64)), 1.0e-12)
    relative = absolute / denominator
    violation = np.maximum(absolute - tolerance, 0.0)

    def percentile(values: Any, value: float) -> float:
        return float(np.percentile(values, value))

    return {
        "passed": bool(within.all()),
        "element_count": int(ref.size),
        "violation_count": int((~within).sum()),
        "violation_fraction": float((~within).sum() / ref.size),
        "absolute_error": {
            "mean": float(absolute.mean()),
            "p50": percentile(absolute, 50.0),
            "p95": percentile(absolute, 95.0),
            "p99": percentile(absolute, 99.0),
            "max": float(absolute.max()),
        },
        "relative_error": {
            "p50": percentile(relative, 50.0),
            "p95": percentile(relative, 95.0),
            "p99": percentile(relative, 99.0),
            "max": float(relative.max()),
        },
        "root_mean_square_error": float(
            np.sqrt(np.mean(np.square(absolute)))
        ),
        "max_tolerance_excess": float(violation.max()),
        "reference_range": {
            "min": float(ref.min()),
            "max": float(ref.max()),
        },
        "candidate_range": {
            "min": float(cand.min()),
            "max": float(cand.max()),
        },
    }


@contextmanager
def _cuda_fp32_policy(torch: Any) -> Iterator[None]:
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    previous_matmul_tf32 = bool(matmul.allow_tf32)
    previous_cudnn_tf32 = bool(cudnn.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    try:
        matmul.allow_tf32 = False
        cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        matmul.allow_tf32 = previous_matmul_tf32
        cudnn.allow_tf32 = previous_cudnn_tf32
        torch.set_float32_matmul_precision(previous_precision)


def _run_model(
    *,
    model: Mapping[str, Any],
    authority: Mapping[str, Any],
    input_values: Any,
    references: Mapping[str, Any],
    device_name: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if CUDA_DEVICE_RE.fullmatch(device_name) is None:
        raise ParityError("--device must be an explicit CUDA device such as cuda:0")
    device_index = int(device_name.split(":", 1)[1])
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise ParityError(f"requested CUDA device is unavailable: {device_name}")
    device = torch.device(device_name)
    package = importlib.import_module(str(model["package"]))
    model_class = getattr(package, str(model["class_name"]), None)
    if not isinstance(model_class, type):
        raise ParityError(
            f"official RF-DETR class is unavailable: {model['class_name']}"
        )

    started = time.monotonic()
    instance: Any = None
    module: Any = None
    input_tensor: Any = None
    with _cuda_fp32_policy(torch):
        try:
            instance = model_class(
                pretrain_weights=str(authority["checkpoint_path"]),
                device=device_name,
            )
            if (
                getattr(getattr(instance, "model", None), "resolution", None)
                != model["resolution"]
            ):
                raise ParityError(
                    f"{model['id']} official class resolution drifted"
                )
            module = getattr(instance.model, "model", None)
            if not isinstance(module, torch.nn.Module):
                raise ParityError(
                    f"{model['id']} lacks its official underlying module"
                )
            module.eval()
            module.to(device=device, dtype=torch.float32)
            export_method = getattr(module, "export", None)
            if not callable(export_method):
                raise ParityError(
                    f"{model['id']} module lacks the official export path"
                )
            export_method()
            if module.training:
                raise ParityError(f"{model['id']} module remained in training mode")

            contiguous_input = np.ascontiguousarray(input_values, dtype=np.float32)
            input_tensor = torch.from_numpy(contiguous_input).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )
            torch.cuda.synchronize(device)
            inference_started = time.monotonic()
            with torch.inference_mode():
                raw = module(input_tensor)
            torch.cuda.synchronize(device)
            inference_seconds = time.monotonic() - inference_started
            mapped = _map_export_outputs(raw, tuple(model["outputs"]))
            output_records: dict[str, Any] = {}
            candidate_outputs: dict[str, Any] = {}
            for output_name, (source_key, tensor) in mapped.items():
                if not isinstance(tensor, torch.Tensor):
                    raise ParityError(
                        f"{model['id']} {output_name} is not a tensor"
                    )
                if tensor.dtype != torch.float32:
                    raise ParityError(
                        f"{model['id']} {output_name} is not float32"
                    )
                expected_shape = tuple(int(v) for v in model["outputs"][output_name])
                if tuple(tensor.shape) != expected_shape:
                    raise ParityError(
                        f"{model['id']} {output_name} shape drifted: "
                        f"expected {expected_shape}, found {tuple(tensor.shape)}"
                    )
                candidate = tensor.detach().cpu().contiguous().numpy()
                if candidate.dtype != np.float32 or not np.isfinite(candidate).all():
                    raise ParityError(
                        f"{model['id']} {output_name} is not finite float32"
                    )
                comparison = _compare_tensors(
                    references[output_name],
                    candidate,
                    atol=atol,
                    rtol=rtol,
                )
                output_records[output_name] = {
                    "pytorch_source_key": source_key,
                    "onnx_output_name": output_name,
                    "shape": list(expected_shape),
                    "dtype": "float32",
                    "comparison": comparison,
                }
                candidate_outputs[output_name] = candidate
            raw_numeric_passed = all(
                row["comparison"]["passed"]
                for row in output_records.values()
            )
            return {
                "class_name": model["class_name"],
                "device": device_name,
                "precision": "float32",
                "tf32": False,
                "autocast": False,
                "training": False,
                "export_path": "underlying_module.export/forward_export",
                "inference_seconds": inference_seconds,
                "model_load_and_inference_seconds": time.monotonic() - started,
                "outputs": output_records,
                "raw_numeric_status": (
                    "passed" if raw_numeric_passed else "diagnostic_drift"
                ),
                "_candidate_outputs": candidate_outputs,
            }
        finally:
            del input_tensor, module, instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _semantic_media_parity(
    *,
    run_dir: Path,
    model: Mapping[str, Any],
    run_model: Mapping[str, Any],
    run_case: Mapping[str, Any],
    corpus_case: Mapping[str, Any],
    candidate_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    validator = _media_validator_module()
    with tempfile.TemporaryDirectory(
        prefix=".pytorch-parity-candidate-", dir=str(run_dir)
    ) as temporary_name:
        temporary = Path(temporary_name)
        os.chmod(temporary, 0o700)
        output_records: dict[str, dict[str, Any]] = {}
        for name, shape in model["outputs"].items():
            candidate = np.asarray(
                candidate_outputs.get(name), dtype="<f4", order="C"
            )
            expected_shape = tuple(int(value) for value in shape)
            if (
                tuple(candidate.shape) != expected_shape
                or not np.isfinite(candidate).all()
            ):
                raise ParityError(
                    f"{model['id']} {name} semantic candidate drifted"
                )
            path = temporary / f"{run_case['id']}__{name}.bin"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(path, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(candidate.tobytes(order="C"))
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            output_records[name] = {
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }

        prepared_model = {
            "id": model["id"],
            "family": model["family"],
            "variant": model["variant"],
            "resolution": model["resolution"],
            "outputs": dict(model["outputs"]),
        }
        prepared_run_model = {
            **dict(run_model),
            "cases": [dict(run_case)],
        }
        candidate_model = {
            "id": model["id"],
            "cases": [
                {
                    "id": run_case["id"],
                    "repeat_stable": True,
                    "outputs": output_records,
                }
            ],
        }
        try:
            report, rows = validator._compare_model(
                run_dir,
                prepared_model,
                prepared_run_model,
                candidate_model,
                [dict(corpus_case)],
            )
        except Exception as exc:
            raise ParityError(
                f"{model['id']} semantic media comparison failed: {exc}"
            ) from exc
        finally:
            for row in locals().get("rows", []):
                close = getattr(row, "close", None)
                if callable(close):
                    close()
    return report


def _ensure_private_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise ParityError(f"private report directory is unsafe: {path}")
    else:
        path.mkdir(parents=True, mode=0o700)
        os.chmod(path, 0o700)
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ParityError(
            f"private report directory must be caller-owned and mode 0700: {path}"
        )
    return path


def _publish_private_report(
    directory: Path,
    report_id: str,
    payload: Mapping[str, Any],
) -> Path:
    report_id = _safe_token(report_id, "report ID")
    root = _ensure_private_directory(directory)
    destination = root / f"{report_id}.json"
    if destination.exists() or destination.is_symlink():
        raise ParityError(f"report evidence already exists: {destination}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_JSON_BYTES:
        raise ParityError(f"report exceeds {MAX_JSON_BYTES} bytes")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{report_id}.", suffix=".tmp", dir=str(root)
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise ParityError(
                f"report evidence already exists: {destination}"
            ) from exc
        directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if stat.S_IMODE(destination.stat().st_mode) != 0o600:
        raise ParityError(f"published report is not mode 0600: {destination}")
    return destination


def _validate_tolerance(value: float, label: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ParityError(f"{label} must be finite and nonnegative")
    return float(value)


def validate(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if os.getuid() == 0:
        raise ParityError("PyTorch parity validation must run as a non-root caller")
    artifact_root = _explicit_artifact_root(
        args.artifact_root
        or os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "")
    )
    matrix = _load_matrix()
    release = dict(matrix["release"])
    selected = _select_models(matrix, args.model)
    case_id = _safe_token(args.case, "case ID")
    report_id = _safe_token(
        args.report_id or _default_report_id(), "report ID"
    )
    atol = _validate_tolerance(args.atol, "--atol")
    rtol = _validate_tolerance(args.rtol, "--rtol")
    run_dir = Path(args.run_dir).expanduser()
    run = _validate_private_run(run_dir, artifact_root)
    run_dir = run_dir.resolve()
    if run.get("release") != release:
        raise ParityError("source run release differs from the current matrix")
    source_environment = _installed_source_environment(release)

    source_cases = [
        row
        for row in run["cases"]
        if isinstance(row, dict) and row.get("id") == case_id
    ]
    if len(source_cases) != 1:
        raise ParityError(f"source run has no unique corpus case {case_id}")

    import torch

    device_name = str(args.device)
    if CUDA_DEVICE_RE.fullmatch(device_name) is None:
        raise ParityError("--device must be an explicit CUDA device such as cuda:0")
    device_index = int(device_name.split(":", 1)[1])
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise ParityError(f"requested CUDA device is unavailable: {device_name}")

    records = []
    for model in selected:
        run_model, run_case = _model_from_run(run, model, case_id)
        authority = _validate_model_authority(
            artifact_root, run_model, model, release
        )
        expected_input = [
            3,
            3,
            int(model["resolution"]),
            int(model["resolution"]),
        ]
        input_path = _validate_tensor_record(
            run_dir,
            run_case.get("input"),
            expected_input,
            f"{model['id']} {case_id} input",
        )
        input_values = _load_float32(input_path, expected_input)
        outputs_record = run_case.get("outputs")
        if (
            not isinstance(outputs_record, dict)
            or set(outputs_record) != set(model["outputs"])
        ):
            raise ParityError(
                f"{model['id']} {case_id} reference output contract drifted"
            )
        references: dict[str, Any] = {}
        reference_authority: dict[str, Any] = {}
        for name, shape in model["outputs"].items():
            path = _validate_tensor_record(
                run_dir,
                outputs_record.get(name),
                shape,
                f"{model['id']} {case_id} {name} reference",
            )
            references[name] = _load_float32(path, shape)
            reference_authority[name] = {
                "path": _relative(path, artifact_root, "reference tensor"),
                "sha256": outputs_record[name]["sha256"],
                "size_bytes": outputs_record[name]["size_bytes"],
            }

        result = _run_model(
            model=model,
            authority=authority,
            input_values=input_values,
            references=references,
            device_name=device_name,
            atol=atol,
            rtol=rtol,
        )
        candidate_outputs = result.pop("_candidate_outputs")
        semantic = _semantic_media_parity(
            run_dir=run_dir,
            model=model,
            run_model=run_model,
            run_case=run_case,
            corpus_case=source_cases[0],
            candidate_outputs=candidate_outputs,
        )
        result["semantic_media_gate"] = semantic
        result["passed"] = semantic.get("automated_status") == "passed"
        records.append(
            {
                "id": model["id"],
                "family": model["family"],
                "variant": model["variant"],
                "checkpoint": {
                    "path": _relative(
                        authority["checkpoint_path"],
                        artifact_root,
                        "checkpoint",
                    ),
                    "size_bytes": model["checkpoint_size_bytes"],
                    "md5": authority["checkpoint_md5"],
                    "sha256": authority["checkpoint_sha256"],
                    "receipt": _relative(
                        authority["checkpoint_receipt_path"],
                        artifact_root,
                        "checkpoint receipt",
                    ),
                    "receipt_sha256": authority[
                        "checkpoint_receipt_sha256"
                    ],
                },
                "immutable_onnx": {
                    "path": _relative(
                        authority["onnx_path"], artifact_root, "ONNX"
                    ),
                    "size_bytes": authority["onnx_size_bytes"],
                    "sha256": authority["onnx_sha256"],
                    "receipt": _relative(
                        authority["onnx_receipt_path"],
                        artifact_root,
                        "ONNX receipt",
                    ),
                    "receipt_sha256": authority["onnx_receipt_sha256"],
                },
                "input": {
                    "path": _relative(input_path, artifact_root, "input tensor"),
                    "shape": expected_input,
                    "dtype": "float32",
                    "sha256": run_case["input"]["sha256"],
                },
                "references": reference_authority,
                **result,
            }
        )
        print(
            f"[PYTORCH] {model['id']} {case_id}: "
            f"{'PASS' if result['passed'] else 'FAIL'}",
            flush=True,
        )

    passed = all(record["passed"] for record in records)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "report_id": report_id,
        "created_at_utc": _utc_now(),
        "status": "passed" if passed else "failed",
        "promotion_status": "unpromoted",
        "runtime_selected": False,
        "training_or_fine_tuning": False,
        "purpose": "PyTorch FP32 export-path versus immutable ONNX CPU reference parity",
        "release": release,
        "source_run": {
            "run_id": run["run_id"],
            "path": _relative(run_dir, artifact_root, "source run"),
            "manifest_sha256": run["manifest_sha256"],
            "case_id": case_id,
        },
        "comparison_contract": {
            "atol": atol,
            "rtol": rtol,
            "element_rule": "abs(candidate-reference) <= atol + rtol*abs(reference)",
            "raw_numeric_role": "diagnostic_only",
            "authoritative_gate": (
                "the same strict all-class and symmetric person-task semantic "
                "media gates used for TensorRT conversion"
            ),
        },
        "environment": {
            **source_environment,
            "python": sys.version.split()[0],
            "torch": importlib.metadata.version("torch"),
            "cuda_runtime": str(torch.version.cuda),
            "device": device_name,
            "device_name": torch.cuda.get_device_name(device_index),
            "tf32": False,
            "autocast": False,
        },
        "privacy": {
            "classification": "owner_private_derived_metrics",
            "contains_source_images_or_tensor_values": False,
            "directory_mode": "0700",
            "file_mode": "0600",
            "commit_or_upload": "forbidden",
        },
        "models": records,
    }
    report["manifest_sha256"] = _json_digest(report)
    output_directory = (
        artifact_root
        / "models"
        / "validation"
        / "rfdetr"
        / RELEASE_VERSION
        / "pytorch-parity"
    )
    destination = _publish_private_report(
        output_directory, report_id, report
    )
    return destination, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GPU-run the official RF-DETR 1.8.3 export path against one "
            "existing immutable ONNX media-reference case."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing owner-private RF-DETR media validation run.",
    )
    parser.add_argument(
        "--artifact-root",
        default="",
        help="DS9 artifact root (or set NOESIS_DS9_ARTIFACT_ROOT).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model ID; repeat or comma-separate. Defaults to "
            + ", ".join(DEFAULT_MODELS)
            + "."
        ),
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help=f"Exactly one prepared case ID (default: {DEFAULT_CASE}).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Explicit CUDA device (default: cuda:0).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"Elementwise absolute tolerance (default: {DEFAULT_ATOL:g}).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"Elementwise relative tolerance (default: {DEFAULT_RTOL:g}).",
    )
    parser.add_argument(
        "--report-id",
        default="",
        help="Unique output report ID; defaults to a UTC timestamp.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination, report = validate(args)
    except ParityError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(f"[REPORT] {destination}")
    print(f"[STATUS] {report['status'].upper()}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
