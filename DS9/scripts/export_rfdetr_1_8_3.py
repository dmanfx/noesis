#!/usr/bin/env python3
"""Download and export the reviewed RF-DETR 1.8.3 DS9 model matrix.

This script intentionally performs CPU-only ONNX export. TensorRT engines must
be built separately with the reviewed DS9 TensorRT image on the deployment GPU.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from packaging.version import Version


DS9_ROOT = Path(__file__).resolve().parents[1]
MODEL_MATRIX_PATH = DS9_ROOT / "config" / "rfdetr_1_8_3_models.json"
PROVENANCE_SCHEMA = "noesis.ds9.rfdetr-artifact-provenance.v1"
RUNTIME_INPUT_CONTRACT = "rgb01_imagenet"
RUNTIME_ADAPTER_REVISION = "sub_div_float32_v1"
RUNTIME_MEAN = (0.485, 0.456, 0.406)
RUNTIME_STD = (0.229, 0.224, 0.225)
RUNTIME_SUB_NODE = "NoesisInputNormalize/Sub"
RUNTIME_DIV_NODE = "NoesisInputNormalize/DivStd"
RUNTIME_CENTERED_TENSOR = "noesis_rfdetr_centered_input"
RUNTIME_NORMALIZED_TENSOR = "noesis_rfdetr_normalized_input"
RUNTIME_MEAN_INITIALIZER = "noesis_rfdetr_imagenet_mean"
RUNTIME_STD_INITIALIZER = "noesis_rfdetr_imagenet_std"
CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 64 * 1024 * 1024
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _safe_token(raw: object, label: str) -> str:
    value = str(raw or "")
    if (
        not SAFE_TOKEN.fullmatch(value)
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise RuntimeError(f"unsafe {label} in RF-DETR model matrix: {value!r}")
    return value


def _load_matrix() -> dict[str, Any]:
    with MODEL_MATRIX_PATH.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    if matrix.get("schema") != "noesis.ds9.rfdetr-model-matrix.v1":
        raise RuntimeError(f"unexpected RF-DETR model matrix schema: {MODEL_MATRIX_PATH}")
    seen: set[str] = set()
    for model in matrix.get("models", []):
        model_id = _safe_token(model.get("id"), "model id")
        if model_id in seen:
            raise RuntimeError(f"duplicate RF-DETR model id: {model_id}")
        seen.add(model_id)
        _safe_token(model.get("checkpoint_filename"), "checkpoint filename")
        _safe_token(model.get("onnx_filename"), "ONNX filename")
        _safe_token(model.get("engine_filename"), "engine filename")
        runtime = model.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError(f"{model_id} is missing its runtime adapter contract")
        if runtime.get("input_contract") != RUNTIME_INPUT_CONTRACT:
            raise RuntimeError(
                f"{model_id} has an unsupported runtime input contract: "
                f"{runtime.get('input_contract')!r}"
            )
        if runtime.get("adapter_revision") != RUNTIME_ADAPTER_REVISION:
            raise RuntimeError(
                f"{model_id} has an unsupported runtime adapter revision: "
                f"{runtime.get('adapter_revision')!r}"
            )
        _safe_token(runtime.get("onnx_filename"), "runtime ONNX filename")
        _safe_token(model.get("class_name"), "class name")
        package = _safe_token(model.get("package"), "package name")
        if package not in {"rfdetr", "rfdetr_plus"}:
            raise RuntimeError(f"unexpected RF-DETR package: {package}")
        if not str(model.get("checkpoint_url", "")).startswith(
            "https://storage.googleapis.com/rfdetr/"
        ):
            raise RuntimeError(f"unexpected checkpoint URL for {model_id}")
    return matrix


def _artifact_root(raw: str | None) -> Path:
    value = str(raw or os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "") or "").strip()
    if not value:
        raise SystemExit(
            "--artifact-root or NOESIS_DS9_ARTIFACT_ROOT is required; "
            "RF-DETR assets must not spill onto the repository/root filesystem"
        )
    root = Path(value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _paths(root: Path) -> dict[str, Path]:
    paths = {
        "checkpoints": root / "models" / "checkpoints" / "rfdetr" / "1.8.3",
        "onnx": root / "models" / "onnx" / "rfdetr" / "1.8.3",
        "runtime_onnx": root
        / "models"
        / "onnx"
        / "rfdetr"
        / "1.8.3"
        / "runtime",
        "engines": root / "models" / "engines" / "rfdetr" / "1.8.3",
        "provenance": root / "models" / "provenance" / "rfdetr" / "1.8.3",
        "work": root / "model-work" / "rfdetr" / "1.8.3",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing or unsafe {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def _receipt_path(paths: dict[str, Path], model: dict[str, Any], kind: str) -> Path:
    return paths["provenance"] / f"{_safe_token(model['id'], 'model id')}.{kind}.json"


def _validate_checkpoint_receipt(
    receipt: dict[str, Any],
    *,
    model: dict[str, Any],
    release: dict[str, Any],
    destination: Path,
    sha256: str,
) -> None:
    expected = {
        "schema": PROVENANCE_SCHEMA,
        "artifact_kind": "checkpoint",
        "model_id": model["id"],
        "family": model["family"],
        "variant": model["variant"],
        "license": model["license"],
        "release": release,
        "source_url": model["checkpoint_url"],
        "filename": destination.name,
        "size_bytes": destination.stat().st_size,
        "md5": model["checkpoint_md5"],
        "sha256": sha256,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"checkpoint receipt mismatch for {model['id']} field {key}: "
                f"expected {value!r}, got {receipt.get(key)!r}"
            )


def _download(
    model: dict[str, Any], paths: dict[str, Path], release: dict[str, Any]
) -> Path:
    filename = _safe_token(model["checkpoint_filename"], "checkpoint filename")
    destination = paths["checkpoints"] / filename
    lock_path = destination.with_name(f".{destination.name}.lock")
    with _exclusive_lock(lock_path):
        return _download_locked(model, paths, release, destination)


def _download_locked(
    model: dict[str, Any],
    paths: dict[str, Path],
    release: dict[str, Any],
    destination: Path,
) -> Path:
    expected_size = int(model["checkpoint_size_bytes"])
    expected_md5 = str(model["checkpoint_md5"])
    if destination.is_file():
        actual_md5, sha256 = _hashes(destination)
        if destination.stat().st_size != expected_size or actual_md5 != expected_md5:
            raise RuntimeError(
                f"existing checkpoint does not match the reviewed release: {destination}"
            )
        receipt = _load_json(
            _receipt_path(paths, model, "checkpoint"), "checkpoint receipt"
        )
        _validate_checkpoint_receipt(
            receipt,
            model=model,
            release=release,
            destination=destination,
            sha256=sha256,
        )
        print(f"[REUSE] {model['id']}: {destination}", flush=True)
        return destination
    else:
        fd, partial_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=str(destination.parent)
        )
        os.close(fd)
        partial = Path(partial_name)
        request = urllib.request.Request(
            str(model["checkpoint_url"]),
            headers={"User-Agent": "Noesis-DS9-RFDETR-Exporter/1.0"},
        )
        print(
            f"[DOWNLOAD] {model['id']}: {model['checkpoint_url']} -> {destination}",
            flush=True,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                downloaded = 0
                next_report = PROGRESS_BYTES
                while chunk := response.read(CHUNK_BYTES):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(
                            f"[DOWNLOAD] {model['id']}: {downloaded}/{expected_size} bytes",
                            flush=True,
                        )
                        next_report += PROGRESS_BYTES
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, urllib.error.URLError):
            partial.unlink(missing_ok=True)
            raise
        actual_size = partial.stat().st_size
        if actual_size != expected_size:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"checkpoint size mismatch for {model['id']}: "
                f"expected {expected_size}, got {actual_size}"
            )
        actual_md5, sha256 = _hashes(partial)
        if actual_md5 != expected_md5:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"checkpoint MD5 mismatch for {model['id']}: "
                f"expected {expected_md5}, got {actual_md5}"
            )
        os.replace(partial, destination)
        print(f"[DOWNLOADED] {model['id']}: {destination}", flush=True)

    _write_json_atomic(
        _receipt_path(paths, model, "checkpoint"),
        {
            "schema": PROVENANCE_SCHEMA,
            "artifact_kind": "checkpoint",
            "model_id": model["id"],
            "family": model["family"],
            "variant": model["variant"],
            "license": model["license"],
            "release": release,
            "source_url": model["checkpoint_url"],
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
            "md5": expected_md5,
            "sha256": sha256,
            "recorded_at_unix": int(time.time()),
        },
    )
    return destination


def _installed_source_commit(package: str) -> str:
    module = importlib.import_module(package)
    module_path = Path(module.__file__).resolve()
    checkout = next(
        (parent for parent in (module_path.parent, *module_path.parents) if (parent / ".git").exists()),
        None,
    )
    if checkout is None:
        raise RuntimeError(
            f"{package} export provenance requires an installed Git checkout, "
            "not an unverified wheel"
        )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"invalid RF-DETR checkout commit: {commit!r}")
    for diff_args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        result = subprocess.run(
            ["git", "-C", str(checkout), *diff_args],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{package} source checkout has tracked changes: {checkout}")
    return commit


def _assert_export_environment(
    release: dict[str, Any], model: dict[str, Any]
) -> dict[str, str]:
    expected = str(release["version"])
    actual = importlib.metadata.version("rfdetr")
    if actual != expected:
        raise RuntimeError(f"RF-DETR export requires rfdetr=={expected}, found {actual}")
    source_commit = _installed_source_commit("rfdetr")
    if source_commit != str(release["git_commit"]):
        raise RuntimeError(
            f"RF-DETR source commit mismatch: expected {release['git_commit']}, "
            f"found {source_commit}"
        )
    versions = {
        "python": sys.version.split()[0],
        "rfdetr": actual,
        "rfdetr_source_commit": source_commit,
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "transformers": importlib.metadata.version("transformers"),
        "onnx": importlib.metadata.version("onnx"),
    }
    if model["package"] == "rfdetr_plus":
        expected_plus = str(model["package_version"])
        actual_plus = importlib.metadata.version("rfdetr-plus")
        if actual_plus != expected_plus:
            raise RuntimeError(
                f"licensed export requires rfdetr-plus=={expected_plus}, found {actual_plus}"
            )
        plus_commit = _installed_source_commit("rfdetr_plus")
        if plus_commit != str(model["package_git_commit"]):
            raise RuntimeError(
                f"RF-DETR Plus source commit mismatch: expected "
                f"{model['package_git_commit']}, found {plus_commit}"
            )
        versions["rfdetr_plus"] = actual_plus
        versions["rfdetr_plus_source_commit"] = plus_commit
    if not Version("5.1.0") <= Version(versions["transformers"]) < Version("6.0.0"):
        raise RuntimeError(
            "RF-DETR 1.8.3 requires transformers>=5.1.0,<6.0.0, "
            f"found {versions['transformers']}"
        )
    if Version(versions["torch"]) < Version("2.2.0"):
        raise RuntimeError(f"RF-DETR 1.8.3 requires torch>=2.2.0, found {versions['torch']}")
    if Version(versions["torchvision"]) < Version("0.17.0"):
        raise RuntimeError(
            f"RF-DETR 1.8.3 requires torchvision>=0.17.0, found {versions['torchvision']}"
        )
    if not Version("1.16.0") <= Version(versions["onnx"]) < Version("2.0.0"):
        raise RuntimeError(
            f"RF-DETR ONNX export requires onnx>=1.16.0,<2.0.0, found {versions['onnx']}"
        )
    return versions


def _shape_values(value: Any) -> list[int | str | None]:
    dims: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def _validate_onnx(
    path: Path, model: dict[str, Any], release: dict[str, Any]
) -> dict[str, Any]:
    import onnx

    graph = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(graph)
    imports = {item.domain or "ai.onnx": int(item.version) for item in graph.opset_import}
    if imports.get("ai.onnx") != int(release["onnx_opset"]):
        raise RuntimeError(f"{model['id']} ONNX has unexpected opset imports: {imports}")

    inputs = {value.name: _shape_values(value) for value in graph.graph.input}
    expected_input = [
        int(release["batch_size"]),
        3,
        int(model["resolution"]),
        int(model["resolution"]),
    ]
    if inputs != {"input": expected_input}:
        raise RuntimeError(
            f"{model['id']} ONNX input mismatch: expected input={expected_input}, got {inputs}"
        )

    outputs = {value.name: _shape_values(value) for value in graph.graph.output}
    expected_outputs = {
        name: [int(dim) for dim in shape]
        for name, shape in dict(model["outputs"]).items()
    }
    if outputs != expected_outputs:
        raise RuntimeError(
            f"{model['id']} ONNX output mismatch: "
            f"expected {expected_outputs}, got {outputs}"
        )

    return {
        "input": inputs,
        "outputs": outputs,
        "opset_imports": imports,
        "node_count": len(graph.graph.node),
        "initializer_count": len(graph.graph.initializer),
    }


def _runtime_adapter_spec() -> dict[str, Any]:
    return {
        "input_contract": RUNTIME_INPUT_CONTRACT,
        "adapter_revision": RUNTIME_ADAPTER_REVISION,
        "input_dtype": "float32",
        "input_range": [0.0, 1.0],
        "operation": "(input - mean) / std",
        "channel_order": "RGB",
        "layout": "NCHW",
        "mean": list(RUNTIME_MEAN),
        "std": list(RUNTIME_STD),
        "nodes": [RUNTIME_SUB_NODE, RUNTIME_DIV_NODE],
    }


def _inject_runtime_normalization(graph: Any) -> Any:
    """Return an ONNX model whose public input is RGB float32 in [0, 1]."""

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.ModelProto()
    model.CopyFrom(graph)
    if len(model.graph.input) != 1 or model.graph.input[0].name != "input":
        raise RuntimeError("RF-DETR runtime adapter requires exactly one input named 'input'")

    reserved_names = {
        RUNTIME_SUB_NODE,
        RUNTIME_DIV_NODE,
        RUNTIME_CENTERED_TENSOR,
        RUNTIME_NORMALIZED_TENSOR,
        RUNTIME_MEAN_INITIALIZER,
        RUNTIME_STD_INITIALIZER,
    }
    observed_names = {
        value.name
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
            *model.graph.initializer,
        )
    }
    observed_names.update(node.name for node in model.graph.node if node.name)
    for node in model.graph.node:
        observed_names.update(node.output)
    collision = sorted(reserved_names & observed_names)
    if collision:
        raise RuntimeError(
            "RF-DETR runtime adapter name collision: " + ", ".join(collision)
        )

    rewired = 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "input":
                node.input[index] = RUNTIME_NORMALIZED_TENSOR
                rewired += 1
    if rewired <= 0:
        raise RuntimeError("RF-DETR ONNX graph does not consume its public input")

    mean = numpy_helper.from_array(
        np.asarray(RUNTIME_MEAN, dtype=np.float32).reshape(1, 3, 1, 1),
        name=RUNTIME_MEAN_INITIALIZER,
    )
    std = numpy_helper.from_array(
        np.asarray(RUNTIME_STD, dtype=np.float32).reshape(1, 3, 1, 1),
        name=RUNTIME_STD_INITIALIZER,
    )
    subtract = helper.make_node(
        "Sub",
        ["input", RUNTIME_MEAN_INITIALIZER],
        [RUNTIME_CENTERED_TENSOR],
        name=RUNTIME_SUB_NODE,
    )
    divide = helper.make_node(
        "Div",
        [RUNTIME_CENTERED_TENSOR, RUNTIME_STD_INITIALIZER],
        [RUNTIME_NORMALIZED_TENSOR],
        name=RUNTIME_DIV_NODE,
    )
    model.graph.initializer.extend([mean, std])
    model.graph.node.insert(0, subtract)
    model.graph.node.insert(1, divide)
    model.doc_string = (
        (model.doc_string + "\n") if model.doc_string else ""
    ) + "Noesis runtime adapter: RGB float32 [0,1] to ImageNet-normalized NCHW."
    onnx.checker.check_model(model)
    return model


def _validate_runtime_onnx(
    path: Path, model: dict[str, Any], release: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    contract = _validate_onnx(path, model, release)
    graph = onnx.load(str(path), load_external_data=True)
    nodes = {node.name: node for node in graph.graph.node if node.name}
    subtract = nodes.get(RUNTIME_SUB_NODE)
    divide = nodes.get(RUNTIME_DIV_NODE)
    if (
        subtract is None
        or subtract.op_type != "Sub"
        or list(subtract.input) != ["input", RUNTIME_MEAN_INITIALIZER]
        or list(subtract.output) != [RUNTIME_CENTERED_TENSOR]
    ):
        raise RuntimeError(f"{model['id']} runtime ONNX lacks the exact subtract adapter")
    if (
        divide is None
        or divide.op_type != "Div"
        or list(divide.input)
        != [RUNTIME_CENTERED_TENSOR, RUNTIME_STD_INITIALIZER]
        or list(divide.output) != [RUNTIME_NORMALIZED_TENSOR]
    ):
        raise RuntimeError(f"{model['id']} runtime ONNX lacks the exact division adapter")

    initializers = {
        value.name: numpy_helper.to_array(value)
        for value in graph.graph.initializer
    }
    expected_mean = np.asarray(RUNTIME_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    expected_std = np.asarray(RUNTIME_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    if not np.array_equal(initializers.get(RUNTIME_MEAN_INITIALIZER), expected_mean):
        raise RuntimeError(f"{model['id']} runtime ONNX ImageNet mean drifted")
    if not np.array_equal(
        initializers.get(RUNTIME_STD_INITIALIZER), expected_std
    ):
        raise RuntimeError(f"{model['id']} runtime ONNX ImageNet std drifted")

    public_input_consumers = [
        node.name for node in graph.graph.node if "input" in node.input
    ]
    if public_input_consumers != [RUNTIME_SUB_NODE]:
        raise RuntimeError(
            f"{model['id']} runtime ONNX bypasses normalization: "
            f"{public_input_consumers}"
        )
    normalized_consumers = sum(
        1
        for node in graph.graph.node
        for name in node.input
        if name == RUNTIME_NORMALIZED_TENSOR
    )
    if normalized_consumers <= 0:
        raise RuntimeError(f"{model['id']} runtime ONNX normalization is unused")
    contract["runtime_adapter"] = _runtime_adapter_spec()
    contract["normalized_input_consumer_count"] = normalized_consumers
    return contract


def _runtime_receipt_path(paths: dict[str, Path], model: dict[str, Any]) -> Path:
    return paths["provenance"] / (
        f"{_safe_token(model['id'], 'model id')}.runtime_onnx."
        f"{RUNTIME_ADAPTER_REVISION}.json"
    )


def _adapt_runtime_onnx(
    model: dict[str, Any], paths: dict[str, Path], release: dict[str, Any]
) -> Path:
    import onnx

    source = paths["onnx"] / _safe_token(model["onnx_filename"], "ONNX filename")
    source_receipt_path = _receipt_path(paths, model, "onnx")
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"missing reviewed normalized-input ONNX: {source}")
    source_receipt = _load_json(source_receipt_path, "ONNX receipt")
    source_contract = _validate_onnx(source, model, release)
    if (
        source_receipt.get("schema") != PROVENANCE_SCHEMA
        or source_receipt.get("artifact_kind") != "onnx"
        or source_receipt.get("model_id") != model["id"]
        or source_receipt.get("filename") != source.name
        or source_receipt.get("sha256") != _sha256(source)
        or source_receipt.get("size_bytes") != source.stat().st_size
        or source_receipt.get("tensor_contract") != source_contract
    ):
        raise RuntimeError(f"{model['id']} normalized-input ONNX receipt drifted")

    runtime = dict(model["runtime"])
    destination = paths["runtime_onnx"] / _safe_token(
        runtime["onnx_filename"], "runtime ONNX filename"
    )
    receipt_path = _runtime_receipt_path(paths, model)
    lock_path = destination.with_name(f".{destination.name}.lock")
    with _exclusive_lock(lock_path):
        if destination.is_file():
            contract = _validate_runtime_onnx(destination, model, release)
        else:
            graph = onnx.load(str(source), load_external_data=True)
            adapted = _inject_runtime_normalization(graph)
            with tempfile.TemporaryDirectory(
                prefix=f"{model['id']}-runtime-adapter-", dir=str(paths["work"])
            ) as work_dir:
                temporary = Path(work_dir) / destination.name
                onnx.save_model(adapted, str(temporary))
                contract = _validate_runtime_onnx(temporary, model, release)
                _publish_onnx(temporary, destination)
            print(f"[ADAPTED] {model['id']}: {destination}", flush=True)

        expected_receipt = {
            "schema": PROVENANCE_SCHEMA,
            "artifact_kind": "runtime_onnx",
            "model_id": model["id"],
            "family": model["family"],
            "variant": model["variant"],
            "license": model["license"],
            "release": release,
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "source": {
                "filename": source.name,
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
                "receipt": source_receipt_path.name,
                "receipt_sha256": _sha256(source_receipt_path),
            },
            "adapter": _runtime_adapter_spec(),
            "tensor_contract": contract,
        }
        if receipt_path.is_file():
            receipt = _load_json(receipt_path, "runtime ONNX receipt")
            for key, value in expected_receipt.items():
                if receipt.get(key) != value:
                    raise RuntimeError(
                        f"runtime ONNX receipt mismatch for {model['id']} "
                        f"field {key}"
                    )
            print(f"[REUSE] validated runtime ONNX: {destination}", flush=True)
        else:
            _write_json_atomic(
                receipt_path,
                {**expected_receipt, "recorded_at_unix": int(time.time())},
            )
    return destination


def _publish_onnx(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_file() and _sha256(destination) == _sha256(source):
            print(f"[REUSE] identical ONNX already published: {destination}", flush=True)
            return
        raise RuntimeError(
            f"refusing to overwrite a different versioned ONNX artifact: {destination}"
        )
    temporary = destination.with_name(f".{destination.name}.publishing-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _export_settings(
    model: dict[str, Any], release: dict[str, Any]
) -> dict[str, Any]:
    settings = {
        "class_name": model["class_name"],
        "device": "cpu",
        "format": "onnx",
        "opset": release["onnx_opset"],
        "batch_size": release["batch_size"],
        "dynamic_batch": release["dynamic_batch"],
        "resolution": [model["resolution"], model["resolution"]],
        "preserve_official_query_count": True,
    }
    if model["package"] == "rfdetr_plus":
        settings["licensed_package"] = {
            "name": "rfdetr-plus",
            "version": model["package_version"],
            "git_tag": model["package_git_tag"],
            "git_commit": model["package_git_commit"],
            "source": model["package_source"],
        }
    return settings


def _validate_onnx_receipt(
    receipt: dict[str, Any],
    *,
    model: dict[str, Any],
    release: dict[str, Any],
    destination: Path,
    checkpoint_md5: str,
    checkpoint_sha256: str,
    tensor_contract: dict[str, Any],
    export_environment: dict[str, str],
) -> None:
    expected = {
        "schema": PROVENANCE_SCHEMA,
        "artifact_kind": "onnx",
        "model_id": model["id"],
        "family": model["family"],
        "variant": model["variant"],
        "license": model["license"],
        "release": release,
        "checkpoint": {
            "filename": _safe_token(
                model["checkpoint_filename"], "checkpoint filename"
            ),
            "md5": checkpoint_md5,
            "sha256": checkpoint_sha256,
        },
        "export": _export_settings(model, release),
        "filename": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "tensor_contract": tensor_contract,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"ONNX receipt mismatch for {model['id']} field {key}: "
                f"expected {value!r}, got {receipt.get(key)!r}"
            )
    recorded_environment = receipt.get("export_environment")
    if not isinstance(recorded_environment, dict):
        raise RuntimeError(f"ONNX receipt lacks export environment for {model['id']}")
    # Receipts made by the first reviewed run predate the explicit source-commit
    # field. Their identical release.git_commit plus exact package versions and
    # artifact/checkpoint hashes remain sufficient for immutable reuse.
    for key, value in export_environment.items():
        if key == "rfdetr_source_commit" and key not in recorded_environment:
            continue
        if recorded_environment.get(key) != value:
            raise RuntimeError(
                f"ONNX receipt environment mismatch for {model['id']} field {key}: "
                f"expected {value!r}, got {recorded_environment.get(key)!r}"
            )


def _export_one(
    model: dict[str, Any], paths: dict[str, Path], release: dict[str, Any]
) -> Path:
    export_environment = _assert_export_environment(release, model)
    checkpoint = paths["checkpoints"] / _safe_token(
        model["checkpoint_filename"], "checkpoint filename"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing reviewed checkpoint: {checkpoint}")
    actual_md5, checkpoint_sha256 = _hashes(checkpoint)
    if actual_md5 != str(model["checkpoint_md5"]):
        raise RuntimeError(f"checkpoint MD5 mismatch before export: {checkpoint}")

    destination = paths["onnx"] / _safe_token(model["onnx_filename"], "ONNX filename")
    lock_path = destination.with_name(f".{destination.name}.lock")
    with _exclusive_lock(lock_path):
        return _export_one_locked(
            model,
            paths,
            release,
            export_environment,
            checkpoint,
            actual_md5,
            checkpoint_sha256,
            destination,
        )


def _export_one_locked(
    model: dict[str, Any],
    paths: dict[str, Path],
    release: dict[str, Any],
    export_environment: dict[str, str],
    checkpoint: Path,
    actual_md5: str,
    checkpoint_sha256: str,
    destination: Path,
) -> Path:
    if destination.is_file():
        contract = _validate_onnx(destination, model, release)
        receipt = _load_json(_receipt_path(paths, model, "onnx"), "ONNX receipt")
        _validate_onnx_receipt(
            receipt,
            model=model,
            release=release,
            destination=destination,
            checkpoint_md5=actual_md5,
            checkpoint_sha256=checkpoint_sha256,
            tensor_contract=contract,
            export_environment=export_environment,
        )
        print(f"[REUSE] validated ONNX: {destination}", flush=True)
        return destination
    else:
        module = importlib.import_module(str(model["package"]))
        model_class = getattr(module, str(model["class_name"]))
        print(
            f"[EXPORT] {model['id']}: {model['class_name']} "
            f"batch={release['batch_size']} resolution={model['resolution']}",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix=f"{model['id']}-", dir=str(paths["work"])
        ) as work_dir:
            instance = model_class(
                pretrain_weights=str(checkpoint.resolve(strict=True)),
                device="cpu",
            )
            exported = instance.export(
                output_dir=work_dir,
                shape=(int(model["resolution"]), int(model["resolution"])),
                batch_size=int(release["batch_size"]),
                dynamic_batch=bool(release["dynamic_batch"]),
                opset_version=int(release["onnx_opset"]),
                verbose=False,
                format="onnx",
                notes=None,
            )
            del instance
            gc.collect()
            exported_path = Path(exported)
            sidecars = [
                item
                for item in exported_path.parent.iterdir()
                if item.is_file() and item != exported_path
            ]
            if sidecars:
                raise RuntimeError(
                    f"{model['id']} produced unsupported ONNX external-data sidecars: "
                    f"{[item.name for item in sidecars]}"
                )
            contract = _validate_onnx(exported_path, model, release)
            _publish_onnx(exported_path, destination)
        print(f"[EXPORTED] {model['id']}: {destination}", flush=True)

    _write_json_atomic(
        _receipt_path(paths, model, "onnx"),
        {
            "schema": PROVENANCE_SCHEMA,
            "artifact_kind": "onnx",
            "model_id": model["id"],
            "family": model["family"],
            "variant": model["variant"],
            "license": model["license"],
            "release": release,
            "checkpoint": {
                "filename": checkpoint.name,
                "md5": actual_md5,
                "sha256": checkpoint_sha256,
            },
            "export": _export_settings(model, release),
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "tensor_contract": contract,
            "export_environment": export_environment,
            "recorded_at_unix": int(time.time()),
        },
    )
    return destination


def _selected_models(
    matrix: dict[str, Any],
    *,
    families: set[str],
    model_ids: set[str],
    include_pml: bool,
) -> list[dict[str, Any]]:
    models = list(matrix["models"])
    known_families = {str(item["family"]) for item in models}
    unknown_families = sorted(families - known_families)
    if unknown_families:
        raise SystemExit(f"unknown RF-DETR families: {', '.join(unknown_families)}")
    known_ids = {str(item["id"]) for item in models}
    unknown_ids = sorted(model_ids - known_ids)
    if unknown_ids:
        raise SystemExit(f"unknown RF-DETR model IDs: {', '.join(unknown_ids)}")

    selected = [
        item
        for item in models
        if (not families or str(item["family"]) in families)
        and (not model_ids or str(item["id"]) in model_ids)
    ]
    blocked = [item for item in selected if not bool(item.get("enabled"))]
    if blocked and not include_pml:
        if families or model_ids:
            details = "; ".join(
                f"{item['id']}: {item.get('blocked_reason', 'license-gated')}"
                for item in blocked
            )
            raise SystemExit(
                f"explicit RF-DETR selection contains license-gated model(s): {details}"
            )
        for item in blocked:
            print(
                f"[SKIP-LICENSE] {item['id']}: {item.get('blocked_reason')}",
                flush=True,
            )
        selected = [item for item in selected if bool(item.get("enabled"))]
    if include_pml:
        if os.environ.get("RFDETR_PML_ACCEPTED") != "1":
            raise SystemExit(
                "--include-pml requires RFDETR_PML_ACCEPTED=1 after the user has "
                "personally accepted the Platform Model License"
            )
    return selected


def _split(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root")
    parser.add_argument(
        "--phase",
        choices=("download", "export", "adapt", "all"),
        default="all",
        help=(
            "Run checkpoint download, normalized-input ONNX export, runtime "
            "RGB01 adapter materialization, or all phases."
        ),
    )
    parser.add_argument(
        "--families",
        default="",
        help="Optional comma-separated subset: detection,segmentation,keypoint.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated model IDs from --list.",
    )
    parser.add_argument(
        "--include-pml",
        action="store_true",
        help="Include licensed detection XL/2XL after explicit PML acceptance.",
    )
    parser.add_argument("--list", action="store_true", help="Print the reviewed model matrix.")
    parser.add_argument("--export-one", help=argparse.SUPPRESS)
    parser.add_argument("--adapt-one", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    matrix = _load_matrix()
    release = dict(matrix["release"])
    if args.list:
        for item in matrix["models"]:
            status = "enabled" if item.get("enabled") else f"blocked: {item.get('blocked_reason')}"
            print(
                f"{item['id']:20} {item['family']:12} {item['variant']:8} "
                f"{item['resolution']:4} {item['license']:10} {status}"
            )
        return 0

    root = _artifact_root(args.artifact_root)
    paths = _paths(root)
    selected = _selected_models(
        matrix,
        families=_split(args.families),
        model_ids=_split(args.models),
        include_pml=bool(args.include_pml),
    )
    if args.export_one:
        matching = [item for item in selected if item["id"] == args.export_one]
        if len(matching) != 1:
            raise SystemExit(f"--export-one model was not selected: {args.export_one}")
        _export_one(matching[0], paths, release)
        return 0
    if args.adapt_one:
        matching = [item for item in selected if item["id"] == args.adapt_one]
        if len(matching) != 1:
            raise SystemExit(f"--adapt-one model was not selected: {args.adapt_one}")
        _adapt_runtime_onnx(matching[0], paths, release)
        return 0
    if not selected:
        raise SystemExit("no RF-DETR models selected")

    if args.phase in {"download", "all"}:
        for model in selected:
            _download(model, paths, release)

    if args.phase in {"export", "all"}:
        for model in selected:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--artifact-root",
                str(root),
                "--phase",
                "export",
                "--models",
                str(model["id"]),
                "--export-one",
                str(model["id"]),
            ]
            if args.include_pml:
                command.append("--include-pml")
            print(f"[SUBPROCESS] {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)

    if args.phase in {"adapt", "all"}:
        for model in selected:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--artifact-root",
                str(root),
                "--phase",
                "adapt",
                "--models",
                str(model["id"]),
                "--adapt-one",
                str(model["id"]),
            ]
            if args.include_pml:
                command.append("--include-pml")
            print(f"[SUBPROCESS] {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)

    print(
        f"[OK] RF-DETR 1.8.3 checkpoint/ONNX/runtime-adapter phase complete for "
        f"{', '.join(str(item['id']) for item in selected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
