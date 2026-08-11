#!/usr/bin/env python3
"""Export and validate the DS9 YOLO26n pose batch-3 ONNX without GPU access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODELS_ROOT = (REPO_ROOT / "models").resolve()
DEFAULT_OUTPUT = SOURCE_MODELS_ROOT / "onnx" / "yolo26n-pose_b3.onnx"
DEFAULT_PROVENANCE_OUTPUT = (
    REPO_ROOT
    / "DS9"
    / "models"
    / "onnx"
    / "yolo26n-pose_b3.onnx.provenance.json"
)
EXPECTED_CHECKPOINT_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
EXPECTED_ULTRALYTICS_VERSION = "8.4.0"
EXPECTED_INPUT = (3, 3, 640, 640)
EXPECTED_OUTPUT = (3, 300, 57)


def _artifact_root() -> Path:
    raw = str(os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "")).strip()
    if not raw:
        raise ValueError(
            "NOESIS_DS9_ARTIFACT_ROOT must name an explicit absolute staging root"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"NOESIS_DS9_ARTIFACT_ROOT must be absolute: {raw}")
    resolved = path.resolve(strict=False)
    ds9_root = REPO_ROOT / "DS9"
    if resolved in {Path("/"), REPO_ROOT.resolve(), ds9_root.resolve()}:
        raise ValueError(f"refusing unsafe artifact root: {resolved}")
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"artifact root must not be inside the checkout: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(value: object) -> tuple[int | str, ...]:
    tensor_type = value.type.tensor_type
    result: list[int | str] = []
    for dimension in tensor_type.shape.dim:
        if dimension.dim_value:
            result.append(int(dimension.dim_value))
        else:
            result.append(str(dimension.dim_param))
    return tuple(result)


def _relative_or_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "models" / "yolo26n-pose.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=DEFAULT_PROVENANCE_OUTPUT,
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    try:
        staging_root = _artifact_root()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    workspace = staging_root / "pose_b3_cpu_export"
    workspace.mkdir(parents=True, exist_ok=True)

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit(f"missing pose checkpoint: {source}")
    source_sha256 = _sha256(source)
    if source_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(
            f"pose checkpoint SHA-256 mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {source_sha256}"
        )
    if output.exists() and not args.replace:
        raise SystemExit(f"output already exists; pass --replace after verifying it: {output}")

    # Hide all CUDA devices before importing Torch or Ultralytics. The explicit
    # CPU device below is a second independent guard.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ULTRALYTICS_SETTINGS"] = str(workspace / "ultralytics-settings.json")
    os.environ["YOLO_CONFIG_DIR"] = str(workspace / "ultralytics-config")

    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise SystemExit(
            f"Ultralytics {EXPECTED_ULTRALYTICS_VERSION} required; found {ultralytics.__version__}"
        )
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise SystemExit("CPU-only export guard failed: Torch can see a CUDA device")

    staged_checkpoint = workspace / "yolo26n-pose_b3_cpu.pt"
    shutil.copy2(source, staged_checkpoint)
    if _sha256(staged_checkpoint) != source_sha256:
        raise SystemExit("staged checkpoint hash mismatch")

    model = YOLO(str(staged_checkpoint))
    exported_raw = model.export(
        format="onnx",
        imgsz=640,
        batch=3,
        dynamic=False,
        simplify=False,
        opset=18,
        device="cpu",
        half=False,
        optimize=False,
    )
    exported = Path(str(exported_raw)).resolve()
    if not exported.is_file() or exported.stat().st_size <= 0:
        raise SystemExit(f"Ultralytics did not produce an ONNX file: {exported}")

    graph = onnx.load(str(exported), load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    inputs = {value.name: _shape(value) for value in graph.graph.input}
    outputs = {value.name: _shape(value) for value in graph.graph.output}
    if inputs != {"images": EXPECTED_INPUT}:
        raise SystemExit(f"unexpected pose ONNX inputs: {inputs}")
    if outputs != {"output0": EXPECTED_OUTPUT}:
        raise SystemExit(f"unexpected pose ONNX outputs: {outputs}")
    external = [tensor.name for tensor in graph.graph.initializer if tensor.data_location == onnx.TensorProto.EXTERNAL]
    if external:
        raise SystemExit(f"pose ONNX unexpectedly uses external tensor data: {external[:10]}")

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(2603)
    one = rng.random((1, 3, 640, 640), dtype=np.float32)
    batch = np.repeat(one, 3, axis=0)
    result = session.run(["output0"], {"images": batch})[0]
    if result.shape != EXPECTED_OUTPUT or not np.isfinite(result).all():
        raise SystemExit(f"CPU ONNX inference returned invalid output: shape={result.shape}")
    batch_delta = float(max(np.max(np.abs(result[0] - result[1])), np.max(np.abs(result[0] - result[2]))))
    if batch_delta > 1e-5:
        raise SystemExit(f"identical batch members diverged during CPU inference: max_delta={batch_delta}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    shutil.copy2(exported, temporary)
    os.replace(temporary, output)
    output_sha256 = _sha256(output)

    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": _relative_or_name(source),
        "source_sha256": source_sha256,
        "output": _relative_or_name(output),
        "output_sha256": output_sha256,
        "output_bytes": output.stat().st_size,
        "input_tensors": {name: list(shape) for name, shape in inputs.items()},
        "output_tensors": {name: list(shape) for name, shape in outputs.items()},
        "external_tensor_count": 0,
        "cpu_inference_provider": session.get_providers(),
        "identical_batch_max_abs_delta": batch_delta,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "python_version": sys.version.split()[0],
    }
    provenance_path = args.provenance_output.resolve()
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
