#!/usr/bin/env python3
"""Export existing YOLO26 detector weights to a separate fixed-resolution ONNX.

Exports run on CPU and never build engines, download weights, or select a runtime
model. The output is the existing DS9 parser contract: FLOAT images in NCHW and
FLOAT output0 rows containing x1, y1, x2, y2, score, class_id, without an NMS op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINTS = {
    "yolo26m": {
        "sha256": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
        "scale": "m",
        "classes": 80,
        "export_version": "8.4.0",
    },
    "yolo26l": {
        "sha256": "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26",
        "scale": "l",
        "classes": 80,
        "export_version": "8.4.0",
    },
    "watch-26m-v1-e44": {
        "sha256": "fe68a238e60fc284a8c1269ae3baac0b71d9192360d6c3178c90f4db81fdf742",
        "scale": "m",
        "classes": 1,
        "export_version": "8.4.173",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(value: Any) -> tuple[int | str, ...]:
    return tuple(
        dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
        for dimension in value.type.tensor_type.shape.dim
    )


def _output_root(raw: Path) -> Path:
    if not raw.is_absolute():
        raise ValueError("--output-root must be an explicit absolute artifact directory")
    root = raw.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if root == Path("/") or root == repo_root or repo_root in root.parents:
        raise ValueError("--output-root must be outside the source checkout")
    return root


def _validate_graph(onnx: Any, graph: Any, imgsz: int) -> dict[str, Any]:
    onnx.checker.check_model(graph, full_check=True)
    inputs = {value.name: _shape(value) for value in graph.graph.input}
    outputs = {value.name: _shape(value) for value in graph.graph.output}
    if inputs != {"images": (3, 3, imgsz, imgsz)}:
        raise ValueError(f"unexpected ONNX inputs: {inputs}")
    if outputs != {"output0": (3, 300, 6)}:
        raise ValueError(f"unexpected ONNX outputs: {outputs}")
    for value in list(graph.graph.input) + list(graph.graph.output):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"{value.name} must have FLOAT I/O")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
        raise ValueError("external ONNX tensor files are unsupported by this export contract")
    if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
        raise ValueError("YOLO26 must retain its NMS-free end-to-end detection head")
    return {
        "inputs": {name: list(shape) for name, shape in inputs.items()},
        "outputs": {name: list(shape) for name, shape in outputs.items()},
        "io_dtype": "float32",
        "output_columns": ["x1", "y1", "x2", "y2", "score", "class_id"],
        "nms": False,
        "external_tensor_count": 0,
    }


def _cpu_smoke(onnx_path: Path, imgsz: int, classes: int) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    pixels = np.empty((3, 3, imgsz, imgsz), dtype=np.float32)
    pixels[0].fill(0.0)
    pixels[1].fill(0.5)
    pixels[2] = np.random.default_rng(26960).random((3, imgsz, imgsz), dtype=np.float32)
    result = session.run(["output0"], {"images": pixels})[0]
    if result.shape != (3, 300, 6) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError(f"invalid CPU inference result: {result.shape}, {result.dtype}")
    if not ((result[..., 4] >= 0.0) & (result[..., 4] <= 1.0)).all():
        raise ValueError("CPU inference scores are outside [0,1]")
    class_ids = result[..., 5]
    if not ((class_ids == np.rint(class_ids)) & (class_ids >= 0) & (class_ids < classes)).all():
        raise ValueError("CPU inference class IDs do not match the checkpoint")
    if not ((result[..., 2] >= result[..., 0]) & (result[..., 3] >= result[..., 1])).all():
        raise ValueError("CPU inference boxes do not follow the xyxy contract")
    return {
        "provider": session.get_providers(),
        "shape": list(result.shape),
        "all_finite": True,
        "scores_in_0_1": True,
        "valid_class_ids": True,
        "xyxy_boxes": True,
        "input": "zeros, constant 0.5, seeded noise (26960)",
        "onnxruntime_version": ort.__version__,
        "quality_acceptance": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Exact existing .pt checkpoint; never downloaded")
    parser.add_argument("--model-id", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960, help="Square input resolution; multiple of 32")
    parser.add_argument("--batch", type=int, choices=(3,), default=3)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-cpu-smoke", action="store_true", help="Validate graph only and mark synthetic inference unverified")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.imgsz < 128 or args.imgsz % 32 or args.imgsz > 1920:
        raise ValueError("--imgsz must be a multiple of 32 between 128 and 1920")
    source = args.source.expanduser().resolve(strict=True)
    if source.suffix != ".pt" or not source.is_file() or source.stat().st_size <= 0:
        raise ValueError("--source must name a nonempty existing .pt checkpoint")
    expected = CHECKPOINTS[args.model_id]
    source_hash = _sha256(source)
    if source_hash != expected["sha256"]:
        raise ValueError(f"source checkpoint hash mismatch for {args.model_id}: {source_hash}")
    root = _output_root(args.output_root.expanduser())
    stem = f"{args.model_id}_{args.imgsz}_b3"
    output = root / "models" / "onnx" / f"{stem}.onnx"
    receipt_path = output.with_name(output.name + ".provenance.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError(f"resolution artifact already exists; use a new output root: {output}")

    # Configure CPU/offline mode before importing anything that can access CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    workspace_root = root / "models" / "export_work"
    workspace_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=stem + "-", dir=workspace_root) as raw_workspace:
        workspace = Path(raw_workspace)
        config_dir = workspace / "ultralytics-config"
        config_dir.mkdir()
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

        import onnx
        import torch
        import ultralytics
        from ultralytics import YOLO

        if ultralytics.__version__ != expected["export_version"]:
            raise ValueError(
                f"Ultralytics {expected['export_version']} required for {args.model_id}; "
                f"found {ultralytics.__version__}"
            )
        if torch.cuda.is_available() or torch.cuda.device_count() != 0:
            raise RuntimeError("CPU export guard failed: CUDA devices remain visible")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        staged = workspace / f"{stem}.pt"
        shutil.copy2(source, staged)
        if _sha256(staged) != source_hash:
            raise ValueError("staged checkpoint hash mismatch")
        model = YOLO(str(staged), task="detect")
        head = model.model.model[-1]
        if model.task != "detect" or not getattr(head, "end2end", False):
            raise ValueError("checkpoint does not have the YOLO26 NMS-free detection head")
        if model.model.yaml.get("scale") != expected["scale"] or head.nc != expected["classes"]:
            raise ValueError("checkpoint model scale/classes differ from the requested model ID")
        if model.names.get(0) != "person":
            raise ValueError("checkpoint class 0 is not person")
        export_args = {
            "format": "onnx", "imgsz": args.imgsz, "batch": 3,
            "dynamic": False, "half": False, "simplify": False,
            "opset": args.opset, "device": "cpu", "nms": False,
            "max_det": 300, "optimize": False,
        }
        exported = Path(str(model.export(**export_args))).resolve(strict=True)
        if workspace not in exported.parents or exported.stat().st_size <= 0:
            raise ValueError("export did not produce an ONNX inside the owned workspace")
        graph = onnx.load(str(exported), load_external_data=False)
        contract = _validate_graph(onnx, graph, args.imgsz)
        smoke = {"status": "not_run", "quality_acceptance": False} if args.skip_cpu_smoke else _cpu_smoke(exported, args.imgsz, head.nc)
        if _sha256(source) != source_hash:
            raise ValueError("original checkpoint changed during export")
        provenance = {
            "schema": "noesis-yolo26-resolution-export.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": args.model_id,
            "source": {"path": str(source), "sha256": source_hash, "bytes": source.stat().st_size, "unchanged": True},
            "checkpoint_metadata": {
                "ultralytics_version": model.ckpt.get("version"),
                "date": model.ckpt.get("date"),
                "epoch": model.ckpt.get("epoch"),
                "model_scale": model.model.yaml.get("scale"),
                "class_count": head.nc,
                "end2end": bool(head.end2end),
            },
            "output": {"path": str(output), "sha256": _sha256(exported), "bytes": exported.stat().st_size},
            "export": export_args,
            "contract": contract,
            "classes": model.names,
            "cpu_smoke": smoke,
            "runtime_selection_changed": False,
            "engine_built": False,
            "quality_acceptance": False,
            "versions": {"python": platform.python_version(), "executable": sys.executable,
                         "torch": torch.__version__, "ultralytics": ultralytics.__version__, "onnx": onnx.__version__},
            "limitations": ["Synthetic validation does not establish real-image numerical parity or detection quality."],
        }
        staged_receipt = workspace / receipt_path.name
        staged_receipt.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Atomic, exclusive links publish complete files without overwriting any
        # pre-existing 640/960 artifact. Workspace and output share an artifact
        # filesystem; TemporaryDirectory later removes only its own link names.
        created_output = False
        try:
            os.link(exported, output)
            created_output = True
            os.link(staged_receipt, receipt_path)
        except Exception:
            if created_output:
                output.unlink(missing_ok=True)
            raise
        print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
