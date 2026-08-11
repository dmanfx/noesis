#!/usr/bin/env python3
"""Guarded DS8 YOLO26-seg TensorRT engine maintenance for n/s/m."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SIZES = ("n", "s", "m")

# This maintenance path is deliberately tied to the restored canonical DS8
# build stack. A future SDK/driver migration must update and re-review it.
EXPECTED_TENSORRT_VERSION = "10.13.3.9"
EXPECTED_TRTEXEC_TAG = "TensorRT v101303"
EXPECTED_TRTEXEC_BUILD = "[b9]"
EXPECTED_CUDA_SDK_VERSION = "13.0.2"
EXPECTED_CUDA_RUNTIME_VERSION = "13.0.96"
EXPECTED_DRIVER_VERSION = "595.71.05"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3060"
EXPECTED_GPU_COMPUTE_CAPABILITY = "8.6"
EXPECTED_GPU_MEMORY_MIB = 12288
GPU_INDEX = 0

BATCH_SIZE = 3
INPUT_NAME = "images"
INPUT_SHAPE = (3, 3, 640, 640)
OUTPUT_NAME = "output0"
OUTPUT_SHAPE = (3, 30, 4102)
WORKSPACE_MIB = 4096
BUILDER_OPTIMIZATION_LEVEL = 3
MAX_AUX_STREAMS = 0
GPU_MEMORY_GUARD_MIB = 11000
MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024
MAX_LOG_BYTES = 32 * 1024 * 1024
MIN_ENGINE_BYTES = 1024 * 1024
LOAD_TIMEOUT_SECONDS = 120

MAX_ENGINE_BYTES = {
    "n": 64 * 1024 * 1024,
    "s": 128 * 1024 * 1024,
    "m": 256 * 1024 * 1024,
}
BUILD_TIMEOUT_SECONDS = {"n": 900, "s": 1200, "m": 1800}

_FAILURE_PATTERNS = (
    re.compile(r"(?m)\[E\]"),
    re.compile(r"(?i)Error\[\d+\]"),
    re.compile(r"(?i)Error Code\s+[1-9]\d*"),
    re.compile(r"(?i)Engine deserialization failed"),
    re.compile(r"(?i)Deserialize engine failed"),
    re.compile(r"(?i)failed to deserialize"),
    re.compile(r"(?i)failed to load (?:the )?engine"),
    re.compile(r"(?i)&&&&\s+FAILED\s+TensorRT\.trtexec"),
)


class MaintenanceError(RuntimeError):
    """A fail-closed maintenance contract violation."""


@dataclass(frozen=True)
class EngineSpec:
    size: str
    onnx_path: Path
    engine_path: Path
    max_engine_bytes: int
    build_timeout_seconds: int


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str
    duration_seconds: float
    max_gpu_memory_mib: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _parse_sizes(raw: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(item.strip().lower() for item in raw.split(",") if item.strip()))
    invalid = [item for item in values if item not in SUPPORTED_SIZES]
    if not values or invalid:
        raise MaintenanceError(
            f"--sizes must be a non-empty subset of {','.join(SUPPORTED_SIZES)}; "
            f"invalid={invalid}"
        )
    return values


def _validate_run_id(raw: str) -> str:
    value = str(raw or "").strip()
    if not value or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value) is None:
        raise MaintenanceError("run id must contain only 1-96 safe filename characters")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str, *, max_bytes: int | None = None) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MaintenanceError(f"{label} is missing or empty: {path}")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise MaintenanceError(
            f"{label} exceeds the {max_bytes}-byte safety bound: "
            f"size={path.stat().st_size} path={path}"
        )


def _engine_spec(repo_root: Path, size: str) -> EngineSpec:
    return EngineSpec(
        size=size,
        onnx_path=repo_root / "models" / f"yolo26{size}-seg_fused.onnx",
        engine_path=repo_root / "models" / "engines" / f"yolo26{size}-seg_fused_b3_fp16.engine",
        max_engine_bytes=MAX_ENGINE_BYTES[size],
        build_timeout_seconds=BUILD_TIMEOUT_SECONDS[size],
    )


def _shape(value: Any) -> tuple[int | str, ...]:
    dims: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(str(dim.dim_param))
    return tuple(dims)


def _validate_onnx_contract(path: Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise MaintenanceError("the onnx Python package is required for source validation") from exc

    model = onnx.load(str(path), load_external_data=False)
    opsets = {str(item.domain): int(item.version) for item in model.opset_import}
    inputs = {item.name: _shape(item) for item in model.graph.input}
    outputs = {item.name: _shape(item) for item in model.graph.output}
    input_types = {item.name: int(item.type.tensor_type.elem_type) for item in model.graph.input}
    output_types = {item.name: int(item.type.tensor_type.elem_type) for item in model.graph.output}
    if opsets.get("") != 18:
        raise MaintenanceError(f"YOLO26-seg ONNX must use opset 18: {path} opsets={opsets}")
    if inputs != {INPUT_NAME: INPUT_SHAPE}:
        raise MaintenanceError(f"unexpected YOLO26-seg input contract: {path} inputs={inputs}")
    if outputs != {OUTPUT_NAME: OUTPUT_SHAPE}:
        raise MaintenanceError(f"unexpected YOLO26-seg output contract: {path} outputs={outputs}")
    if input_types != {INPUT_NAME: int(onnx.TensorProto.FLOAT)}:
        raise MaintenanceError(f"YOLO26-seg input must be FLOAT: {path} types={input_types}")
    if output_types != {OUTPUT_NAME: int(onnx.TensorProto.FLOAT)}:
        raise MaintenanceError(f"YOLO26-seg output must be FLOAT: {path} types={output_types}")
    return {
        "opset": 18,
        "input": {"name": INPUT_NAME, "shape": list(INPUT_SHAPE), "type": "float32"},
        "output": {"name": OUTPUT_NAME, "shape": list(OUTPUT_SHAPE), "type": "float32"},
    }


def _validate_template_contract(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    required_once = ("@ONNX_PATH@", "@ENGINE_PATH@", "@LABELS_PATH@", "@CUSTOM_LIB@")
    for token in required_once:
        if text.count(token) != 1:
            raise MaintenanceError(f"template must contain {token} exactly once: {template_path}")
    required_lines = (
        "batch-size=3",
        "network-type=3",
        "network-mode=2",
        "disable-output-host-copy=1",
        "output-blob-names=output0",
        "parse-bbox-instance-mask-func-name=NvDsInferParseYolo26Seg",
        "topk=30",
    )
    missing = [line for line in required_lines if line not in text]
    if missing:
        raise MaintenanceError(f"template contract drift: missing={missing} path={template_path}")


def _resolve_executable(raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        path = candidate
    else:
        resolved = shutil.which(raw)
        if not resolved:
            raise MaintenanceError(f"executable not found on PATH: {raw}")
        path = Path(resolved)
    if not path.exists() or not os.access(path, os.X_OK):
        raise MaintenanceError(f"executable is missing or not executable: {path}")
    return path.resolve()


def _cuda_versions(cuda_home: Path) -> tuple[str, str]:
    version_path = cuda_home / "version.json"
    _require_file(version_path, "CUDA version manifest", max_bytes=4 * 1024 * 1024)
    payload = json.loads(version_path.read_text(encoding="utf-8"))
    sdk = str((payload.get("cuda") or {}).get("version") or "")
    runtime = str((payload.get("cuda_cudart") or {}).get("version") or "")
    if sdk != EXPECTED_CUDA_SDK_VERSION or runtime != EXPECTED_CUDA_RUNTIME_VERSION:
        raise MaintenanceError(
            "DS8 CUDA profile mismatch: "
            f"expected sdk={EXPECTED_CUDA_SDK_VERSION} runtime={EXPECTED_CUDA_RUNTIME_VERSION}; "
            f"observed sdk={sdk} runtime={runtime}"
        )
    return sdk, runtime


def _host_static_profile(cuda_home: Path, trtexec: Path) -> dict[str, Any]:
    try:
        tensorrt_version = importlib.metadata.version("tensorrt")
    except importlib.metadata.PackageNotFoundError as exc:
        raise MaintenanceError("TensorRT Python distribution is not installed") from exc
    if tensorrt_version != EXPECTED_TENSORRT_VERSION:
        raise MaintenanceError(
            f"DS8 TensorRT profile mismatch: expected {EXPECTED_TENSORRT_VERSION}, "
            f"observed {tensorrt_version}"
        )
    cuda_sdk, cuda_runtime = _cuda_versions(cuda_home)
    return {
        "deepstream": "8.0",
        "tensorrt_python": tensorrt_version,
        "trtexec_path": str(trtexec),
        "trtexec_sha256": _sha256(trtexec),
        "cuda_home": str(cuda_home),
        "cuda_sdk": cuda_sdk,
        "cuda_runtime": cuda_runtime,
        "driver": EXPECTED_DRIVER_VERSION,
        "gpu_index": GPU_INDEX,
        "gpu_name": EXPECTED_GPU_NAME,
        "gpu_compute_capability": EXPECTED_GPU_COMPUTE_CAPABILITY,
        "gpu_memory_mib": EXPECTED_GPU_MEMORY_MIB,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    _require_file(path, "maintenance input")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _build_command(trtexec: Path, spec: EngineSpec, temporary_engine: Path) -> list[str]:
    return [
        str(trtexec),
        f"--onnx={spec.onnx_path}",
        "--fp16",
        f"--memPoolSize=workspace:{WORKSPACE_MIB}",
        f"--builderOptimizationLevel={BUILDER_OPTIMIZATION_LEVEL}",
        f"--maxAuxStreams={MAX_AUX_STREAMS}",
        f"--saveEngine={temporary_engine}",
        "--skipInference",
    ]


def _load_command(trtexec: Path, temporary_engine: Path) -> list[str]:
    return [str(trtexec), f"--loadEngine={temporary_engine}", "--skipInference"]


def _collect_plan(
    *,
    repo_root: Path,
    sizes: tuple[str, ...],
    evidence_root: Path,
    run_id: str,
    trtexec: Path,
    cuda_home: Path,
) -> dict[str, Any]:
    template = repo_root / "pipelines" / "config_infer_primary_yolo26_seg.template.ini"
    parser_binary = repo_root / "pipelines" / "nvdsinfer_yolo26_seg" / "libnvdsinfer_yolo26_seg.so"
    parser_source = repo_root / "pipelines" / "nvdsinfer_yolo26_seg" / "nvdsinfer_yolo26_seg.cpp"
    preprocess_template = repo_root / "pipelines" / "config_preproc.ini"
    labels = repo_root / "models" / "coco_labels.txt"
    for path, label, bound in (
        (template, "YOLO26-seg nvinfer template", 1024 * 1024),
        (parser_binary, "YOLO26-seg parser binary", 32 * 1024 * 1024),
        (parser_source, "YOLO26-seg parser source", 4 * 1024 * 1024),
        (preprocess_template, "YOLO26 preprocess template", 1024 * 1024),
        (labels, "COCO labels", 1024 * 1024),
    ):
        _require_file(path, label, max_bytes=bound)
    _validate_template_contract(template)

    specs = [_engine_spec(repo_root, size) for size in sizes]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        _require_file(spec.onnx_path, f"YOLO26{spec.size}-seg ONNX", max_bytes=1024 * 1024 * 1024)
        source = _artifact_record(spec.onnx_path)
        source["contract"] = _validate_onnx_contract(spec.onnx_path)
        prior: dict[str, Any] = {"exists": False, "path": str(spec.engine_path)}
        if spec.engine_path.exists():
            _require_file(
                spec.engine_path,
                f"existing YOLO26{spec.size}-seg engine",
                max_bytes=spec.max_engine_bytes,
            )
            prior.update(
                {
                    "exists": True,
                    "size_bytes": spec.engine_path.stat().st_size,
                    "sha256": _sha256(spec.engine_path),
                }
            )
        temporary = spec.engine_path.with_name(f".{spec.engine_path.name}.building-{run_id}")
        rows.append(
            {
                "size": spec.size,
                "source": source,
                "target": str(spec.engine_path),
                "temporary": str(temporary),
                "prior": prior,
                "max_engine_bytes": spec.max_engine_bytes,
                "build_timeout_seconds": spec.build_timeout_seconds,
                "build_command": _build_command(trtexec, spec, temporary),
                "load_command": _load_command(trtexec, temporary),
                "installed_load_command": _load_command(trtexec, spec.engine_path),
            }
        )

    return {
        "schema_version": 1,
        "contract": "noesis.ds8.yolo26_seg_engine_maintenance",
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "repo_root": str(repo_root),
        "evidence_root": str(evidence_root),
        "profile": {
            **_host_static_profile(cuda_home, trtexec),
            "precision": "fp16_with_fp32_fallback",
            "batch_size": BATCH_SIZE,
            "input": {"name": INPUT_NAME, "shape": list(INPUT_SHAPE)},
            "output": {"name": OUTPUT_NAME, "shape": list(OUTPUT_SHAPE)},
            "workspace_mib": WORKSPACE_MIB,
            "builder_optimization_level": BUILDER_OPTIMIZATION_LEVEL,
            "max_aux_streams": MAX_AUX_STREAMS,
            "gpu_memory_guard_mib": GPU_MEMORY_GUARD_MIB,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "max_log_bytes": MAX_LOG_BYTES,
        },
        "shared_inputs": {
            "nvinfer_template": _artifact_record(template),
            "parser_binary": _artifact_record(parser_binary),
            "parser_source": _artifact_record(parser_source),
            "preprocess_template": _artifact_record(preprocess_template),
            "labels": _artifact_record(labels),
        },
        "engines": rows,
    }


def _print_plan(plan: dict[str, Any]) -> None:
    print("[PLAN] DS8 YOLO26-seg engine maintenance (CPU/read-only; no subprocesses run)")
    print(f"[PLAN] run_id={plan['run_id']} evidence_root={plan['evidence_root']}")
    profile = plan["profile"]
    print(
        "[PLAN] profile "
        f"TensorRT={profile['tensorrt_python']} CUDA={profile['cuda_sdk']} "
        f"driver={profile['driver']} GPU={profile['gpu_name']} cc={profile['gpu_compute_capability']} "
        f"batch={profile['batch_size']} precision={profile['precision']}"
    )
    for name, record in plan["shared_inputs"].items():
        print(f"[PLAN] shared {name} sha256={record['sha256']} path={record['path']}")
    for row in plan["engines"]:
        prior = row["prior"]
        prior_text = prior.get("sha256", "missing")
        print(
            f"[PLAN] size={row['size']} source_sha256={row['source']['sha256']} "
            f"prior_sha256={prior_text}"
        )
        print(f"[PLAN] build: {shlex.join(row['build_command'])}")
        print(f"[PLAN] validate: {shlex.join(row['load_command'])}")
        print(f"[PLAN] validate-installed: {shlex.join(row['installed_load_command'])}")
    print("[PLAN] real mode preserves prior bytes, builds to a sibling temporary file, requires")
    print("[PLAN] positive deserialization evidence, then atomically installs. Nothing changed.")


def _nvidia_smi() -> str:
    binary = shutil.which("nvidia-smi")
    if not binary:
        raise MaintenanceError("nvidia-smi is required for guarded real builds")
    return binary


def _run_capture(command: Sequence[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _query_gpu_profile() -> dict[str, Any]:
    proc = _run_capture(
        [
            _nvidia_smi(),
            f"--id={GPU_INDEX}",
            "--query-gpu=index,name,uuid,compute_cap,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        raise MaintenanceError(f"unable to query GPU profile: {(proc.stderr or proc.stdout).strip()}")
    values = [item.strip() for item in proc.stdout.strip().split(",")]
    if len(values) != 6:
        raise MaintenanceError(f"unexpected nvidia-smi GPU profile: {proc.stdout!r}")
    profile = {
        "index": int(values[0]),
        "name": values[1],
        "uuid": values[2],
        "compute_capability": values[3],
        "driver": values[4],
        "memory_total_mib": int(values[5]),
    }
    expected = {
        "index": GPU_INDEX,
        "name": EXPECTED_GPU_NAME,
        "compute_capability": EXPECTED_GPU_COMPUTE_CAPABILITY,
        "driver": EXPECTED_DRIVER_VERSION,
        "memory_total_mib": EXPECTED_GPU_MEMORY_MIB,
    }
    mismatches = {key: (expected[key], profile[key]) for key in expected if profile[key] != expected[key]}
    if mismatches:
        raise MaintenanceError(f"DS8 GPU build profile mismatch: {mismatches}")
    return profile


def _query_compute_owners() -> list[dict[str, Any]]:
    proc = _run_capture(
        [
            _nvidia_smi(),
            f"--id={GPU_INDEX}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        raise MaintenanceError(f"unable to query GPU compute owners: {(proc.stderr or proc.stdout).strip()}")
    owners: list[dict[str, Any]] = []
    for raw in proc.stdout.splitlines():
        if not raw.strip() or raw.lower().startswith("no running"):
            continue
        values = [item.strip() for item in raw.split(",", 2)]
        if len(values) != 3:
            raise MaintenanceError(f"unexpected compute-owner row: {raw!r}")
        owners.append({"pid": int(values[0]), "process_name": values[1], "used_memory_mib": int(values[2])})
    return owners


def _require_no_compute_owners(*, allowed_pid: int | None = None) -> None:
    unexpected = [row for row in _query_compute_owners() if row["pid"] != allowed_pid]
    if unexpected:
        raise MaintenanceError(f"GPU compute owner(s) are active: {unexpected}")


def _query_gpu_memory_used() -> int:
    proc = _run_capture(
        [
            _nvidia_smi(),
            f"--id={GPU_INDEX}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        raise MaintenanceError(f"unable to query GPU memory: {(proc.stderr or proc.stdout).strip()}")
    return int(proc.stdout.strip().splitlines()[0])


def _validate_trtexec_profile_probe(proc: subprocess.CompletedProcess[str]) -> str:
    output = (proc.stdout or "") + (proc.stderr or "")
    failures = _failure_signatures(output)
    if (
        proc.returncode != 0
        or EXPECTED_TRTEXEC_TAG not in output
        or EXPECTED_TRTEXEC_BUILD not in output
        or failures
    ):
        raise MaintenanceError(
            "trtexec does not match the exact DS8 TensorRT 10.13.3.9 b9 profile: "
            f"returncode={proc.returncode} signatures={failures}"
        )
    return f"{EXPECTED_TRTEXEC_TAG} {EXPECTED_TRTEXEC_BUILD}"


def _validate_real_platform(plan: dict[str, Any], trtexec: Path) -> dict[str, Any]:
    _require_no_compute_owners()
    gpu = _query_gpu_profile()
    # TensorRT 10.13 trtexec has no functional --version mode: it prints the
    # version banner, then fails with "Model missing". --help emits the same
    # exact binary banner and exits successfully without creating a CUDA build.
    proc = _run_capture([str(trtexec), "--help"], timeout=30)
    version_evidence = _validate_trtexec_profile_probe(proc)
    if _sha256(trtexec) != plan["profile"]["trtexec_sha256"]:
        raise MaintenanceError("trtexec bytes changed after plan collection")
    _require_no_compute_owners()
    return {"gpu": gpu, "trtexec_version_probe": version_evidence}


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise MaintenanceError(f"no existing ancestor for path: {path}")
        current = parent
    return current


def _require_capacity(plan: dict[str, Any]) -> None:
    requirements: dict[int, int] = {}
    samples: dict[int, Path] = {}

    def add(path: Path, amount: int) -> None:
        sample = _existing_ancestor(path)
        device = sample.stat().st_dev
        requirements[device] = requirements.get(device, 0) + amount
        samples[device] = sample

    evidence_root = Path(plan["evidence_root"])
    for row in plan["engines"]:
        add(Path(row["target"]).parent, int(row["max_engine_bytes"]))
        prior_bytes = int(row["prior"].get("size_bytes", 0))
        add(evidence_root, prior_bytes + 3 * MAX_LOG_BYTES + 1024 * 1024)
    for device, required_growth in requirements.items():
        available = shutil.disk_usage(samples[device]).free
        required = required_growth + MIN_FREE_BYTES
        if available < required:
            raise MaintenanceError(
                "insufficient maintenance capacity: "
                f"device={device} available={available} required={required} "
                f"residual_headroom={MIN_FREE_BYTES}"
            )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    source_hash = _sha256(source)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    try:
        with source.open("rb") as read_handle, temporary.open("xb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle, length=4 * 1024 * 1024)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        if _sha256(temporary) != source_hash:
            raise MaintenanceError(f"prior-engine preservation hash mismatch: {source}")
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "size_bytes": destination.stat().st_size, "sha256": source_hash}


def _terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def _run_bounded(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
    env: dict[str, str],
) -> CommandResult:
    _require_no_compute_owners()
    started = time.monotonic()
    max_memory = 0
    with log_path.open("xb") as log_handle:
        proc = subprocess.Popen(
            [str(value) for value in command],
            cwd=str(cwd),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > timeout_seconds:
                    raise MaintenanceError(f"command exceeded {timeout_seconds}s timeout")
                if log_path.stat().st_size > MAX_LOG_BYTES:
                    raise MaintenanceError(f"command log exceeded {MAX_LOG_BYTES} bytes")
                _require_no_compute_owners(allowed_pid=proc.pid)
                max_memory = max(max_memory, _query_gpu_memory_used())
                if max_memory > GPU_MEMORY_GUARD_MIB:
                    raise MaintenanceError(
                        f"GPU memory exceeded {GPU_MEMORY_GUARD_MIB} MiB guard: {max_memory} MiB"
                    )
                time.sleep(1.0)
        except Exception:
            _terminate_process(proc)
            raise
        returncode = int(proc.wait())
    if log_path.stat().st_size > MAX_LOG_BYTES:
        raise MaintenanceError(f"command log exceeded {MAX_LOG_BYTES} bytes")
    output = log_path.read_text(encoding="utf-8", errors="replace")
    return CommandResult(
        returncode=returncode,
        output=output,
        duration_seconds=time.monotonic() - started,
        max_gpu_memory_mib=max_memory,
    )


def _failure_signatures(output: str) -> list[str]:
    return [pattern.pattern for pattern in _FAILURE_PATTERNS if pattern.search(output)]


def _validate_build_result(result: CommandResult, temporary: Path, max_bytes: int) -> None:
    failures = _failure_signatures(result.output)
    if result.returncode != 0 or failures:
        raise MaintenanceError(
            f"TensorRT build failed: returncode={result.returncode} signatures={failures}"
        )
    _require_file(temporary, "new TensorRT engine", max_bytes=max_bytes)
    if temporary.stat().st_size < MIN_ENGINE_BYTES:
        raise MaintenanceError(
            f"new TensorRT engine is implausibly small: {temporary.stat().st_size} bytes"
        )


def _validate_load_result(result: CommandResult) -> None:
    failures = _failure_signatures(result.output)
    positive = {
        "loaded_engine_size": "Loaded engine size:" in result.output,
        "engine_deserialized": "Engine deserialized in" in result.output,
        "inference_skipped": "Skipped inference phase since --skipInference is added." in result.output,
    }
    if result.returncode != 0 or failures or not all(positive.values()):
        raise MaintenanceError(
            "TensorRT load validation failed: "
            f"returncode={result.returncode} signatures={failures} positive={positive}"
        )


def _verify_plan_inputs(plan: dict[str, Any]) -> None:
    for record in plan["shared_inputs"].values():
        path = Path(record["path"])
        if _sha256(path) != record["sha256"]:
            raise MaintenanceError(f"shared maintenance input changed after planning: {path}")
    for row in plan["engines"]:
        path = Path(row["source"]["path"])
        if _sha256(path) != row["source"]["sha256"]:
            raise MaintenanceError(f"ONNX source changed after planning: {path}")


def _clean_build_env(cuda_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        env.pop(name, None)
    env["CUDA_HOME"] = str(cuda_home)
    return env


@contextmanager
def _maintenance_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MaintenanceError(f"another YOLO26-seg engine maintenance run holds {path}") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _execute_plan_locked(plan: dict[str, Any], *, trtexec: Path, cuda_home: Path) -> Path:
    platform = _validate_real_platform(plan, trtexec)
    _verify_plan_inputs(plan)
    _require_no_compute_owners()
    _require_capacity(plan)

    evidence_root = Path(plan["evidence_root"])
    evidence_root.mkdir(parents=True, exist_ok=True)
    run_dir = evidence_root / str(plan["run_id"])
    run_dir.mkdir(mode=0o700, exist_ok=False)
    prior_dir = run_dir / "prior"
    logs_dir = run_dir / "logs"
    prior_dir.mkdir(mode=0o700)
    logs_dir.mkdir(mode=0o700)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        **plan,
        "mode": "build",
        "status": "running",
        "started_at_utc": _utc_now(),
        "observed_platform": platform,
    }
    _atomic_write_json(manifest_path, manifest)
    active_temporary: Path | None = None
    try:
        for row in manifest["engines"]:
            target = Path(row["target"])
            planned_prior = row["prior"]
            if planned_prior.get("exists"):
                current_hash = _sha256(target)
                if current_hash != planned_prior["sha256"]:
                    raise MaintenanceError(f"prior engine changed after planning: {target}")
                row["preserved_prior"] = _copy_verified(target, prior_dir / target.name)
            else:
                row["preserved_prior"] = {"exists": False}
        _atomic_write_json(manifest_path, manifest)

        env = _clean_build_env(cuda_home)
        for row in manifest["engines"]:
            size = str(row["size"])
            temporary = Path(row["temporary"])
            target = Path(row["target"])
            active_temporary = temporary
            if temporary.exists():
                raise MaintenanceError(f"refusing pre-existing temporary engine: {temporary}")
            target.parent.mkdir(parents=True, exist_ok=True)
            build_result = _run_bounded(
                row["build_command"],
                cwd=Path(plan["repo_root"]),
                log_path=logs_dir / f"build-{size}.log",
                timeout_seconds=int(row["build_timeout_seconds"]),
                env=env,
            )
            _validate_build_result(build_result, temporary, int(row["max_engine_bytes"]))
            new_hash = _sha256(temporary)
            load_result = _run_bounded(
                row["load_command"],
                cwd=Path(plan["repo_root"]),
                log_path=logs_dir / f"load-{size}.log",
                timeout_seconds=LOAD_TIMEOUT_SECONDS,
                env=env,
            )
            _validate_load_result(load_result)
            _require_no_compute_owners()
            row["result"] = {
                "sha256": new_hash,
                "size_bytes": temporary.stat().st_size,
                "build_duration_seconds": build_result.duration_seconds,
                "candidate_load_duration_seconds": load_result.duration_seconds,
                "max_gpu_memory_mib": max(
                    build_result.max_gpu_memory_mib, load_result.max_gpu_memory_mib
                ),
                "build_log": str(logs_dir / f"build-{size}.log"),
                "candidate_load_log": str(logs_dir / f"load-{size}.log"),
            }
            os.replace(temporary, target)
            active_temporary = None
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            if _sha256(target) != new_hash:
                raise MaintenanceError(f"installed engine hash mismatch: {target}")
            installed_load_result = _run_bounded(
                row["installed_load_command"],
                cwd=Path(plan["repo_root"]),
                log_path=logs_dir / f"load-installed-{size}.log",
                timeout_seconds=LOAD_TIMEOUT_SECONDS,
                env=env,
            )
            _validate_load_result(installed_load_result)
            _require_no_compute_owners()
            row["result"]["installed_load_duration_seconds"] = (
                installed_load_result.duration_seconds
            )
            row["result"]["installed_load_log"] = str(
                logs_dir / f"load-installed-{size}.log"
            )
            row["result"]["max_gpu_memory_mib"] = max(
                int(row["result"]["max_gpu_memory_mib"]),
                installed_load_result.max_gpu_memory_mib,
            )
            row["result"]["installed_at_utc"] = _utc_now()
            _atomic_write_json(manifest_path, manifest)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        return manifest_path
    except Exception as exc:
        if active_temporary is not None:
            active_temporary.unlink(missing_ok=True)
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = _utc_now()
        manifest["error"] = str(exc)
        _atomic_write_json(manifest_path, manifest)
        raise


def _execute_plan(plan: dict[str, Any], *, trtexec: Path, cuda_home: Path) -> Path:
    lock_path = (
        Path(plan["repo_root"])
        / "models"
        / "engines"
        / ".noesis-yolo26-seg-engine-maintenance.lock"
    )
    with _maintenance_lock(lock_path):
        return _execute_plan_locked(plan, trtexec=trtexec, cuda_home=cuda_home)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--plan",
        "--dry-run",
        dest="plan",
        action="store_true",
        help="Hash and validate inputs, then print exact commands without subprocesses or writes.",
    )
    mode.add_argument(
        "--build",
        action="store_true",
        help="Run the guarded exclusive-GPU build after explicit authorization.",
    )
    parser.add_argument("--sizes", default="n,s,m", help="Comma-separated subset of n,s,m.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--trtexec", default=os.environ.get("TRTEXEC", "trtexec"))
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        repo_root = args.repo_root.expanduser().resolve()
        sizes = _parse_sizes(args.sizes)
        run_id = _validate_run_id(args.run_id or ("PLAN" if args.plan else _default_run_id()))
        evidence_root = (
            args.evidence_root.expanduser()
            if args.evidence_root is not None
            else repo_root / "models" / "engine_maintenance" / "yolo26_seg"
        )
        if not evidence_root.is_absolute():
            evidence_root = repo_root / evidence_root
        trtexec = _resolve_executable(args.trtexec)
        cuda_home = args.cuda_home.expanduser().resolve()
        plan = _collect_plan(
            repo_root=repo_root,
            sizes=sizes,
            evidence_root=evidence_root,
            run_id=run_id,
            trtexec=trtexec,
            cuda_home=cuda_home,
        )
        if args.plan:
            _print_plan(plan)
            return 0
        manifest = _execute_plan(plan, trtexec=trtexec, cuda_home=cuda_home)
        print(f"[OK] DS8 YOLO26-seg engine maintenance complete: {manifest}")
        return 0
    except (MaintenanceError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
