#!/usr/bin/env python3
"""
Collects GPU / CUDA / TensorRT / DeepStream environment details and prints them as JSON.
This helps document the exact runtime used during MapAnything ONNX → TensorRT workflows.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _run_command(cmd: str) -> Optional[str]:
    try:
        result = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL
        ).decode("utf-8", "ignore")
        return result.strip() or None
    except Exception:
        return None


def _import_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    except Exception:
        return None
    return None


def gather_environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    info["python"] = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }

    info["system"] = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    info["packages"] = {
        "torch": _import_version("torch"),
        "onnx": _import_version("onnx"),
        "onnxruntime": _import_version("onnxruntime"),
        "tensorrt": _import_version("tensorrt"),
    }

    info["cuda"] = {
        "nvcc": _run_command("nvcc --version"),
        "cuda_home": str(Path("/usr/local/cuda")) if Path("/usr/local/cuda").exists() else None,
    }

    info["tensorRT"] = {
        "dpkg": _run_command("dpkg -l | grep TensorRT"),
        "trtexec_version": _run_command("trtexec --version"),
    }

    info["deepstream"] = {
        "version_all": _run_command("deepstream-app --version-all"),
    }

    gpu_query = (
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
    )
    info["gpu"] = _run_command(gpu_query)

    return info


def main() -> None:
    data = gather_environment()
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
