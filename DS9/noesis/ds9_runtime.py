#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent


def _ensure_runtime_sys_path() -> None:
    for value in (str(DS9_ROOT), str(REPO_ROOT)):
        if value in sys.path:
            sys.path.remove(value)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(DS9_ROOT))


def _prepend_env_path(name: str, value: Path) -> None:
    current = os.environ.get(name, "")
    value_str = str(value)
    parts = [value_str]
    if current:
        parts.append(current)
    os.environ[name] = os.pathsep.join(parts)


def _prepend_sys_path(value: Path) -> None:
    value_str = str(value)
    if value_str in sys.path:
        sys.path.remove(value_str)
    sys.path.insert(0, value_str)


def _ds9_system_python_site() -> Path:
    return Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")


def _truthy_env(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _set_ds9_environment() -> None:
    os.environ.setdefault("NOESIS_DEEPSTREAM_MAJOR", "9")
    os.environ.setdefault("NOESIS_DEEPSTREAM_HOME", "/opt/nvidia/deepstream/deepstream-9.0")
    os.environ.setdefault("NOESIS_MODEL_DIR", str(DS9_ROOT / "models"))
    os.environ.setdefault("NOESIS_ONNX_DIR", str(DS9_ROOT / "models" / "onnx"))
    os.environ.setdefault("NOESIS_ENGINE_DIR", str(DS9_ROOT / "models" / "engines"))
    os.environ.setdefault("NOESIS_PIPELINE_DIR", str(DS9_ROOT / "pipelines"))
    os.environ.setdefault("NOESIS_BUILD_DIR", str(DS9_ROOT / "build"))
    os.environ.setdefault("NOESIS_NATIVE_EXT_DIR", str(DS9_ROOT / "native_extensions"))
    os.environ.setdefault("NOESIS_NATIVE_BUILD_SCRIPT_DIR", str(DS9_ROOT / "scripts"))
    os.environ.setdefault("NOESIS_GST_PLUGIN_DIR", str(DS9_ROOT / "gst-plugins"))
    os.environ.setdefault("NOESIS_RFDETR_TRT_PLUGIN_LIB", str(DS9_ROOT / "plugins" / "libnvdsinfer_custom_impl_Yolo_seg.so"))
    os.environ.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "1")
    os.environ.setdefault("NOESIS_DEPTH_ENABLE_SECONDS", "0")

    ds9_site = _ds9_system_python_site()
    if (ds9_site / "pyservicemaker" / "_pydeepstream.so").exists():
        _prepend_env_path("PYTHONPATH", ds9_site)
        _prepend_sys_path(ds9_site)

    _prepend_env_path("PYTHONPATH", DS9_ROOT / "native_extensions")
    _prepend_sys_path(DS9_ROOT / "native_extensions")
    _prepend_env_path("LD_LIBRARY_PATH", Path(os.environ["NOESIS_DEEPSTREAM_HOME"]) / "lib")
    _prepend_env_path("GST_PLUGIN_PATH", Path(os.environ["NOESIS_GST_PLUGIN_DIR"]))


def _run_preflight() -> int:
    script = DS9_ROOT / "scripts" / "ds9_preflight.py"
    return subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT)).returncode


def _preload_ds9_python_runtime() -> None:
    if str(os.environ.get("NOESIS_DS9_PRELOAD_TORCH", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return
    try:
        import torch  # type: ignore
    except Exception as exc:
        print(f"[WARN] DS9 Torch preload skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return
    try:
        if bool(torch.cuda.is_available()):
            _ = torch.tensor([1.0], device="cuda:0")
    except Exception as exc:
        print(f"[WARN] DS9 Torch CUDA warmup skipped: {type(exc).__name__}: {exc}", file=sys.stderr)


def main() -> int:
    _ensure_runtime_sys_path()
    _set_ds9_environment()
    if _run_preflight() != 0:
        return 1

    _preload_ds9_python_runtime()

    from noesis.ds9_runtime_core import main as _runtime_main

    if "--pipeline-config" not in sys.argv:
        sys.argv.extend(["--pipeline-config", str(DS9_ROOT / "config" / "infer.yaml")])
    if "--cameras-config" not in sys.argv:
        sys.argv.extend(["--cameras-config", str(REPO_ROOT / "config" / "cameras.yaml")])
    if "--storage-base" not in sys.argv:
        sys.argv.extend(["--storage-base", str(DS9_ROOT / "data" / "depth")])
    return int(_runtime_main())


if __name__ == "__main__":
    _exit_code = int(main())
    if _truthy_env("NOESIS_DS9_BYPASS_NATIVE_GC_ON_EXIT", True):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_exit_code)
    raise SystemExit(_exit_code)
