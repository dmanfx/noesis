#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_CPU_MATH_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _configure_cpu_math_threads() -> None:
    raw_default = str(os.environ.get("NOESIS_CPU_MATH_THREADS", "1") or "1").strip()
    if raw_default.lower() in ("0", "off", "false", "no"):
        return
    try:
        default_threads = str(max(1, int(raw_default)))
    except Exception:
        default_threads = "1"
    for name in _CPU_MATH_THREAD_ENV_VARS:
        os.environ.setdefault(name, default_threads)


_configure_cpu_math_threads()


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
for _bootstrap_path in (str(DS9_ROOT), str(REPO_ROOT)):
    while _bootstrap_path in sys.path:
        sys.path.remove(_bootstrap_path)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DS9_ROOT))

from noesis.runtime_paths import (  # noqa: E402
    configure_ds9_runtime_import_paths,
    load_ds9_native_extensions,
    service_maker_system_site,
)
from noesis.native_artifact_provenance import (  # noqa: E402
    attest_ds9_native_artifacts,
)
from noesis_core.servicemaker_shutdown import (  # noqa: E402
    synthetic_stub_lifecycle_evidence,
)


_ENV_TRUE = {"1", "true", "yes", "y", "on"}


def _ensure_runtime_sys_path() -> None:
    configure_ds9_runtime_import_paths(
        ds9_root=DS9_ROOT,
        repo_root=REPO_ROOT,
        system_site=_ds9_system_python_site(),
    )


def _prepend_env_path(name: str, value: Path) -> None:
    current = os.environ.get(name, "")
    value_str = str(value)
    parts = [value_str]
    if current:
        parts.append(current)
    os.environ[name] = os.pathsep.join(parts)


def _ds9_system_python_site() -> Path | None:
    return service_maker_system_site()


def _configured_model_root() -> Path:
    raw = str(os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "") or "").strip()
    if not raw:
        return (DS9_ROOT / "models").resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise SystemExit(f"[FATAL] NOESIS_DS9_ARTIFACT_ROOT must be absolute: {raw}")
    artifact_root = candidate.resolve(strict=False)
    if artifact_root in {Path("/"), REPO_ROOT.resolve(), DS9_ROOT.resolve()}:
        raise SystemExit(f"[FATAL] refusing unsafe NOESIS_DS9_ARTIFACT_ROOT: {artifact_root}")
    try:
        artifact_root.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit(
            f"[FATAL] NOESIS_DS9_ARTIFACT_ROOT must not be inside the checkout: {artifact_root}"
        )
    return artifact_root / "models"


def _set_ds9_environment() -> None:
    model_root = _configured_model_root()
    os.environ.setdefault("NOESIS_DEEPSTREAM_MAJOR", "9")
    os.environ.setdefault("NOESIS_DEEPSTREAM_HOME", "/opt/nvidia/deepstream/deepstream-9.1")
    os.environ.setdefault("NOESIS_MODEL_DIR", str(model_root))
    os.environ.setdefault("NOESIS_ONNX_DIR", str(model_root / "onnx"))
    os.environ.setdefault("NOESIS_ENGINE_DIR", str(model_root / "engines"))
    os.environ.setdefault("NOESIS_PIPELINE_DIR", str(DS9_ROOT / "pipelines"))
    os.environ.setdefault("NOESIS_BUILD_DIR", str(DS9_ROOT / "build"))
    os.environ.setdefault("NOESIS_NATIVE_EXT_DIR", str(DS9_ROOT / "native_extensions"))
    os.environ.setdefault("NOESIS_NATIVE_BUILD_SCRIPT_DIR", str(DS9_ROOT / "scripts"))
    os.environ.setdefault("NOESIS_GST_PLUGIN_DIR", str(DS9_ROOT / "gst-plugins"))
    os.environ.setdefault("NOESIS_RFDETR_TRT_PLUGIN_LIB", str(DS9_ROOT / "plugins" / "libnvdsinfer_custom_impl_Yolo_seg.so"))
    os.environ.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "1")
    os.environ.setdefault("NOESIS_DEPTH_ENABLE_SECONDS", "0")

    ds9_site = _ds9_system_python_site()
    if (
        ds9_site is not None
        and (ds9_site / "pyservicemaker" / "_pydeepstream.so").exists()
    ):
        _prepend_env_path("PYTHONPATH", ds9_site)

    native_dir = Path(os.environ["NOESIS_NATIVE_EXT_DIR"])
    _prepend_env_path("PYTHONPATH", native_dir)
    attest_ds9_native_artifacts(ds9_root=DS9_ROOT, native_dir=native_dir)
    native_dir = configure_ds9_runtime_import_paths(
        ds9_root=DS9_ROOT,
        repo_root=REPO_ROOT,
        system_site=ds9_site,
    )
    load_ds9_native_extensions(native_dir)
    _prepend_env_path("LD_LIBRARY_PATH", Path(os.environ["NOESIS_DEEPSTREAM_HOME"]) / "lib")
    _prepend_env_path("GST_PLUGIN_PATH", Path(os.environ["NOESIS_GST_PLUGIN_DIR"]))


def _run_preflight(config_path: Path, cameras_path: Path) -> int:
    script = DS9_ROOT / "scripts" / "ds9_preflight.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--cameras-config",
            str(cameras_path),
        ],
        cwd=str(REPO_ROOT),
    ).returncode


def _argv_value(name: str) -> str | None:
    for index, value in enumerate(sys.argv[1:], start=1):
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        prefix = f"{name}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _requested_tracking_mode() -> str:
    raw = _argv_value("--tracking-mode")
    if raw is None:
        if "--v3dt" in sys.argv[1:]:
            return "v3dt"
        raw = os.environ.get("NOESIS_TRACKING_MODE", "")
    mode = str(raw or "").strip().lower()
    if mode == "mv3dt":
        return "mv3dt"
    if mode in {"v3dt", "sv3dt", "3d"}:
        return "v3dt"
    if mode in {"", "auto", "2d", "baseline", "standard", "default"}:
        return "baseline"
    raise SystemExit(
        "[FATAL] Unsupported DS9 tracking mode "
        f"{raw!r}; expected baseline, v3dt, mv3dt, or auto"
    )


def _v3dt_requested() -> bool:
    return _requested_tracking_mode() in {"v3dt", "mv3dt"}


def _selected_launch_paths() -> tuple[Path, Path, bool, bool]:
    tracking_mode = _requested_tracking_mode()
    pipeline_raw = _argv_value("--pipeline-config")
    cameras_raw = _argv_value("--cameras-config")
    pipeline_explicit = pipeline_raw is not None
    cameras_explicit = cameras_raw is not None
    if pipeline_raw is None:
        pipeline_raw = os.environ.get("NOESIS_DS9_PIPELINE_CONFIG", "").strip()
    if cameras_raw is None:
        cameras_raw = os.environ.get("NOESIS_CAMERAS_CONFIG", "").strip()
    if not pipeline_raw:
        default_pipeline = (
            "infer_mv3dt.yaml"
            if tracking_mode == "mv3dt"
            else "infer_v3dt.yaml"
            if tracking_mode == "v3dt"
            else "infer.yaml"
        )
        pipeline_raw = str(DS9_ROOT / "config" / default_pipeline)
    if not cameras_raw:
        cameras_raw = str(
            DS9_ROOT / "config" / "cameras_v3dt.yaml"
            if tracking_mode in {"v3dt", "mv3dt"}
            else REPO_ROOT / "config" / "cameras.yaml"
        )
    return (
        Path(pipeline_raw).expanduser().resolve(),
        Path(cameras_raw).expanduser().resolve(),
        pipeline_explicit,
        cameras_explicit,
    )


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


def _synthetic_stub_requested() -> bool:
    """Return true only for the DS9-owned explicit synthetic test selector."""

    return str(os.environ.get("NOESIS_DS9_STUB_PIPELINE", "")).strip().lower() in _ENV_TRUE


def main() -> int:
    _ensure_runtime_sys_path()
    if _requested_tracking_mode() == "mv3dt":
        print(
            "[FATAL] MV3DT activation is deferred until Kitchen geometry and "
            "synchronized occupied Kitchen/Family-Room overlap evidence are ready; "
            "Living Room has no MV3DT peer edge.",
            file=sys.stderr,
        )
        return 78
    _set_ds9_environment()
    pipeline_path, cameras_path, pipeline_explicit, cameras_explicit = _selected_launch_paths()
    synthetic_stub = _synthetic_stub_requested()
    if synthetic_stub:
        print(
            json.dumps(
                {
                    "event": "pipeline_backend_selected",
                    **synthetic_stub_lifecycle_evidence(),
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
    else:
        if _run_preflight(pipeline_path, cameras_path) != 0:
            return 1
        _preload_ds9_python_runtime()

    from noesis.ds9_runtime_core import main as _runtime_main

    if not pipeline_explicit:
        sys.argv.extend(["--pipeline-config", str(pipeline_path)])
    if not cameras_explicit:
        sys.argv.extend(["--cameras-config", str(cameras_path)])
    if "--storage-base" not in sys.argv:
        sys.argv.extend(["--storage-base", str(DS9_ROOT / "data" / "depth")])
    return int(_runtime_main())


if __name__ == "__main__":
    raise SystemExit(int(main()))
