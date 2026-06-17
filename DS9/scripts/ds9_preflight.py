#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
DS9_HOME = Path(os.environ.get("NOESIS_DEEPSTREAM_HOME", "/opt/nvidia/deepstream/deepstream-9.0"))
DS9_GST_PLUGIN_DIR = Path(os.environ.get("NOESIS_GST_PLUGIN_DIR", DS9_ROOT / "gst-plugins"))


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _ds9_system_python_site() -> Path:
    return Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")


def _prepend_env_path(name: str, value: Path) -> None:
    current = os.environ.get(name, "")
    value_str = str(value)
    parts = [value_str]
    if current:
        parts.append(current)
    os.environ[name] = os.pathsep.join(parts)


def _prefer_ds9_pyservicemaker() -> None:
    ds9_site = _ds9_system_python_site()
    if not (ds9_site / "pyservicemaker" / "_pydeepstream.so").exists():
        return
    site_str = str(ds9_site)
    if site_str in sys.path:
        sys.path.remove(site_str)
    sys.path.insert(0, site_str)
    _prepend_env_path("PYTHONPATH", ds9_site)
    importlib.invalidate_caches()


def _resolve(raw: str, *, base: Path) -> Path:
    path = Path(str(raw or "").strip())
    if path.is_absolute():
        return path
    if str(raw).startswith(("config/", "models/", "pipelines/", "build/", "DS9/")):
        return (REPO_ROOT / path).resolve()
    return (base / path).resolve()


def _deepstream_version_ok() -> bool:
    if not DS9_HOME.exists():
        _fail(f"DeepStream 9 home missing: {DS9_HOME}")
        return False
    resolved = DS9_HOME.resolve()
    if "deepstream-8.0" in str(resolved):
        _fail(f"DeepStream home resolves to DS8, not DS9: {resolved}")
        return False
    if "9.0" not in DS9_HOME.name and "9.0" not in str(resolved):
        _fail(f"DeepStream home is not explicitly DS9: {DS9_HOME} -> {resolved}")
        return False
    _ok(f"DeepStream home: {DS9_HOME}")
    return True


def _python_modules_ok() -> bool:
    ok = True
    if sys.version_info[:2] != (3, 12):
        _fail(f"Python 3.12 required for DS9; found {sys.version.split()[0]}")
        ok = False
    else:
        _ok(f"Python {sys.version.split()[0]}")

    try:
        mod = importlib.import_module("pyservicemaker")
    except Exception as exc:
        _fail(f"Python module missing: pyservicemaker ({exc})")
        ok = False
    else:
        _ok(f"pyservicemaker: {getattr(mod, '__file__', '<builtin>')}")

    try:
        mod = importlib.import_module("pyds")
    except Exception as exc:
        _warn(f"pyds unavailable in DS9 container ({exc}); DS9 path must rely on Service Maker/native extensions")
    else:
        _ok(f"pyds: {getattr(mod, '__file__', '<builtin>')}")
    return ok


def _trtexec_ok() -> bool:
    trtexec = shutil.which("trtexec")
    if not trtexec:
        _fail("trtexec not found on PATH")
        return False
    proc = subprocess.run([trtexec, "--version"], text=True, capture_output=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    if "v10140" not in text and "10.14" not in text:
        _fail("TensorRT 10.14.x required for DS9 engine rebuilds; trtexec output did not report 10.14")
        summary = "\n".join(text.strip().splitlines()[:8])
        if summary:
            print(summary)
        return False
    _ok("TensorRT trtexec reports DS9-compatible 10.14.x")
    return True


def _plugins_ok() -> bool:
    gst_inspect = shutil.which("gst-inspect-1.0")
    if not gst_inspect:
        _fail("gst-inspect-1.0 not found")
        return False
    if DS9_GST_PLUGIN_DIR.exists():
        current = os.environ.get("GST_PLUGIN_PATH", "")
        parts = [str(DS9_GST_PLUGIN_DIR)]
        if current:
            parts.append(current)
        os.environ["GST_PLUGIN_PATH"] = os.pathsep.join(parts)
    required = [
        "nvmultiurisrcbin",
        "nvstreammux",
        "nvdspreprocess",
        "nvinfer",
        "nvtracker",
        "nvdsanalytics",
        "nvdsroiexclude",
        "nvmultistreamtiler",
        "nvdsosd",
        "nvrtspoutsinkbin",
    ]
    ok = True
    for element in required:
        proc = subprocess.run([gst_inspect, element], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode != 0:
            _fail(f"GStreamer element unavailable: {element}")
            ok = False
        else:
            _ok(f"GStreamer element: {element}")
    return ok


def _config_assets_ok(config_path: Path) -> bool:
    if not config_path.exists():
        _fail(f"Pipeline config missing: {config_path}")
        return False
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        _fail(f"Pipeline config must be a mapping: {config_path}")
        return False

    ok = True
    base = config_path.parent
    models = cfg.get("models") if isinstance(cfg.get("models"), dict) else {}
    for name, model_cfg in models.items():
        if not isinstance(model_cfg, dict) or model_cfg.get("enable") is False:
            continue
        for key in ("config-file-path", "engine"):
            raw = str(model_cfg.get(key, "") or "").strip()
            if not raw:
                continue
            path = _resolve(raw, base=base)
            if not path.exists() or path.stat().st_size <= 0:
                _fail(f"models.{name}.{key} missing or empty: {path}")
                ok = False
            elif key == "engine" and str(path).startswith(str((REPO_ROOT / "models" / "engines").resolve())):
                _fail(f"models.{name}.engine points to DS8/root engine instead of DS9 engine: {path}")
                ok = False
            else:
                _ok(f"models.{name}.{key}: {path}")

    tracker = cfg.get("tracker") if isinstance(cfg.get("tracker"), dict) else {}
    tracker_lib = str(tracker.get("ll-lib-file", "") or "")
    if "deepstream-8.0" in tracker_lib:
        _fail(f"tracker.ll-lib-file points to DS8: {tracker_lib}")
        ok = False

    native_dir = Path(os.environ.get("NOESIS_NATIVE_EXT_DIR", DS9_ROOT / "native_extensions"))
    for module in (
        "noesis_pose_meta_ext",
        "noesis_v3dt_meta_ext",
        "noesis_reid_meta_ext",
        "noesis_latency_ext",
        "noesis_depth_meta_ext",
        "noesis_depth_tracking_tensor_ext",
    ):
        if not any(native_dir.glob(f"{module}*.so")):
            _fail(f"DS9 native extension missing: {native_dir}/{module}*.so")
            ok = False

    gst_plugin = DS9_GST_PLUGIN_DIR / "libgstnvdsroiexclude.so"
    if not gst_plugin.exists() or gst_plugin.stat().st_size <= 0:
        _fail(f"DS9 GStreamer plugin missing: {gst_plugin}")
        ok = False

    for rel in (
        "nvdsinfer_yolo11_seg/libnvdsinfer_yolo11_seg.so",
        "nvdsinfer_yolo26_seg/libnvdsinfer_yolo26_seg.so",
        "nvdsinfer_yolo_detect/libnvdsparsebbox_yolo.so",
        "nvdsinfer_rfdetr/libnvdsinfer_rfdetr.so",
        "nvdsinfer_rfdetr_seg/libnvdsinfer_rfdetr_seg.so",
    ):
        parser_path = DS9_ROOT / "pipelines" / rel
        if not parser_path.exists() or parser_path.stat().st_size <= 0:
            _fail(f"DS9 parser missing: {parser_path}")
            ok = False

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepStream 9 Noesis preflight")
    parser.add_argument("--config", type=Path, default=DS9_ROOT / "config" / "infer.yaml")
    parser.add_argument("--env-only", action="store_true", help="Skip model/parser/native artifact checks.")
    args = parser.parse_args()

    _prefer_ds9_pyservicemaker()
    checks = [_deepstream_version_ok(), _python_modules_ok(), _trtexec_ok(), _plugins_ok()]
    if not args.env_only:
        checks.append(_config_assets_ok(args.config.resolve()))
    else:
        _warn("Skipping DS9 artifact checks (--env-only)")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
