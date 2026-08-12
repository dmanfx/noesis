#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (str(DS9_ROOT), str(REPO_ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DS9_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from noesis.runtime_paths import (  # noqa: E402
    DS9_NATIVE_EXTENSION_MODULES,
    DS9NativeExtensionOriginError,
    configure_ds9_runtime_import_paths,
    load_ds9_native_extensions,
)
from noesis.native_artifact_provenance import (  # noqa: E402
    DS9NativeArtifactProvenanceError,
    attest_ds9_native_artifacts,
)
from engine_maintenance_common import (  # noqa: E402
    DS9_TRTEXEC_BANNER,
    EngineMaintenanceError,
    validate_trtexec_probe,
)
from noesis_core.runtime_secrets import (  # noqa: E402
    RuntimeSecretError,
    materialize_pipeline_config,
)
from noesis.reid_swin_profile import (  # noqa: E402
    load_reid_swin_nvinfer_properties,
    validate_reid_swin_model_config,
    validate_reid_swin_nvinfer_properties,
    validate_reid_swin_onnx_source,
)
from noesis.v3dt_assets import V3DTAssetError, validate_v3dt_assets  # noqa: E402

DS9_HOME = Path(
    os.environ.get("NOESIS_DEEPSTREAM_HOME", "/opt/nvidia/deepstream/deepstream-9.1")
)
DS9_GST_PLUGIN_DIR = Path(
    os.environ.get("NOESIS_GST_PLUGIN_DIR", DS9_ROOT / "gst-plugins")
)
DS9_MIN_DRIVER_VERSION = (595, 58, 3)
DS9_MIN_DRIVER_LABEL = "595.58.03"
DS9_OWNED_GST_PLUGINS = {
    "nvdsroiexclude": "libgstnvdsroiexclude.so",
    "noesisforceidr": "libgstnoesisforceidr.so",
    "noesiseos": "libgstnoesiseos.so",
}
NVDSROIEXCLUDE_REQUIRED_PROPERTIES = frozenset(
    {
        "config-file",
        "id-mode",
        "reload-request-sequence",
        "reload-accepted-sequence",
        "reload-failed-sequence",
        "last-reload-ok",
        "expected-config-sha256",
        "active-config-sha256",
        "reload-error-count",
        "objects-removed-count",
        "last-reload-error",
    }
)
WEBRTC_GI_VERSIONS = (
    ("Gst", "1.0"),
    ("GstSdp", "1.0"),
    ("GstWebRTC", "1.0"),
)
WEBRTC_GI_MODULES = ("GLib", "Gst", "GstSdp", "GstWebRTC")
WEBRTC_GATEWAY_GST_FACTORIES = (
    "shmsink",
    "shmsrc",
    "appsink",
    "appsrc",
    "h264parse",
    "rtph264pay",
    "webrtcbin",
    "queue",
    "fakesink",
)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _ds9_system_python_site() -> Path:
    return Path(
        f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"
    )


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


def _driver_version(raw: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)(?:\.(\d+))?\s*", str(raw or ""))
    if match is None:
        return None
    return tuple(int(value or 0) for value in match.groups())


def _driver_version_ok() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        _fail("nvidia-smi not found; cannot verify the DeepStream 9 driver floor")
        return False
    proc = subprocess.run(
        [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown error").strip().splitlines()
        suffix = f" ({detail[0]})" if detail else ""
        _fail(f"unable to query the NVIDIA driver version{suffix}")
        return False
    raw_versions = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    parsed = [_driver_version(value) for value in raw_versions]
    if not parsed or any(value is None for value in parsed):
        _fail(f"unable to parse NVIDIA driver version(s): {raw_versions!r}")
        return False
    incompatible = [
        raw
        for raw, version in zip(raw_versions, parsed, strict=True)
        if version is not None and version < DS9_MIN_DRIVER_VERSION
    ]
    if incompatible:
        _fail(
            "NVIDIA driver below the DeepStream 9 minimum "
            f"{DS9_MIN_DRIVER_LABEL}: {', '.join(incompatible)}"
        )
        return False
    _ok(
        "NVIDIA driver satisfies the DeepStream 9 minimum "
        f"{DS9_MIN_DRIVER_LABEL}: {', '.join(raw_versions)}"
    )
    return True


def _resolve(raw: str, *, base: Path) -> Path:
    value = str(raw or "").strip()
    path = Path(value)
    if path.is_absolute():
        return path
    model_override = str(os.environ.get("NOESIS_MODEL_DIR", "") or "").strip()
    artifact_override = str(
        os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "") or ""
    ).strip()
    if model_override:
        model_root = Path(model_override).expanduser()
    elif artifact_override:
        model_root = Path(artifact_override).expanduser() / "models"
    else:
        model_root = DS9_ROOT / "models"
    onnx_root = Path(
        os.environ.get("NOESIS_ONNX_DIR", model_root / "onnx")
    ).expanduser()
    engine_root = Path(
        os.environ.get("NOESIS_ENGINE_DIR", model_root / "engines")
    ).expanduser()
    for prefix, root in (
        ("DS9/models/engines/", engine_root),
        ("DS9/models/onnx/", onnx_root),
        ("DS9/models/", model_root),
        ("models/engines/", engine_root),
        ("models/onnx/", onnx_root),
        ("models/", model_root),
    ):
        if value.startswith(prefix):
            return (root / value[len(prefix) :]).resolve()
    if value.startswith(("config/", "pipelines/", "build/", "DS9/")):
        resolved = (REPO_ROOT / path).resolve()
    else:
        resolved = (base / path).resolve()
    try:
        model_relative = resolved.relative_to((DS9_ROOT / "models").resolve())
    except ValueError:
        return resolved
    return (model_root / model_relative).resolve()


def _deepstream_version_ok() -> bool:
    if not DS9_HOME.exists():
        _fail(f"DeepStream 9 home missing: {DS9_HOME}")
        return False
    resolved = DS9_HOME.resolve()
    if "deepstream-8.0" in str(resolved):
        _fail(f"DeepStream home resolves to DS8, not DS9: {resolved}")
        return False
    if "9.1" not in DS9_HOME.name and "9.1" not in str(resolved):
        _fail(f"DeepStream home is not explicitly DS9.1: {DS9_HOME} -> {resolved}")
        return False
    _ok(f"DeepStream home: {DS9_HOME}")
    return True


def _webrtc_python_stack_ok() -> bool:
    """Verify the exact PyGObject namespaces imported by the WebRTC gateway."""

    try:
        gi = importlib.import_module("gi")
        for namespace, version in WEBRTC_GI_VERSIONS:
            gi.require_version(namespace, version)
        modules = {
            namespace: importlib.import_module(f"gi.repository.{namespace}")
            for namespace in WEBRTC_GI_MODULES
        }
        modules["Gst"].init(None)
    except Exception as exc:
        _fail(
            "DS9 WebRTC Python stack unavailable; required PyGObject namespaces "
            "are GLib, Gst 1.0, GstSdp 1.0, and GstWebRTC 1.0 "
            f"({exc})"
        )
        return False
    _ok("PyGObject WebRTC namespaces: GLib, Gst 1.0, GstSdp 1.0, GstWebRTC 1.0")
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
        _warn(
            f"pyds unavailable in DS9 container ({exc}); DS9 path must rely on Service Maker/native extensions"
        )
    else:
        _ok(f"pyds: {getattr(mod, '__file__', '<builtin>')}")
    if not _webrtc_python_stack_ok():
        ok = False
    return ok


def _native_extensions_ok() -> bool:
    native_dir = configure_ds9_runtime_import_paths(
        ds9_root=DS9_ROOT,
        repo_root=REPO_ROOT,
        system_site=_ds9_system_python_site(),
    )
    missing = [
        module
        for module in DS9_NATIVE_EXTENSION_MODULES
        if not any(native_dir.glob(f"{module}*.so"))
    ]
    if missing:
        for module in missing:
            _fail(f"DS9 native extension missing: {native_dir}/{module}*.so")
        return False
    try:
        attest_ds9_native_artifacts(ds9_root=DS9_ROOT, native_dir=native_dir)
        origins = load_ds9_native_extensions(native_dir)
    except (DS9NativeArtifactProvenanceError, DS9NativeExtensionOriginError) as exc:
        _fail(f"DS9 native extension provenance/load/origin validation failed: {exc}")
        return False
    _ok(f"DS9 native extensions loaded: {len(origins)} directly owned binaries")
    return True


def _trtexec_ok() -> bool:
    trtexec = shutil.which("trtexec")
    if not trtexec:
        _fail("trtexec not found on PATH")
        return False
    proc = subprocess.run([trtexec, "--help"], text=True, capture_output=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    try:
        validate_trtexec_probe(proc.returncode, text)
    except EngineMaintenanceError as exc:
        _fail(
            f"TensorRT probe rejected; required exact DS9 banner {DS9_TRTEXEC_BANNER}: {exc}"
        )
        summary = "\n".join(text.strip().splitlines()[:8])
        if summary:
            print(summary)
        return False
    _ok(f"TensorRT trtexec reports exact DS9 version: {DS9_TRTEXEC_BANNER}")
    return True


def _gst_inspect_filename(output: str) -> Path:
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Filename"):
            continue
        raw = stripped[len("Filename") :].strip()
        if raw:
            return Path(raw).expanduser()
    raise ValueError("gst-inspect output did not report a plugin Filename")


def _validate_owned_plugin_origin(
    element: str, output: str, expected_binary: Path
) -> Path:
    actual = _gst_inspect_filename(output)
    if not actual.is_absolute():
        raise ValueError(
            f"GStreamer element {element} reported a non-absolute plugin origin: {actual}"
        )
    expected_resolved = expected_binary.expanduser().resolve(strict=False)
    actual_resolved = actual.resolve(strict=False)
    if actual_resolved != expected_resolved:
        raise ValueError(
            f"GStreamer element {element} loaded from {actual_resolved}; "
            f"required DS9-owned binary is {expected_resolved}"
        )
    return actual_resolved


def _gst_inspect_property_names(output: str) -> frozenset[str]:
    properties: set[str] = set()
    in_properties = False
    for line in output.splitlines():
        if line.strip() == "Element Properties:":
            in_properties = True
            continue
        if not in_properties:
            continue
        match = re.match(r"^\s{2}([a-z][a-z0-9-]*)\s*:", line)
        if match is not None:
            properties.add(match.group(1))
    return frozenset(properties)


def _validate_nvdsroiexclude_contract(output: str) -> None:
    present = _gst_inspect_property_names(output)
    missing = sorted(NVDSROIEXCLUDE_REQUIRED_PROPERTIES - present)
    if missing:
        raise ValueError(
            "DS9-owned nvdsroiexclude binary is stale or incompatible; "
            f"missing exact properties: {', '.join(missing)}"
        )


def _plugins_ok() -> bool:
    gst_inspect = shutil.which("gst-inspect-1.0")
    if not gst_inspect:
        _fail("gst-inspect-1.0 not found")
        return False
    inspect_env = dict(os.environ)
    plugin_dir = DS9_GST_PLUGIN_DIR.expanduser().resolve(strict=False)
    for env_name in ("GST_PLUGIN_PATH", "GST_PLUGIN_PATH_1_0"):
        current = inspect_env.get(env_name, "")
        inherited = [part for part in current.split(os.pathsep) if part]
        inspect_env[env_name] = os.pathsep.join(
            [
                str(plugin_dir),
                *[
                    part
                    for part in inherited
                    if Path(part).resolve(strict=False) != plugin_dir
                ],
            ]
        )
    required = [
        "nvmultiurisrcbin",
        "nvstreammux",
        "nvdspreprocess",
        "nvinfer",
        "nvtracker",
        "nvdsanalytics",
        "nvdsroiexclude",
        "noesisforceidr",
        "noesiseos",
        "nvmultistreamtiler",
        "nvdsosd",
        "nvrtspoutsinkbin",
        *WEBRTC_GATEWAY_GST_FACTORIES,
    ]
    ok = True
    for element in required:
        expected_name = DS9_OWNED_GST_PLUGINS.get(element)
        expected_binary = plugin_dir / expected_name if expected_name else None
        if expected_binary is not None:
            if expected_binary.is_symlink():
                _fail(
                    f"DS9-owned GStreamer plugin must not be a symlink: {expected_binary}"
                )
                ok = False
                continue
            if not expected_binary.is_file() or expected_binary.stat().st_size <= 0:
                _fail(f"DS9-owned GStreamer plugin missing or empty: {expected_binary}")
                ok = False
                continue
        proc = subprocess.run(
            [gst_inspect, element],
            text=True,
            capture_output=True,
            env=inspect_env,
            check=False,
        )
        if proc.returncode != 0:
            _fail(f"GStreamer element unavailable: {element}")
            ok = False
        elif expected_binary is not None:
            try:
                origin = _validate_owned_plugin_origin(
                    element, proc.stdout or "", expected_binary
                )
            except ValueError as exc:
                _fail(str(exc))
                ok = False
            else:
                if element == "nvdsroiexclude":
                    try:
                        _validate_nvdsroiexclude_contract(proc.stdout or "")
                    except ValueError as exc:
                        _fail(str(exc))
                        ok = False
                        continue
                _ok(f"GStreamer element: {element} (DS9-owned: {origin})")
        else:
            _ok(f"GStreamer element: {element}")
    return ok


def _config_assets_ok(config_path: Path, cameras_path: Path) -> bool:
    if not config_path.exists():
        _fail(f"Pipeline config missing: {config_path}")
        return False
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        _fail(f"Pipeline config must be a mapping: {config_path}")
        return False

    try:
        materialize_pipeline_config(cfg)
    except RuntimeSecretError as exc:
        _fail(f"Camera source secret contract failed: {exc}")
        return False
    _ok("Camera source secret references resolve through owner-only state")

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
            elif key == "engine" and str(path).startswith(
                str((REPO_ROOT / "models" / "engines").resolve())
            ):
                _fail(
                    f"models.{name}.engine points to DS8/root engine instead of DS9 engine: {path}"
                )
                ok = False
            else:
                _ok(f"models.{name}.{key}: {path}")

    reid_cfg = models.get("reid") if isinstance(models, dict) else None
    if isinstance(reid_cfg, dict) and bool(reid_cfg.get("enable", True)):
        try:
            validate_reid_swin_model_config(reid_cfg)
            reid_config_path = _resolve(
                str(reid_cfg.get("config-file-path") or ""), base=base
            )
            props = load_reid_swin_nvinfer_properties(reid_config_path)
            validate_reid_swin_nvinfer_properties(props)
            onnx_path = _resolve(
                str(props.get("onnx-file") or ""), base=reid_config_path.parent
            )
            validate_reid_swin_onnx_source(onnx_path)
        except (OSError, ValueError) as exc:
            _fail(f"Canonical ReID profile contract failed: {exc}")
            ok = False
        else:
            _ok(f"Canonical ReID profile: TAO Swin-Tiny fc_pred/256 ({onnx_path})")

    if isinstance(cfg.get("v3dt"), dict):
        try:
            bundle = validate_v3dt_assets(
                config_path,
                cameras_config=cameras_path,
                require_engines=True,
            )
        except (OSError, V3DTAssetError) as exc:
            _fail(str(exc))
            ok = False
        else:
            _ok(
                "DS9-owned V3DT profile: "
                f"tracker={bundle.tracker_config} cameras={len(bundle.camera_models)}"
            )

    tracker = cfg.get("tracker") if isinstance(cfg.get("tracker"), dict) else {}
    tracker_lib = str(tracker.get("ll-lib-file", "") or "")
    if "/deepstream-9.1/" not in tracker_lib:
        _fail(f"tracker.ll-lib-file must bind explicitly to DeepStream 9.1: {tracker_lib}")
        ok = False

    gst_plugin = DS9_GST_PLUGIN_DIR / "libgstnvdsroiexclude.so"
    if not gst_plugin.exists() or gst_plugin.stat().st_size <= 0:
        _fail(f"DS9 GStreamer plugin missing: {gst_plugin}")
        ok = False

    force_idr_plugin = DS9_GST_PLUGIN_DIR / "libgstnoesisforceidr.so"
    if not force_idr_plugin.exists() or force_idr_plugin.stat().st_size <= 0:
        _fail(f"DS9 force-IDR GStreamer plugin missing: {force_idr_plugin}")
        ok = False

    eos_plugin = DS9_GST_PLUGIN_DIR / "libgstnoesiseos.so"
    if not eos_plugin.exists() or eos_plugin.stat().st_size <= 0:
        _fail(f"DS9 orderly-EOS GStreamer plugin missing: {eos_plugin}")
        ok = False

    for rel in (
        "nvdsinfer_yolo11_seg/libnvdsinfer_yolo11_seg.so",
        "nvdsinfer_yolo26_seg/libnvdsinfer_yolo26_seg.so",
        "nvdsinfer_yolo_detect/libnvdsparsebbox_yolo.so",
        "nvdsinfer_rfdetr/libnvdsinfer_rfdetr.so",
        "nvdsinfer_rfdetr_seg/libnvdsinfer_rfdetr_seg.so",
        "nvdsinfer_deimv2_wholebody49/libnvdsinfer_deimv2_wholebody49.so",
    ):
        parser_path = DS9_ROOT / "pipelines" / rel
        if not parser_path.exists() or parser_path.stat().st_size <= 0:
            _fail(f"DS9 parser missing: {parser_path}")
            ok = False

    return ok


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepStream 9 Noesis preflight")
    parser.add_argument(
        "--config", type=Path, default=DS9_ROOT / "config" / "infer.yaml"
    )
    parser.add_argument(
        "--cameras-config",
        type=Path,
        default=REPO_ROOT / "config" / "cameras.yaml",
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="Skip model/parser checks; required native runtime dependencies remain checked.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()

    _prefer_ds9_pyservicemaker()
    configure_ds9_runtime_import_paths(
        ds9_root=DS9_ROOT,
        repo_root=REPO_ROOT,
        system_site=_ds9_system_python_site(),
    )
    checks = [
        _driver_version_ok(),
        _deepstream_version_ok(),
        _python_modules_ok(),
        _native_extensions_ok(),
        _trtexec_ok(),
        _plugins_ok(),
    ]
    if not args.env_only:
        checks.append(
            _config_assets_ok(args.config.resolve(), args.cameras_config.resolve())
        )
    else:
        _warn("Skipping DS9 artifact checks (--env-only)")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
