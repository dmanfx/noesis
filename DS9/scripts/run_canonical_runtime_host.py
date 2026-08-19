#!/usr/bin/env python3
"""Canonical native-host supervisor for the DS9.1 application.

Operations:
  check  validate native platform, venv origin, artifacts, secrets, and ports
         without binding 6008/8080 or opening cameras
  run    exec DS9/noesis/ds9_runtime.py as the long-lived process

This supervisor does not speak Docker or read a deployment-selector file.
Host paths come from existing environment variables; no /home or storage path is
hardcoded here. Health-contract identity is emitted from explicit env fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

sys.dont_write_bytecode = True

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DS9_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(DS9_ROOT / "scripts"))

from pydantic import ValidationError  # noqa: E402
from noesis.server.internal_auth import load_internal_token  # noqa: E402
from noesis_core.appliance import (  # noqa: E402
    ApplianceConfigurationError,
    DeploymentHealthBinding,
    runtime_context_environment,
)
from noesis_core.private_paths import (  # noqa: E402
    PrivatePathError,
    validate_private_file,
)
from noesis_core.runtime_secrets import (  # noqa: E402
    RuntimeSecretError,
    load_camera_uri_registry,
    load_mapanything_api_key,
)


CONTRACT = "noesis.ds9.canonical_runtime_host"
SCHEMA_VERSION = 1
REQUIRED_DEEPSTREAM_HOME = Path("/opt/nvidia/deepstream/deepstream-9.1")
REQUIRED_CUDA_HOME = Path("/usr/local/cuda-13.2")
REQUIRED_CUDA_VERSION = "13.2"
REQUIRED_TENSORRT_BANNER = "TensorRT v101600"
REQUIRED_TENSORRT_VERSION = "10.16.0.72"
REQUIRED_GSTREAMER = "1.24.2"
REQUIRED_DRIVER_FLOOR = (595, 58, 3)
REQUIRED_PYBIND11 = "2.12.0"
REQUIRED_PYTHON = (3, 12)
NATIVE_DEPENDENCY_CONSTRAINTS = DS9_ROOT / "native" / "requirements.host.constraints.txt"
CANONICAL_WS_PORT = 6008
CANONICAL_REST_PORT = 8080
CANONICAL_PORTS = (CANONICAL_WS_PORT, CANONICAL_REST_PORT)
CANONICAL_PROFILE = "yolo26"
CANONICAL_SIZE = "m"
CANONICAL_TRACKING_MODE = "baseline"
PIPELINE_CONFIG = "DS9/config/infer.yaml"
CAMERAS_CONFIG = "config/cameras.yaml"
OPT_IN_TRACKING_MODES = (CANONICAL_TRACKING_MODE, "mv3dt")
MV3DT_PROFILE = "yolo26"
MV3DT_SIZE = "m"
MV3DT_PIPELINE_CONFIG = "DS9/config/infer_mv3dt.yaml"
MV3DT_CAMERAS_CONFIG = "DS9/config/cameras_v3dt.yaml"
REALIZATION_FILENAME = "asset_realization.json"
REALIZATION_CONTRACT = "noesis.ds9.asset_realization"
REALIZATION_SHA256 = "9c3815bbf86eb41a94efb504fad79a9208c543e05f98c68d017cb1a8dcfd2b26"
CANONICAL_ENGINE_IDS = (
    "engine.yolo26_detect_m",
    "engine.reid_swin_tiny",
    "engine.pose_yolo26",
    "engine.depth_tracking_dav2",
    "engine.mapanything",
)
FORBIDDEN_ORIGIN_MARKERS = (
    "deepstream-8.0",
    "deepstream-9.0",
    "/var/lib/docker/",
    "overlay2",
    "noesis-ds9-docker",
)
STRIP_ENV_KEYS = (
    "NOESIS_APPLIANCE_RUNTIME_CONTEXT",
    "NOESIS_APPLIANCE_RUNTIME_CONTEXT_SHA256",
    "NOESIS_DEPLOYMENT_SELECTOR_FILE",
    "NOESIS_DEPLOYMENT_SELECTOR_SHA256",
    "NOESIS_STATE_RELEASE_MANIFEST",
    "NOESIS_STATE_RELEASE_ROOT",
    "NOESIS_STATE_RELEASE_LEASE",
    "NOESIS_STATE_RELEASE_LEASE_FILE",
    "NOESIS_DEPLOYMENT_LEASE_FILE",
    "MENON_STATE_RELEASE_LEASE",
    "MENON_STATE_RELEASE_LEASE_FILE",
    "MENON_APPLIANCE_STATE_LEASE_FILE",
    "NOESIS_DS9_DOCKER_ROOT",
    "DOCKER_HOST",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)
REQUIRED_ENV_PATHS = (
    "NOESIS_DS91_NATIVE_ROOT",
    "NOESIS_DS9_ARTIFACT_ROOT",
    "NOESIS_DS9_RUNTIME_ROOT",
    "NOESIS_WORLD_JOURNAL_PATH",
    "NOESIS_IDENTITY_V2_STORE",
    "NOESIS_ANALYTICS_CONFIG",
    "NOESIS_ANALYTICS_EXCLUDE_CONFIG",
    "NOESIS_SCENE_STORE_PATH",
    "NOESIS_VIRTUAL_TWIN_ROOT",
    "NOESIS_CAMERA_SECRETS_FILE",
    "NOESIS_MAPANYTHING_API_KEY_FILE",
    "NOESIS_INTERNAL_AUTH_TOKEN_FILE",
)


class NativeRuntimeError(RuntimeError):
    """Fail-closed native host supervisor error."""


@dataclass(frozen=True)
class SecretFiles:
    cameras: Path
    mapanything: Path
    internal_auth: Path


@dataclass(frozen=True)
class NativeHostConfig:
    repo_root: Path
    ds9_root: Path
    native_root: Path
    artifacts: Path
    runtime: Path
    venv_python: Path
    world_store: Path
    identity_store: Path
    analytics_config: Path
    analytics_exclude: Path
    scene_store: Path
    virtual_twin: Path
    secrets: SecretFiles
    gst_registry: Path
    deepstream_home: Path
    cuda_home: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fZ").lower()
    return f"{stamp}-{os.urandom(3).hex()}"


def _fail(message: str) -> None:
    raise NativeRuntimeError(message)


def _absolute_path(raw: str, *, label: str, must_exist: bool = True) -> Path:
    text = str(raw or "").strip()
    if not text:
        _fail(f"{label} must be set")
    path = Path(text).expanduser()
    if not path.is_absolute():
        _fail(f"{label} must be an explicit absolute path")
    path = Path(os.path.abspath(os.fspath(path)))
    if any(value in os.fspath(path) for value in ("\x00", "\n", "\r")):
        _fail(f"{label} contains an unsupported character")
    if path in {Path("/"), Path("/var"), Path("/var/lib"), Path("/tmp")}:
        _fail(f"refusing unsafe {label}: {path}")
    if must_exist and not path.exists():
        _fail(f"{label} is missing: {path}")
    return path


def _require_env_path(env: Mapping[str, str], name: str, *, directory: bool) -> Path:
    path = _absolute_path(env.get(name, ""), label=name, must_exist=True)
    if directory and not path.is_dir():
        _fail(f"{name} must be a directory: {path}")
    if not directory and not path.is_file():
        _fail(f"{name} must be a regular file: {path}")
    return path


def _venv_python(native_root: Path) -> Path:
    candidate = native_root / "venv" / "bin" / "python"
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        _fail(f"native virtualenv python is missing: {candidate}")
    return candidate


def load_native_config(env: Mapping[str, str] | None = None) -> NativeHostConfig:
    env = dict(os.environ if env is None else env)
    native_root = _require_env_path(env, "NOESIS_DS91_NATIVE_ROOT", directory=True)
    artifacts = _require_env_path(env, "NOESIS_DS9_ARTIFACT_ROOT", directory=True)
    runtime = _absolute_path(
        env.get("NOESIS_DS9_RUNTIME_ROOT", ""),
        label="NOESIS_DS9_RUNTIME_ROOT",
        must_exist=False,
    )
    secrets = SecretFiles(
        cameras=_require_env_path(env, "NOESIS_CAMERA_SECRETS_FILE", directory=False),
        mapanything=_require_env_path(
            env, "NOESIS_MAPANYTHING_API_KEY_FILE", directory=False
        ),
        internal_auth=_require_env_path(
            env, "NOESIS_INTERNAL_AUTH_TOKEN_FILE", directory=False
        ),
    )
    gst_registry = native_root / "gst-registry" / "registry.bin"
    return NativeHostConfig(
        repo_root=REPO_ROOT,
        ds9_root=DS9_ROOT,
        native_root=native_root,
        artifacts=artifacts,
        runtime=runtime,
        venv_python=_venv_python(native_root),
        world_store=_require_env_path(env, "NOESIS_WORLD_JOURNAL_PATH", directory=False),
        identity_store=_require_env_path(
            env, "NOESIS_IDENTITY_V2_STORE", directory=False
        ),
        analytics_config=_require_env_path(
            env, "NOESIS_ANALYTICS_CONFIG", directory=False
        ),
        analytics_exclude=_require_env_path(
            env, "NOESIS_ANALYTICS_EXCLUDE_CONFIG", directory=False
        ),
        scene_store=_require_env_path(env, "NOESIS_SCENE_STORE_PATH", directory=False),
        virtual_twin=_require_env_path(
            env, "NOESIS_VIRTUAL_TWIN_ROOT", directory=True
        ),
        secrets=secrets,
        gst_registry=gst_registry,
        deepstream_home=REQUIRED_DEEPSTREAM_HOME,
        cuda_home=REQUIRED_CUDA_HOME,
    )


def _runtime_lane(tracking_mode: str) -> dict[str, str]:
    mode = str(tracking_mode or "").strip().lower()
    if mode == CANONICAL_TRACKING_MODE:
        return {
            "name": "baseline",
            "tracking_mode": CANONICAL_TRACKING_MODE,
            "pgie_profile": CANONICAL_PROFILE,
            "model_size": CANONICAL_SIZE,
            "pipeline_config": PIPELINE_CONFIG,
            "cameras_config": CAMERAS_CONFIG,
        }
    if mode == "mv3dt":
        return {
            "name": "mv3dt",
            "tracking_mode": "mv3dt",
            "pgie_profile": MV3DT_PROFILE,
            "model_size": MV3DT_SIZE,
            "pipeline_config": MV3DT_PIPELINE_CONFIG,
            "cameras_config": MV3DT_CAMERAS_CONFIG,
        }
    _fail(
        "native host tracking mode must be one of: "
        + ", ".join(OPT_IN_TRACKING_MODES)
    )


def canonical_runtime_arguments(
    *,
    storage_base: Path,
    tracking_mode: str = CANONICAL_TRACKING_MODE,
) -> list[str]:
    lane = _runtime_lane(tracking_mode)
    return [
        "DS9/noesis/ds9_runtime.py",
        "--pipeline-config",
        lane["pipeline_config"],
        "--cameras-config",
        lane["cameras_config"],
        "--pgie-profile",
        lane["pgie_profile"],
        "--size",
        lane["model_size"],
        "--tracking-mode",
        lane["tracking_mode"],
        "--ws-host",
        "127.0.0.1",
        "--ws-port",
        str(CANONICAL_WS_PORT),
        "--rest-host",
        "127.0.0.1",
        "--rest-port",
        str(CANONICAL_REST_PORT),
        "--enable-rest",
        "--storage-base",
        str(storage_base),
        "--log-level",
        "INFO",
    ]


def build_health_context_env(
    env: Mapping[str, str],
    *,
    tracking_mode: str = CANONICAL_TRACKING_MODE,
) -> dict[str, str]:
    """Emit existing health-contract identity without reading a selector file."""

    selected_mode = _runtime_lane(tracking_mode)["tracking_mode"]
    deployment_id = str(env.get("NOESIS_DEPLOYMENT_ID") or "").strip()
    selector_sha256 = str(env.get("NOESIS_HEALTH_SELECTOR_SHA256") or "").strip()
    state_release_id = str(env.get("NOESIS_STATE_RELEASE_ID") or "").strip()
    software_revision = str(env.get("NOESIS_SOFTWARE_REVISION") or "").strip()
    if not all((deployment_id, selector_sha256, state_release_id, software_revision)):
        _fail(
            "native health identity requires NOESIS_DEPLOYMENT_ID, "
            "NOESIS_HEALTH_SELECTOR_SHA256, NOESIS_STATE_RELEASE_ID, and "
            "NOESIS_SOFTWARE_REVISION"
        )
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError) as exc:
        _fail(f"kernel boot identity is unavailable: {exc}")
    binding = DeploymentHealthBinding(
        deployment_id=deployment_id,
        selector_sha256=selector_sha256,
        state_release_id=state_release_id,
        runtime_family="ds9",
        runtime_variant=(
            "ds9:v3dt" if selected_mode == "mv3dt" else "ds9:baseline"
        ),
        software_revision=software_revision,
        boot_id=boot_id,
    )
    try:
        return runtime_context_environment(binding)
    except (ApplianceConfigurationError, ValidationError, ValueError) as exc:
        _fail(f"native health identity is not canonical: {exc}")


def validate_secret_files(secrets: SecretFiles) -> None:
    resolved: dict[str, Path] = {}
    mapping = {
        "camera source secrets": secrets.cameras,
        "MapAnything RPC key": secrets.mapanything,
        "internal auth token": secrets.internal_auth,
    }
    for label, path in mapping.items():
        try:
            resolved[label] = validate_private_file(path, label=label)
        except PrivatePathError as exc:
            _fail(str(exc))
    if len(set(resolved.values())) != len(resolved):
        _fail("runtime secret files must be three distinct files")
    try:
        load_camera_uri_registry(resolved["camera source secrets"])
        load_mapanything_api_key(resolved["MapAnything RPC key"])
        load_internal_token(resolved["internal auth token"])
    except (RuntimeSecretError, Exception) as exc:
        _fail(f"runtime secret validation failed: {type(exc).__name__}")


def _origin_is_forbidden(raw: str) -> str | None:
    text = raw.replace("\\", "/")
    for marker in FORBIDDEN_ORIGIN_MARKERS:
        if marker in text:
            return marker
    return None


def _run_text(command: Sequence[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"command failed: {' '.join(command[:3])}: {exc}")
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def _parse_driver(raw: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)(?:\.(\d+))?\s*", str(raw or ""))
    if match is None:
        return None
    return tuple(int(value or 0) for value in match.groups())  # type: ignore[return-value]


def verify_platform_versions() -> dict[str, str]:
    if not REQUIRED_DEEPSTREAM_HOME.is_dir():
        _fail(f"DeepStream 9.1 home missing: {REQUIRED_DEEPSTREAM_HOME}")
    resolved_ds = REQUIRED_DEEPSTREAM_HOME.resolve()
    if resolved_ds.name != "deepstream-9.1":
        _fail(f"DeepStream home is not 9.1: {resolved_ds}")
    if not REQUIRED_CUDA_HOME.is_dir():
        _fail(f"CUDA 13.2 home missing: {REQUIRED_CUDA_HOME}")
    cuda_link = Path("/usr/local/cuda")
    if cuda_link.exists() and cuda_link.resolve() != REQUIRED_CUDA_HOME.resolve():
        _fail(
            "canonical CUDA link is not 13.2: "
            f"{cuda_link} -> {cuda_link.resolve()}"
        )

    smi = shutil.which("nvidia-smi")
    if not smi:
        _fail("nvidia-smi not found")
    code, text = _run_text(
        [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    if code != 0:
        _fail("nvidia-smi failed")
    gpu_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if "," not in gpu_line:
        _fail("nvidia-smi GPU identity is incomplete")
    gpu_name, driver_version = [part.strip() for part in gpu_line.split(",", 1)]
    parsed_driver = _parse_driver(driver_version)
    if parsed_driver is None or parsed_driver < REQUIRED_DRIVER_FLOOR:
        _fail(
            f"driver {driver_version!r} is below DS9.1 floor "
            f"{REQUIRED_DRIVER_FLOOR[0]}.{REQUIRED_DRIVER_FLOOR[1]:02d}.{REQUIRED_DRIVER_FLOOR[2]:02d}"
        )

    nvcc = REQUIRED_CUDA_HOME / "bin" / "nvcc"
    if not nvcc.is_file():
        _fail(f"nvcc missing: {nvcc}")
    _, nvcc_text = _run_text([str(nvcc), "--version"])
    if f"release {REQUIRED_CUDA_VERSION}," not in nvcc_text:
        _fail(f"nvcc is not CUDA {REQUIRED_CUDA_VERSION}")

    trtexec = shutil.which("trtexec")
    if not trtexec:
        _fail("trtexec not found on PATH")
    _, trt_text = _run_text([trtexec, "--help"])
    if REQUIRED_TENSORRT_BANNER not in trt_text:
        _fail(f"trtexec is not {REQUIRED_TENSORRT_BANNER}")

    deepstream_app = shutil.which("deepstream-app") or str(
        REQUIRED_DEEPSTREAM_HOME / "bin" / "deepstream-app"
    )
    _, ds_text = _run_text([deepstream_app, "--version-all"])
    if "DeepStreamSDK 9.1.0" not in ds_text:
        _fail("deepstream-app is not DeepStream 9.1.0")
    if "TensorRT Version: 10.16" not in ds_text:
        _fail("deepstream-app is not linked to TensorRT 10.16")

    gst_launch = shutil.which("gst-launch-1.0")
    if not gst_launch:
        _fail("gst-launch-1.0 not found")
    _, gst_text = _run_text([gst_launch, "--version"])
    if REQUIRED_GSTREAMER not in gst_text:
        _fail(f"GStreamer is not {REQUIRED_GSTREAMER}")

    inspect_env = os.environ.copy()
    plugin_dir = str(REQUIRED_DEEPSTREAM_HOME / "lib" / "gst-plugins")
    inspect_env["GST_PLUGIN_PATH"] = plugin_dir
    for element in ("nvinfer", "nvtracker", "nvstreammux", "nvvideoconvert"):
        proc = subprocess.run(
            ["gst-inspect-1.0", element],
            check=False,
            capture_output=True,
            text=True,
            env=inspect_env,
            timeout=30,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            _fail(f"GStreamer element unavailable: {element}")
        filename = ""
        for line in output.splitlines():
            if line.strip().startswith("Filename"):
                filename = line.split(":", 1)[-1].strip()
                break
        if "/deepstream-9.1/" not in filename.replace("\\", "/"):
            _fail(f"{element} does not resolve under DeepStream 9.1: {filename or '<missing>'}")
        forbidden = _origin_is_forbidden(filename)
        if forbidden:
            _fail(f"{element} origin is forbidden ({forbidden}): {filename}")

    return {
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "cuda": REQUIRED_CUDA_VERSION,
        "tensorrt": REQUIRED_TENSORRT_VERSION,
        "deepstream": "9.1.0",
        "gstreamer": REQUIRED_GSTREAMER,
    }


def verify_python_origin(python: Path) -> dict[str, str]:
    script = r"""
import json, os, sys
from pathlib import Path
info = {
  "executable": sys.executable,
  "prefix": sys.prefix,
  "base_prefix": getattr(sys, "base_prefix", sys.prefix),
  "version": "%d.%d.%d" % sys.version_info[:3],
  "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
}
try:
    import pybind11
    info["pybind11"] = pybind11.__version__
    info["pybind11_file"] = getattr(pybind11, "__file__", "")
except Exception as exc:
    raise SystemExit("pybind11 import failed: %s" % exc)
try:
    import pyservicemaker
    info["pyservicemaker"] = getattr(pyservicemaker, "__file__", "")
except Exception as exc:
    raise SystemExit("pyservicemaker import failed: %s" % exc)
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstSdp", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    from gi.repository import Gst, GstSdp, GstWebRTC
    info["gi"] = getattr(gi, "__file__", "")
    info["gst"] = getattr(Gst, "__file__", "") or "gi"
except Exception as exc:
    raise SystemExit("gi Gst/GstSdp/GstWebRTC import failed: %s" % exc)
try:
    import torch
    info["torch"] = torch.__version__
    info["torch_cuda"] = bool(torch.cuda.is_available())
    info["torch_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
except Exception as exc:
    raise SystemExit("torch import failed: %s" % exc)
for name in ("yaml", "numpy", "cv2", "fastapi"):
    __import__(name)
if "noesis.ds8_runtime" in sys.modules:
    raise SystemExit("DS8 runtime module was imported")
forbidden = (
    "deepstream-8.0", "deepstream-9.0", "/var/lib/docker/", "overlay2", "noesis-ds9-docker",
)
for name, module in list(sys.modules.items()):
    raw = getattr(module, "__file__", None)
    if not raw:
        continue
    text = str(raw)
    for marker in forbidden:
        if marker in text.replace("\\", "/"):
            raise SystemExit("%s origin is forbidden (%s): %s" % (name, marker, text))
print(json.dumps(info))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [str(python), "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        _fail(detail[-1] if detail else "native Python origin check failed")
    payload = json.loads(result.stdout)
    if not payload.get("in_venv"):
        _fail("native Python is not a virtual environment")
    if tuple(int(part) for part in str(payload["version"]).split(".")[:2]) != REQUIRED_PYTHON:
        _fail(f"native Python is not 3.12: {payload['version']}")
    if payload.get("pybind11") != REQUIRED_PYBIND11:
        _fail(f"pybind11 is not {REQUIRED_PYBIND11}: {payload.get('pybind11')}")
    pysm = Path(str(payload.get("pyservicemaker") or ""))
    venv_root = python.parent.parent.resolve()
    if not pysm.exists() or venv_root not in pysm.resolve().parents:
        _fail(f"pyservicemaker is not from the native venv: {pysm}")
    if not payload.get("torch_cuda"):
        _fail("Torch does not see CUDA")
    ds8 = payload.get("ds8_spec")
    if ds8:
        # Presence of the archived module file is allowed; executing it is not.
        # The check process must not import it. find_spec locating the file is
        # reported but not a failure unless the origin is already loaded.
        pass
    return {
        "python": payload["executable"],
        "pyservicemaker": str(pysm),
        "pybind11": payload["pybind11"],
        "torch": payload["torch"],
        "gpu": payload.get("torch_device") or "",
    }


def verify_native_dependency_constraints(
    python: Path,
    constraints_path: Path = NATIVE_DEPENDENCY_CONSTRAINTS,
) -> dict[str, Any]:
    """Require the native venv to match the accepted image's drifting deps."""

    expected: dict[str, str] = {}
    try:
        lines = constraints_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _fail(f"native dependency constraints cannot be read: {exc}")
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if match is None:
            _fail(
                "native dependency constraint must be an exact package pin: "
                f"{constraints_path}:{line_number}"
            )
        expected[match.group(1)] = match.group(2)
    if not expected:
        _fail("native dependency constraints contain no package pins")

    script = r"""
import json, sys
from importlib.metadata import PackageNotFoundError, version
expected = json.loads(sys.argv[1])
actual = {}
for name in expected:
    try:
        actual[name] = version(name)
    except PackageNotFoundError:
        actual[name] = "<missing>"
print(json.dumps(actual, sort_keys=True))
"""
    result = subprocess.run(
        [str(python), "-c", script, json.dumps(expected, sort_keys=True)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        _fail("native dependency version probe failed")
    try:
        actual = json.loads(result.stdout)
    except json.JSONDecodeError:
        _fail("native dependency version probe returned invalid JSON")
    mismatches = [
        f"{name} expected={wanted} actual={actual.get(name, '<missing>')}"
        for name, wanted in sorted(expected.items())
        if actual.get(name) != wanted
    ]
    if mismatches:
        _fail("native Python dependency mismatch: " + "; ".join(mismatches))
    return {
        "constraints_path": str(constraints_path),
        "count": len(expected),
        "versions": actual,
    }


def verify_artifact_realization(config: NativeHostConfig) -> dict[str, Any]:
    realization_path = config.artifacts / REALIZATION_FILENAME
    if not realization_path.is_file():
        _fail(f"asset realization missing: {realization_path}")
    digest = hashlib.sha256(realization_path.read_bytes()).hexdigest()
    expected = str(
        os.environ.get("NOESIS_DS9_ASSET_REALIZATION_SHA256", REALIZATION_SHA256)
        or REALIZATION_SHA256
    ).strip()
    if digest != expected:
        _fail(
            "selected DS9.1 artifact realization digest mismatch: "
            f"expected={expected} actual={digest}"
        )
    from validate_asset_manifest import validate_asset_realization

    result = validate_asset_realization(
        DS9_ROOT / "asset_manifest.yaml",
        realization_path,
        config.artifacts,
        profile="canonical",
        check_files=True,
        require_provenance=True,
    )
    if not bool(result.get("ok")):
        _fail("authoritative DS9 asset realization validation did not pass")
    return {
        "realization_path": str(realization_path),
        "realization_sha256": digest,
        "required_engine_ids": list(CANONICAL_ENGINE_IDS),
        "ok": True,
    }


def port_occupancy() -> dict[str, Any]:
    occupied: list[int] = []
    for port in CANONICAL_PORTS:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                occupied.append(port)
        except OSError:
            continue
    return {
        "ports": list(CANONICAL_PORTS),
        "occupied": occupied,
        "probe": "connect",
    }


def build_run_environment(
    config: NativeHostConfig,
    *,
    session_id: str,
    storage_base: Path,
    evidence_root: Path,
    build_root: Path,
    tracking_mode: str = CANONICAL_TRACKING_MODE,
) -> dict[str, str]:
    lane = _runtime_lane(tracking_mode)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in STRIP_ENV_KEYS
    }
    ds_lib = str(config.deepstream_home / "lib")
    ds_plugins = str(config.deepstream_home / "lib" / "gst-plugins")
    cuda_bin = str(config.cuda_home / "bin")
    cuda_lib = str(config.cuda_home / "lib64")
    current_path = env.get("PATH", "/usr/bin:/bin")
    env["PATH"] = os.pathsep.join(
        [str(config.venv_python.parent), cuda_bin, current_path]
    )
    ld_parts = [ds_lib, cuda_lib]
    inherited_ld = env.get("LD_LIBRARY_PATH", "")
    if inherited_ld:
        ld_parts.append(inherited_ld)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
    gst_parts = [ds_plugins]
    inherited_gst = env.get("GST_PLUGIN_PATH", "")
    if inherited_gst:
        gst_parts.append(inherited_gst)
    env["GST_PLUGIN_PATH"] = os.pathsep.join(gst_parts)
    env["GST_REGISTRY"] = str(config.gst_registry)
    env["CUDA_HOME"] = str(config.cuda_home)
    env["VIRTUAL_ENV"] = str(config.native_root / "venv")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NOESIS_DEEPSTREAM_HOME"] = str(config.deepstream_home)
    env["NOESIS_DEEPSTREAM_MAJOR"] = "9"
    env["NOESIS_DS9_ARTIFACT_ROOT"] = str(config.artifacts)
    env["NOESIS_DS91_NATIVE_ROOT"] = str(config.native_root)
    env["NOESIS_DS9_RUNTIME_ROOT"] = str(config.runtime)
    env["NOESIS_BUILD_DIR"] = str(build_root)
    env["HOME"] = str(storage_base / "home")
    env["XDG_CACHE_HOME"] = str(build_root / "cache")
    env["XDG_RUNTIME_DIR"] = str(build_root / "xdg-runtime")
    env["CUDA_CACHE_PATH"] = str(build_root / "cuda-cache")
    env["NOESIS_CAMERA_SECRETS_FILE"] = str(config.secrets.cameras)
    env["NOESIS_MAPANYTHING_API_KEY_FILE"] = str(config.secrets.mapanything)
    env["NOESIS_INTERNAL_AUTH_MODE"] = "required"
    env["NOESIS_INTERNAL_AUTH_TOKEN_FILE"] = str(config.secrets.internal_auth)
    env["NOESIS_WORLD_JOURNAL_PATH"] = str(config.world_store)
    env["NOESIS_IDENTITY_V2_STORE"] = str(config.identity_store)
    env["NOESIS_IDENTITY_V2_EVIDENCE_PATH"] = str(evidence_root / "identity_v2.jsonl")
    env["NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID"] = session_id
    env["NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME"] = "ds9"
    env["NOESIS_REID_ALIAS_FILE"] = str(storage_base / "reid_aliases.json")
    env["NOESIS_REID_SID_POOL_FILE"] = str(storage_base / "sid_pool.json")
    env["NOESIS_REID_GALLERY_FILE"] = str(storage_base / "reid_gallery.npz")
    env["NOESIS_ANALYTICS_CONFIG"] = str(config.analytics_config)
    env["NOESIS_ANALYTICS_EXCLUDE_CONFIG"] = str(config.analytics_exclude)
    env["NOESIS_SCENE_STORE_PATH"] = str(config.scene_store)
    env["NOESIS_VIRTUAL_TWIN_ROOT"] = str(config.virtual_twin)
    env["NOESIS_CALIBRATION_AUDIT_DIR"] = str(evidence_root / "calibration")
    env["NOESIS_V3DT_DIAG_DIR"] = str(evidence_root / "v3dt")
    env["NOESIS_V3DT_DIAG_SESSION"] = session_id
    env["NOESIS_PGIE_PROFILE"] = lane["pgie_profile"]
    env["NOESIS_TRACKING_MODE"] = lane["tracking_mode"]
    env["NOESIS_MANUAL_DEPTH_MODEL"] = str(
        env.get("NOESIS_MANUAL_DEPTH_MODEL", "mapanything") or "mapanything"
    ).strip().lower()
    env["NOESIS_MOSAIC_RTSP_ENABLED"] = "0"
    env["NOESIS_MOSAIC_WEBRTC_ENABLED"] = "1"
    env["NOESIS_REID_ENABLED"] = "1"
    env["NOESIS_SHUTDOWN_GRACE_SECONDS"] = str(
        env.get("NOESIS_SHUTDOWN_GRACE_SECONDS", "75") or "75"
    )
    env.setdefault("NOESIS_CPU_MATH_THREADS", "1")
    env["NOESIS_SEMANTIC_CAPTURE_ROOT"] = str(
        storage_base / "semantic-seg" / "captures"
    )
    env["NOESIS_DEV_CONSOLE_LAUNCH_DIR"] = str(build_root)
    if env["NOESIS_MANUAL_DEPTH_MODEL"] not in {"mapanything", "da3metric-large"}:
        _fail("NOESIS_MANUAL_DEPTH_MODEL must be mapanything or da3metric-large")
    env.update(build_health_context_env(env, tracking_mode=lane["tracking_mode"]))
    return env


def prepare_session_directories(config: NativeHostConfig, session_id: str) -> dict[str, Path]:
    storage_base = config.runtime / "state" / session_id
    evidence_root = config.runtime / "evidence" / session_id / "runtime"
    build_root = config.runtime / "build" / session_id
    depth_root = config.runtime / "depth" / session_id
    for path in (
        storage_base,
        evidence_root,
        build_root,
        depth_root,
        config.gst_registry.parent,
        storage_base / "home",
        build_root / "cache",
        build_root / "xdg-runtime",
        build_root / "cuda-cache",
    ):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    return {
        "storage_base": depth_root,
        "session_state": storage_base,
        "evidence_root": evidence_root,
        "build_root": build_root,
    }


def run_preflight(
    config: NativeHostConfig,
    env: Mapping[str, str],
    *,
    tracking_mode: str = CANONICAL_TRACKING_MODE,
) -> None:
    lane = _runtime_lane(tracking_mode)
    script = DS9_ROOT / "scripts" / "ds9_preflight.py"
    result = subprocess.run(
        [
            str(config.venv_python),
            str(script),
            "--config",
            str(config.repo_root / lane["pipeline_config"]),
            "--cameras-config",
            str(config.repo_root / lane["cameras_config"]),
        ],
        check=False,
        cwd=str(config.repo_root),
        env=dict(env),
        timeout=180,
    )
    if result.returncode != 0:
        _fail(f"ds9_preflight failed with exit {result.returncode}")


def check_native(
    config: NativeHostConfig,
    *,
    tracking_mode: str = CANONICAL_TRACKING_MODE,
) -> dict[str, Any]:
    started = time.monotonic()
    lane = _runtime_lane(tracking_mode)
    validate_secret_files(config.secrets)
    platform = verify_platform_versions()
    python_origin: dict[str, Any] = verify_python_origin(config.venv_python)
    python_origin["dependency_constraints"] = verify_native_dependency_constraints(
        config.venv_python
    )
    artifacts = verify_artifact_realization(config)
    occupancy = port_occupancy()
    check_env = build_run_environment(
        config,
        session_id="check",
        storage_base=config.runtime / "depth" / "check",
        evidence_root=config.runtime / "evidence" / "check" / "runtime",
        build_root=config.runtime / "build" / "check",
        tracking_mode=lane["tracking_mode"],
    )
    run_preflight(config, check_env, tracking_mode=lane["tracking_mode"])
    return {
        "ok": True,
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "backend": "native_host",
        "checked_at_utc": utc_now(),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "platform": platform,
        "python": python_origin,
        "artifacts": artifacts,
        "ports": occupancy,
        "lane": {
            key: lane[key]
            for key in ("name", "pgie_profile", "model_size", "tracking_mode")
        },
    }


def build_run_command(
    config: NativeHostConfig,
    *,
    session_id: str,
    storage_base: Path,
    tracking_mode: str = CANONICAL_TRACKING_MODE,
) -> list[str]:
    return [
        str(config.venv_python),
        str(config.repo_root / "DS9" / "noesis" / "ds9_runtime.py"),
        *canonical_runtime_arguments(
            storage_base=storage_base,
            tracking_mode=tracking_mode,
        )[1:],
    ]


def run_native(
    config: NativeHostConfig,
    *,
    tracking_mode: str = CANONICAL_TRACKING_MODE,
) -> NoReturn:
    lane = _runtime_lane(tracking_mode)
    occupancy = port_occupancy()
    if occupancy["occupied"]:
        _fail(
            "canonical ports are occupied; stop the current runtime before native run: "
            + ",".join(str(port) for port in occupancy["occupied"])
        )
    session_id = new_session_id()
    directories = prepare_session_directories(config, session_id)
    env = build_run_environment(
        config,
        session_id=session_id,
        storage_base=directories["session_state"],
        evidence_root=directories["evidence_root"],
        build_root=directories["build_root"],
        tracking_mode=lane["tracking_mode"],
    )
    command = build_run_command(
        config,
        session_id=session_id,
        storage_base=directories["storage_base"],
        tracking_mode=lane["tracking_mode"],
    )
    os.execve(command[0], command, env)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical DS9.1 native-host supervisor")
    parser.add_argument("mode", choices=("check", "run"))
    parser.add_argument(
        "--tracking-mode",
        choices=OPT_IN_TRACKING_MODES,
        default=CANONICAL_TRACKING_MODE,
        help=(
            "Runtime tracking lane. Baseline remains the default; mv3dt is an "
            "explicit Kitchen/Family Room opt-in."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_native_config()
        if args.mode == "check":
            payload = check_native(config, tracking_mode=args.tracking_mode)
            json.dump(payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        run_native(config, tracking_mode=args.tracking_mode)
    except NativeRuntimeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
