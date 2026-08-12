#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.contracts.health import (  # noqa: E402
    CapabilityHealth,
    CapabilityStatus,
)
from noesis_core.private_paths import (  # noqa: E402
    atomic_create_private_file,
    ensure_private_directory,
    read_private_file,
    validate_private_file,
)

CANONICAL_PGIE_PROFILE = "yolo26"
CANONICAL_MODEL_SIZE = "m"
CANONICAL_TRACKING_MODE = "baseline"
CANONICAL_WS_HOST = "127.0.0.1"
CANONICAL_WS_PORT = 6008
CANONICAL_REST_HOST = "127.0.0.1"
CANONICAL_REST_PORT = 8080
PRODUCTION_SHUTDOWN_GRACE_S = 75.0
PRODUCTION_SHUTDOWN_TIMEOUT_S = 90.0
MAX_HEALTH_RESPONSE_BYTES = 64 * 1024
MAX_REPORT_BYTES = 1024 * 1024
MAX_ANALYTICS_CONFIG_BYTES = 4 * 1024 * 1024
REQUIRED_CAPABILITIES = ("tracking_observations", "global_world")
ANALYTICS_CONFIG_RELATIVE = Path("config/nvdsanalytics.yaml")
CANARY_STATE_MARKER_NAME = ".noesis-ds8-canary-state.json"
CANARY_STATE_MARKER = b'{"contract":"noesis.ds8.canary-state","version":1}\n'
CANARY_STATE_TOP_LEVEL_NAMES = frozenset(
    {
        CANARY_STATE_MARKER_NAME,
        "analytics",
        "build",
        "calibration",
        "depth",
        "diagnostics",
        "evidence",
        "home",
        "scene",
        "tmp",
        "virtual-twin",
        "world",
        "xdg",
    }
)
PASSTHROUGH_ENV_KEYS = (
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "DISPLAY",
    "GI_TYPELIB_PATH",
    "GST_PLUGIN_PATH",
    "GST_PLUGIN_PATH_1_0",
    "GST_PLUGIN_SCANNER",
    "GST_PLUGIN_SYSTEM_PATH",
    "GST_PLUGIN_SYSTEM_PATH_1_0",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "TZ",
    "XAUTHORITY",
)

ERROR_SIGNATURES = (
    "Fatal Python error",
    "Segmentation fault",
    "SIGSEGV",
    "Traceback",
    "CRITICAL",
    "ERROR",
    "Wait thread did not terminate cleanly",
    "pipeline stop timed out",
)

COMMON_REQUIRED_LOG_MARKERS = (
    "DS8 pipeline activated successfully",
    "MosaicH264ShmFeeder ready:",
    "WebRTC gateway capacity: 1 max, 1 warm slot(s)",
    "REST server listening on http://",
    "Orderly pipeline EOS request initiated",
    "Orderly pipeline EOS accepted:",
    "EOS received on pipeline (reason=shutdown_requested)",
    "pyservicemaker wait() returned (pipeline stopped)",
    "Shutdown complete",
)

SHUTDOWN_SUCCESS_MARKERS = {
    "orderly_eos_requested": "Orderly pipeline EOS request initiated",
    "orderly_eos_accepted": "Orderly pipeline EOS accepted:",
    "shutdown_eos_callback": "EOS received on pipeline (reason=shutdown_requested)",
    "servicemaker_wait_returned": "pyservicemaker wait() returned (pipeline stopped)",
    "shutdown_complete": "Shutdown complete",
}

BASELINE_REQUIRED_LOG_MARKERS = (
    "PGIE profile: yolo26",
    "YOLO26 m detection PGIE selected",
    *COMMON_REQUIRED_LOG_MARKERS,
)

V3DT_REQUIRED_LOG_MARKERS = (
    "PGIE profile: yolo26_seg",
    "YOLO26 PGIE size: s",
    "Tracking mode 'v3dt' resolved pipeline=",
    "V3DT patched nvtracker enabled:",
    *COMMON_REQUIRED_LOG_MARKERS,
)

# Backward-compatible alias for callers that mean the default baseline canary.
REQUIRED_LOG_MARKERS = BASELINE_REQUIRED_LOG_MARKERS

SANITIZED_ENV_KEYS = (
    "CODEX_HOME",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MENON_OWNER_PASSWORD",
    "MENON_OWNER_BOOTSTRAP_TOKEN",
    "HS_PASS",
    "NOESIS_DEPTH_API_FORCE_STUB",
    "NOESIS_DS8_FORCE_NATIVE_TEST_PIPELINE",
    "NOESIS_DS8_STUB_PIPELINE",
    "NOESIS_DS9_STUB_PIPELINE",
    "NOESIS_REID_TEST_MODE",
    "NOESIS_SKIP_CUDA_PREFLIGHT",
    "NOESIS_OBJECT_DEPTH_ALLOW_HOST_ROI_COPY",
    "NOESIS_REST_CORS_ALLOW_ALL",
    "NOESIS_REST_CORS_ORIGIN_REGEX",
    "NOESIS_REST_CORS_ORIGINS",
    "NOESIS_INFLUX_TOKEN",
    "NOESIS_MQTT_PASSWORD",
    "NOESIS_MOSAIC_WEBRTC_STUN_SERVER",
    "NOESIS_MOSAIC_WEBRTC_TURN_SERVER",
    "NOESIS_MOSAIC_H264_SHM",
    "NOESIS_GST_PLUGIN_DIR",
    "NOESIS_V3DT_AUTOGEN_CAMINFO",
    "NOESIS_V3DT_META_EXTRACT",
    "NOESIS_V3DT_OBJ3D_CACHE",
    "NOESIS_V3DT_USE_PATCHED_NVTRACKER",
    "NOESIS_WS_PORT_FALLBACK_TRIES",
    "NOESIS_WS_BIND_RETRY_TRIES",
    "NOESIS_CAMERA_SECRETS_FILE",
    "NOESIS_MAPANYTHING_API_KEY_FILE",
    "NOESIS_ANALYTICS_CONFIG",
    "NOESIS_ANALYTICS_EXCLUDE_CONFIG",
    "NOESIS_BUILD_DIR",
    "NOESIS_CALIBRATION_AUDIT_DIR",
    "NOESIS_CAMERA_CALIBRATION_FILE",
    "NOESIS_IDENTITY_V2_STORE",
    "NOESIS_IDENTITY_V2_EVIDENCE_PATH",
    "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID",
    "NOESIS_IDENTITY_V2_MIGRATION_REVIEW_REPORT",
    "NOESIS_REID_GALLERY_FILE",
    "NOESIS_REID_ALIAS_FILE",
    "NOESIS_REID_SID_POOL_FILE",
    "NOESIS_PLY_ALIGNMENT_FILE",
    "NOESIS_SCENE_STORE_PATH",
    "NOESIS_VIRTUAL_TWIN_ROOT",
    "NOESIS_WORLD_JOURNAL_PATH",
    "NOESIS_V3DT_DIAG_DIR",
    "NOESIS_V3DT_DIAG_LOG",
    "NOESIS_V3DT_DIAG_SESSION",
    "GST_DEBUG_DUMP_DOT_DIR",
    "GST_REGISTRY_1_0",
)


@dataclass(frozen=True)
class RuntimeContract:
    name: str
    entrypoint: Path
    pipeline_config: Path
    cameras_config: Path
    pgie_profile: str
    model_size: str
    tracking_mode: str
    required_log_markers: tuple[str, ...]
    scope: str
    expected_tracker_config: str | None = None
    expected_config_tracking_mode: str | None = None
    expected_reid_config: str | None = None
    expected_reid_engine: str | None = None
    expected_reid_layer: str | None = None
    expected_reid_dimension: int | None = None
    required_regular_artifacts: tuple[Path, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CanaryStatePaths:
    root: Path
    home: Path
    xdg_state: Path
    xdg_cache: Path
    xdg_config: Path
    xdg_data: Path
    xdg_runtime: Path
    python_user_base: Path
    cuda_cache: Path
    temporary: Path
    build: Path
    depth: Path
    analytics: Path
    analytics_config: Path
    analytics_exclude: Path
    household: Path
    identity_store: Path
    reid_gallery: Path
    reid_aliases: Path
    reid_sid_pool: Path
    world: Path
    world_journal: Path
    scene: Path
    scene_store: Path
    virtual_twin: Path
    calibration: Path
    calibration_audit: Path
    camera_calibration: Path
    ply_alignment: Path
    diagnostics: Path
    gst_registry: Path
    evidence: Path


@dataclass(frozen=True)
class RuntimeSecretPaths:
    camera_sources: Path
    mapanything_key: Path
    internal_auth_token: Path


BASELINE_RUNTIME_CONTRACT = RuntimeContract(
    name="baseline",
    entrypoint=Path("noesis/ds8_runtime.py"),
    pipeline_config=Path("config/infer.yaml"),
    cameras_config=Path("config/cameras.yaml"),
    pgie_profile=CANONICAL_PGIE_PROFILE,
    model_size=CANONICAL_MODEL_SIZE,
    tracking_mode=CANONICAL_TRACKING_MODE,
    required_log_markers=BASELINE_REQUIRED_LOG_MARKERS,
    scope="canonical_ds8_production_canary",
    expected_reid_config="pipelines/config_infer_secondary_reid_swin.ini",
    expected_reid_engine="models/engines/reid_swin_tiny_aicity156_dyn_b16_fp16.engine",
    expected_reid_layer="fc_pred",
    expected_reid_dimension=256,
)

V3DT_RUNTIME_CONTRACT = RuntimeContract(
    name="v3dt",
    entrypoint=Path("noesis/ds8_runtime_v3dt_reimpl.py"),
    pipeline_config=Path("config/infer_v3dt_reimpl_fast1056_mp4.yaml"),
    cameras_config=Path("config/cameras_v3dt_baseline.yaml"),
    pgie_profile="yolo26_seg",
    model_size="s",
    tracking_mode="v3dt",
    required_log_markers=V3DT_REQUIRED_LOG_MARKERS,
    scope="canonical_ds8_v3dt_production_canary",
    expected_tracker_config="config/v3dt/reimpl/nvtracker_sv3dt_yolo26s_fast.yml",
    expected_config_tracking_mode="v3dt",
    expected_reid_config="pipelines/config_infer_secondary_reid_swin.ini",
    expected_reid_engine="models/engines/reid_swin_tiny_aicity156_dyn_b16_fp16.engine",
    expected_reid_layer="fc_pred",
    expected_reid_dimension=256,
    required_regular_artifacts=(
        Path("build/gst-plugins-deepstream/libnvdsgst_tracker.so"),
    ),
    environment=(
        ("NOESIS_V3DT_AUTOGEN_CAMINFO", "0"),
        ("NOESIS_V3DT_META_EXTRACT", "1"),
        ("NOESIS_V3DT_USE_PATCHED_NVTRACKER", "1"),
    ),
)

RUNTIME_CONTRACTS = {
    contract.name: contract
    for contract in (BASELINE_RUNTIME_CONTRACT, V3DT_RUNTIME_CONTRACT)
}


@dataclass(frozen=True)
class CapabilityEvidence:
    instance_id: str
    run_id: str
    generated_at_us: int
    sequences: dict[str, int]

    def public_payload(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "run_id": self.run_id,
            "generated_at_us": self.generated_at_us,
            "sequences": dict(self.sequences),
        }


@dataclass(frozen=True)
class ReadinessResult:
    ok: bool
    elapsed_s: float
    evidence: CapabilityEvidence | None
    error: str = ""


@dataclass(frozen=True)
class StopResult:
    returncode: int | None
    elapsed_s: float
    forced_kill: bool
    error: str = ""
    signal_sent: bool = False


def _absolute_lexical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _trusted_runtime_pythonpath(
    *,
    repo_root: Path,
    operator_home: Path,
) -> str:
    """Preserve installed imports without retaining a writable user base.

    The child cannot use its isolated ``PYTHONUSERBASE`` to discover the host's
    installed application dependencies.  Carry only already-active absolute
    import directories that resolve below the operator's read-only user
    package tree or system library roots.  The checkout itself is deliberately
    excluded: the absolute child entrypoint inserts its own repository root,
    so a live checkout must never shadow a checkpoint launch.  In particular,
    do not forward an arbitrary ``PYTHONPATH`` environment value.
    """

    repository = repo_root.resolve(strict=True)
    user_library = (operator_home.resolve(strict=True) / ".local" / "lib")
    allowed_roots = (
        user_library,
        Path("/usr/lib"),
        Path("/usr/local/lib"),
    )
    result: list[str] = []
    for raw in sys.path:
        if not str(raw or "").strip():
            continue
        candidate = _absolute_lexical_path(raw)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not resolved.is_dir():
            continue
        if (
            resolved == repository
            or resolved.is_relative_to(repository)
            or repository.is_relative_to(resolved)
        ):
            continue
        if not any(
            resolved == root or resolved.is_relative_to(root)
            for root in allowed_roots
        ):
            continue
        rendered = os.fspath(resolved)
        if rendered not in result:
            result.append(rendered)
    if not result:
        raise RuntimeError("trusted DS8 runtime Python import path is incomplete")
    return os.pathsep.join(result)


def _ensure_private_descendant_directory(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("private state directory must be root-relative")
    current = root
    for component in relative.parts:
        current = ensure_private_directory(
            current / component,
            label="DS8 canary state",
        )
    return current


def _require_external_state_root(
    requested: Path,
    *,
    repo_root: Path,
) -> Path:
    expanded = Path(requested).expanduser()
    if not expanded.is_absolute():
        raise RuntimeError("DS8 canary state root must be an explicit absolute path")
    candidate = _absolute_lexical_path(expanded)
    repository = repo_root.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=False)
    if (
        resolved_candidate == repository
        or resolved_candidate.is_relative_to(repository)
        or repository.is_relative_to(resolved_candidate)
    ):
        raise RuntimeError(
            "DS8 canary state root must be external to and must not contain the checkout"
        )
    operator_home = _operator_home(os.environ).resolve(strict=True)
    if (
        resolved_candidate == operator_home
        or resolved_candidate.is_relative_to(operator_home)
        or operator_home.is_relative_to(resolved_candidate)
    ):
        raise RuntimeError(
            "DS8 canary state root must be separate from the operator home"
        )
    for ancestor in (candidate, *candidate.parents):
        if (ancestor / "SEALED.json").is_file():
            raise RuntimeError(
                "DS8 canary state root must not be inside a sealed checkpoint"
            )

    root = ensure_private_directory(candidate, label="DS8 canary state root")
    marker = root / CANARY_STATE_MARKER_NAME
    entries = {path.name for path in root.iterdir()}
    unexpected = entries - CANARY_STATE_TOP_LEVEL_NAMES
    if unexpected:
        raise RuntimeError(
            "DS8 canary state root contains unexpected entries: "
            + ", ".join(sorted(unexpected))
        )
    if marker.exists() or marker.is_symlink():
        payload = read_private_file(
            marker,
            label="DS8 canary state marker",
            max_bytes=1024,
        )
        if payload != CANARY_STATE_MARKER:
            raise RuntimeError("DS8 canary state marker contract drifted")
    elif entries:
        raise RuntimeError(
            "existing DS8 canary state root is not gate-owned; use an empty root"
        )
    else:
        atomic_create_private_file(
            marker,
            CANARY_STATE_MARKER,
            label="DS8 canary state marker",
            max_bytes=1024,
        )
    return root


def _load_or_seed_state_file(
    source: Path,
    destination: Path,
    *,
    label: str,
    max_bytes: int,
) -> None:
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"canonical {label} cannot be read") from exc
    if not payload or len(payload) > int(max_bytes):
        raise RuntimeError(f"canonical {label} is empty or oversized")
    if destination.exists() or destination.is_symlink():
        existing = read_private_file(
            destination,
            label=f"DS8 canary {label}",
            max_bytes=max_bytes,
        )
        if existing != payload:
            raise RuntimeError(
                f"reused DS8 canary state has {label} drift; use a fresh state root"
            )
        return
    atomic_create_private_file(
        destination,
        payload,
        label=f"DS8 canary {label}",
        max_bytes=max_bytes,
    )


def _prepare_canary_state(
    requested: Path,
    *,
    repo_root: Path,
) -> CanaryStatePaths:
    root = _require_external_state_root(requested, repo_root=repo_root)
    home = _ensure_private_descendant_directory(root, Path("home"))
    household = _ensure_private_descendant_directory(
        root,
        Path("home/.noesis/household"),
    )
    xdg_state = _ensure_private_descendant_directory(root, Path("xdg/state"))
    xdg_cache = _ensure_private_descendant_directory(root, Path("xdg/cache"))
    xdg_config = _ensure_private_descendant_directory(root, Path("xdg/config"))
    xdg_data = _ensure_private_descendant_directory(root, Path("xdg/data"))
    xdg_runtime = _ensure_private_descendant_directory(root, Path("xdg/runtime"))
    python_user_base = _ensure_private_descendant_directory(
        root,
        Path("home/.local"),
    )
    temporary = _ensure_private_descendant_directory(root, Path("tmp"))
    build = _ensure_private_descendant_directory(root, Path("build"))
    cuda_cache = _ensure_private_descendant_directory(root, Path("build/cuda-cache"))
    depth = _ensure_private_descendant_directory(root, Path("depth"))
    analytics = _ensure_private_descendant_directory(root, Path("analytics"))
    world = _ensure_private_descendant_directory(root, Path("world"))
    scene = _ensure_private_descendant_directory(root, Path("scene"))
    virtual_twin = _ensure_private_descendant_directory(root, Path("virtual-twin"))
    calibration = _ensure_private_descendant_directory(root, Path("calibration"))
    calibration_audit = _ensure_private_descendant_directory(
        root,
        Path("calibration/raw"),
    )
    diagnostics = _ensure_private_descendant_directory(root, Path("diagnostics"))
    evidence = _ensure_private_descendant_directory(root, Path("evidence"))
    state = CanaryStatePaths(
        root=root,
        home=home,
        xdg_state=xdg_state,
        xdg_cache=xdg_cache,
        xdg_config=xdg_config,
        xdg_data=xdg_data,
        xdg_runtime=xdg_runtime,
        python_user_base=python_user_base,
        cuda_cache=cuda_cache,
        temporary=temporary,
        build=build,
        depth=depth,
        analytics=analytics,
        analytics_config=analytics / "nvdsanalytics.yaml",
        analytics_exclude=analytics / "config_nvdsanalytics_exclude.ini",
        household=household,
        identity_store=household / "identity_v2.sqlite3",
        reid_gallery=household / "reid_gallery.npz",
        reid_aliases=household / "reid_aliases.json",
        reid_sid_pool=household / "sid_pool.json",
        world=world,
        world_journal=world / "world.sqlite3",
        scene=scene,
        scene_store=scene / "scene_releases.sqlite3",
        virtual_twin=virtual_twin,
        calibration=calibration,
        calibration_audit=calibration_audit,
        camera_calibration=calibration / "camera_calibration.json",
        ply_alignment=calibration / "ply_alignment.json",
        diagnostics=diagnostics,
        gst_registry=build / "gst-registry.bin",
        evidence=evidence,
    )
    _load_or_seed_state_file(
        repo_root / ANALYTICS_CONFIG_RELATIVE,
        state.analytics_config,
        label="analytics config",
        max_bytes=MAX_ANALYTICS_CONFIG_BYTES,
    )
    _load_or_seed_state_file(
        repo_root / "config/camera_calibration.json",
        state.camera_calibration,
        label="camera calibration",
        max_bytes=MAX_ANALYTICS_CONFIG_BYTES,
    )
    _load_or_seed_state_file(
        repo_root / "config/ply_alignment.json",
        state.ply_alignment,
        label="PLY alignment",
        max_bytes=MAX_ANALYTICS_CONFIG_BYTES,
    )
    return state


def _operator_home(env: Mapping[str, str]) -> Path:
    configured = str(env.get("HOME", "") or "").strip()
    return _absolute_lexical_path(configured or Path.home())


def _operator_file_path(
    env: Mapping[str, str],
    key: str,
    *,
    default: Path,
) -> Path:
    configured = str(env.get(key, "") or "").strip()
    return _absolute_lexical_path(configured or default)


def _resolve_runtime_secret_paths(
    env: Mapping[str, str],
    *,
    token_file: Path,
) -> RuntimeSecretPaths:
    home = _operator_home(env)
    return RuntimeSecretPaths(
        camera_sources=_operator_file_path(
            env,
            "NOESIS_CAMERA_SECRETS_FILE",
            default=home / ".local/state/noesis/secrets/camera_sources.json",
        ),
        mapanything_key=_operator_file_path(
            env,
            "NOESIS_MAPANYTHING_API_KEY_FILE",
            default=home / ".local/state/noesis/secrets/mapanything_rpc.key",
        ),
        internal_auth_token=_absolute_lexical_path(token_file),
    )


def _validate_runtime_secret_files(paths: RuntimeSecretPaths) -> None:
    for path, label in (
        (paths.camera_sources, "camera source secrets"),
        (paths.mapanything_key, "MapAnything RPC key"),
        (paths.internal_auth_token, "internal auth token"),
    ):
        validate_private_file(path, label=label)
        ensure_private_directory(path.parent, label=f"{label} parent")


def _resolve_evidence_path(
    explicit: Path | None,
    *,
    state: CanaryStatePaths,
    default_prefix: str,
    suffix: str,
    label: str,
) -> Path:
    if explicit is None:
        candidate = state.evidence / (
            f"{default_prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
            f"{secrets.token_hex(4)}{suffix}"
        )
    else:
        expanded = Path(explicit).expanduser()
        if not expanded.is_absolute():
            raise RuntimeError(f"{label} must be an explicit absolute path")
        candidate = _absolute_lexical_path(expanded)
    if not candidate.is_relative_to(state.evidence):
        raise RuntimeError(f"{label} must be inside the DS8 canary evidence directory")
    relative_parent = candidate.parent.relative_to(state.evidence)
    _ensure_private_descendant_directory(state.evidence, relative_parent)
    if candidate.exists() or candidate.is_symlink():
        raise RuntimeError(f"{label} already exists; use a caller-unique name")
    return candidate


def _open_private_log(path: Path) -> TextIO:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError("DS8 canary log cannot be created securely") from exc
    try:
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
        ):
            raise RuntimeError("DS8 canary log is not an owner-private regular file")
        return os.fdopen(descriptor, "w", encoding="utf-8", closefd=True)
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=False))


def _emit_report(payload: dict[str, Any], report_path: Path | None) -> None:
    """Print the verdict and optionally publish one immutable private report."""

    if report_path is not None:
        rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        atomic_create_private_file(
            report_path.expanduser().absolute(),
            rendered,
            label="DS8 canary report",
            max_bytes=MAX_REPORT_BYTES,
        )
    _json_print(payload)


def _reserve_requested_port(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)
    except Exception:
        sock.close()
        raise
    return sock


def _scan_log(
    log_path: Path,
    *,
    required_markers: Sequence[str] = REQUIRED_LOG_MARKERS,
) -> tuple[list[str], list[str], list[str]]:
    found: set[str] = set()
    samples: list[str] = []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return [], list(required_markers), []

    lines = text.splitlines()
    for line in lines:
        for signature in ERROR_SIGNATURES:
            if signature in line:
                found.add(signature)
                if len(samples) < 8:
                    samples.append(line.rstrip())

    missing_markers = [marker for marker in required_markers if marker not in text]
    if not any(
        marker in missing_markers for marker in SHUTDOWN_SUCCESS_MARKERS.values()
    ):
        positions = {
            name: text.rfind(marker)
            for name, marker in SHUTDOWN_SUCCESS_MARKERS.items()
        }
        requested = positions["orderly_eos_requested"]
        accepted = positions["orderly_eos_accepted"]
        callback = positions["shutdown_eos_callback"]
        wait_returned = positions["servicemaker_wait_returned"]
        complete = positions["shutdown_complete"]
        if requested > accepted:
            missing_markers.append(
                "orderly_eos_requested_before_orderly_eos_accepted"
            )
        if requested > callback:
            missing_markers.append(
                "orderly_eos_requested_before_shutdown_eos_callback"
            )
        if max(accepted, callback) > wait_returned:
            missing_markers.append(
                "orderly_eos_acceptance_and_callback_before_servicemaker_wait_returned"
            )
        if wait_returned > complete:
            missing_markers.append(
                "servicemaker_wait_returned_before_shutdown_complete"
            )
    if not samples and lines:
        samples.extend(line.rstrip() for line in lines[-8:] if line.strip())
    return sorted(found), missing_markers, samples[:8]


def _resolve_auth_token_file(
    explicit: Path | None,
    env: Mapping[str, str],
    *,
    home: Path | None = None,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().absolute()
    configured = str(env.get("NOESIS_INTERNAL_AUTH_TOKEN_FILE", "") or "").strip()
    if configured:
        return Path(configured).expanduser().absolute()
    base_home = Path.home() if home is None else Path(home)
    return (base_home / ".local" / "state" / "noesis" / "gateway-token").absolute()


def _load_auth_token(token_file: Path) -> str:
    from noesis.server.internal_auth import load_internal_token

    return load_internal_token(token_file)


def _build_runtime_env(
    base_env: Mapping[str, str],
    *,
    state: CanaryStatePaths,
    secret_paths: RuntimeSecretPaths,
    runtime_contract: RuntimeContract = BASELINE_RUNTIME_CONTRACT,
) -> dict[str, str]:
    operator_home = _operator_home(base_env)
    env = {
        key: str(base_env[key])
        for key in PASSTHROUGH_ENV_KEYS
        if str(base_env.get(key, "") or "").strip()
    }
    env.setdefault("PATH", os.defpath)
    env.update(
        {
            "HOME": str(state.home),
            "XDG_STATE_HOME": str(state.xdg_state),
            "XDG_CACHE_HOME": str(state.xdg_cache),
            "XDG_CONFIG_HOME": str(state.xdg_config),
            "XDG_DATA_HOME": str(state.xdg_data),
            "XDG_RUNTIME_DIR": str(state.xdg_runtime),
            "TMPDIR": str(state.temporary),
            "CUDA_CACHE_PATH": str(state.cuda_cache),
            "GST_REGISTRY_1_0": str(state.gst_registry),
            "GST_DEBUG_DUMP_DOT_DIR": str(state.diagnostics),
            "PYTHONUSERBASE": str(state.python_user_base),
            "PYTHONPATH": _trusted_runtime_pythonpath(
                repo_root=REPO_ROOT,
                operator_home=operator_home,
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "NOESIS_INTERNAL_AUTH_MODE": "required",
            "NOESIS_INTERNAL_AUTH_TOKEN_FILE": str(
                secret_paths.internal_auth_token
            ),
            "NOESIS_CAMERA_SECRETS_FILE": str(secret_paths.camera_sources),
            "NOESIS_MAPANYTHING_API_KEY_FILE": str(secret_paths.mapanything_key),
            "NOESIS_BUILD_DIR": str(state.build),
            "NOESIS_ANALYTICS_CONFIG": str(state.analytics_config),
            "NOESIS_ANALYTICS_EXCLUDE_CONFIG": str(state.analytics_exclude),
            "NOESIS_IDENTITY_V2_STORE": str(state.identity_store),
            "NOESIS_REID_GALLERY_FILE": str(state.reid_gallery),
            "NOESIS_REID_ALIAS_FILE": str(state.reid_aliases),
            "NOESIS_REID_SID_POOL_FILE": str(state.reid_sid_pool),
            "NOESIS_WORLD_JOURNAL_PATH": str(state.world_journal),
            "NOESIS_SCENE_STORE_PATH": str(state.scene_store),
            "NOESIS_VIRTUAL_TWIN_ROOT": str(state.virtual_twin),
            "NOESIS_CALIBRATION_AUDIT_DIR": str(state.calibration_audit),
            "NOESIS_CAMERA_CALIBRATION_FILE": str(state.camera_calibration),
            "NOESIS_PLY_ALIGNMENT_FILE": str(state.ply_alignment),
            "NOESIS_V3DT_DIAG_DIR": str(state.diagnostics),
            "NOESIS_HOUSEHOLD_ARCHIVE_STATE": "0",
            "NOESIS_PGIE_PROFILE": runtime_contract.pgie_profile,
            "NOESIS_TRACKING_MODE": runtime_contract.tracking_mode,
            "NOESIS_MOSAIC_RTSP_ENABLED": "0",
            "NOESIS_MOSAIC_WEBRTC_ENABLED": "1",
            "NOESIS_MOSAIC_WEBRTC_MAX_CLIENTS": "1",
            "NOESIS_MOSAIC_WEBRTC_INITIAL_CLIENTS": "1",
            "NOESIS_MOSAIC_H264_SHM": str(state.temporary / "mosaic-h264.sock"),
            "NOESIS_REID_ENABLED": "1",
            "NOESIS_SHUTDOWN_GRACE_SECONDS": str(int(PRODUCTION_SHUTDOWN_GRACE_S)),
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    env.update(dict(runtime_contract.environment))
    return env


def _build_runtime_command(
    *,
    repo_root: Path,
    ws_host: str,
    ws_port: int,
    rest_host: str,
    rest_port: int,
    storage_base: Path,
    python_bin: str = sys.executable,
    runtime_contract: RuntimeContract = BASELINE_RUNTIME_CONTRACT,
) -> list[str]:
    return [
        python_bin,
        "-X",
        "faulthandler",
        str(repo_root / runtime_contract.entrypoint),
        "--pipeline-config",
        str(repo_root / runtime_contract.pipeline_config),
        "--cameras-config",
        str(repo_root / runtime_contract.cameras_config),
        "--pgie-profile",
        runtime_contract.pgie_profile,
        "--size",
        runtime_contract.model_size,
        "--tracking-mode",
        runtime_contract.tracking_mode,
        "--ws-host",
        ws_host,
        "--ws-port",
        str(ws_port),
        "--rest-host",
        rest_host,
        "--rest-port",
        str(rest_port),
        "--storage-base",
        str(storage_base),
        "--enable-rest",
        "--log-level",
        "INFO",
    ]


def _contract_path(repo_root: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"runtime contract path is not repo-relative: {relative_path}")
    root = repo_root.resolve()
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"runtime contract path escapes the repository: {relative_path}")
    return resolved


def _validate_runtime_contract(
    runtime_contract: RuntimeContract,
    *,
    repo_root: Path,
) -> None:
    entrypoint = _contract_path(repo_root, runtime_contract.entrypoint)
    pipeline_config = _contract_path(repo_root, runtime_contract.pipeline_config)
    cameras_config = _contract_path(repo_root, runtime_contract.cameras_config)
    for label, path in (
        ("runtime entrypoint", entrypoint),
        ("pipeline config", pipeline_config),
        ("cameras config", cameras_config),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is missing: {path}")

    for relative_path in runtime_contract.required_regular_artifacts:
        lexical_artifact = repo_root.resolve() / relative_path
        if lexical_artifact.is_symlink():
            raise RuntimeError(
                "required repo-owned runtime artifact must be a regular, non-symlink "
                f"file: {lexical_artifact}"
            )
        artifact = _contract_path(repo_root, relative_path)
        if not artifact.is_file():
            raise RuntimeError(
                "required repo-owned runtime artifact must be a regular, non-symlink "
                f"file: {artifact}"
            )

    try:
        payload = yaml.safe_load(pipeline_config.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("pipeline config is not valid YAML") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("pipeline config must be a mapping")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("pipeline config must define at least one live source")
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise RuntimeError(f"source {index} must be a mapping")
        secret_ref = str(source.get("uri_secret") or "").strip()
        inline_uri = str(source.get("uri") or "").strip()
        if not secret_ref or inline_uri:
            raise RuntimeError(
                f"source {index} must use uri_secret exclusively for the live canary"
            )
        if str(source.get("element") or "").strip() != "nvurisrcbin":
            raise RuntimeError(
                f"source {index} must use the canonical nvurisrcbin live-source path"
            )

    streammux = payload.get("streammux")
    if not isinstance(streammux, Mapping) or streammux.get("live-source") != 1:
        raise RuntimeError("pipeline config must keep streammux.live-source=1")

    mosaic = payload.get("mosaic_output")
    if not isinstance(mosaic, Mapping):
        raise RuntimeError("pipeline config must define mosaic_output")
    if mosaic.get("rtsp_enabled") is not False:
        raise RuntimeError("pipeline config must disable optional RTSP output")
    if mosaic.get("mosaic_webrtc_enabled") is not True:
        raise RuntimeError("pipeline config must enable the WebRTC gateway")
    if not str(mosaic.get("mosaic_h264_shm_socket") or "").strip():
        raise RuntimeError("pipeline config must define the mosaic H.264 SHM socket")
    if str(mosaic.get("encoder") or "").strip() != "nvv4l2h264enc":
        raise RuntimeError("pipeline config must select the canonical GPU H.264 encoder")
    if int(mosaic.get("video_bitrate_kbps") or 0) != 12000:
        raise RuntimeError("pipeline config must select the 12000 kbps mosaic bitrate")
    if int(mosaic.get("h264_iframeinterval") or 0) != 10:
        raise RuntimeError("pipeline config must select a 10-frame I-frame interval")
    if int(mosaic.get("h264_idrinterval") or 0) != 10:
        raise RuntimeError("pipeline config must select a 10-frame IDR interval")

    expected_reid = (
        runtime_contract.expected_reid_config,
        runtime_contract.expected_reid_engine,
        runtime_contract.expected_reid_layer,
        runtime_contract.expected_reid_dimension,
    )
    if any(value is not None for value in expected_reid):
        models = payload.get("models")
        reid = models.get("reid") if isinstance(models, Mapping) else None
        if not isinstance(reid, Mapping) or reid.get("enable") is not True:
            raise RuntimeError("pipeline config must enable the contract-pinned ReID model")
        actual_reid = (
            str(reid.get("config-file-path") or "").strip(),
            str(reid.get("engine") or "").strip(),
            str(reid.get("layer") or "").strip(),
            reid.get("embedding_dim"),
        )
        if actual_reid != expected_reid:
            raise RuntimeError(
                "pipeline config does not select the contract-pinned ReID model"
            )
        reid_config = _contract_path(
            repo_root, Path(runtime_contract.expected_reid_config or "")
        )
        if not reid_config.is_file():
            raise RuntimeError(f"contract-pinned ReID config is missing: {reid_config}")
        reid_engine = repo_root.resolve() / Path(
            runtime_contract.expected_reid_engine or ""
        )
        if not reid_engine.is_file() or reid_engine.stat().st_size <= 0:
            raise RuntimeError(f"contract-pinned ReID engine is missing: {reid_engine}")

    if runtime_contract.expected_tracker_config is not None:
        tracker = payload.get("tracker")
        actual_tracker = (
            str(tracker.get("config-file") or "").strip()
            if isinstance(tracker, Mapping)
            else ""
        )
        if actual_tracker != runtime_contract.expected_tracker_config:
            raise RuntimeError(
                "pipeline config does not select the contract-pinned V3DT tracker"
            )
        tracker_path = _contract_path(repo_root, Path(actual_tracker))
        if not tracker_path.is_file():
            raise RuntimeError(f"contract-pinned V3DT tracker is missing: {tracker_path}")

    if runtime_contract.expected_config_tracking_mode is not None:
        v3dt = payload.get("v3dt")
        actual_mode = (
            str(v3dt.get("tracking_mode") or "").strip().lower()
            if isinstance(v3dt, Mapping)
            else ""
        )
        if actual_mode != runtime_contract.expected_config_tracking_mode:
            raise RuntimeError(
                "pipeline config does not declare the contract-pinned V3DT tracking mode"
            )


def _validate_capability_payload(payload: Any) -> CapabilityEvidence:
    health = CapabilityHealth.model_validate(payload)
    by_name = {row.capability: row for row in health.capabilities}
    sequences: dict[str, int] = {}
    for capability in REQUIRED_CAPABILITIES:
        row = by_name.get(capability)
        if (
            row is None
            or row.status != CapabilityStatus.HEALTHY
            or row.last_success_at_us is None
        ):
            raise ValueError(f"required capability is not healthy: {capability}")
        sequence = row.evidence.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError(
                f"required capability has invalid sequence evidence: {capability}"
            )
        sequences[capability] = sequence
    return CapabilityEvidence(
        instance_id=health.instance_id,
        run_id=health.run_id,
        generated_at_us=int(health.generated_at_us),
        sequences=sequences,
    )


def _require_advancement(
    previous: CapabilityEvidence, current: CapabilityEvidence
) -> None:
    if current.instance_id != previous.instance_id or current.run_id != previous.run_id:
        raise ValueError("runtime identity changed during readiness")
    for capability in REQUIRED_CAPABILITIES:
        before = previous.sequences[capability]
        after = current.sequences[capability]
        if after <= before:
            raise ValueError(
                f"required capability did not advance: {capability} before={before} after={after}"
            )


def _probe_capability_health(
    host: str, port: int, token: str, *, timeout_s: float
) -> CapabilityEvidence:
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.request(
            "GET",
            "/api/v1/health/capabilities",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        raw = response.read(MAX_HEALTH_RESPONSE_BYTES + 1)
        if response.status != 200:
            raise RuntimeError(f"capability endpoint returned HTTP {response.status}")
        if len(raw) > MAX_HEALTH_RESPONSE_BYTES:
            raise RuntimeError("capability response exceeded size bound")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("capability endpoint returned invalid JSON") from exc
        return _validate_capability_payload(payload)
    finally:
        connection.close()


def _probe_websocket_health(
    host: str, port: int, token: str, *, timeout_s: float
) -> None:
    from websockets.sync.client import connect

    expected = {
        "type": "health",
        "contract": "noesis.ws.health",
        "contract_version": 1,
    }
    with connect(
        f"ws://{host}:{port}/healthz",
        additional_headers={"Authorization": f"Bearer {token}"},
        compression=None,
        proxy=None,
        open_timeout=timeout_s,
        close_timeout=timeout_s,
        max_size=4096,
    ) as websocket:
        raw = websocket.recv(timeout=timeout_s)
    if not isinstance(raw, str):
        raise RuntimeError("WebSocket health response was not text")
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise RuntimeError("WebSocket health response was invalid JSON") from exc
    if payload != expected:
        raise RuntimeError("WebSocket health contract mismatch")


def _probe_webrtc_decoded_media(
    *,
    ws_host: str,
    ws_port: int,
    auth_token_file: Path,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "webrtc_gateway_smoke_test.py"),
        "--ws",
        f"ws://{ws_host}:{ws_port}",
        "--duration",
        "5",
        "--min-rtp",
        "10",
        "--min-decoded",
        "1",
        "--auth-token-file",
        str(auth_token_file),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no output").strip()[-2000:]
        raise RuntimeError(
            f"decoded WebRTC media probe failed with code {completed.returncode}: {detail}"
        )
    try:
        payload = json.loads(completed.stdout)
    except Exception as exc:
        raise RuntimeError("decoded WebRTC media probe returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("decoded WebRTC media probe did not report ok=true")
    return payload


def _wait_for_readiness(
    proc: subprocess.Popen[str],
    *,
    token: str,
    ws_host: str,
    ws_port: int,
    rest_host: str,
    rest_port: int,
    timeout_s: float,
    interval_s: float = 0.5,
) -> ReadinessResult:
    start = time.monotonic()
    deadline = start + timeout_s
    previous: CapabilityEvidence | None = None
    last_error = "runtime did not become ready"
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            return ReadinessResult(
                ok=False,
                elapsed_s=time.monotonic() - start,
                evidence=None,
                error=f"runtime exited before readiness with code {returncode}",
            )
        try:
            current = _probe_capability_health(
                rest_host, rest_port, token, timeout_s=2.0
            )
            if previous is not None:
                _require_advancement(previous, current)
                _probe_websocket_health(ws_host, ws_port, token, timeout_s=2.0)
                return ReadinessResult(
                    ok=True,
                    elapsed_s=time.monotonic() - start,
                    evidence=current,
                )
            previous = current
            last_error = "waiting for a second advancing capability sample"
        except Exception as exc:
            last_error = str(exc) or type(exc).__name__
        time.sleep(min(interval_s, max(0.0, deadline - time.monotonic())))
    return ReadinessResult(
        ok=False,
        elapsed_s=time.monotonic() - start,
        evidence=previous,
        error=last_error,
    )


def _probe_active_runtime(
    previous: CapabilityEvidence,
    *,
    token: str,
    ws_host: str,
    ws_port: int,
    rest_host: str,
    rest_port: int,
) -> CapabilityEvidence:
    current = _probe_capability_health(rest_host, rest_port, token, timeout_s=3.0)
    _require_advancement(previous, current)
    _probe_websocket_health(ws_host, ws_port, token, timeout_s=3.0)
    return current


def _stop_process(
    proc: subprocess.Popen[str], timeout_s: float = PRODUCTION_SHUTDOWN_TIMEOUT_S
) -> StopResult:
    start = time.monotonic()
    if proc.poll() is not None:
        return StopResult(returncode=proc.returncode, elapsed_s=0.0, forced_kill=False)
    try:
        proc.terminate()
    except ProcessLookupError:
        return StopResult(
            returncode=proc.poll(),
            elapsed_s=time.monotonic() - start,
            forced_kill=False,
            error="runtime disappeared before SIGTERM could be delivered",
        )
    except Exception as exc:
        error = f"SIGTERM failed: {exc}; sent SIGKILL to runtime process group"
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            error = f"{error}; process group did not exit after SIGKILL"
        return StopResult(
            returncode=proc.returncode,
            elapsed_s=time.monotonic() - start,
            forced_kill=True,
            error=error,
        )
    try:
        proc.wait(timeout=timeout_s)
        return StopResult(
            returncode=proc.returncode,
            elapsed_s=time.monotonic() - start,
            forced_kill=False,
            signal_sent=True,
        )
    except subprocess.TimeoutExpired:
        pass

    error = (
        "shutdown exceeded production timeout; sent SIGKILL to runtime process group"
    )
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        error = f"{error}; process group did not exit after SIGKILL"
    return StopResult(
        returncode=proc.returncode,
        elapsed_s=time.monotonic() - start,
        forced_kill=True,
        error=error,
        signal_sent=True,
    )


def _gate_ok(
    *,
    readiness_ok: bool,
    active_completed: bool,
    active_probe_ok: bool,
    running_before_shutdown: bool,
    stop: StopResult,
    signatures_found: Sequence[str],
    markers_missing: Sequence[str],
) -> bool:
    return bool(
        readiness_ok
        and active_completed
        and active_probe_ok
        and running_before_shutdown
        and stop.signal_sent
        and stop.returncode == 0
        and not stop.forced_kill
        and not stop.error
        and not signatures_found
        and not markers_missing
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical bounded DS8/V3DT production canary"
    )
    parser.add_argument(
        "--runtime-profile",
        choices=tuple(RUNTIME_CONTRACTS),
        default=BASELINE_RUNTIME_CONTRACT.name,
        help="Fixed runtime/config/model contract to validate (default: baseline)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=30.0,
        help="Required active duration after authenticated readiness",
    )
    parser.add_argument("--startup-timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--shutdown-timeout-s", type=float, default=PRODUCTION_SHUTDOWN_TIMEOUT_S
    )
    parser.add_argument("--ws-port", type=int, default=CANONICAL_WS_PORT)
    parser.add_argument("--rest-port", type=int, default=CANONICAL_REST_PORT)
    parser.add_argument(
        "--state-root",
        type=Path,
        required=True,
        help=(
            "Required owner-private external root for all mutable canary state; "
            "may be reused by sequential baseline/V3DT runs."
        ),
    )
    parser.add_argument("--auth-token-file", type=Path, default=None)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Fresh absolute path below <state-root>/evidence; generated if omitted.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help=(
            "Optional fresh absolute immutable JSON verdict path below "
            "<state-root>/evidence."
        ),
    )
    return parser.parse_args()


def _invalid_args(args: argparse.Namespace) -> str | None:
    if args.duration_s <= 0:
        return "duration must be > 0"
    if args.startup_timeout_s <= 0:
        return "startup timeout must be > 0"
    if args.shutdown_timeout_s < PRODUCTION_SHUTDOWN_GRACE_S + 15.0:
        return "shutdown timeout must exceed the production shutdown grace by at least 15 seconds"
    for label, port in (("ws", args.ws_port), ("rest", args.rest_port)):
        if not 1 <= int(port) <= 65535:
            return f"{label} port must be between 1 and 65535"
    if int(args.ws_port) == int(args.rest_port):
        return "WebSocket and REST ports must be distinct"
    return None


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    invalid = _invalid_args(args)
    if invalid:
        _emit_report({"ok": False, "notes": invalid}, None)
        return 2

    runtime_contract = RUNTIME_CONTRACTS[args.runtime_profile]
    try:
        _validate_runtime_contract(runtime_contract, repo_root=REPO_ROOT)
    except Exception as exc:
        _emit_report(
            {
                "ok": False,
                "runtime_profile": runtime_contract.name,
                "notes": f"runtime contract invalid: {exc}",
            },
            None,
        )
        return 2

    try:
        state = _prepare_canary_state(args.state_root, repo_root=REPO_ROOT)
        log_path = _resolve_evidence_path(
            args.log_path,
            state=state,
            default_prefix=f"ds8-{runtime_contract.name}-canary",
            suffix=".log",
            label="DS8 canary log path",
        )
        report_path = (
            _resolve_evidence_path(
                args.report_path,
                state=state,
                default_prefix=f"ds8-{runtime_contract.name}-canary",
                suffix=".json",
                label="DS8 canary report path",
            )
            if args.report_path is not None
            else None
        )
        if report_path == log_path:
            raise RuntimeError("DS8 canary log and report paths must be distinct")
    except Exception as exc:
        _emit_report(
            {
                "ok": False,
                "runtime_profile": runtime_contract.name,
                "notes": f"canary state invalid: {exc}",
            },
            None,
        )
        return 2

    token_file = _resolve_auth_token_file(args.auth_token_file, os.environ)
    secret_paths = _resolve_runtime_secret_paths(
        os.environ,
        token_file=token_file,
    )
    try:
        _validate_runtime_secret_files(secret_paths)
        token = _load_auth_token(token_file)
    except Exception as exc:
        _emit_report(
            {
                "ok": False,
                "log_path": str(log_path),
                "notes": f"required internal auth token unavailable: {exc}",
            },
            report_path,
        )
        return 2

    locks: list[socket.socket] = []
    try:
        for host, port in (
            (CANONICAL_WS_HOST, int(args.ws_port)),
            (CANONICAL_REST_HOST, int(args.rest_port)),
        ):
            locks.append(_reserve_requested_port(host, port))
    except OSError as exc:
        for lock in locks:
            lock.close()
        _emit_report(
            {
                "ok": False,
                "ws_port": int(args.ws_port),
                "rest_port": int(args.rest_port),
                "log_path": str(log_path),
                "notes": f"required canonical endpoint unavailable: {exc}",
            },
            report_path,
        )
        return 2

    cmd = _build_runtime_command(
        repo_root=REPO_ROOT,
        ws_host=CANONICAL_WS_HOST,
        ws_port=int(args.ws_port),
        rest_host=CANONICAL_REST_HOST,
        rest_port=int(args.rest_port),
        storage_base=state.depth,
        runtime_contract=runtime_contract,
    )
    child_env = _build_runtime_env(
        os.environ,
        state=state,
        secret_paths=secret_paths,
        runtime_contract=runtime_contract,
    )
    canonical_endpoints = (
        int(args.ws_port) == CANONICAL_WS_PORT
        and int(args.rest_port) == CANONICAL_REST_PORT
    )

    proc: subprocess.Popen[str] | None = None
    readiness = ReadinessResult(
        ok=False, elapsed_s=0.0, evidence=None, error="not started"
    )
    active_evidence: CapabilityEvidence | None = None
    webrtc_media_evidence: dict[str, Any] | None = None
    active_completed = False
    active_probe_ok = False
    running_before_shutdown = False
    stop = StopResult(
        returncode=None, elapsed_s=0.0, forced_kill=False, error="runtime not started"
    )
    notes: list[str] = []
    total_start = time.monotonic()
    active_elapsed_s = 0.0

    try:
        with _open_private_log(log_path) as logf:
            for lock in locks:
                lock.close()
            locks.clear()
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            readiness = _wait_for_readiness(
                proc,
                token=token,
                ws_host=CANONICAL_WS_HOST,
                ws_port=int(args.ws_port),
                rest_host=CANONICAL_REST_HOST,
                rest_port=int(args.rest_port),
                timeout_s=float(args.startup_timeout_s),
            )
            if not readiness.ok:
                notes.append(f"readiness failed: {readiness.error}")
            else:
                active_start = time.monotonic()
                deadline = active_start + float(args.duration_s)
                while time.monotonic() < deadline:
                    returncode = proc.poll()
                    if returncode is not None:
                        notes.append(
                            f"runtime exited during active window with code {returncode}"
                        )
                        break
                    time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
                active_elapsed_s = time.monotonic() - active_start
                active_completed = proc.poll() is None and active_elapsed_s >= float(
                    args.duration_s
                )
                if active_completed and readiness.evidence is not None:
                    try:
                        active_evidence = _probe_active_runtime(
                            readiness.evidence,
                            token=token,
                            ws_host=CANONICAL_WS_HOST,
                            ws_port=int(args.ws_port),
                            rest_host=CANONICAL_REST_HOST,
                            rest_port=int(args.rest_port),
                        )
                        webrtc_media_evidence = _probe_webrtc_decoded_media(
                            ws_host=CANONICAL_WS_HOST,
                            ws_port=int(args.ws_port),
                            auth_token_file=token_file,
                        )
                        active_probe_ok = True
                    except Exception as exc:
                        notes.append(f"end-of-window readiness failed: {exc}")
            running_before_shutdown = proc.poll() is None
    except KeyboardInterrupt:
        notes.append("interrupted")
    except Exception as exc:
        notes.append(f"gate execution failed: {type(exc).__name__}: {exc}")
    finally:
        for lock in locks:
            lock.close()
        if proc is not None:
            stop = _stop_process(proc, timeout_s=float(args.shutdown_timeout_s))

    signatures_found, markers_missing, samples = _scan_log(
        log_path,
        required_markers=runtime_contract.required_log_markers,
    )
    ok = _gate_ok(
        readiness_ok=readiness.ok,
        active_completed=active_completed,
        active_probe_ok=active_probe_ok,
        running_before_shutdown=running_before_shutdown,
        stop=stop,
        signatures_found=signatures_found,
        markers_missing=markers_missing,
    )
    if stop.error:
        notes.append(stop.error)
    if signatures_found:
        notes.append("fatal/error signatures found in runtime log")
    if markers_missing:
        notes.append("required runtime evidence markers missing")
    if not notes:
        notes.append("passed")

    payload: dict[str, Any] = {
        "ok": ok,
        "scope": (
            runtime_contract.scope
            if canonical_endpoints
            else f"{runtime_contract.scope}_loopback_override"
        ),
        "runtime_profile": runtime_contract.name,
        "profile": runtime_contract.pgie_profile,
        "model_size": runtime_contract.model_size,
        "tracking_mode": runtime_contract.tracking_mode,
        "canonical_endpoints": canonical_endpoints,
        "state_root": str(state.root),
        "mutable_state_isolated": True,
        "requested_active_duration_s": float(args.duration_s),
        "startup_elapsed_s": round(readiness.elapsed_s, 3),
        "active_elapsed_s": round(active_elapsed_s, 3),
        "shutdown_elapsed_s": round(stop.elapsed_s, 3),
        "total_elapsed_s": round(time.monotonic() - total_start, 3),
        "ws": f"ws://{CANONICAL_WS_HOST}:{int(args.ws_port)}",
        "rest": f"http://{CANONICAL_REST_HOST}:{int(args.rest_port)}",
        "mosaic_h264_shm": str(state.temporary / "mosaic-h264.sock"),
        "webrtc_media": webrtc_media_evidence,
        "log_path": str(log_path),
        "process_exit": stop.returncode,
        "forced_kill": stop.forced_kill,
        "sigterm_sent": stop.signal_sent,
        "readiness": (
            readiness.evidence.public_payload() if readiness.evidence else None
        ),
        "end_of_window_readiness": (
            active_evidence.public_payload() if active_evidence else None
        ),
        "signatures_found": signatures_found,
        "markers_missing": markers_missing,
        "notes": "; ".join(notes),
    }
    if samples:
        payload["samples"] = samples
    _emit_report(payload, report_path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
