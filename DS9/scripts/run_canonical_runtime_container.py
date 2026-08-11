#!/usr/bin/env python3
"""Launch the canonical DS9 runtime inside the isolated secondary Docker daemon.

The default ``plan`` command is write-free and never creates a container.  The
``run`` command is intentionally separate and requires an explicit GPU-runtime
authorization flag.  Both modes fail closed on provenance, ownership, port,
secret, and mount-contract drift.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import fcntl
import hashlib
import http.client
import json
import math
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

sys.dont_write_bytecode = True


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.server.internal_auth import load_internal_token  # noqa: E402
from noesis_core.private_paths import (  # noqa: E402
    PrivatePathError,
    atomic_write_private_file,
    ensure_private_directory,
    read_private_file,
    validate_private_file,
)
from noesis_core.appliance import (  # noqa: E402
    ApplianceBinding,
    ApplianceConfigurationError,
    FORBIDDEN_NESTED_LEASE_ENV,
    compute_noesis_checkout_identity,
    load_appliance_binding,
    require_appliance_state_current,
    runtime_context_environment,
)
from noesis_core.contracts.appliance import DS9RuntimeSelector  # noqa: E402
from noesis_core.runtime_secrets import (  # noqa: E402
    RuntimeSecretError,
    load_camera_uri_registry,
    load_mapanything_api_key,
)
from noesis_core.strict_json import (  # noqa: E402
    StrictJSONError,
    strict_json_loads,
)


CONTRACT = "noesis.ds9.canonical_runtime_container"
SCHEMA_VERSION = 1

IMAGE_REF = "noesis-ds9-runtime:9.0-20260710"
IMAGE_ID = "sha256:ca33b4c6a84fc56b86b71feee2a444299cb2ce7f33ab018cae43ac730aaef5fc"
PARENT_BUILD_IMAGE_REF = "noesis-ds9-dev:9.0-20260710"
PARENT_BUILD_IMAGE_ID = "sha256:7476b1021376cd67793c95d949cdc7d46eef7704ab98a5a76feed461e4f907a4"
BASE_DIGEST = "sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a"
DOCKERFILE_SHA256 = "f3345a4c87483a70dc48a7185330880805d38dbc97d18d57c4b08ab1e415d51e"
BUILD_DOCKERFILE_SHA256 = "c1df566ec73759a5bef0275dc5e9edcdc41be3f6b3f3a102a19d89a5f9860c9e"
REQUIREMENTS_SHA256 = "de35fb439f5c9bfd05d7fbc23436122aee033bd7b584b2eb139358e50211be48"
RUNTIME_ROOTFS_LAYER_DELTA = 2
REALIZATION_CONTRACT = "noesis.ds9.asset_realization"
REALIZATION_FILENAME = "asset_realization.json"
REALIZATION_MAX_BYTES = 16 * 1024 * 1024
ARTIFACT_TRANSACTION_LOCK_FILENAME = ".noesis-ds9-artifact-transaction.lock"
BASE_MANIFEST_RELATIVE = "DS9/asset_manifest.yaml"
SOURCE_CONTRACTS_RELATIVE = "DS9/config/engine_source_contracts.json"
OWNERSHIP_MATRIX_RELATIVE = "DS9/docs/runtime_ownership.yaml"
ANALYTICS_SEED_RELATIVE = "config/nvdsanalytics.yaml"
ANALYTICS_STATE_DIRECTORY = "analytics"
ANALYTICS_CONFIG_FILENAME = "nvdsanalytics.yaml"
ANALYTICS_EXCLUDE_FILENAME = "config_nvdsanalytics_exclude.ini"
ANALYTICS_YAML_MAX_BYTES = 4 * 1024 * 1024
ANALYTICS_EXCLUDE_INI_MAX_BYTES = 1024 * 1024
ANALYTICS_SEED_DIRECTORY_PREFIX = f".{ANALYTICS_STATE_DIRECTORY}.seed-"
_RENAME_NOREPLACE = 1
CANONICAL_ENGINE_ARTIFACT_IDS = frozenset(
    {
        "engine.yolo26_detect_m",
        "engine.reid_swin_tiny",
        "engine.pose_yolo26",
        "engine.depth_tracking_dav2",
        "engine.mapanything",
    }
)
V3DT_ENGINE_ARTIFACT_IDS = frozenset(
    {
        "engine.yolo26_seg_s",
        "engine.reid_swin_tiny",
        "engine.pose_yolo26",
        "engine.mapanything",
        "engine.v3dt_bodypose",
        "engine.v3dt_tracker_reid",
    }
)
WHOLEBODY49_S_ENGINE_ARTIFACT_IDS = frozenset(
    {
        "engine.wholebody49_s_masks",
        "engine.reid_swin_tiny",
        "engine.pose_yolo26",
        "engine.depth_tracking_dav2",
        "engine.mapanything",
        "engine.yolo26_detect_m",
    }
)
WHOLEBODY49_X_ENGINE_ARTIFACT_IDS = frozenset(
    {
        "engine.wholebody49_x_boxes",
        "engine.reid_swin_tiny",
        "engine.pose_yolo26",
        "engine.depth_tracking_dav2",
        "engine.mapanything",
        "engine.yolo26_detect_m",
    }
)

CANONICAL_PROFILE = "yolo26"
CANONICAL_SIZE = "m"
CANONICAL_TRACKING_MODE = "baseline"
V3DT_PROFILE = "yolo26_seg"
V3DT_SIZE = "s"
ALTERNATE_ONLY_PROFILES = ("yolo11_seg",)
CANONICAL_WS_PORT = 6008
CANONICAL_REST_PORT = 8080
CANONICAL_RTSP_PORT = 8554
CANONICAL_RTSP_PATH = "mosaic"
CANONICAL_PORTS = (CANONICAL_WS_PORT, CANONICAL_REST_PORT, CANONICAL_RTSP_PORT)
CANONICAL_ENDPOINTS = {
    "websocket": f"ws://127.0.0.1:{CANONICAL_WS_PORT}",
    "rest": f"http://127.0.0.1:{CANONICAL_REST_PORT}",
    "rtsp": f"rtsp://127.0.0.1:{CANONICAL_RTSP_PORT}/{CANONICAL_RTSP_PATH}",
}
RUNTIME_IDENTITY_CONTRACT = "noesis.ds9.supervisor_runtime_identity"
RUNTIME_IDENTITY_FILENAME = "runtime-identity.json"
RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
RUNTIME_MEMORY_BYTES = 26 * 1024 * 1024 * 1024
RESOURCE_SOAK_CONTRACT = "noesis.ds9.runtime_resource_soak"
RESOURCE_SOAK_SCHEMA_VERSION = 2
RESOURCE_SOAK_CONTRACT_VERSION = 2
RESOURCE_SOAK_MIN_DURATION_SECONDS = 300.0
RESOURCE_SOAK_SAMPLE_INTERVAL_SECONDS = 5.0
RESOURCE_SOAK_WARMUP_SECONDS = 60.0
RESOURCE_SOAK_MAX_SAMPLE_GAP_SECONDS = 7.5
RESOURCE_SOAK_MAX_INITIAL_SAMPLE_DELAY_SECONDS = 1.0
RESOURCE_SOAK_MAX_MEMORY_SLOPE_BYTES_PER_SECOND = 1 * 1024 * 1024
RESOURCE_SOAK_MAX_MEMORY_GROWTH_BYTES = 512 * 1024 * 1024
RESOURCE_SOAK_MAX_MEMORY_BYTES = 20 * 1024 * 1024 * 1024
RESOURCE_SOAK_MAX_PIDS_EXCLUSIVE = 4096
RESOURCE_SOAK_GPU_MEMORY_LIMIT_MIB: Mapping[str, int] = {
    "v3dt": 11_000,
    "wholebody49-s": 10_000,
    "wholebody49-x": 11_000,
}
RESOURCE_SOAK_PRIMARY_ENGINE_ID: Mapping[str, str] = {
    "v3dt": "engine.v3dt_tracker_reid",
    "wholebody49-s": "engine.wholebody49_s_masks",
    "wholebody49-x": "engine.wholebody49_x_boxes",
}
RESOURCE_SOAK_MIN_SAMPLE_COUNT = (
    int(RESOURCE_SOAK_MIN_DURATION_SECONDS / RESOURCE_SOAK_SAMPLE_INTERVAL_SECONDS) + 1
)
RESOURCE_SOAK_MIN_POST_WARMUP_SAMPLE_COUNT = (
    int(
        (RESOURCE_SOAK_MIN_DURATION_SECONDS - RESOURCE_SOAK_WARMUP_SECONDS)
        / RESOURCE_SOAK_SAMPLE_INTERVAL_SECONDS
    )
    + 1
)
RESOURCE_SOAK_SAMPLES_FILENAME = "runtime-resource-soak-samples.json"
RESOURCE_SOAK_REPORT_FILENAME = "runtime-resource-soak-report.json"
HOST_CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
HOST_PROC_ROOT = Path("/proc")
CHECKOUT_BINARY_ARTIFACT_KINDS = frozenset(
    {
        "native_extension",
        "gstreamer_plugin",
        "tensorrt_plugin",
        "nvinfer_parser",
    }
)

CONTAINER_REPO_ROOT = Path("/workspace")
CONTAINER_ARTIFACT_ROOT = Path("/opt/noesis/ds9-artifacts")
CONTAINER_BUILD_ROOT = Path("/var/lib/noesis/build")
CONTAINER_STATE_ROOT = Path("/var/lib/noesis/state")
CONTAINER_DEPTH_ROOT = Path("/var/lib/noesis/depth")
CONTAINER_EVIDENCE_ROOT = Path("/var/lib/noesis/evidence")
CONTAINER_SECRET_ROOT = Path("/run/noesis-secrets")
CONTAINER_ANALYTICS_ROOT = CONTAINER_STATE_ROOT / ANALYTICS_STATE_DIRECTORY
CONTAINER_ANALYTICS_CONFIG = CONTAINER_ANALYTICS_ROOT / ANALYTICS_CONFIG_FILENAME
CONTAINER_ANALYTICS_EXCLUDE_CONFIG = (
    CONTAINER_ANALYTICS_ROOT / ANALYTICS_EXCLUDE_FILENAME
)

ROLE_LABEL = "com.noesis.role=ds9-runtime"
SESSION_LABEL_KEY = "com.noesis.session"
LANE_LABEL_KEY = "com.noesis.runtime-lane"
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,47}$")
_SNAPSHOT_OPENAT_TEST_HOOK: Callable[[str, Path, Path], None] | None = None
EXPECTED_SECURITY_OPTIONS = ("no-new-privileges", "label=disable")
EXPECTED_APPARMOR_PROFILE = "docker-default"
EXPECTED_SENSITIVE_HOST_DEFAULTS: Mapping[str, object] = {
    "CgroupnsMode": "private",
    "OomKillDisable": None,
    "AutoRemove": False,
    "PublishAllPorts": False,
    "PortBindings": {},
    "ExtraHosts": None,
    "GroupAdd": None,
    "Links": None,
    "UTSMode": "",
    "VolumesFrom": None,
    "DeviceCgroupRules": None,
    "CgroupParent": "",
    "Dns": None,
    "DnsOptions": [],
    "DnsSearch": [],
    "ContainerIDFile": "",
    "VolumeDriver": "",
    "OomScoreAdj": 0,
    "Isolation": "",
    "Cgroup": "",
    "ShmSize": 64 * 1024 * 1024,
}
EXPECTED_ABSENT_SENSITIVE_HOST_FIELDS = frozenset({"StorageOpt", "Sysctls"})
EXPECTED_READONLY_PATHS = (
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger",
)
EXPECTED_MASKED_PATHS = (
    "/proc/acpi",
    "/proc/asound",
    "/proc/interrupts",
    "/proc/kcore",
    "/proc/keys",
    "/proc/latency_stats",
    "/proc/sched_debug",
    "/proc/scsi",
    "/proc/timer_list",
    "/proc/timer_stats",
    "/sys/devices/virtual/powercap",
    "/sys/firmware",
    *(f"/sys/devices/system/cpu/cpu{index}/thermal_throttle" for index in range(12)),
)

SHUTDOWN_MARKERS = (
    "Orderly pipeline EOS request initiated",
    "Orderly pipeline EOS accepted:",
    "EOS received on pipeline (reason=shutdown_requested)",
    "pyservicemaker wait() returned (pipeline stopped)",
    "Shutdown complete",
)
SHUTDOWN_FAILURE_SIGNATURES = (
    "Fatal Python error",
    "Segmentation fault",
    "SIGSEGV",
    "malloc():",
    "double free",
    "corrupted size",
    "corrupted double-linked list",
    "terminate called without an active exception",
    "Aborted (core dumped)",
    "Orderly pipeline EOS request failed",
    "Pipeline quiescence was not proven",
    "Native pipeline teardown failed",
    "pyservicemaker wait loop error",
    "Error joining wait thread",
    "wait thread did not terminate",
    "wait-thread did not terminate",
    "wait timed out",
    "wait() timed out",
    "wait timeout",
    "wait() timeout",
    "wait-timeout",
    "pipeline stop timed out",
    "failed to quiesce pipeline control timers",
    "GStreamer teardown failed",
    "free(): invalid pointer",
    "invalid pointer",
    "munmap_chunk",
    "pure virtual method called",
    "stack smashing detected",
    "bus error",
    "terminate called after throwing",
    "Failed to persist exclusion config",
    "Failed to write analytics config",
)


class RuntimeContainerError(RuntimeError):
    """Raised when the canonical runtime-container contract is not satisfied."""


@dataclass(frozen=True)
class RuntimeLane:
    """One reviewed runtime graph; arbitrary profile/config input is forbidden."""

    name: str
    artifact_profiles: tuple[str, ...]
    required_engine_ids: frozenset[str]
    pipeline_config: str
    cameras_config: str
    pgie_profile: str
    model_size: str
    tracking_mode: str


BASELINE_LANE = RuntimeLane(
    name="baseline",
    artifact_profiles=("canonical",),
    required_engine_ids=CANONICAL_ENGINE_ARTIFACT_IDS,
    pipeline_config="DS9/config/infer.yaml",
    cameras_config="config/cameras.yaml",
    pgie_profile=CANONICAL_PROFILE,
    model_size=CANONICAL_SIZE,
    tracking_mode=CANONICAL_TRACKING_MODE,
)
V3DT_LANE = RuntimeLane(
    name="v3dt",
    artifact_profiles=("v3dt",),
    required_engine_ids=V3DT_ENGINE_ARTIFACT_IDS,
    pipeline_config="DS9/config/infer_v3dt.yaml",
    cameras_config="DS9/config/cameras_v3dt.yaml",
    pgie_profile=V3DT_PROFILE,
    model_size=V3DT_SIZE,
    tracking_mode="v3dt",
)
WHOLEBODY49_S_LANE = RuntimeLane(
    name="wholebody49-s",
    artifact_profiles=(
        "runtime_common",
        "artifact:parser.wholebody49",
        "artifact:parser.yolo_detect",
        "artifact:engine.depth_tracking_dav2",
        "artifact:engine.yolo26_detect_m",
        "artifact:engine.wholebody49_s_masks",
    ),
    required_engine_ids=WHOLEBODY49_S_ENGINE_ARTIFACT_IDS,
    pipeline_config="DS9/config/infer.yaml",
    cameras_config="config/cameras.yaml",
    pgie_profile="wholebody49",
    model_size="s",
    tracking_mode="baseline",
)
WHOLEBODY49_X_LANE = RuntimeLane(
    name="wholebody49-x",
    artifact_profiles=(
        "runtime_common",
        "artifact:parser.wholebody49",
        "artifact:parser.yolo_detect",
        "artifact:engine.depth_tracking_dav2",
        "artifact:engine.yolo26_detect_m",
        "artifact:engine.wholebody49_x_boxes",
    ),
    required_engine_ids=WHOLEBODY49_X_ENGINE_ARTIFACT_IDS,
    pipeline_config="DS9/config/infer.yaml",
    cameras_config="config/cameras.yaml",
    pgie_profile="wholebody49",
    model_size="x",
    tracking_mode="baseline",
)
RUNTIME_LANES: Mapping[str, RuntimeLane] = {
    lane.name: lane
    for lane in (
        BASELINE_LANE,
        V3DT_LANE,
        WHOLEBODY49_S_LANE,
        WHOLEBODY49_X_LANE,
    )
}


def resolve_runtime_lane(value: str | RuntimeLane) -> RuntimeLane:
    if isinstance(value, RuntimeLane):
        reviewed = RUNTIME_LANES.get(value.name)
        if reviewed != value:
            raise RuntimeContainerError(f"unreviewed DS9 runtime lane: {value.name!r}")
        return value
    name = str(value or "").strip().lower()
    lane = RUNTIME_LANES.get(name)
    if lane is None:
        raise RuntimeContainerError(
            f"unsupported DS9 runtime lane {value!r}; expected one of {sorted(RUNTIME_LANES)}"
        )
    return lane


@dataclass(frozen=True)
class HostRoots:
    docker: Path
    artifacts: Path
    runtime: Path

    @property
    def docker_socket(self) -> Path:
        return self.docker / "run" / "docker.sock"

    @property
    def docker_data(self) -> Path:
        return self.docker / "data"


@dataclass(frozen=True)
class SecretFiles:
    cameras: Path
    mapanything: Path
    internal_auth: Path


@dataclass(frozen=True)
class SessionPaths:
    session_id: str
    build: Path
    state: Path
    depth: Path
    evidence: Path
    runtime_evidence: Path
    launcher_evidence: Path
    persistent_analytics: Path

    @classmethod
    def from_root(cls, root: Path, session_id: str) -> "SessionPaths":
        evidence = root / "evidence" / session_id
        return cls(
            session_id=session_id,
            build=root / "build" / session_id,
            state=root / "state" / session_id,
            depth=root / "depth" / session_id,
            evidence=evidence,
            runtime_evidence=evidence / "runtime",
            launcher_evidence=evidence / "launcher",
            persistent_analytics=root / "persistent" / ANALYTICS_STATE_DIRECTORY,
        )

    def writable_mounts(self) -> tuple[tuple[Path, Path], ...]:
        return (
            (self.build, CONTAINER_BUILD_ROOT),
            (self.state, CONTAINER_STATE_ROOT),
            (self.depth, CONTAINER_DEPTH_ROOT),
            (self.runtime_evidence, CONTAINER_EVIDENCE_ROOT),
            (self.persistent_analytics, CONTAINER_ANALYTICS_ROOT),
        )


def _appliance_analytics_root(binding: ApplianceBinding) -> Path:
    config = binding.state.runtime_files["analytics_config"]
    exclude = binding.state.runtime_files["analytics_exclude"]
    if config.parent != exclude.parent:
        raise RuntimeContainerError(
            "selected appliance analytics files do not share one state directory"
        )
    return config.parent


def _runtime_writable_mounts(
    session: SessionPaths,
    appliance_binding: ApplianceBinding | None,
) -> tuple[tuple[Path, Path], ...]:
    if appliance_binding is None:
        return session.writable_mounts()
    state = appliance_binding.state
    return (
        (state.build_directory, CONTAINER_BUILD_ROOT),
        (session.state, CONTAINER_STATE_ROOT),
        (
            state.runtime_files["world_store"],
            CONTAINER_STATE_ROOT / "world_ds9.sqlite3",
        ),
        (
            state.runtime_files["identity_store"],
            CONTAINER_STATE_ROOT / "household" / "identity_v2.sqlite3",
        ),
        (_appliance_analytics_root(appliance_binding), CONTAINER_ANALYTICS_ROOT),
        (session.depth, CONTAINER_DEPTH_ROOT),
        (session.runtime_evidence, CONTAINER_EVIDENCE_ROOT),
    )


@dataclass(frozen=True)
class CheckoutSnapshot:
    digest: str
    file_count: int
    byte_count: int
    entries: Mapping[str, str]
    manifest_binary_paths: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "sha256": self.digest,
            "file_count": self.file_count,
            "byte_count": self.byte_count,
            "manifest_binary_count": len(self.manifest_binary_paths),
            "manifest_binary_paths": list(self.manifest_binary_paths),
        }


@dataclass(frozen=True)
class DockerState:
    daemon_id: str
    docker_root: str
    default_runtime: str
    image_id: str
    base_digest: str
    parent_build_image_reference: str
    parent_build_image_id: str
    parent_rootfs_layer_count: int
    runtime_rootfs_layer_count: int
    parent_rootfs_sha256: str
    runtime_rootfs_sha256: str
    networks: Mapping[str, str]


class CommandRunner:
    """Small subprocess boundary so tests can exercise a fake Docker binary."""

    def run(
        self,
        command: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeContainerError(
                f"command failed to execute: {_command_name(command)}: {exc}"
            ) from exc
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeContainerError(
                f"command exited {result.returncode}: {_command_name(command)}"
                + (f": {detail}" if detail else "")
            )
        return result


def _command_name(command: Sequence[str]) -> str:
    return " ".join(str(value) for value in command[:4])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fZ").lower()
    return f"{stamp}-{secrets.token_hex(3)}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that treats duplicate mapping keys as corruption."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _path_text_safe(path: Path, *, label: str) -> None:
    text = os.fspath(path)
    if any(value in text for value in ("\x00", "\n", "\r", ",")):
        raise RuntimeContainerError(
            f"{label} contains a character unsupported by Docker bind mounts"
        )


def _absolute_without_symlinks(
    raw: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        raise RuntimeContainerError(f"{label} must be an explicit absolute path")
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    _path_text_safe(candidate, label=label)
    if candidate in {Path("/"), Path("/var"), Path("/var/lib"), Path("/tmp")}:
        raise RuntimeContainerError(f"refusing unsafe {label}: {candidate}")
    cursor = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        cursor /= component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise RuntimeContainerError(f"{label} cannot be inspected") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeContainerError(f"{label} must not contain symlink components")
    if must_exist:
        try:
            info = candidate.lstat()
        except FileNotFoundError as exc:
            raise RuntimeContainerError(f"{label} is missing: {candidate}") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeContainerError(f"{label} must be a directory: {candidate}")
    return candidate


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_disjoint_paths(paths: Mapping[str, Path]) -> None:
    items = list(paths.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if _contains(left, right) or _contains(right, left):
                raise RuntimeContainerError(
                    f"{left_name} and {right_name} must be disjoint: {left} / {right}"
                )


def resolve_host_roots(env: Mapping[str, str]) -> HostRoots:
    values: dict[str, Path] = {}
    for name, key, must_exist in (
        ("secondary Docker root", "NOESIS_DS9_DOCKER_ROOT", True),
        ("DS9 artifact root", "NOESIS_DS9_ARTIFACT_ROOT", True),
        ("DS9 runtime root", "NOESIS_DS9_RUNTIME_ROOT", False),
    ):
        raw = str(env.get(key, "") or "").strip()
        if not raw:
            raise RuntimeContainerError(f"{key} must be set explicitly")
        values[key] = _absolute_without_symlinks(
            raw,
            label=name,
            must_exist=must_exist,
        )
    repo = REPO_ROOT.absolute()
    _require_disjoint_paths(
        {
            "repository checkout": repo,
            "secondary Docker root": values["NOESIS_DS9_DOCKER_ROOT"],
            "DS9 artifact root": values["NOESIS_DS9_ARTIFACT_ROOT"],
            "DS9 runtime root": values["NOESIS_DS9_RUNTIME_ROOT"],
        }
    )
    return HostRoots(
        docker=values["NOESIS_DS9_DOCKER_ROOT"],
        artifacts=values["NOESIS_DS9_ARTIFACT_ROOT"],
        runtime=values["NOESIS_DS9_RUNTIME_ROOT"],
    )


def resolve_secret_files(env: Mapping[str, str], roots: HostRoots) -> SecretFiles:
    home = Path.home()
    raw_paths = {
        "camera source secrets": env.get(
            "NOESIS_CAMERA_SECRETS_FILE",
            str(home / ".local/state/noesis/secrets/camera_sources.json"),
        ),
        "MapAnything RPC key": env.get(
            "NOESIS_MAPANYTHING_API_KEY_FILE",
            str(home / ".local/state/noesis/secrets/mapanything_rpc.key"),
        ),
        "internal auth token": env.get(
            "NOESIS_INTERNAL_AUTH_TOKEN_FILE",
            str(home / ".local/state/noesis/gateway-token"),
        ),
    }
    resolved: dict[str, Path] = {}
    for label, raw in raw_paths.items():
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            raise RuntimeContainerError(f"{label} path must be absolute")
        path = Path(os.path.abspath(os.fspath(path)))
        _path_text_safe(path, label=label)
        try:
            resolved[label] = validate_private_file(path, label=label)
            ensure_private_directory(path.parent, label=f"{label} parent")
        except PrivatePathError as exc:
            raise RuntimeContainerError(str(exc)) from exc
        for root_label, root in (
            ("repository checkout", REPO_ROOT.absolute()),
            ("DS9 artifact root", roots.artifacts),
            ("DS9 runtime root", roots.runtime),
            ("secondary Docker root", roots.docker),
        ):
            if _contains(root, path):
                raise RuntimeContainerError(
                    f"{label} must be mounted only as an individual file; it is inside {root_label}"
                )
    if len(set(resolved.values())) != len(resolved):
        raise RuntimeContainerError("runtime secret files must be three distinct files")
    try:
        load_camera_uri_registry(resolved["camera source secrets"])
        load_mapanything_api_key(resolved["MapAnything RPC key"])
        load_internal_token(resolved["internal auth token"])
    except (RuntimeSecretError, Exception) as exc:
        # The broad catch intentionally normalizes the internal-auth error type
        # without including secret contents in the launcher transcript.
        raise RuntimeContainerError(f"runtime secret validation failed: {exc}") from exc
    return SecretFiles(
        cameras=resolved["camera source secrets"],
        mapanything=resolved["MapAnything RPC key"],
        internal_auth=resolved["internal auth token"],
    )


def validate_repository_provenance() -> dict[str, str]:
    files = {
        "runtime_dockerfile": (
            DS9_ROOT / "docker" / "Dockerfile.runtime",
            DOCKERFILE_SHA256,
        ),
        "build_dockerfile": (
            DS9_ROOT / "docker" / "Dockerfile",
            BUILD_DOCKERFILE_SHA256,
        ),
        "requirements_lock": (
            DS9_ROOT / "docker" / "requirements.lock.txt",
            REQUIREMENTS_SHA256,
        ),
    }
    result: dict[str, str] = {}
    for label, (path, expected) in files.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeContainerError(f"pinned {label} is missing or linked: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeContainerError(
                f"pinned {label} digest changed: expected={expected} actual={actual}"
            )
        result[label] = actual
    return result


def _strict_json_value(payload: bytes | str, *, label: str) -> Any:
    try:
        return strict_json_loads(payload, label=label)
    except StrictJSONError as exc:
        raise RuntimeContainerError(
            f"{label} failed strict JSON validation: {exc.reason}"
        ) from exc


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    decoded = _strict_json_value(payload, label=label)
    if not isinstance(decoded, Mapping):
        raise RuntimeContainerError(f"{label} root must be an object")
    return decoded


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise RuntimeContainerError(
            f"{label} keys drifted: missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}"
        )


def _utc_timestamp(value: object, *, label: str) -> datetime:
    raw = str(value or "").strip()
    if not raw.endswith("Z"):
        raise RuntimeContainerError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeContainerError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeContainerError(f"{label} must be UTC")
    return parsed


def _realization_structure(
    realization_path: Path,
    *,
    base_manifest: Mapping[str, Any],
    required_engine_ids: frozenset[str] = CANONICAL_ENGINE_ARTIFACT_IDS,
) -> tuple[Mapping[str, Any], dict[str, str], str]:
    try:
        realization_path = validate_private_file(
            realization_path,
            label="DS9 asset realization",
        )
        payload = read_private_file(
            realization_path,
            label="DS9 asset realization",
            max_bytes=REALIZATION_MAX_BYTES,
        )
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc
    realization_sha256 = hashlib.sha256(payload).hexdigest()
    realization = _strict_json(payload, label="DS9 asset realization")
    _require_exact_keys(
        realization,
        {
            "schema_version",
            "contract",
            "base_manifest",
            "source_contracts",
            "created_at_utc",
            "updated_at_utc",
            "artifacts",
        },
        label="DS9 asset realization",
    )
    if realization.get("schema_version") != 1:
        raise RuntimeContainerError("DS9 asset realization schema_version must be 1")
    if realization.get("contract") != REALIZATION_CONTRACT:
        raise RuntimeContainerError("DS9 asset realization contract identifier drifted")
    created = _utc_timestamp(realization.get("created_at_utc"), label="created_at_utc")
    updated = _utc_timestamp(realization.get("updated_at_utc"), label="updated_at_utc")
    if updated < created:
        raise RuntimeContainerError(
            "DS9 asset realization updated_at_utc predates creation"
        )

    anchors: dict[str, str] = {}
    for key, relative in (
        ("base_manifest", BASE_MANIFEST_RELATIVE),
        ("source_contracts", SOURCE_CONTRACTS_RELATIVE),
    ):
        raw_anchor = realization.get(key)
        if not isinstance(raw_anchor, Mapping):
            raise RuntimeContainerError(
                f"DS9 asset realization {key} must be an object"
            )
        _require_exact_keys(raw_anchor, {"path", "sha256"}, label=f"realization {key}")
        if raw_anchor.get("path") != relative:
            raise RuntimeContainerError(
                f"DS9 asset realization {key} path drifted: {raw_anchor.get('path')!r}"
            )
        expected = _sha256_file(REPO_ROOT / relative)
        observed = str(raw_anchor.get("sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", observed):
            raise RuntimeContainerError(
                f"DS9 asset realization {key} digest is invalid"
            )
        if observed != expected:
            raise RuntimeContainerError(
                f"DS9 asset realization {key} is stale: expected={expected} observed={observed}"
            )
        anchors[key] = expected

    base_rows = base_manifest.get("artifacts")
    if not isinstance(base_rows, list):
        raise RuntimeContainerError(
            "tracked DS9 asset manifest artifacts must be a list"
        )
    base_by_id: dict[str, Mapping[str, Any]] = {}
    for row in base_rows:
        if not isinstance(row, Mapping):
            continue
        artifact_id = str(row.get("id") or "")
        if artifact_id in base_by_id:
            raise RuntimeContainerError(
                f"tracked DS9 asset manifest repeats artifact ID {artifact_id!r}"
            )
        base_by_id[artifact_id] = row
    artifacts = realization.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeContainerError("DS9 asset realization artifacts must be an object")
    unknown_required = sorted(set(required_engine_ids) - set(base_by_id))
    if unknown_required:
        raise RuntimeContainerError(
            "reviewed DS9 runtime lane references unknown engine IDs: "
            + ", ".join(unknown_required)
        )
    missing_required = sorted(set(required_engine_ids) - set(artifacts))
    if missing_required:
        raise RuntimeContainerError(
            "DS9 asset realization lacks required engine entries: "
            + ", ".join(missing_required)
        )
    for artifact_id, overlay in artifacts.items():
        artifact_id = str(artifact_id)
        base = base_by_id.get(artifact_id)
        if base is None:
            raise RuntimeContainerError(
                f"DS9 asset realization contains unknown artifact ID {artifact_id!r}"
            )
        if base.get("kind") != "tensorrt_engine":
            raise RuntimeContainerError(
                f"DS9 asset realization may override only TensorRT engines: {artifact_id}"
            )
        if not isinstance(overlay, Mapping):
            raise RuntimeContainerError(
                f"DS9 asset realization entry {artifact_id} must be an object"
            )
        _require_exact_keys(
            overlay,
            {"state", "provenance"},
            label=f"realization artifact {artifact_id}",
        )
        if overlay.get("state") not in {"staged_unverified", "validated"}:
            raise RuntimeContainerError(
                f"DS9 asset realization entry {artifact_id} has invalid state"
            )
        provenance = overlay.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise RuntimeContainerError(
                f"DS9 asset realization entry {artifact_id} lacks provenance"
            )
    return realization, anchors, realization_sha256


def _canonical_engine_host_compatibility(
    realization: Mapping[str, Any],
    host_gpu: Mapping[str, object],
    *,
    required_engine_ids: frozenset[str] = CANONICAL_ENGINE_ARTIFACT_IDS,
) -> dict[str, object]:
    """Prove every selected engine was built on the current physical GPU/driver."""

    expected_host = {
        "driver_version": str(host_gpu.get("driver_version") or ""),
        "gpu_name": str(host_gpu.get("name") or ""),
        "gpu_uuid": str(host_gpu.get("uuid") or ""),
        "gpu_compute_capability": str(host_gpu.get("compute_capability") or ""),
        "gpu_memory_mib": int(host_gpu.get("memory_mib") or 0),
    }
    if (
        not expected_host["driver_version"]
        or not expected_host["gpu_name"]
        or not expected_host["gpu_uuid"].startswith("GPU-")
        or not re.fullmatch(r"\d+\.\d+", expected_host["gpu_compute_capability"])
        or expected_host["gpu_memory_mib"] <= 0
    ):
        raise RuntimeContainerError("current GPU 0 identity is incomplete")

    artifacts = realization.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeContainerError("DS9 asset realization artifacts must be an object")
    builds: dict[str, dict[str, object]] = {}
    mismatches: list[str] = []
    for artifact_id in sorted(required_engine_ids):
        overlay = artifacts.get(artifact_id)
        provenance = overlay.get("provenance") if isinstance(overlay, Mapping) else None
        maintenance = (
            provenance.get("maintenance") if isinstance(provenance, Mapping) else None
        )
        gpu = maintenance.get("gpu") if isinstance(maintenance, Mapping) else None
        if not isinstance(maintenance, Mapping) or not isinstance(gpu, Mapping):
            raise RuntimeContainerError(
                f"required engine lacks realized build-platform provenance: {artifact_id}"
            )
        try:
            observed = {
                "driver_version": str(maintenance.get("driver_version") or ""),
                "gpu_name": str(gpu.get("name") or ""),
                "gpu_uuid": str(gpu.get("uuid") or ""),
                "gpu_compute_capability": str(gpu.get("compute_capability") or ""),
                "gpu_memory_mib": int(gpu.get("memory_mib") or 0),
            }
        except (TypeError, ValueError) as exc:
            raise RuntimeContainerError(
                f"required engine has invalid build-platform provenance: {artifact_id}"
            ) from exc
        builds[artifact_id] = observed
        if observed != expected_host:
            differing = sorted(
                key
                for key, expected in expected_host.items()
                if observed[key] != expected
            )
            mismatches.append(f"{artifact_id} ({', '.join(differing)})")
    if mismatches:
        raise RuntimeContainerError(
            "required engine build host differs from current GPU 0: "
            + "; ".join(mismatches)
        )
    return {
        "ok": True,
        "policy": "exact_physical_gpu_and_driver",
        "host_gpu": dict(host_gpu),
        "required_engine_builds": builds,
        # Retained for readers of baseline evidence written before lane support.
        "canonical_engine_builds": builds,
    }


def validate_runtime_artifacts(
    artifact_root: Path,
    *,
    host_gpu: Mapping[str, object],
    lane: RuntimeLane = BASELINE_LANE,
) -> dict[str, object]:
    """Require the exact external realization for one reviewed runtime lane."""

    lane = resolve_runtime_lane(lane)

    scripts_root = DS9_ROOT / "scripts"
    scripts_text = str(scripts_root)
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)
    try:
        from validate_asset_manifest import validate_asset_realization

        base_manifest = yaml.safe_load(
            (DS9_ROOT / "asset_manifest.yaml").read_text(encoding="utf-8")
        )
        if not isinstance(base_manifest, Mapping):
            raise RuntimeContainerError("DS9 asset manifest root must be a mapping")
        realization_path = artifact_root / REALIZATION_FILENAME
        realization, anchors, realization_sha256 = _realization_structure(
            realization_path,
            base_manifest=base_manifest,
            required_engine_ids=lane.required_engine_ids,
        )
        profile_results: dict[str, Mapping[str, Any]] = {}
        for profile in lane.artifact_profiles:
            result = validate_asset_realization(
                base_manifest,
                realization_path,
                artifact_root,
                profile=profile,
                check_files=True,
                require_provenance=True,
            )
            if not bool(result.get("ok")):
                raise RuntimeContainerError(
                    "authoritative DS9 asset realization validation did not pass "
                    f"for lane={lane.name} profile={profile}"
                )
            profile_results[profile] = dict(result)
        after_payload = read_private_file(
            realization_path,
            label="DS9 asset realization",
            max_bytes=REALIZATION_MAX_BYTES,
        )
        if hashlib.sha256(after_payload).hexdigest() != realization_sha256:
            raise RuntimeContainerError(
                "DS9 asset realization changed during authoritative validation"
            )
        host_compatibility = _canonical_engine_host_compatibility(
            realization,
            host_gpu,
            required_engine_ids=lane.required_engine_ids,
        )
        selected_output_sha256: dict[str, str] = {}
        for artifact_id in sorted(lane.required_engine_ids):
            artifact = realization.get("artifacts", {}).get(artifact_id)
            provenance = (
                artifact.get("provenance") if isinstance(artifact, Mapping) else None
            )
            output_sha256 = str(
                provenance.get("output_sha256")
                if isinstance(provenance, Mapping)
                else ""
            )
            if re.fullmatch(r"[0-9a-f]{64}", output_sha256) is None:
                raise RuntimeContainerError(
                    f"selected runtime engine output digest is invalid: {artifact_id}"
                )
            selected_output_sha256[artifact_id] = output_sha256
    except Exception as exc:
        return {
            "ok": False,
            "lane": lane.name,
            "profiles": list(lane.artifact_profiles),
            "errors": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
            "blockers": [],
            "realization_path": str(artifact_root / REALIZATION_FILENAME),
        }
    normalized: dict[str, object] = {
        "ok": True,
        "lane": lane.name,
        "profiles": list(lane.artifact_profiles),
        "profile_results": profile_results,
    }
    normalized["realization_path"] = str(realization_path)
    normalized["realization_sha256"] = realization_sha256
    normalized["base_manifest_sha256"] = anchors["base_manifest"]
    normalized["engine_source_contracts_sha256"] = anchors["source_contracts"]
    normalized["required_engine_ids"] = sorted(lane.required_engine_ids)
    normalized["selected_output_sha256"] = selected_output_sha256
    if lane is BASELINE_LANE:
        normalized["required_canonical_engine_ids"] = sorted(
            CANONICAL_ENGINE_ARTIFACT_IDS
        )
    normalized["host_compatibility"] = host_compatibility
    return normalized


def _canonical_source_id(value: object, *, label: str) -> str:
    if isinstance(value, bool):
        raise RuntimeContainerError(f"{label} must be a non-negative integer")
    if isinstance(value, int):
        source_id = value
    elif isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]{0,5})", value):
        source_id = int(value)
    else:
        raise RuntimeContainerError(
            f"{label} must be a canonical non-negative decimal source ID"
        )
    if source_id < 0 or source_id > 999999:
        raise RuntimeContainerError(f"{label} is outside the supported source-ID range")
    return str(source_id)


def _canonical_source_ids_from_configs(
    pipeline: Mapping[str, Any],
    cameras_config: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, ...]:
    """Resolve source IDs using the exact pipeline and camera-loader contracts."""

    sources = pipeline.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeContainerError(f"{label} pipeline sources must be a non-empty list")
    pipeline_ids: list[str] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise RuntimeContainerError(
                f"{label} pipeline source {index} must be a mapping"
            )
        pipeline_ids.append(
            _canonical_source_id(
                source.get("source-id", index),
                label=f"{label} pipeline source {index} source-id",
            )
        )
    if len(pipeline_ids) != len(set(pipeline_ids)):
        raise RuntimeContainerError(f"{label} pipeline source IDs are not unique")

    cameras = cameras_config.get("cameras")
    if not isinstance(cameras, Mapping) or not cameras:
        raise RuntimeContainerError(f"{label} cameras must be a non-empty mapping")
    camera_ids: list[str] = []
    for key, entry in cameras.items():
        if not isinstance(entry, Mapping):
            raise RuntimeContainerError(f"{label} camera {key!r} must be a mapping")
        overrides = [name for name in ("source_id", "source", "sensor_id") if name in entry]
        if len(overrides) > 1:
            raise RuntimeContainerError(
                f"{label} camera {key!r} has ambiguous source-ID overrides: {overrides}"
            )
        candidate = entry[overrides[0]] if overrides else key
        camera_ids.append(
            _canonical_source_id(candidate, label=f"{label} camera {key!r} source ID")
        )
    if len(camera_ids) != len(set(camera_ids)):
        raise RuntimeContainerError(f"{label} camera source IDs are not unique")
    if set(pipeline_ids) != set(camera_ids):
        raise RuntimeContainerError(
            f"{label} pipeline/camera source-ID coverage drifted: "
            f"pipeline={sorted(pipeline_ids, key=int)} "
            f"cameras={sorted(camera_ids, key=int)}"
        )
    return tuple(sorted(pipeline_ids, key=int))


def _load_lane_source_ids(lane: RuntimeLane) -> tuple[str, ...]:
    lane = resolve_runtime_lane(lane)
    pipeline_path = REPO_ROOT / lane.pipeline_config
    cameras_path = REPO_ROOT / lane.cameras_config
    for name, path in (("pipeline", pipeline_path), ("cameras", cameras_path)):
        if path.is_symlink() or not path.is_file():
            raise RuntimeContainerError(
                f"reviewed {lane.name} {name} source configuration is missing or linked"
            )
    try:
        pipeline = yaml.load(
            pipeline_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        )
        cameras = yaml.load(
            cameras_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeContainerError(
            f"reviewed {lane.name} source configuration is invalid: {exc}"
        ) from exc
    if not isinstance(pipeline, Mapping) or not isinstance(cameras, Mapping):
        raise RuntimeContainerError(
            f"reviewed {lane.name} pipeline and cameras roots must be mappings"
        )
    return _canonical_source_ids_from_configs(
        pipeline,
        cameras,
        label=f"reviewed {lane.name}",
    )


def validate_canonical_config(
    lane: RuntimeLane = BASELINE_LANE,
) -> dict[str, object]:
    lane = resolve_runtime_lane(lane)
    pipeline = REPO_ROOT / lane.pipeline_config
    cameras = REPO_ROOT / lane.cameras_config
    for label, path in (("pipeline config", pipeline), ("camera config", cameras)):
        if path.is_symlink() or not path.is_file():
            raise RuntimeContainerError(
                f"reviewed {lane.name} {label} is missing or linked: {path}"
            )
    try:
        loaded = yaml.load(
            pipeline.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        ) or {}
    except Exception as exc:
        raise RuntimeContainerError(
            f"reviewed {lane.name} pipeline config is invalid: {exc}"
        ) from exc
    mosaic = loaded.get("mosaic_output") if isinstance(loaded, Mapping) else None
    if not isinstance(mosaic, Mapping):
        raise RuntimeContainerError(
            f"reviewed {lane.name} pipeline lacks mosaic_output"
        )
    expected = {
        "rtsp_enabled": False,
        "rtsp_port": CANONICAL_RTSP_PORT,
        "rtsp_path": CANONICAL_RTSP_PATH,
        "mosaic_webrtc_enabled": True,
    }
    for key, value in expected.items():
        actual = mosaic.get(key)
        if actual != value:
            raise RuntimeContainerError(
                f"reviewed {lane.name} pipeline {key} drifted: "
                f"expected={value!r} actual={actual!r}"
            )
    models = loaded.get("models") if isinstance(loaded, Mapping) else None
    pgie = models.get("pgie") if isinstance(models, Mapping) else None
    if not isinstance(pgie, Mapping):
        raise RuntimeContainerError(f"reviewed {lane.name} pipeline lacks models.pgie")
    if lane is BASELINE_LANE:
        expected_pgie = {
            "config-file-path": "DS9/pipelines/config_infer_primary_yolo26_m.ini",
            "engine": "DS9/models/engines/yolo26m_b3_fp16.engine",
            "batch_size": 3,
            "gie_id": 1,
        }
    elif lane is V3DT_LANE:
        expected_pgie = {
            "config-file-path": "DS9/pipelines/config_infer_primary_yolo26_seg_s.ini",
            "engine": "DS9/models/engines/yolo26s-seg_fused_b3_fp16.engine",
            "batch_size": 3,
            "gie_id": 1,
        }
    else:
        # Wholebody49 is an explicit runtime overlay over the reviewed baseline
        # graph.  The runtime materializer owns the generated INI paths; the
        # supervisor never accepts an arbitrary config or model profile.
        expected_pgie = {
            "batch_size": 3,
            "gie_id": 1,
        }
        for required in (
            DS9_ROOT
            / "pipelines"
            / f"config_infer_primary_deimv2_wholebody49_{'masks' if lane.model_size == 's' else 'boxes'}.template.ini",
            DS9_ROOT / "pipelines" / "config_preproc.ini",
            DS9_ROOT
            / "pipelines"
            / "nvdsinfer_deimv2_wholebody49"
            / "libnvdsinfer_deimv2_wholebody49.so",
            DS9_ROOT / "models" / "deimv2_wholebody49" / "classes.txt",
        ):
            if (
                required.is_symlink()
                or not required.is_file()
                or required.stat().st_size <= 0
            ):
                raise RuntimeContainerError(
                    f"reviewed {lane.name} runtime materializer dependency is missing or unsafe: {required}"
                )
    for key, value in expected_pgie.items():
        actual = pgie.get(key)
        if actual != value:
            raise RuntimeContainerError(
                f"reviewed {lane.name} PGIE {key} drifted: "
                f"expected={value!r} actual={actual!r}"
            )
    if lane is V3DT_LANE:
        streammux = loaded.get("streammux")
        tracker = loaded.get("tracker")
        v3dt = loaded.get("v3dt")
        expected_v3dt = {
            "streammux.width": (streammux or {}).get("width"),
            "streammux.height": (streammux or {}).get("height"),
            "streammux.batch-size": (streammux or {}).get("batch-size"),
            "streammux.enable-padding": (streammux or {}).get("enable-padding"),
            "tracker.config-file": (tracker or {}).get("config-file"),
            "v3dt.profile": (v3dt or {}).get("profile"),
            "v3dt.world_frame": (v3dt or {}).get("world_frame"),
            "v3dt.caminfo_world_axes": (v3dt or {}).get("caminfo_world_axes"),
            "v3dt.camera_order": (v3dt or {}).get("camera_order"),
        }
        required_v3dt = {
            "streammux.width": 1920,
            "streammux.height": 1080,
            "streammux.batch-size": 3,
            "streammux.enable-padding": 0,
            "tracker.config-file": "DS9/config/v3dt/nvtracker_v3dt.yaml",
            "v3dt.profile": "sv3dt",
            "v3dt.world_frame": "backend_world_m",
            "v3dt.caminfo_world_axes": "xzy",
            "v3dt.camera_order": ["living-room", "kitchen", "family-room"],
        }
        if expected_v3dt != required_v3dt:
            raise RuntimeContainerError(
                f"reviewed v3dt geometry contract drifted: {expected_v3dt!r}"
            )
        depth = models.get("depth_tracking") if isinstance(models, Mapping) else None
        if not isinstance(depth, Mapping) or depth.get("enable") is not False:
            raise RuntimeContainerError(
                "reviewed v3dt lane requires the duplicate baseline depth-tracking lane disabled"
            )
    validate_lane_selection(
        lane,
        tracking_mode=lane.tracking_mode,
        profile=lane.pgie_profile,
        size=lane.model_size,
    )
    source_ids = _load_lane_source_ids(lane)
    return {
        "lane": lane.name,
        "pipeline": str(pipeline.relative_to(REPO_ROOT)),
        "cameras": str(cameras.relative_to(REPO_ROOT)),
        "pgie_profile": lane.pgie_profile,
        "model_size": lane.model_size,
        "tracking_mode": lane.tracking_mode,
        "artifact_profiles": list(lane.artifact_profiles),
        "required_engine_ids": sorted(lane.required_engine_ids),
        "source_ids": list(source_ids),
        "profile_policy": {
            "baseline_readiness": {
                "profile": CANONICAL_PROFILE,
                "size": CANONICAL_SIZE,
            },
            "v3dt_readiness": {"profile": V3DT_PROFILE, "size": V3DT_SIZE},
            "alternate_only": list(ALTERNATE_ONLY_PROFILES),
        },
        "ports": {
            "websocket": CANONICAL_WS_PORT,
            "rest": CANONICAL_REST_PORT,
            "rtsp": CANONICAL_RTSP_PORT,
        },
        "endpoints": dict(CANONICAL_ENDPOINTS),
    }


def validate_lane_selection(
    lane: RuntimeLane,
    *,
    tracking_mode: str,
    profile: str,
    size: str,
) -> None:
    lane = resolve_runtime_lane(lane)
    actual = (
        str(tracking_mode).strip().lower(),
        str(profile).strip().lower(),
        str(size).strip().lower(),
    )
    expected = (lane.tracking_mode, lane.pgie_profile, lane.model_size)
    if actual != expected:
        raise RuntimeContainerError(
            f"reviewed DS9 runtime lane {lane.name} selection drifted: "
            f"expected={expected!r} actual={actual!r}"
        )


def validate_readiness_profile(
    *,
    tracking_mode: str,
    profile: str,
    size: str,
) -> None:
    """Reject profiles that cannot satisfy canonical DS9 readiness."""

    selection = (str(profile).strip().lower(), str(size).strip().lower())
    mode = str(tracking_mode).strip().lower()
    if selection[0] in ALTERNATE_ONLY_PROFILES:
        raise RuntimeContainerError(
            f"{selection[0]} is an explicit alternate-only profile and cannot satisfy readiness"
        )
    expected = (
        (CANONICAL_PROFILE, CANONICAL_SIZE)
        if mode == "baseline"
        else (V3DT_PROFILE, V3DT_SIZE)
        if mode == "v3dt"
        else None
    )
    if expected is None:
        raise RuntimeContainerError(
            f"unsupported readiness tracking mode: {tracking_mode!r}"
        )
    if selection != expected:
        raise RuntimeContainerError(
            "canonical readiness profile mismatch: "
            f"tracking_mode={mode} expected={expected[0]}/{expected[1]} "
            f"actual={selection[0]}/{selection[1]}"
        )


def _docker_command(roots: HostRoots, *arguments: str) -> list[str]:
    return [
        "docker",
        "--host",
        f"unix://{roots.docker_socket}",
        *arguments,
    ]


def _parse_image_inspect(raw: str, *, label: str) -> Mapping[str, Any]:
    payload = _strict_json_value(raw, label=f"{label} image inspection")
    if not isinstance(payload, Mapping):
        raise RuntimeContainerError(f"{label} image inspection was not an object")
    return payload


def _validated_rootfs_layers(
    image: Mapping[str, Any], *, label: str
) -> tuple[str, ...]:
    rootfs = image.get("RootFS")
    if not isinstance(rootfs, Mapping) or rootfs.get("Type") != "layers":
        raise RuntimeContainerError(f"{label} image RootFS type must be 'layers'")
    raw_layers = rootfs.get("Layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise RuntimeContainerError(f"{label} image RootFS layers must be non-empty")
    layers = tuple(str(value) for value in raw_layers)
    if any(re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None for value in layers):
        raise RuntimeContainerError(
            f"{label} image RootFS layers contain an invalid digest"
        )
    return layers


def _layer_list_sha256(layers: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(layers) + "\n").encode("ascii")).hexdigest()


def inspect_docker_state(roots: HostRoots, runner: CommandRunner) -> DockerState:
    try:
        socket_info = roots.docker_socket.lstat()
    except FileNotFoundError as exc:
        raise RuntimeContainerError(
            f"secondary Docker socket is missing: {roots.docker_socket}"
        ) from exc
    if not stat.S_ISSOCK(socket_info.st_mode):
        raise RuntimeContainerError(
            f"secondary Docker endpoint is not a Unix socket: {roots.docker_socket}"
        )
    info_raw = runner.run(
        _docker_command(roots, "info", "--format", "{{json .}}"),
        timeout=15.0,
    ).stdout
    info = _strict_json_value(info_raw, label="secondary Docker info")
    actual_root = Path(str(info.get("DockerRootDir") or "")).absolute()
    if actual_root != roots.docker_data:
        raise RuntimeContainerError(
            f"secondary Docker data root drifted: expected={roots.docker_data} actual={actual_root}"
        )
    default_runtime = str(info.get("DefaultRuntime") or "")
    if default_runtime != "runc":
        raise RuntimeContainerError(
            f"secondary Docker default runtime must remain runc; found {default_runtime!r}"
        )
    daemon_id = str(info.get("ID") or "").strip()
    if not daemon_id:
        raise RuntimeContainerError("secondary Docker daemon ID is missing")
    runtimes = info.get("Runtimes")
    if (
        not isinstance(runtimes, Mapping)
        or "nvidia" not in runtimes
        or "runc" not in runtimes
    ):
        raise RuntimeContainerError(
            "secondary Docker lacks the required runc/nvidia runtime pair"
        )

    networks: dict[str, str] = {}
    rows = runner.run(
        _docker_command(roots, "network", "ls", "--format", "{{json .}}"),
        timeout=15.0,
    ).stdout.splitlines()
    for row in rows:
        if not row.strip():
            continue
        payload = _strict_json_value(
            row,
            label="secondary Docker network listing row",
        )
        networks[str(payload.get("Name") or "")] = str(payload.get("Driver") or "")
    if networks != {"host": "host", "none": "null"}:
        raise RuntimeContainerError(
            f"secondary Docker network isolation drifted: {networks!r}"
        )

    image_raw = runner.run(
        _docker_command(roots, "image", "inspect", IMAGE_REF, "--format", "{{json .}}"),
        timeout=15.0,
    ).stdout
    image = _parse_image_inspect(image_raw, label="derived DS9 runtime")
    image_id = str(image.get("Id") or "")
    image_config = image.get("Config")
    labels = (
        image_config.get("Labels")
        if isinstance(image_config, Mapping)
        and isinstance(image_config.get("Labels"), Mapping)
        else {}
    )
    base_digest = str(labels.get("org.opencontainers.image.base.digest") or "")
    parent_build_image_reference = str(
        labels.get("com.noesis.engine-build.image.reference") or ""
    )
    parent_build_image_id = str(
        labels.get("com.noesis.engine-build.image.id") or ""
    )
    if image_id != IMAGE_ID:
        raise RuntimeContainerError(
            f"derived DS9 image ID drifted: expected={IMAGE_ID} actual={image_id}"
        )
    if base_digest != BASE_DIGEST:
        raise RuntimeContainerError(
            f"derived DS9 base digest drifted: expected={BASE_DIGEST} actual={base_digest}"
        )
    if parent_build_image_reference != PARENT_BUILD_IMAGE_REF:
        raise RuntimeContainerError(
            "derived DS9 runtime parent reference drifted: "
            f"expected={PARENT_BUILD_IMAGE_REF} actual={parent_build_image_reference}"
        )
    if parent_build_image_id != PARENT_BUILD_IMAGE_ID:
        raise RuntimeContainerError(
            "derived DS9 runtime parent image ID drifted: "
            f"expected={PARENT_BUILD_IMAGE_ID} actual={parent_build_image_id}"
        )
    parent_raw = runner.run(
        _docker_command(
            roots,
            "image",
            "inspect",
            PARENT_BUILD_IMAGE_ID,
            "--format",
            "{{json .}}",
        ),
        timeout=15.0,
    ).stdout
    parent_image = _parse_image_inspect(parent_raw, label="DS9 engine-build parent")
    if str(parent_image.get("Id") or "") != PARENT_BUILD_IMAGE_ID:
        raise RuntimeContainerError("DS9 engine-build parent image ID drifted")
    parent_config = parent_image.get("Config")
    parent_labels = (
        parent_config.get("Labels")
        if isinstance(parent_config, Mapping)
        and isinstance(parent_config.get("Labels"), Mapping)
        else {}
    )
    if str(parent_labels.get("org.opencontainers.image.base.digest") or "") != BASE_DIGEST:
        raise RuntimeContainerError("DS9 engine-build parent base digest drifted")
    parent_layers = _validated_rootfs_layers(
        parent_image, label="DS9 engine-build parent"
    )
    runtime_layers = _validated_rootfs_layers(image, label="derived DS9 runtime")
    if runtime_layers[: len(parent_layers)] != parent_layers:
        raise RuntimeContainerError(
            "derived DS9 runtime RootFS is not based on the exact engine-build image"
        )
    if len(runtime_layers) - len(parent_layers) != RUNTIME_ROOTFS_LAYER_DELTA:
        raise RuntimeContainerError(
            "derived DS9 runtime RootFS layer delta drifted: "
            f"expected={RUNTIME_ROOTFS_LAYER_DELTA} "
            f"actual={len(runtime_layers) - len(parent_layers)}"
        )
    return DockerState(
        daemon_id=daemon_id,
        docker_root=str(actual_root),
        default_runtime=default_runtime,
        image_id=image_id,
        base_digest=base_digest,
        parent_build_image_reference=parent_build_image_reference,
        parent_build_image_id=parent_build_image_id,
        parent_rootfs_layer_count=len(parent_layers),
        runtime_rootfs_layer_count=len(runtime_layers),
        parent_rootfs_sha256=_layer_list_sha256(parent_layers),
        runtime_rootfs_sha256=_layer_list_sha256(runtime_layers),
        networks=networks,
    )


def existing_runtime_containers(roots: HostRoots, runner: CommandRunner) -> list[str]:
    output = runner.run(
        _docker_command(
            roots,
            "ps",
            "-a",
            "--filter",
            f"label={ROLE_LABEL}",
            "--format",
            "{{.ID}} {{.Names}} {{.Status}}",
        ),
        timeout=15.0,
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def existing_host_runtime_processes() -> list[str]:
    matches: list[str] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        args = [os.fsdecode(value) for value in raw.split(b"\0") if value]
        if any(
            value == "DS9/noesis/ds9_runtime.py"
            or value.endswith("/DS9/noesis/ds9_runtime.py")
            for value in args
        ):
            matches.append(f"pid={entry.name} argv0={args[0] if args else '<unknown>'}")
    return sorted(matches)


def gpu_compute_owners(runner: CommandRunner) -> list[str]:
    result = runner.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=15.0,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def host_gpu_identity(runner: CommandRunner) -> dict[str, object]:
    """Return the exact physical identity/runtime compatibility tuple for GPU 0."""

    result = runner.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=index,name,uuid,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=15.0,
    )
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise RuntimeContainerError(
            "nvidia-smi did not return exactly one complete identity row for GPU 0"
        )
    index, name, uuid, compute_capability, memory_mib, driver_version = (
        value.strip() for value in rows[0]
    )
    try:
        memory_value = int(memory_mib)
    except ValueError as exc:
        raise RuntimeContainerError(
            "GPU 0 memory total is not an integer MiB value"
        ) from exc
    if (
        index != "0"
        or not name
        or not uuid.startswith("GPU-")
        or not re.fullmatch(r"\d+\.\d+", compute_capability)
        or memory_value <= 0
        or not re.fullmatch(r"\d+(?:\.\d+)+", driver_version)
    ):
        raise RuntimeContainerError("GPU 0 identity returned by nvidia-smi is invalid")
    return {
        "index": 0,
        "name": name,
        "uuid": uuid,
        "compute_capability": compute_capability,
        "memory_mib": memory_value,
        "driver_version": driver_version,
    }


def _validated_lock_descriptor(path: Path, flags: int) -> int:
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeContainerError(
            f"DS9 artifact transaction lock cannot be opened: {path}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeContainerError(
                "DS9 artifact transaction lock must be a single-link regular file"
            )
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeContainerError(
                "DS9 artifact transaction lock must be owned by the runtime user with mode 0600"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def artifact_transaction_lock_status(artifact_root: Path) -> dict[str, object]:
    path = artifact_root / ARTIFACT_TRANSACTION_LOCK_FILENAME
    if not path.exists() and not path.is_symlink():
        return {"available": True, "path": str(path), "exists": False}
    descriptor = _validated_lock_descriptor(path, os.O_RDONLY)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            return {"available": False, "path": str(path), "exists": True}
        finally:
            if acquired:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        os.close(descriptor)
    return {"available": True, "path": str(path), "exists": True}


def acquire_artifact_transaction_lock(artifact_root: Path) -> tuple[int, Path]:
    path = artifact_root / ARTIFACT_TRANSACTION_LOCK_FILENAME
    descriptor = _validated_lock_descriptor(path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeContainerError(
            f"another DS9 artifact transaction owns {path}"
        ) from exc
    return descriptor, path


def release_artifact_transaction_lock(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _pid_descends_from(pid: int, ancestor: int) -> bool:
    current = int(pid)
    target = int(ancestor)
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == target:
            return True
        seen.add(current)
        try:
            stat_fields = (
                (Path("/proc") / str(current) / "stat")
                .read_text(encoding="utf-8")
                .split()
            )
            current = int(stat_fields[3])
        except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
            return False
    return current == target


def confirm_container_gpu_ownership(
    inspect: Mapping[str, Any],
    runner: CommandRunner,
    *,
    timeout_seconds: float = 15.0,
) -> dict[str, object]:
    state = inspect.get("State") or {}
    try:
        container_init_pid = int(state.get("Pid") or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeContainerError(
            "container inspection lacks a valid host PID"
        ) from exc
    if container_init_pid <= 1:
        raise RuntimeContainerError("container inspection lacks a valid host PID")
    deadline = time.monotonic() + float(timeout_seconds)
    last_owners: list[str] = []
    while time.monotonic() < deadline:
        last_owners = gpu_compute_owners(runner)
        parsed: list[tuple[int, str]] = []
        for row in last_owners:
            raw_pid = row.split(",", 1)[0].strip()
            if raw_pid.isdigit():
                parsed.append((int(raw_pid), row))
        if len(parsed) != len(last_owners):
            raise RuntimeContainerError(
                f"malformed GPU compute-owner evidence: {last_owners}"
            )
        owned = [
            row for pid, row in parsed if _pid_descends_from(pid, container_init_pid)
        ]
        foreign = [
            row
            for pid, row in parsed
            if not _pid_descends_from(pid, container_init_pid)
        ]
        if foreign:
            raise RuntimeContainerError(
                f"foreign GPU owner appeared during DS9 startup: {foreign}"
            )
        if owned:
            return {
                "container_init_pid": container_init_pid,
                "compute_owners": owned,
            }
        time.sleep(0.2)
    raise RuntimeContainerError(
        "DS9 reached port readiness without a confirmed container GPU compute owner: "
        f"{last_owners}"
    )


def _read_bounded_host_text(
    path: Path, *, label: str, max_bytes: int = 64 * 1024
) -> str:
    """Read one host kernel evidence file without following a symlink."""

    try:
        lexical = path.lstat()
    except OSError as exc:
        raise RuntimeContainerError(f"{label} is unavailable: {path}") from exc
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise RuntimeContainerError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeContainerError(f"{label} cannot be opened: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeContainerError(f"{label} is not a regular file: {path}")
        if (opened.st_dev, opened.st_ino) != (lexical.st_dev, lexical.st_ino):
            raise RuntimeContainerError(f"{label} changed while opening: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, min(4096, max_bytes + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            if size > max_bytes:
                raise RuntimeContainerError(f"{label} exceeds {max_bytes} bytes")
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeContainerError(f"{label} changed while reading: {path}")
    finally:
        os.close(descriptor)
    try:
        return b"".join(chunks).decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeContainerError(f"{label} must contain ASCII text") from exc


def _container_id_and_init_pid(inspect: Mapping[str, Any]) -> tuple[str, int]:
    container_id = str(inspect.get("Id") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        raise RuntimeContainerError(
            "resource-soak container inspection lacks an exact 64-character ID"
        )
    state = inspect.get("State")
    try:
        init_pid = int(state.get("Pid") if isinstance(state, Mapping) else 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeContainerError(
            "resource-soak container inspection lacks a valid host PID"
        ) from exc
    if init_pid <= 1:
        raise RuntimeContainerError(
            "resource-soak container inspection lacks a valid host PID"
        )
    return container_id, init_pid


def resolve_container_cgroup_v2_path(
    inspect: Mapping[str, Any],
    *,
    cgroup_root: Path = HOST_CGROUP_V2_ROOT,
    proc_root: Path = HOST_PROC_ROOT,
) -> Path:
    """Resolve the exact host cgroup-v2 directory owned by one container PID."""

    container_id, init_pid = _container_id_and_init_pid(inspect)
    try:
        root_info = cgroup_root.lstat()
    except OSError as exc:
        raise RuntimeContainerError(
            f"host cgroup-v2 root is unavailable: {cgroup_root}"
        ) from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeContainerError(
            f"host cgroup-v2 root must be a non-symlink directory: {cgroup_root}"
        )
    controllers = _read_bounded_host_text(
        cgroup_root / "cgroup.controllers",
        label="host cgroup-v2 controllers",
    ).strip()
    if not controllers:
        raise RuntimeContainerError("host cgroup-v2 controllers are empty")

    membership = _read_bounded_host_text(
        proc_root / str(init_pid) / "cgroup",
        label="container init cgroup membership",
    )
    lines = [line.strip() for line in membership.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith("0::/"):
        raise RuntimeContainerError(
            "container init process is not in one exact unified cgroup-v2 hierarchy"
        )
    raw_relative = lines[0][3:].lstrip("/")
    relative = Path(raw_relative)
    if (
        not raw_relative
        or relative.is_absolute()
        or ".." in relative.parts
        or any(token in raw_relative for token in ("\x00", "\n", "\r"))
    ):
        raise RuntimeContainerError("container cgroup-v2 membership path is unsafe")
    accepted_components = {container_id, f"docker-{container_id}.scope"}
    if not accepted_components.intersection(relative.parts):
        raise RuntimeContainerError(
            "container cgroup-v2 membership does not identify the exact container ID"
        )

    candidate = cgroup_root
    for component in relative.parts:
        candidate /= component
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise RuntimeContainerError(
                f"container cgroup-v2 directory is unavailable: {candidate}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeContainerError(
                f"container cgroup-v2 path contains a symlink: {candidate}"
            )
    if not stat.S_ISDIR(candidate.lstat().st_mode):
        raise RuntimeContainerError(
            f"container cgroup-v2 path is not a directory: {candidate}"
        )
    return candidate


def _cgroup_nonnegative_integer(path: Path, *, label: str) -> int:
    raw = _read_bounded_host_text(path, label=label).strip()
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", raw) is None:
        raise RuntimeContainerError(f"{label} is not a non-negative integer")
    return int(raw)


def _cgroup_memory_events(path: Path) -> dict[str, int]:
    raw = _read_bounded_host_text(path, label="container memory.events")
    events: dict[str, int] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0]:
            raise RuntimeContainerError("container memory.events is malformed")
        key, value = parts
        if key in events or re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
            raise RuntimeContainerError("container memory.events is malformed")
        events[key] = int(value)
    missing = {"oom", "oom_kill"} - set(events)
    if missing:
        raise RuntimeContainerError(
            f"container memory.events lacks required counters: {sorted(missing)}"
        )
    return events


def read_container_cgroup_v2_sample(cgroup_path: Path) -> dict[str, int]:
    """Read every required resource value from one resolved cgroup-v2 path."""

    return {
        "memory_current_bytes": _cgroup_nonnegative_integer(
            cgroup_path / "memory.current",
            label="container memory.current",
        ),
        "memory_peak_bytes": _cgroup_nonnegative_integer(
            cgroup_path / "memory.peak",
            label="container memory.peak",
        ),
        "pids_current": _cgroup_nonnegative_integer(
            cgroup_path / "pids.current",
            label="container pids.current",
        ),
        **{
            f"memory_events_{key}": value
            for key, value in _cgroup_memory_events(
                cgroup_path / "memory.events"
            ).items()
        },
    }


def characterize_confirmed_gpu_memory(
    confirmation: Mapping[str, object],
) -> dict[str, int]:
    """Parse exact per-process GPU memory from confirmed container owners."""

    owners = confirmation.get("compute_owners")
    if not isinstance(owners, list) or not owners:
        raise RuntimeContainerError(
            "resource-soak GPU ownership confirmation has no compute owners"
        )
    memory_values: list[int] = []
    for raw in owners:
        rows = list(csv.reader([str(raw)], skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 3:
            raise RuntimeContainerError("resource-soak GPU ownership row is malformed")
        pid, process_name, memory_mib = (value.strip() for value in rows[0])
        if (
            not pid.isdigit()
            or not process_name
            or re.fullmatch(r"(?:0|[1-9][0-9]*)", memory_mib) is None
        ):
            raise RuntimeContainerError("resource-soak GPU ownership row is malformed")
        memory_values.append(int(memory_mib))
    return {
        "gpu_compute_owner_count": len(memory_values),
        "gpu_used_memory_mib": sum(memory_values),
        "gpu_largest_process_memory_mib": max(memory_values),
    }


def secret_identities(secret_files: SecretFiles) -> dict[str, tuple[object, ...]]:
    result: dict[str, tuple[object, ...]] = {}
    for label, path, limit in (
        ("cameras", secret_files.cameras, 64 * 1024),
        ("mapanything", secret_files.mapanything, 1024),
        ("internal_auth", secret_files.internal_auth, 4096),
    ):
        try:
            payload = read_private_file(path, label=f"{label} secret", max_bytes=limit)
            info = path.lstat()
        except PrivatePathError as exc:
            raise RuntimeContainerError(str(exc)) from exc
        result[label] = (
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_size),
            int(info.st_mtime_ns),
            hashlib.sha256(payload).digest(),
        )
    return result


def assert_immediate_run_preconditions(
    *,
    roots: HostRoots,
    secrets_: SecretFiles,
    source_before: CheckoutSnapshot,
    planned_artifact_readiness: Mapping[str, object],
    runner: CommandRunner,
    lane: RuntimeLane = BASELINE_LANE,
) -> dict[str, tuple[object, ...]]:
    lane = resolve_runtime_lane(lane)
    containers = existing_runtime_containers(roots, runner)
    if containers:
        raise RuntimeContainerError(
            f"DS9 runtime container owner appeared after planning: {containers}"
        )
    processes = existing_host_runtime_processes()
    if processes:
        raise RuntimeContainerError(
            f"DS9 runtime process owner appeared after planning: {processes}"
        )
    owners = gpu_compute_owners(runner)
    if owners:
        raise RuntimeContainerError(
            f"GPU compute owner appeared after planning: {owners}"
        )
    ports = unavailable_canonical_ports()
    if ports:
        raise RuntimeContainerError(
            f"canonical host-network port appeared after planning: {ports}"
        )
    current_source = snapshot_checkout()
    comparison = compare_snapshots(source_before, current_source)
    if not bool(comparison["unchanged"]):
        raise RuntimeContainerError(
            "checkout changed between planning and Docker launch"
        )
    current_gpu = host_gpu_identity(runner)
    planned_compatibility = planned_artifact_readiness.get("host_compatibility")
    planned_gpu = (
        planned_compatibility.get("host_gpu")
        if isinstance(planned_compatibility, Mapping)
        else None
    )
    if not isinstance(planned_gpu, Mapping) or dict(planned_gpu) != current_gpu:
        raise RuntimeContainerError(
            "GPU 0 identity or driver changed between planning and launch"
        )
    current_artifacts = validate_runtime_artifacts(
        roots.artifacts,
        host_gpu=current_gpu,
        lane=lane,
    )
    if not bool(current_artifacts.get("ok")):
        raise RuntimeContainerError(
            "canonical DS9 artifact realization failed immediate pre-launch validation"
        )
    planned_realization = str(
        planned_artifact_readiness.get("realization_sha256") or ""
    )
    current_realization = str(current_artifacts.get("realization_sha256") or "")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", planned_realization)
        or current_realization != planned_realization
    ):
        raise RuntimeContainerError(
            "canonical DS9 artifact realization changed between planning and launch"
        )
    return secret_identities(secrets_)


def _port_is_free(port: int) -> bool:
    sockets: list[socket.socket] = []
    try:
        for family, address in (
            (socket.AF_INET, ("0.0.0.0", int(port))),
            (socket.AF_INET6, ("::", int(port), 0, 0)),
        ):
            try:
                probe = socket.socket(family, socket.SOCK_STREAM)
            except OSError:
                continue
            sockets.append(probe)
            try:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError:
                pass
            if family == socket.AF_INET6:
                try:
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                except OSError:
                    pass
            try:
                probe.bind(address)
            except OSError:
                return False
        return True
    finally:
        for probe in sockets:
            probe.close()


def unavailable_canonical_ports() -> list[int]:
    return [port for port in CANONICAL_PORTS if not _port_is_free(port)]


def _port_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.4):
            return True
    except OSError:
        return False


def _snapshot_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _snapshot_path_observation(
    repo: Path,
    relative: str,
) -> tuple[os.stat_result, bytes | str | None]:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeContainerError(f"unsafe checkout snapshot path: {relative!r}")
    root_before = repo.lstat()
    if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
        raise RuntimeContainerError("checkout snapshot root must be a non-symlink directory")
    root_fd = os.open(
        repo,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptors = [root_fd]
    parents: list[tuple[int, str, int, os.stat_result]] = []
    try:
        root_open = os.fstat(root_fd)
        if _snapshot_identity(root_open) != _snapshot_identity(root_before):
            raise RuntimeContainerError("checkout snapshot root changed while opening")
        current_fd = root_fd
        for component in path.parts[:-1]:
            before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise RuntimeContainerError(
                    f"checkout snapshot parent is not a real directory: {relative}"
                )
            child_fd = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=current_fd,
            )
            descriptors.append(child_fd)
            opened = os.fstat(child_fd)
            if _snapshot_identity(opened) != _snapshot_identity(before):
                raise RuntimeContainerError(
                    f"checkout snapshot parent changed while opening: {relative}"
                )
            parents.append((current_fd, component, child_fd, opened))
            current_fd = child_fd

        name = path.parts[-1]
        before = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
        payload: bytes | str | None = None
        opened_final: os.stat_result | None = None
        if stat.S_ISREG(before.st_mode):
            file_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=current_fd,
            )
            descriptors.append(file_fd)
            opened_final = os.fstat(file_fd)
            if _snapshot_identity(opened_final) != _snapshot_identity(before):
                raise RuntimeContainerError(
                    f"checkout file changed while opening: {relative}"
                )
            if _SNAPSHOT_OPENAT_TEST_HOOK is not None:
                _SNAPSHOT_OPENAT_TEST_HOOK(relative, repo, path)
            chunks: list[bytes] = []
            while True:
                block = os.read(file_fd, 4 * 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            payload = b"".join(chunks)
            after_fd = os.fstat(file_fd)
            if (
                _snapshot_identity(after_fd) != _snapshot_identity(opened_final)
                or len(payload) != opened_final.st_size
            ):
                raise RuntimeContainerError(
                    f"checkout file changed while hashing: {relative}"
                )
        elif stat.S_ISLNK(before.st_mode):
            if _SNAPSHOT_OPENAT_TEST_HOOK is not None:
                _SNAPSHOT_OPENAT_TEST_HOOK(relative, repo, path)
            payload = os.readlink(name, dir_fd=current_fd)
        else:
            if _SNAPSHOT_OPENAT_TEST_HOOK is not None:
                _SNAPSHOT_OPENAT_TEST_HOOK(relative, repo, path)

        named_after = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
        expected_final = opened_final if opened_final is not None else before
        if _snapshot_identity(named_after) != _snapshot_identity(expected_final):
            raise RuntimeContainerError(
                f"checkout snapshot name was replaced while hashing: {relative}"
            )
        for parent_fd, component, child_fd, opened in reversed(parents):
            named_parent = os.stat(
                component, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                _snapshot_identity(named_parent) != _snapshot_identity(opened)
                or _snapshot_identity(os.fstat(child_fd)) != _snapshot_identity(opened)
            ):
                raise RuntimeContainerError(
                    f"checkout snapshot parent was replaced: {relative}"
                )
        if (
            _snapshot_identity(repo.lstat()) != _snapshot_identity(root_open)
            or _snapshot_identity(os.fstat(root_fd)) != _snapshot_identity(root_open)
        ):
            raise RuntimeContainerError("checkout snapshot root was replaced")
        return before, payload
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _snapshot_entry(repo: Path, relative: str) -> tuple[str, int]:
    info, payload = _snapshot_path_observation(repo, relative)
    mode = stat.S_IMODE(info.st_mode)
    if stat.S_ISREG(info.st_mode):
        if not isinstance(payload, bytes):
            raise RuntimeContainerError(f"checkout file read failed: {relative}")
        digest = hashlib.sha256(payload).hexdigest()
        return f"file:{mode:04o}:{info.st_size}:{digest}", int(info.st_size)
    if stat.S_ISLNK(info.st_mode):
        if not isinstance(payload, str):
            raise RuntimeContainerError(f"checkout symlink read failed: {relative}")
        encoded = os.fsencode(payload)
        return f"symlink:{mode:04o}:{hashlib.sha256(encoded).hexdigest()}", len(encoded)
    if stat.S_ISDIR(info.st_mode):
        return f"directory:{mode:04o}", 0
    return f"special:{stat.S_IFMT(info.st_mode):o}:{mode:04o}", 0


def _manifest_owned_checkout_binary_paths(repo: Path) -> tuple[str, ...]:
    """Expand every repository-owned binary output declared by the manifest.

    These outputs are intentionally ignored by Git, but the canonical runtime
    bind-mounts them from the checkout.  They therefore belong in the same
    before/immediate/after content snapshot as tracked application source.
    TensorRT engines are excluded because their external realization has a
    separate locked provenance contract.
    """

    manifest_path = repo / BASE_MANIFEST_RELATIVE
    try:
        manifest_info, manifest_raw = _snapshot_path_observation(
            repo, BASE_MANIFEST_RELATIVE
        )
    except FileNotFoundError:
        return ()
    if not stat.S_ISREG(manifest_info.st_mode) or not isinstance(manifest_raw, bytes):
        raise RuntimeContainerError(
            f"DS9 asset manifest must be a regular non-symlink file: {manifest_path}"
        )
    try:
        # The authoritative asset manifest uses YAML merge anchors with
        # explicit per-artifact overrides.  Parse it with the same SafeLoader
        # semantics as validate_asset_manifest.py; that validator has already
        # enforced the manifest contract before the runtime snapshot is taken.
        manifest = yaml.safe_load(manifest_raw.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeContainerError(
            f"unable to read DS9 asset manifest for checkout snapshot: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeContainerError(
            "DS9 asset manifest root must be a mapping for checkout snapshot"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeContainerError(
            "DS9 asset manifest artifacts must be a list for checkout snapshot"
        )

    repo_absolute = repo.absolute()
    paths: set[str] = set()
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise RuntimeContainerError(
                "DS9 asset manifest contains a malformed artifact row"
            )
        if str(row.get("kind") or "") not in CHECKOUT_BINARY_ARTIFACT_KINDS:
            continue
        output = str(row.get("output") or "").strip()
        relative_pattern = Path(output)
        if (
            not output
            or relative_pattern.is_absolute()
            or ".." in relative_pattern.parts
            or not relative_pattern.parts
            or relative_pattern.parts[0] != "DS9"
            or any(token in output for token in ("\x00", "\n", "\r"))
        ):
            raise RuntimeContainerError(
                "DS9 manifest checkout-binary output must be a safe DS9-relative "
                f"pattern: {output!r}"
            )
        try:
            matches = tuple(repo.glob(output))
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeContainerError(
                f"unable to expand DS9 manifest checkout-binary output: {output!r}"
            ) from exc
        for candidate in matches:
            lexical = Path(os.path.abspath(os.fspath(candidate)))
            try:
                relative = lexical.relative_to(repo_absolute)
            except ValueError as exc:
                raise RuntimeContainerError(
                    f"DS9 manifest checkout-binary output escapes the repository: {candidate}"
                ) from exc
            paths.add(relative.as_posix())
    return tuple(sorted(paths))


def snapshot_checkout(repo: Path = REPO_ROOT) -> CheckoutSnapshot:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=False,
            capture_output=True,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeContainerError(
            f"unable to enumerate checkout files: {exc}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeContainerError(
            "unable to enumerate checkout files: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    raw_paths = {value for value in result.stdout.split(b"\0") if value}
    manifest_binary_paths = _manifest_owned_checkout_binary_paths(repo)
    raw_paths.update(os.fsencode(value) for value in manifest_binary_paths)
    entries: dict[str, str] = {}
    byte_count = 0
    for raw in sorted(raw_paths):
        relative = os.fsdecode(raw)
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in {"", "."}:
            raise RuntimeContainerError(
                f"git returned an unsafe checkout path: {relative!r}"
            )
        try:
            entry, size = _snapshot_entry(repo, relative)
        except FileNotFoundError:
            entry, size = "missing", 0
        entries[relative] = entry
        byte_count += size
    digest = hashlib.sha256()
    for relative, entry in entries.items():
        encoded = relative.encode("utf-8", errors="surrogateescape")
        payload = entry.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return CheckoutSnapshot(
        digest=digest.hexdigest(),
        file_count=len(entries),
        byte_count=byte_count,
        entries=entries,
        manifest_binary_paths=manifest_binary_paths,
    )


def compare_snapshots(
    before: CheckoutSnapshot, after: CheckoutSnapshot
) -> dict[str, object]:
    before_keys = set(before.entries)
    after_keys = set(after.entries)
    changed = sorted(
        key
        for key in before_keys & after_keys
        if before.entries[key] != after.entries[key]
    )
    return {
        "unchanged": before.digest == after.digest,
        "before": before.summary(),
        "after": after.summary(),
        "added": sorted(after_keys - before_keys)[:200],
        "removed": sorted(before_keys - after_keys)[:200],
        "changed": changed[:200],
        "differences_truncated": (
            len(after_keys - before_keys) + len(before_keys - after_keys) + len(changed)
        )
        > 200,
    }


def _mount(source: Path, target: Path, *, readonly: bool) -> str:
    value = f"type=bind,src={source},dst={target},bind-propagation=rprivate"
    return f"{value},readonly" if readonly else value


def canonical_runtime_arguments(
    lane: RuntimeLane = BASELINE_LANE,
) -> list[str]:
    lane = resolve_runtime_lane(lane)
    validate_lane_selection(
        lane,
        tracking_mode=lane.tracking_mode,
        profile=lane.pgie_profile,
        size=lane.model_size,
    )
    return [
        "DS9/noesis/ds9_runtime.py",
        "--pipeline-config",
        lane.pipeline_config,
        "--cameras-config",
        lane.cameras_config,
        "--pgie-profile",
        lane.pgie_profile,
        "--size",
        lane.model_size,
        "--tracking-mode",
        lane.tracking_mode,
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
        str(CONTAINER_DEPTH_ROOT),
        "--log-level",
        "INFO",
    ]


def build_container_command(
    roots: HostRoots,
    secret_files: SecretFiles,
    session: SessionPaths,
    lane: RuntimeLane = BASELINE_LANE,
    appliance_context_env: Mapping[str, str] | None = None,
    appliance_binding: ApplianceBinding | None = None,
) -> list[str]:
    lane = resolve_runtime_lane(lane)
    manual_depth_model = str(
        os.environ.get("NOESIS_MANUAL_DEPTH_MODEL", "mapanything")
        or "mapanything"
    ).strip().lower()
    if manual_depth_model not in {"mapanything", "da3metric-large"}:
        raise RuntimeContainerError(
            "NOESIS_MANUAL_DEPTH_MODEL must be mapanything or da3metric-large"
        )
    uid = os.geteuid()
    gid = os.getegid()
    container_name = f"noesis-ds9-runtime-{session.session_id}"
    env_values = {
        "HOME": str(CONTAINER_STATE_ROOT / "home"),
        "XDG_CACHE_HOME": str(CONTAINER_BUILD_ROOT / "cache"),
        "XDG_RUNTIME_DIR": str(CONTAINER_BUILD_ROOT / "xdg-runtime"),
        "CUDA_CACHE_PATH": str(CONTAINER_BUILD_ROOT / "cuda-cache"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NOESIS_DS9_ARTIFACT_ROOT": str(CONTAINER_ARTIFACT_ROOT),
        "NOESIS_BUILD_DIR": str(CONTAINER_BUILD_ROOT),
        "NOESIS_SEMANTIC_CAPTURE_ROOT": str(CONTAINER_STATE_ROOT / "semantic-seg" / "captures"),
        "NOESIS_DEV_CONSOLE_LAUNCH_DIR": str(CONTAINER_BUILD_ROOT),
        "NOESIS_CAMERA_SECRETS_FILE": str(
            CONTAINER_SECRET_ROOT / "camera_sources.json"
        ),
        "NOESIS_MAPANYTHING_API_KEY_FILE": str(
            CONTAINER_SECRET_ROOT / "mapanything_rpc.key"
        ),
        "NOESIS_INTERNAL_AUTH_MODE": "required",
        "NOESIS_INTERNAL_AUTH_TOKEN_FILE": str(CONTAINER_SECRET_ROOT / "gateway-token"),
        "NOESIS_WORLD_JOURNAL_PATH": str(CONTAINER_STATE_ROOT / "world_ds9.sqlite3"),
        "NOESIS_IDENTITY_V2_STORE": str(
            CONTAINER_STATE_ROOT / "household" / "identity_v2.sqlite3"
        ),
        "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(
            CONTAINER_EVIDENCE_ROOT / "identity_v2.jsonl"
        ),
        "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": session.session_id,
        "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME": "ds9",
        "NOESIS_REID_ALIAS_FILE": str(CONTAINER_STATE_ROOT / "reid_aliases.json"),
        "NOESIS_REID_SID_POOL_FILE": str(CONTAINER_STATE_ROOT / "sid_pool.json"),
        "NOESIS_REID_GALLERY_FILE": str(CONTAINER_STATE_ROOT / "reid_gallery.npz"),
        "NOESIS_ANALYTICS_CONFIG": str(CONTAINER_ANALYTICS_CONFIG),
        "NOESIS_ANALYTICS_EXCLUDE_CONFIG": str(CONTAINER_ANALYTICS_EXCLUDE_CONFIG),
        "NOESIS_SCENE_STORE_PATH": str(CONTAINER_STATE_ROOT / "scene_releases.sqlite3"),
        "NOESIS_VIRTUAL_TWIN_ROOT": str(CONTAINER_STATE_ROOT / "virtual_twin"),
        "NOESIS_CALIBRATION_AUDIT_DIR": str(CONTAINER_EVIDENCE_ROOT / "calibration"),
        "NOESIS_V3DT_DIAG_DIR": str(CONTAINER_EVIDENCE_ROOT / "v3dt"),
        "NOESIS_V3DT_DIAG_SESSION": session.session_id,
        "NOESIS_PGIE_PROFILE": lane.pgie_profile,
        "NOESIS_TRACKING_MODE": lane.tracking_mode,
        "NOESIS_MANUAL_DEPTH_MODEL": manual_depth_model,
        "NOESIS_MOSAIC_RTSP_ENABLED": "1",
        "NOESIS_MOSAIC_WEBRTC_ENABLED": "1",
        "NOESIS_REID_ENABLED": "1",
        "NOESIS_SHUTDOWN_GRACE_SECONDS": "75",
    }
    context_env = dict(appliance_context_env or {})
    expected_context_keys = {
        "NOESIS_APPLIANCE_RUNTIME_CONTEXT",
        "NOESIS_APPLIANCE_RUNTIME_CONTEXT_SHA256",
    }
    if context_env and set(context_env) != expected_context_keys:
        raise RuntimeContainerError("appliance runtime context environment is incomplete")
    env_values.update(context_env)
    command = _docker_command(
        roots,
        "run",
        "--detach",
        "--name",
        container_name,
        "--label",
        ROLE_LABEL,
        "--label",
        f"{SESSION_LABEL_KEY}={session.session_id}",
        "--label",
        f"{LANE_LABEL_KEY}={lane.name}",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--security-opt=label=disable",
        "--user",
        f"{uid}:{gid}",
        "--init",
        "--pids-limit",
        "4096",
        "--memory",
        str(RUNTIME_MEMORY_BYTES),
        "--memory-swap",
        str(RUNTIME_MEMORY_BYTES),
        "--ulimit",
        "nofile=65536:65536",
        "--log-driver=json-file",
        "--log-opt",
        "max-size=50m",
        "--log-opt",
        "max-file=2",
        "--network=host",
        "--ipc=host",
        "--runtime=nvidia",
        "--gpus",
        "device=0",
    )
    for key, value in env_values.items():
        command.extend(("--env", f"{key}={value}"))
    command.extend(
        (
            "--tmpfs",
            f"/tmp:rw,exec,nosuid,nodev,size=2147483648,uid={uid},gid={gid},mode=0700",
            "--tmpfs",
            f"{CONTAINER_SECRET_ROOT}:rw,noexec,nosuid,nodev,size=65536,uid={uid},gid={gid},mode=0700",
            "--mount",
            _mount(REPO_ROOT.absolute(), CONTAINER_REPO_ROOT, readonly=True),
            "--mount",
            _mount(roots.artifacts, CONTAINER_ARTIFACT_ROOT, readonly=True),
        )
    )
    for source, target in _runtime_writable_mounts(session, appliance_binding):
        command.extend(("--mount", _mount(source, target, readonly=False)))
    for source, target in (
        (secret_files.cameras, CONTAINER_SECRET_ROOT / "camera_sources.json"),
        (secret_files.mapanything, CONTAINER_SECRET_ROOT / "mapanything_rpc.key"),
        (secret_files.internal_auth, CONTAINER_SECRET_ROOT / "gateway-token"),
    ):
        command.extend(("--mount", _mount(source, target, readonly=True)))
    command.extend(
        (
            "--workdir",
            str(CONTAINER_REPO_ROOT),
            "--entrypoint",
            "python3",
            IMAGE_ID,
            *canonical_runtime_arguments(lane),
        )
    )
    return command


def redacted_command(command: Sequence[str], secret_files: SecretFiles) -> list[str]:
    replacements = {
        str(secret_files.cameras): "<camera-secret-file>",
        str(secret_files.mapanything): "<mapanything-key-file>",
        str(secret_files.internal_auth): "<internal-auth-file>",
    }
    result: list[str] = []
    for argument in command:
        value = str(argument)
        for raw, replacement in replacements.items():
            value = value.replace(raw, replacement)
        result.append(value)
    return result


def _validate_mounts(
    inspect: Mapping[str, Any],
    *,
    roots: HostRoots,
    secrets_: SecretFiles,
    session: SessionPaths,
    appliance_binding: ApplianceBinding | None = None,
) -> None:
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list):
        raise RuntimeContainerError("container inspect lacks a mount list")
    actual: dict[str, tuple[str, bool]] = {}
    for mount in mounts:
        if not isinstance(mount, Mapping):
            raise RuntimeContainerError("container inspect contains a malformed mount")
        if str(mount.get("Type") or "") != "bind":
            raise RuntimeContainerError("container inspect contains a non-bind mount")
        destination = str(mount.get("Destination") or "")
        if not destination or destination in actual:
            raise RuntimeContainerError(
                f"container inspect contains an ambiguous mount destination: {destination!r}"
            )
        actual[destination] = (
            str(mount.get("Source") or ""),
            bool(mount.get("RW")),
        )
    expected = {
        str(CONTAINER_REPO_ROOT): (str(REPO_ROOT.absolute()), False),
        str(CONTAINER_ARTIFACT_ROOT): (str(roots.artifacts), False),
        str(CONTAINER_SECRET_ROOT / "camera_sources.json"): (
            str(secrets_.cameras),
            False,
        ),
        str(CONTAINER_SECRET_ROOT / "mapanything_rpc.key"): (
            str(secrets_.mapanything),
            False,
        ),
        str(CONTAINER_SECRET_ROOT / "gateway-token"): (
            str(secrets_.internal_auth),
            False,
        ),
    }
    expected.update(
        {
            str(target): (str(source), True)
            for source, target in _runtime_writable_mounts(
                session,
                appliance_binding,
            )
        }
    )
    if set(actual) != set(expected):
        raise RuntimeContainerError(
            f"container mount destinations drifted: expected={sorted(expected)} actual={sorted(actual)}"
        )
    for destination, (source, writable) in expected.items():
        actual_source, actual_writable = actual[destination]
        if source is not None and actual_source != source:
            raise RuntimeContainerError(
                f"container mount source drifted for {destination}: {actual_source}"
            )
        if actual_writable is not writable:
            raise RuntimeContainerError(
                f"container mount writability drifted for {destination}: {actual_writable}"
            )


def validate_container_inspect(
    inspect: Mapping[str, Any],
    *,
    roots: HostRoots,
    secrets_: SecretFiles,
    session: SessionPaths,
    lane: RuntimeLane = BASELINE_LANE,
    appliance_context_env: Mapping[str, str] | None = None,
    appliance_binding: ApplianceBinding | None = None,
) -> None:
    lane = resolve_runtime_lane(lane)
    config = inspect.get("Config") or {}
    host = inspect.get("HostConfig") or {}
    if str(inspect.get("Image") or "") != IMAGE_ID:
        raise RuntimeContainerError(
            "running container does not use the pinned DS9 image ID"
        )
    if str(config.get("User") or "") != f"{os.geteuid()}:{os.getegid()}":
        raise RuntimeContainerError("running container user contract drifted")
    if list(config.get("Entrypoint") or []) != ["python3"]:
        raise RuntimeContainerError("running container entrypoint contract drifted")
    if list(config.get("Cmd") or []) != canonical_runtime_arguments(lane):
        raise RuntimeContainerError(
            "running container canonical runtime arguments drifted"
        )
    env_values: dict[str, str] = {}
    for item in config.get("Env") or []:
        key, separator, value = str(item).partition("=")
        if separator:
            if key in env_values:
                raise RuntimeContainerError(
                    f"running container environment contains duplicate key {key}"
                )
            env_values[key] = value
    required_env = {
        "NOESIS_PGIE_PROFILE": lane.pgie_profile,
        "NOESIS_TRACKING_MODE": lane.tracking_mode,
        "NOESIS_MANUAL_DEPTH_MODEL": str(
            os.environ.get("NOESIS_MANUAL_DEPTH_MODEL", "mapanything")
            or "mapanything"
        ).strip().lower(),
        "NOESIS_INTERNAL_AUTH_MODE": "required",
        "NOESIS_DS9_ARTIFACT_ROOT": str(CONTAINER_ARTIFACT_ROOT),
        "NOESIS_BUILD_DIR": str(CONTAINER_BUILD_ROOT),
        "NOESIS_WORLD_JOURNAL_PATH": str(CONTAINER_STATE_ROOT / "world_ds9.sqlite3"),
        "NOESIS_IDENTITY_V2_STORE": str(
            CONTAINER_STATE_ROOT / "household" / "identity_v2.sqlite3"
        ),
        "NOESIS_ANALYTICS_CONFIG": str(CONTAINER_ANALYTICS_CONFIG),
        "NOESIS_ANALYTICS_EXCLUDE_CONFIG": str(CONTAINER_ANALYTICS_EXCLUDE_CONFIG),
    }
    context_env = dict(appliance_context_env or {})
    expected_context_keys = {
        "NOESIS_APPLIANCE_RUNTIME_CONTEXT",
        "NOESIS_APPLIANCE_RUNTIME_CONTEXT_SHA256",
    }
    if context_env and set(context_env) != expected_context_keys:
        raise RuntimeContainerError("expected appliance runtime context is incomplete")
    required_env.update(context_env)
    if not context_env and expected_context_keys & set(env_values):
        raise RuntimeContainerError(
            "non-appliance DS9 container inherited appliance runtime context"
        )
    for key, value in required_env.items():
        if env_values.get(key) != value:
            raise RuntimeContainerError(
                f"running container environment contract drifted for {key}"
            )
    forbidden_env = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MENON_OWNER_PASSWORD",
        "MENON_OWNER_BOOTSTRAP_TOKEN",
        "HS_PASS",
    } | set(FORBIDDEN_NESTED_LEASE_ENV)
    leaked = sorted(forbidden_env & set(env_values))
    if leaked:
        raise RuntimeContainerError(
            f"running container inherited forbidden host secret variables: {leaked}"
        )
    if not bool(host.get("ReadonlyRootfs")):
        raise RuntimeContainerError(
            "running container root filesystem is not read-only"
        )
    if bool(host.get("Privileged")):
        raise RuntimeContainerError(
            "running container unexpectedly has privileged mode"
        )
    if host.get("CapAdd"):
        raise RuntimeContainerError("running container unexpectedly adds capabilities")
    if int(host.get("Memory") or 0) != RUNTIME_MEMORY_BYTES:
        raise RuntimeContainerError("running container memory ceiling drifted")
    if int(host.get("MemorySwap") or 0) != RUNTIME_MEMORY_BYTES:
        raise RuntimeContainerError("running container memory+swap ceiling drifted")
    # On this swapless cgroup-v2 host Docker accepts --memory-swappiness but
    # discards it and reports null.  The enforceable no-swap contract is the
    # exact equality MemorySwap == Memory; reject any nonzero explicit value.
    memory_swappiness = host.get("MemorySwappiness")
    if memory_swappiness not in (None, 0):
        raise RuntimeContainerError("running container memory swappiness drifted")
    if {str(value).upper() for value in (host.get("CapDrop") or [])} != {"ALL"}:
        raise RuntimeContainerError(
            "running container capability-drop contract drifted"
        )
    if host.get("SecurityOpt") != list(EXPECTED_SECURITY_OPTIONS):
        raise RuntimeContainerError(
            "running container security-option contract drifted"
        )
    if inspect.get("AppArmorProfile") != EXPECTED_APPARMOR_PROFILE:
        raise RuntimeContainerError("running container AppArmor profile drifted")
    for field, expected in EXPECTED_SENSITIVE_HOST_DEFAULTS.items():
        if field not in host or host[field] != expected:
            raise RuntimeContainerError(
                f"running container sensitive HostConfig default drifted: {field}"
            )
    for field in EXPECTED_ABSENT_SENSITIVE_HOST_FIELDS:
        if field in host:
            raise RuntimeContainerError(
                f"running container sensitive HostConfig default drifted: {field}"
            )
    if host.get("Devices") != [] or host.get("Binds") is not None:
        raise RuntimeContainerError("running container device/bind defaults drifted")
    if host.get("PidMode") != "" or host.get("UsernsMode") != "":
        raise RuntimeContainerError("running container PID/user namespace defaults drifted")
    if host.get("ReadonlyPaths") != list(EXPECTED_READONLY_PATHS):
        raise RuntimeContainerError("running container read-only path policy drifted")
    if host.get("MaskedPaths") != list(EXPECTED_MASKED_PATHS):
        raise RuntimeContainerError("running container masked path policy drifted")
    if str(host.get("NetworkMode") or "") != "host":
        raise RuntimeContainerError(
            "running container network mode is not the authorized host mode"
        )
    if str(host.get("IpcMode") or "") != "host":
        raise RuntimeContainerError(
            "running container IPC mode is not the authorized host mode"
        )
    if str(host.get("Runtime") or "") != "nvidia":
        raise RuntimeContainerError(
            "running container runtime is not the explicitly authorized NVIDIA mode"
        )
    labels = config.get("Labels") or {}
    if str(labels.get("com.noesis.role") or "") != "ds9-runtime":
        raise RuntimeContainerError("running container role label drifted")
    if str(labels.get(SESSION_LABEL_KEY) or "") != session.session_id:
        raise RuntimeContainerError("running container session label drifted")
    if str(labels.get(LANE_LABEL_KEY) or "") != lane.name:
        raise RuntimeContainerError("running container lane label drifted")
    device_requests = host.get("DeviceRequests") or []
    if len(device_requests) != 1 or not isinstance(device_requests[0], Mapping):
        raise RuntimeContainerError("running container GPU device request drifted")
    request = device_requests[0]
    if [str(value) for value in (request.get("DeviceIDs") or [])] != ["0"]:
        raise RuntimeContainerError(
            "running container is not restricted to GPU device 0"
        )
    capabilities = request.get("Capabilities") or []
    if not any("gpu" in {str(value).lower() for value in row} for row in capabilities):
        raise RuntimeContainerError(
            "running container lacks an explicit GPU capability request"
        )
    tmpfs = host.get("Tmpfs") or {}
    if set(tmpfs) != {"/tmp", str(CONTAINER_SECRET_ROOT)}:
        raise RuntimeContainerError("running container tmpfs destinations drifted")
    secret_tmpfs = str(tmpfs.get(str(CONTAINER_SECRET_ROOT)) or "")
    if "mode=0700" not in secret_tmpfs:
        raise RuntimeContainerError("runtime secret tmpfs mode drifted from 0700")
    _validate_mounts(
        inspect,
        roots=roots,
        secrets_=secrets_,
        session=session,
        appliance_binding=appliance_binding,
    )


def validate_shutdown_log(text: str) -> dict[str, object]:
    folded = text.casefold()
    failures = [
        signature
        for signature in SHUTDOWN_FAILURE_SIGNATURES
        if signature.casefold() in folded
    ]
    analytics_markers = (
        "analytics",
        "nvdsanalytics",
        "config_nvdsanalytics_exclude",
        str(CONTAINER_ANALYTICS_ROOT).casefold(),
    )
    for line in text.splitlines():
        folded_line = line.casefold()
        if ("erofs" in folded_line or "read-only file system" in folded_line) and any(
            marker in folded_line for marker in analytics_markers
        ):
            failures.append("analytics state path is read-only")
            break
    failures = list(dict.fromkeys(failures))
    severity = [
        token
        for token in ("ERROR", "CRITICAL")
        if re.search(rf"(^|[\s:]){token}([\s:])", text, flags=re.MULTILINE)
    ]
    positions = {marker: text.rfind(marker) for marker in SHUTDOWN_MARKERS}
    missing = [
        marker for marker, position in positions.items() if position < 0
    ]
    request_position = positions[SHUTDOWN_MARKERS[0]]
    accepted_position = positions[SHUTDOWN_MARKERS[1]]
    eos_position = positions[SHUTDOWN_MARKERS[2]]
    wait_position = positions[SHUTDOWN_MARKERS[3]]
    complete_position = positions[SHUTDOWN_MARKERS[4]]
    ordered = bool(
        not missing
        and request_position < accepted_position
        and request_position < eos_position
        and max(accepted_position, eos_position) < wait_position < complete_position
    )
    return {
        "ok": not failures and not severity and not missing and ordered,
        "missing_markers": missing,
        "ordered": ordered,
        "failure_signatures": failures,
        "severity_signatures": severity,
    }


def _json_object(raw: str, *, label: str) -> Mapping[str, Any]:
    payload = _strict_json_value(raw, label=label)
    if (
        isinstance(payload, list)
        and len(payload) == 1
        and isinstance(payload[0], Mapping)
    ):
        return payload[0]
    if not isinstance(payload, Mapping):
        raise RuntimeContainerError(f"{label} must be a JSON object")
    return payload


def _encoded_private_json(payload: Mapping[str, object]) -> bytes:
    """Return the one JSON byte representation persisted by this supervisor."""

    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_private_json(path: Path, payload: Mapping[str, object]) -> None:
    try:
        atomic_write_private_file(
            path,
            _encoded_private_json(payload),
            label=path.name,
        )
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc


def _write_private_text(path: Path, text: str) -> None:
    try:
        atomic_write_private_file(
            path,
            text.encode("utf-8", errors="replace"),
            label=path.name,
        )
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc


def analytics_state_contract(
    session: SessionPaths,
    lane: RuntimeLane = BASELINE_LANE,
    appliance_binding: ApplianceBinding | None = None,
) -> dict[str, object]:
    """Return the exact planned host/container analytics-state boundary."""

    lane = resolve_runtime_lane(lane)
    host_root = (
        session.persistent_analytics
        if appliance_binding is None
        else _appliance_analytics_root(appliance_binding)
    )
    contract = {
        "reviewed_seed": ANALYTICS_SEED_RELATIVE,
        "canonical_source_ids": list(_load_lane_source_ids(lane)),
        "host_paths": {
            "config": str(host_root / ANALYTICS_CONFIG_FILENAME),
            "exclude_config": str(host_root / ANALYTICS_EXCLUDE_FILENAME),
        },
        "mount": {
            "source": str(host_root),
            "destination": str(CONTAINER_ANALYTICS_ROOT),
            "writable": True,
            "nested_after_session_state": True,
        },
        "container_environment": {
            "NOESIS_ANALYTICS_CONFIG": str(CONTAINER_ANALYTICS_CONFIG),
            "NOESIS_ANALYTICS_EXCLUDE_CONFIG": str(CONTAINER_ANALYTICS_EXCLUDE_CONFIG),
        },
        "limits_bytes": {
            "yaml": ANALYTICS_YAML_MAX_BYTES,
            "exclude_ini": ANALYTICS_EXCLUDE_INI_MAX_BYTES,
        },
        "session_evidence": {
            phase: {
                "config": str(
                    session.launcher_evidence
                    / f"analytics-{phase}-{ANALYTICS_CONFIG_FILENAME}"
                ),
                "exclude_config": str(
                    session.launcher_evidence
                    / f"analytics-{phase}-{ANALYTICS_EXCLUDE_FILENAME}"
                ),
                "metadata": str(
                    session.launcher_evidence / f"analytics-state-{phase}.json"
                ),
            }
            for phase in ("before", "after")
        },
        "initialization": "seed_persistent_pair_once_or_preserve_exact_private_pair",
    }
    if appliance_binding is not None:
        contract.update(
            {
                "reviewed_seed": None,
                "state_release_id": appliance_binding.state_release_id,
                "initialization": (
                    "validate_selected_mutable_pair_without_seed_copy_or_fallback"
                ),
            }
        )
    return contract


def _validate_plan_analytics_state(
    plan: Mapping[str, object],
    session: SessionPaths,
    lane: RuntimeLane = BASELINE_LANE,
    appliance_binding: ApplianceBinding | None = None,
) -> None:
    expected = analytics_state_contract(session, lane, appliance_binding)
    if plan.get("analytics_state") != expected:
        raise RuntimeContainerError(
            "runtime plan analytics-state contract is missing or drifted"
        )


def _read_reviewed_analytics_seed() -> bytes:
    path = _absolute_without_symlinks(
        REPO_ROOT / ANALYTICS_SEED_RELATIVE,
        label="reviewed analytics seed",
        must_exist=False,
    )
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeContainerError(
            f"reviewed analytics seed is missing: {path}"
        ) from exc
    except OSError as exc:
        raise RuntimeContainerError(
            "reviewed analytics seed cannot be inspected"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeContainerError("reviewed analytics seed must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeContainerError(
            "reviewed analytics seed cannot be opened securely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeContainerError("reviewed analytics seed changed while opening")
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeContainerError(
                "reviewed analytics seed must be a regular file"
            )
        body = bytearray()
        while len(body) <= ANALYTICS_YAML_MAX_BYTES:
            block = os.read(
                descriptor, min(65536, ANALYTICS_YAML_MAX_BYTES + 1 - len(body))
            )
            if not block:
                break
            body.extend(block)
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise RuntimeContainerError(
            "reviewed analytics seed changed while reading"
        ) from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_after != identity_before:
        raise RuntimeContainerError("reviewed analytics seed changed while reading")
    if len(body) > ANALYTICS_YAML_MAX_BYTES:
        raise RuntimeContainerError(
            f"reviewed analytics seed exceeds {ANALYTICS_YAML_MAX_BYTES} bytes"
        )
    return bytes(body)


def _validate_analytics_stream_coverage(
    config: Mapping[str, Any],
    canonical_source_ids: Sequence[str],
    *,
    label: str,
) -> None:
    expected = {
        _canonical_source_id(value, label=f"{label} canonical source ID")
        for value in canonical_source_ids
    }
    analytics = config.get("analytics")
    stages = analytics.get("stages") if isinstance(analytics, Mapping) else None
    exclude = stages.get("exclude") if isinstance(stages, Mapping) else None
    streams = exclude.get("streams") if isinstance(exclude, Mapping) else None
    if not isinstance(streams, Mapping):
        raise RuntimeContainerError(
            f"{label} must define analytics.stages.exclude.streams"
        )
    actual = {
        _canonical_source_id(value, label=f"{label} exclusion stream key")
        for value in streams
    }
    if actual != expected or len(actual) != len(streams):
        raise RuntimeContainerError(
            f"{label} exclusion stream coverage drifted: "
            f"missing={sorted(expected - actual, key=int)} "
            f"unexpected={sorted(actual - expected, key=int)}"
        )


def _load_analytics_config(
    payload: bytes,
    *,
    label: str,
    canonical_source_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if len(payload) > ANALYTICS_YAML_MAX_BYTES:
        raise RuntimeContainerError(
            f"{label} exceeds {ANALYTICS_YAML_MAX_BYTES} bytes"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeContainerError(f"{label} must be UTF-8") from exc
    try:
        loaded = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise RuntimeContainerError(f"{label} contains invalid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeContainerError(f"{label} root must be a mapping")
    if loaded.get("version") != 1:
        raise RuntimeContainerError(f"{label} version must be 1")
    analytics = loaded.get("analytics")
    stages = analytics.get("stages") if isinstance(analytics, Mapping) else None
    exclude = stages.get("exclude") if isinstance(stages, Mapping) else None
    if not isinstance(exclude, dict):
        raise RuntimeContainerError(f"{label} must define analytics.stages.exclude")
    for stage_name, stage_config in stages.items():
        if not isinstance(stage_config, Mapping):
            continue
        streams = stage_config.get("streams")
        if streams is None:
            continue
        if not isinstance(streams, Mapping):
            raise RuntimeContainerError(
                f"{label} analytics stage {stage_name!r} streams must be a mapping"
            )
        canonical_keys: dict[str, object] = {}
        for stream_key in streams:
            canonical = str(stream_key)
            if canonical in canonical_keys:
                raise RuntimeContainerError(
                    f"{label} analytics stage {stage_name!r} has ambiguous stream keys "
                    f"{canonical_keys[canonical]!r} and {stream_key!r}"
                )
            canonical_keys[canonical] = stream_key
    try:
        from DS9.noesis.server import analytics_api

        analytics_api._normalize_stream_keys(loaded)
    except Exception as exc:
        raise RuntimeContainerError(
            f"{label} stream normalization failed: {exc}"
        ) from exc
    if canonical_source_ids is not None:
        _validate_analytics_stream_coverage(
            loaded,
            canonical_source_ids,
            label=label,
        )
    return loaded


def _render_analytics_exclude_ini(
    config: Mapping[str, Any], *, private_directory: Path
) -> bytes:
    analytics = config.get("analytics")
    stages = analytics.get("stages") if isinstance(analytics, Mapping) else None
    exclude = stages.get("exclude") if isinstance(stages, Mapping) else None
    if not isinstance(exclude, dict):
        raise RuntimeContainerError(
            "analytics state must define analytics.stages.exclude"
        )
    try:
        ensure_private_directory(
            private_directory,
            label="DS9 analytics state",
        )
        from DS9.noesis.server import analytics_api

        if (
            analytics_api.ANALYTICS_YAML_MAX_BYTES != ANALYTICS_YAML_MAX_BYTES
            or analytics_api.ANALYTICS_EXCLUDE_INI_MAX_BYTES
            != ANALYTICS_EXCLUDE_INI_MAX_BYTES
        ):
            raise RuntimeContainerError(
                "DS9 analytics API and supervisor size contracts drifted"
            )
        rendered = analytics_api._render_exclude_ini(exclude).encode("utf-8")
        if not rendered or len(rendered) > ANALYTICS_EXCLUDE_INI_MAX_BYTES:
            raise RuntimeContainerError(
                "rendered analytics exclusion config is empty or oversized"
            )
        return rendered
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc
    except RuntimeContainerError:
        raise
    except Exception as exc:
        raise RuntimeContainerError(
            f"analytics exclusion config render failed: {exc}"
        ) from exc


def _read_analytics_state_pair(
    session: SessionPaths,
    lane: RuntimeLane,
    appliance_binding: ApplianceBinding | None = None,
) -> tuple[bytes, bytes, tuple[str, ...]]:
    """Read and validate the appliance-persistent analytics pair without mutation."""

    lane = resolve_runtime_lane(lane)
    analytics_directory = (
        session.persistent_analytics
        if appliance_binding is None
        else _appliance_analytics_root(appliance_binding)
    )
    if not analytics_directory.exists() or analytics_directory.is_symlink():
        raise RuntimeContainerError(
            "DS9 appliance-persistent analytics state is missing or linked"
        )
    try:
        directory = ensure_private_directory(
            analytics_directory,
            label="DS9 analytics state",
        )
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc
    config_path = directory / ANALYTICS_CONFIG_FILENAME
    exclude_path = directory / ANALYTICS_EXCLUDE_FILENAME
    allowed_names = {config_path.name, exclude_path.name}
    if appliance_binding is None:
        unexpected = sorted(
            path.name for path in directory.iterdir() if path.name not in allowed_names
        )
        if unexpected:
            raise RuntimeContainerError(
                f"DS9 analytics state contains unexpected entries: {unexpected}"
            )

    config_present = config_path.exists() or config_path.is_symlink()
    exclude_present = exclude_path.exists() or exclude_path.is_symlink()
    if not config_present or not exclude_present:
        raise RuntimeContainerError(
            "DS9 analytics state is inconsistent: config and exclusion INI must both exist"
        )
    try:
        config_payload = read_private_file(
            config_path,
            label="DS9 analytics config",
            max_bytes=ANALYTICS_YAML_MAX_BYTES,
        )
        exclude_payload = read_private_file(
            exclude_path,
            label="DS9 analytics exclusion config",
            max_bytes=ANALYTICS_EXCLUDE_INI_MAX_BYTES,
        )
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc
    source_ids = _load_lane_source_ids(lane)
    config = _load_analytics_config(
        config_payload,
        label="DS9 analytics config",
        canonical_source_ids=source_ids,
    )
    expected_exclude = _render_analytics_exclude_ini(
        config,
        private_directory=directory,
    )
    if exclude_payload != expected_exclude:
        raise RuntimeContainerError(
            "DS9 analytics state is inconsistent: exclusion INI does not match YAML"
        )
    return config_payload, exclude_payload, source_ids


def _publish_analytics_seed_pair(
    *,
    parent: Path,
    destination: Path,
    config_payload: bytes,
    exclude_payload: bytes,
) -> None:
    """Atomically publish the initial analytics directory as one no-replace unit."""

    candidate = parent / f"{ANALYTICS_SEED_DIRECTORY_PREFIX}{secrets.token_hex(12)}"
    published = False
    try:
        directory = ensure_private_directory(
            candidate,
            label="DS9 analytics seed candidate",
        )
        atomic_write_private_file(
            directory / ANALYTICS_CONFIG_FILENAME,
            config_payload,
            label="DS9 analytics seed config",
        )
        atomic_write_private_file(
            directory / ANALYTICS_EXCLUDE_FILENAME,
            exclude_payload,
            label="DS9 analytics seed exclusion config",
        )
        observed_config = read_private_file(
            directory / ANALYTICS_CONFIG_FILENAME,
            label="DS9 analytics seed config",
            max_bytes=ANALYTICS_YAML_MAX_BYTES,
        )
        observed_exclude = read_private_file(
            directory / ANALYTICS_EXCLUDE_FILENAME,
            label="DS9 analytics seed exclusion config",
            max_bytes=ANALYTICS_EXCLUDE_INI_MAX_BYTES,
        )
        if observed_config != config_payload or observed_exclude != exclude_payload:
            raise RuntimeContainerError(
                "DS9 analytics seed candidate failed exact read-back verification"
            )

        parent_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(libc, "renameat2", None)
            if renameat2 is None:
                raise RuntimeContainerError(
                    "atomic no-replace analytics directory publication is unavailable"
                )
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                parent_descriptor,
                os.fsencode(candidate.name),
                parent_descriptor,
                os.fsencode(destination.name),
                _RENAME_NOREPLACE,
            )
            if result != 0:
                error_number = ctypes.get_errno()
                if error_number == errno.EEXIST:
                    raise RuntimeContainerError(
                        "DS9 analytics state appeared during seed publication"
                    )
                raise RuntimeContainerError(
                    "atomic analytics directory publication failed"
                ) from OSError(
                    error_number,
                    os.strerror(error_number),
                    destination.name,
                )
            published = True
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc
    finally:
        if not published:
            try:
                candidate_info = candidate.lstat()
            except FileNotFoundError:
                candidate_info = None
            if candidate_info is not None and stat.S_ISDIR(candidate_info.st_mode):
                for name in (
                    ANALYTICS_CONFIG_FILENAME,
                    ANALYTICS_EXCLUDE_FILENAME,
                ):
                    path = candidate / name
                    try:
                        info = path.lstat()
                    except FileNotFoundError:
                        continue
                    if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        path.unlink()
                try:
                    candidate.rmdir()
                except OSError:
                    pass


def prepare_analytics_state(
    session: SessionPaths,
    lane: RuntimeLane = BASELINE_LANE,
    appliance_binding: ApplianceBinding | None = None,
) -> dict[str, object]:
    """Seed once or preserve the private appliance-persistent analytics pair."""

    lane = resolve_runtime_lane(lane)
    if appliance_binding is not None:
        directory = _appliance_analytics_root(appliance_binding)
        config_payload, exclude_payload, source_ids = _read_analytics_state_pair(
            session,
            lane,
            appliance_binding,
        )
        return {
            "action": "selected",
            "persistence": "state_release",
            "state_release_id": appliance_binding.state_release_id,
            "canonical_source_ids": list(source_ids),
            "config_path": str(directory / ANALYTICS_CONFIG_FILENAME),
            "config_sha256": hashlib.sha256(config_payload).hexdigest(),
            "exclude_config_path": str(directory / ANALYTICS_EXCLUDE_FILENAME),
            "exclude_config_sha256": hashlib.sha256(exclude_payload).hexdigest(),
            "directory_mode": "0700",
            "file_mode": "0600",
        }
    try:
        persistent_root = ensure_private_directory(
            session.persistent_analytics.parent,
            label="DS9 persistent state root",
        )
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc
    seed_residue = sorted(
        path.name
        for path in persistent_root.iterdir()
        if path.name.startswith(ANALYTICS_SEED_DIRECTORY_PREFIX)
    )
    if seed_residue:
        raise RuntimeContainerError(
            "unresolved DS9 analytics seed transaction requires recovery: "
            f"{seed_residue}"
        )

    destination_present = (
        session.persistent_analytics.exists()
        or session.persistent_analytics.is_symlink()
    )
    action = "preserved"
    if destination_present:
        try:
            directory = ensure_private_directory(
                session.persistent_analytics,
                label="DS9 analytics state",
            )
        except PrivatePathError as exc:
            raise RuntimeContainerError(str(exc)) from exc
    else:
        config_payload = _read_reviewed_analytics_seed()
        source_ids = _load_lane_source_ids(lane)
        config = _load_analytics_config(
            config_payload,
            label="reviewed analytics seed",
            canonical_source_ids=source_ids,
        )
        exclude_payload = _render_analytics_exclude_ini(
            config,
            private_directory=persistent_root,
        )
        _publish_analytics_seed_pair(
            parent=persistent_root,
            destination=session.persistent_analytics,
            config_payload=config_payload,
            exclude_payload=exclude_payload,
        )
        try:
            directory = ensure_private_directory(
                session.persistent_analytics,
                label="DS9 analytics state",
            )
        except PrivatePathError as exc:
            raise RuntimeContainerError(str(exc)) from exc
        action = "seeded"

    config_path = directory / ANALYTICS_CONFIG_FILENAME
    exclude_path = directory / ANALYTICS_EXCLUDE_FILENAME
    allowed_names = {config_path.name, exclude_path.name}
    unexpected = sorted(
        path.name for path in directory.iterdir() if path.name not in allowed_names
    )
    if unexpected:
        raise RuntimeContainerError(
            f"DS9 analytics state contains unexpected entries: {unexpected}"
        )
    config_present = config_path.exists() or config_path.is_symlink()
    exclude_present = exclude_path.exists() or exclude_path.is_symlink()
    if config_present != exclude_present:
        raise RuntimeContainerError(
            "DS9 analytics state is inconsistent: config and exclusion INI must both exist or both be absent"
        )
    if not config_present:
        raise RuntimeContainerError(
            "DS9 analytics state is inconsistent: atomic seed directory contains no pair"
        )

    config_payload, exclude_payload, source_ids = _read_analytics_state_pair(
        session,
        lane,
    )
    return {
        "action": action,
        "persistence": "appliance",
        "canonical_source_ids": list(source_ids),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "exclude_config_path": str(exclude_path),
        "exclude_config_sha256": hashlib.sha256(exclude_payload).hexdigest(),
        "directory_mode": "0700",
        "file_mode": "0600",
    }


def capture_analytics_state_evidence(
    session: SessionPaths,
    lane: RuntimeLane,
    *,
    phase: str,
    action: str,
    appliance_binding: ApplianceBinding | None = None,
) -> dict[str, object]:
    """Copy one validated persistent state snapshot into session-local evidence."""

    if phase not in {"before", "after"}:
        raise RuntimeContainerError(f"unsupported analytics evidence phase: {phase!r}")
    config_payload, exclude_payload, source_ids = _read_analytics_state_pair(
        session,
        lane,
        appliance_binding,
    )
    analytics_root = (
        session.persistent_analytics
        if appliance_binding is None
        else _appliance_analytics_root(appliance_binding)
    )
    config_snapshot = (
        session.launcher_evidence / f"analytics-{phase}-{ANALYTICS_CONFIG_FILENAME}"
    )
    exclude_snapshot = (
        session.launcher_evidence / f"analytics-{phase}-{ANALYTICS_EXCLUDE_FILENAME}"
    )
    metadata_path = session.launcher_evidence / f"analytics-state-{phase}.json"
    for path in (config_snapshot, exclude_snapshot, metadata_path):
        if path.exists() or path.is_symlink():
            raise RuntimeContainerError(
                f"refusing to replace immutable analytics evidence: {path}"
            )
    try:
        atomic_write_private_file(
            config_snapshot,
            config_payload,
            label=f"analytics {phase} YAML evidence",
        )
        atomic_write_private_file(
            exclude_snapshot,
            exclude_payload,
            label=f"analytics {phase} INI evidence",
        )
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc
    metadata: dict[str, object] = {
        "phase": phase,
        "action": action,
        "persistence": "appliance",
        "canonical_source_ids": list(source_ids),
        "config_path": str(analytics_root / ANALYTICS_CONFIG_FILENAME),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "config_snapshot": str(config_snapshot),
        "exclude_config_path": str(
            analytics_root / ANALYTICS_EXCLUDE_FILENAME
        ),
        "exclude_config_sha256": hashlib.sha256(exclude_payload).hexdigest(),
        "exclude_config_snapshot": str(exclude_snapshot),
    }
    _write_private_json(metadata_path, metadata)
    return metadata


def prepare_session_paths(
    roots: HostRoots,
    session: SessionPaths,
    appliance_binding: ApplianceBinding | None = None,
) -> None:
    try:
        ensure_private_directory(roots.runtime, label="DS9 runtime root")
        for category in ("build", "state", "depth", "evidence", "persistent"):
            ensure_private_directory(
                roots.runtime / category,
                label=f"DS9 runtime {category} root",
            )
        session_paths = (
            session.state,
            session.depth,
            session.evidence,
            session.runtime_evidence,
            session.launcher_evidence,
        )
        if appliance_binding is None:
            session_paths = (session.build, *session_paths)
        for path in session_paths:
            if path.exists() or path.is_symlink():
                raise RuntimeContainerError(
                    f"refusing to reuse a runtime session path: {path}"
                )
            ensure_private_directory(path, label=f"DS9 session {path.name}")
        build_root = (
            session.build
            if appliance_binding is None
            else appliance_binding.state.build_directory
        )
        for path in (
            session.state / "home",
            build_root / "cache",
            build_root / "xdg-runtime",
            build_root / "cuda-cache",
            session.state / "household",
        ):
            ensure_private_directory(path, label=f"DS9 session directory {path.name}")
    except PrivatePathError as exc:
        raise RuntimeContainerError(str(exc)) from exc


def _wait_for_readiness(
    roots: HostRoots,
    runner: CommandRunner,
    container_name: str,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        state = runner.run(
            _docker_command(
                roots,
                "inspect",
                container_name,
                "--format",
                "{{.State.Running}} {{.State.ExitCode}}",
            ),
            timeout=10.0,
        ).stdout.strip()
        if not state.startswith("true "):
            raise RuntimeContainerError(
                f"DS9 container exited before canonical ports were ready: {state}"
            )
        if all(_port_listening(port) for port in CANONICAL_PORTS):
            return
        time.sleep(0.25)
    raise RuntimeContainerError(
        f"DS9 container did not open canonical ports {CANONICAL_PORTS} within {timeout_seconds}s"
    )


def observe_runtime_identity(
    internal_auth_file: Path,
    *,
    session_id: str,
    runtime_lane: str,
) -> dict[str, object]:
    """Bind one authenticated capability-health identity to the supervisor run."""

    token = load_internal_token(internal_auth_file)
    connection = http.client.HTTPConnection(
        "127.0.0.1", CANONICAL_REST_PORT, timeout=5.0
    )
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
        raw = response.read(64 * 1024 + 1)
    finally:
        connection.close()
    if response.status != 200 or len(raw) > 64 * 1024:
        raise RuntimeContainerError(
            "authenticated runtime identity endpoint did not return one bounded success"
        )
    payload = _strict_json_value(
        raw,
        label="authenticated runtime identity endpoint response",
    )
    if not isinstance(payload, Mapping):
        raise RuntimeContainerError("runtime identity payload must be an object")
    instance_id = str(payload.get("instance_id") or "")
    run_id = str(payload.get("run_id") or "")
    generated_at_us = payload.get("generated_at_us")
    if (
        payload.get("contract") != "noesis.capability.health"
        or payload.get("contract_version") != 1
        or RUNTIME_ID_RE.fullmatch(instance_id) is None
        or RUNTIME_ID_RE.fullmatch(run_id) is None
        or isinstance(generated_at_us, bool)
        or not isinstance(generated_at_us, int)
        or generated_at_us <= 0
    ):
        raise RuntimeContainerError("runtime identity capability contract drifted")
    return {
        "schema_version": 1,
        "contract": RUNTIME_IDENTITY_CONTRACT,
        "contract_version": 1,
        "session_id": session_id,
        "runtime_lane": runtime_lane,
        "runtime_instance_id": instance_id,
        "runtime_run_id": run_id,
        "health_generated_at_us": generated_at_us,
        "observed_at_utc": utc_now(),
        "endpoints": dict(CANONICAL_ENDPOINTS),
    }


def _resource_soak_lane(runtime_lane: str) -> RuntimeLane:
    lane = resolve_runtime_lane(runtime_lane)
    if lane.name not in RESOURCE_SOAK_GPU_MEMORY_LIMIT_MIB:
        raise RuntimeContainerError(
            "resource-soak acceptance requires v3dt or a reviewed Wholebody49 lane"
        )
    return lane


def _resource_soak_thresholds(runtime_lane: str) -> dict[str, int | float]:
    lane = _resource_soak_lane(runtime_lane)
    return {
        "minimum_duration_seconds": RESOURCE_SOAK_MIN_DURATION_SECONDS,
        "sample_interval_seconds": RESOURCE_SOAK_SAMPLE_INTERVAL_SECONDS,
        "warmup_seconds": RESOURCE_SOAK_WARMUP_SECONDS,
        "maximum_sample_gap_seconds": RESOURCE_SOAK_MAX_SAMPLE_GAP_SECONDS,
        "maximum_initial_sample_delay_seconds": (
            RESOURCE_SOAK_MAX_INITIAL_SAMPLE_DELAY_SECONDS
        ),
        "minimum_sample_count": RESOURCE_SOAK_MIN_SAMPLE_COUNT,
        "minimum_post_warmup_sample_count": (
            RESOURCE_SOAK_MIN_POST_WARMUP_SAMPLE_COUNT
        ),
        "maximum_memory_slope_bytes_per_second": (
            RESOURCE_SOAK_MAX_MEMORY_SLOPE_BYTES_PER_SECOND
        ),
        "maximum_memory_growth_bytes": RESOURCE_SOAK_MAX_MEMORY_GROWTH_BYTES,
        "maximum_absolute_memory_bytes_exclusive": RESOURCE_SOAK_MAX_MEMORY_BYTES,
        "maximum_gpu_process_memory_mib_exclusive": (
            RESOURCE_SOAK_GPU_MEMORY_LIMIT_MIB[lane.name]
        ),
        "maximum_pids_exclusive": RESOURCE_SOAK_MAX_PIDS_EXCLUSIVE,
    }


def _normalize_resource_runtime_binding(
    binding: Mapping[str, object], *, runtime_lane: str
) -> dict[str, str]:
    lane = _resource_soak_lane(runtime_lane)
    expected_keys = {
        "container_id",
        "runtime_image_id",
        "checkout_sha256",
        "realization_sha256",
        "primary_engine_artifact_id",
        "primary_engine_sha256",
        "pipeline_config",
        "pipeline_config_sha256",
        "cameras_config",
        "cameras_config_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_keys:
        raise RuntimeContainerError("resource-soak runtime binding schema drifted")
    normalized = {key: str(binding.get(key) or "") for key in expected_keys}
    if re.fullmatch(r"[0-9a-f]{64}", normalized["container_id"]) is None:
        raise RuntimeContainerError("resource-soak container binding is invalid")
    if (
        not normalized["runtime_image_id"].startswith("sha256:")
        or re.fullmatch(
            r"[0-9a-f]{64}",
            normalized["runtime_image_id"].removeprefix("sha256:"),
        )
        is None
    ):
        raise RuntimeContainerError("resource-soak image binding is invalid")
    for key in (
        "checkout_sha256",
        "realization_sha256",
        "primary_engine_sha256",
        "pipeline_config_sha256",
        "cameras_config_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", normalized[key]) is None:
            raise RuntimeContainerError(f"resource-soak {key} binding is invalid")
    if (
        normalized["primary_engine_artifact_id"]
        != RESOURCE_SOAK_PRIMARY_ENGINE_ID[lane.name]
        or normalized["pipeline_config"] != lane.pipeline_config
        or normalized["cameras_config"] != lane.cameras_config
    ):
        raise RuntimeContainerError("resource-soak lane/engine/config binding drifted")
    return normalized


def build_resource_runtime_binding(
    plan: Mapping[str, object],
    inspect: Mapping[str, Any],
    *,
    lane: RuntimeLane,
) -> dict[str, str]:
    lane = _resource_soak_lane(lane.name)
    readiness = plan.get("artifact_readiness")
    checkout = plan.get("checkout_snapshot")
    image = plan.get("image")
    output_sha256 = (
        readiness.get("selected_output_sha256")
        if isinstance(readiness, Mapping)
        else None
    )
    engine_id = RESOURCE_SOAK_PRIMARY_ENGINE_ID[lane.name]
    binding = {
        "container_id": str(inspect.get("Id") or "").strip().lower(),
        "runtime_image_id": str(image.get("id") if isinstance(image, Mapping) else ""),
        "checkout_sha256": str(
            checkout.get("sha256") if isinstance(checkout, Mapping) else ""
        ),
        "realization_sha256": str(
            readiness.get("realization_sha256")
            if isinstance(readiness, Mapping)
            else ""
        ),
        "primary_engine_artifact_id": engine_id,
        "primary_engine_sha256": str(
            output_sha256.get(engine_id)
            if isinstance(output_sha256, Mapping)
            else ""
        ),
        "pipeline_config": lane.pipeline_config,
        "pipeline_config_sha256": _sha256_file(REPO_ROOT / lane.pipeline_config),
        "cameras_config": lane.cameras_config,
        "cameras_config_sha256": _sha256_file(REPO_ROOT / lane.cameras_config),
    }
    return _normalize_resource_runtime_binding(binding, runtime_lane=lane.name)


def _sample_nonnegative_integer(sample: Mapping[str, object], key: str) -> int:
    value = sample.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeContainerError(
            f"resource-soak sample {key} must be a non-negative integer"
        )
    return value


def _normalize_resource_soak_samples(
    samples: Sequence[Mapping[str, object]],
) -> list[dict[str, int | float | bool | str]]:
    normalized: list[dict[str, int | float | bool | str]] = []
    previous_elapsed = -1.0
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise RuntimeContainerError(
                f"resource-soak sample {index} must be an object"
            )
        raw_elapsed = sample.get("elapsed_seconds")
        if isinstance(raw_elapsed, bool) or not isinstance(raw_elapsed, (int, float)):
            raise RuntimeContainerError(
                f"resource-soak sample {index} lacks elapsed_seconds"
            )
        elapsed = float(raw_elapsed)
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed <= previous_elapsed:
            raise RuntimeContainerError(
                "resource-soak sample elapsed_seconds must be finite and strictly increasing"
            )
        previous_elapsed = elapsed
        row: dict[str, int | float | bool | str] = {
            "elapsed_seconds": elapsed,
            "captured_at_utc": str(sample.get("captured_at_utc") or ""),
            "memory_current_bytes": _sample_nonnegative_integer(
                sample, "memory_current_bytes"
            ),
            "memory_peak_bytes": _sample_nonnegative_integer(
                sample, "memory_peak_bytes"
            ),
            "memory_events_oom": _sample_nonnegative_integer(
                sample, "memory_events_oom"
            ),
            "memory_events_oom_kill": _sample_nonnegative_integer(
                sample, "memory_events_oom_kill"
            ),
            "pids_current": _sample_nonnegative_integer(sample, "pids_current"),
            "gpu_compute_owner_count": _sample_nonnegative_integer(
                sample, "gpu_compute_owner_count"
            ),
            "gpu_used_memory_mib": _sample_nonnegative_integer(
                sample, "gpu_used_memory_mib"
            ),
            "gpu_largest_process_memory_mib": _sample_nonnegative_integer(
                sample, "gpu_largest_process_memory_mib"
            ),
            "gpu_owner_verified": sample.get("gpu_owner_verified") is True,
        }
        if int(row["memory_peak_bytes"]) < int(row["memory_current_bytes"]):
            raise RuntimeContainerError(
                "resource-soak memory.peak is below memory.current"
            )
        if int(row["gpu_compute_owner_count"]) <= 0:
            raise RuntimeContainerError(
                "resource-soak sample has no confirmed GPU compute owner"
            )
        normalized.append(row)
    return normalized


def _least_squares_slope(
    rows: Sequence[Mapping[str, int | float | bool | str]],
) -> float | None:
    if len(rows) < 2:
        return None
    x_values = [float(row["elapsed_seconds"]) for row in rows]
    y_values = [float(row["memory_current_bytes"]) for row in rows]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0:
        return None
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    return numerator / denominator


def evaluate_resource_soak_samples(
    samples: Sequence[Mapping[str, object]],
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    requested_duration_seconds: float,
    stop_reason: str,
    runtime_binding: Mapping[str, object],
    sampling_error: str | None = None,
) -> dict[str, object]:
    """Evaluate one synthetic or live reviewed-lane resource-soak sample series."""

    session_id = str(session_id).strip().lower()
    runtime_lane = str(runtime_lane).strip().lower()
    if SESSION_RE.fullmatch(session_id) is None:
        raise RuntimeContainerError(
            "resource-soak session ID must match [a-z0-9][a-z0-9-]{5,47}"
        )
    lane = _resource_soak_lane(runtime_lane)
    if (
        RUNTIME_ID_RE.fullmatch(str(runtime_instance_id)) is None
        or RUNTIME_ID_RE.fullmatch(str(runtime_run_id)) is None
    ):
        raise RuntimeContainerError("resource-soak runtime identity is invalid")
    validation_errors: list[str] = []
    try:
        normalized_binding = _normalize_resource_runtime_binding(
            runtime_binding,
            runtime_lane=lane.name,
        )
        requested_duration = float(requested_duration_seconds)
        if not math.isfinite(requested_duration) or requested_duration < 0:
            raise RuntimeContainerError(
                "resource-soak requested duration must be finite and non-negative"
            )
        normalized = _normalize_resource_soak_samples(samples)
    except (RuntimeContainerError, TypeError, ValueError) as exc:
        try:
            requested_duration = float(requested_duration_seconds)
        except (TypeError, ValueError):
            requested_duration = 0.0
        if not math.isfinite(requested_duration) or requested_duration < 0:
            requested_duration = 0.0
        normalized = []
        normalized_binding = {}
        validation_errors.append(str(exc))

    sample_count = len(normalized)
    first_elapsed = float(normalized[0]["elapsed_seconds"]) if normalized else 0.0
    last_elapsed = float(normalized[-1]["elapsed_seconds"]) if normalized else 0.0
    actual_duration = last_elapsed if normalized else 0.0
    gaps = [
        float(right["elapsed_seconds"]) - float(left["elapsed_seconds"])
        for left, right in zip(normalized, normalized[1:])
    ]
    maximum_gap = max(gaps, default=0.0)
    post_warmup = [
        row
        for row in normalized
        if float(row["elapsed_seconds"]) >= RESOURCE_SOAK_WARMUP_SECONDS
    ]
    slope = _least_squares_slope(post_warmup)
    post_memory = [int(row["memory_current_bytes"]) for row in post_warmup]
    post_growth = max(post_memory) - min(post_memory) if post_memory else None
    current_memory = [int(row["memory_current_bytes"]) for row in normalized]
    peak_memory = [int(row["memory_peak_bytes"]) for row in normalized]
    oom = [int(row["memory_events_oom"]) for row in normalized]
    oom_kill = [int(row["memory_events_oom_kill"]) for row in normalized]
    event_counters_monotonic = bool(normalized) and all(
        left <= right
        for values in (oom, oom_kill)
        for left, right in zip(values, values[1:])
    )
    oom_increment = oom[-1] - oom[0] if event_counters_monotonic else None
    oom_kill_increment = (
        oom_kill[-1] - oom_kill[0] if event_counters_monotonic else None
    )
    gpu_memory = [int(row["gpu_used_memory_mib"]) for row in normalized]
    pids = [int(row["pids_current"]) for row in normalized]
    thresholds = _resource_soak_thresholds(lane.name)

    checks = {
        "samples_well_formed": not validation_errors,
        "sampling_completed_without_error": sampling_error is None,
        "duration_stop_completed": stop_reason == "duration_complete",
        "requested_duration_sufficient": (
            requested_duration >= RESOURCE_SOAK_MIN_DURATION_SECONDS
        ),
        "observed_duration_sufficient": (
            actual_duration >= RESOURCE_SOAK_MIN_DURATION_SECONDS
        ),
        "sample_count_sufficient": sample_count >= RESOURCE_SOAK_MIN_SAMPLE_COUNT,
        "post_warmup_sample_count_sufficient": (
            len(post_warmup) >= RESOURCE_SOAK_MIN_POST_WARMUP_SAMPLE_COUNT
        ),
        "sampling_cadence_sufficient": bool(normalized)
        and first_elapsed <= RESOURCE_SOAK_MAX_INITIAL_SAMPLE_DELAY_SECONDS
        and maximum_gap <= RESOURCE_SOAK_MAX_SAMPLE_GAP_SECONDS,
        "gpu_ownership_continuous": bool(normalized)
        and all(bool(row["gpu_owner_verified"]) for row in normalized),
        "memory_event_counters_monotonic": event_counters_monotonic,
        "oom_increment_zero": oom_increment == 0,
        "oom_kill_increment_zero": oom_kill_increment == 0,
        "post_warmup_slope_within_limit": slope is not None
        and slope <= RESOURCE_SOAK_MAX_MEMORY_SLOPE_BYTES_PER_SECOND,
        "post_warmup_growth_within_limit": post_growth is not None
        and post_growth <= RESOURCE_SOAK_MAX_MEMORY_GROWTH_BYTES,
        "absolute_memory_below_limit": bool(normalized)
        and max((*current_memory, *peak_memory), default=RESOURCE_SOAK_MAX_MEMORY_BYTES)
        < RESOURCE_SOAK_MAX_MEMORY_BYTES,
        "gpu_process_memory_below_limit": bool(normalized)
        and max(
            (
                int(row["gpu_used_memory_mib"])
                for row in normalized
            ),
            default=int(thresholds["maximum_gpu_process_memory_mib_exclusive"]),
        )
        < int(thresholds["maximum_gpu_process_memory_mib_exclusive"])
        and max(
            (
                int(row["gpu_largest_process_memory_mib"])
                for row in normalized
            ),
            default=int(thresholds["maximum_gpu_process_memory_mib_exclusive"]),
        )
        < int(thresholds["maximum_gpu_process_memory_mib_exclusive"]),
        "pids_below_limit": bool(normalized)
        and max(pids, default=RESOURCE_SOAK_MAX_PIDS_EXCLUSIVE)
        < RESOURCE_SOAK_MAX_PIDS_EXCLUSIVE,
        "runtime_binding_complete": bool(normalized_binding),
    }
    failed_checks = sorted(key for key, value in checks.items() if not value)
    if sampling_error:
        validation_errors.append(sampling_error)
    metrics: dict[str, object] = {
        "requested_duration_seconds": requested_duration,
        "observed_duration_seconds": actual_duration,
        "first_sample_elapsed_seconds": first_elapsed if normalized else None,
        "last_sample_elapsed_seconds": last_elapsed if normalized else None,
        "maximum_sample_gap_seconds": maximum_gap if normalized else None,
        "sample_count": sample_count,
        "post_warmup_sample_count": len(post_warmup),
        "memory_current_start_bytes": current_memory[0] if normalized else None,
        "memory_current_end_bytes": current_memory[-1] if normalized else None,
        "memory_current_max_bytes": max(current_memory) if normalized else None,
        "memory_peak_max_bytes": max(peak_memory) if normalized else None,
        "post_warmup_slope_bytes_per_second": slope,
        "post_warmup_growth_bytes": post_growth,
        "oom_increment": oom_increment,
        "oom_kill_increment": oom_kill_increment,
        "pids_current_max": max(pids) if normalized else None,
        "gpu_used_memory_mib_start": gpu_memory[0] if normalized else None,
        "gpu_used_memory_mib_end": gpu_memory[-1] if normalized else None,
        "gpu_used_memory_mib_max": max(gpu_memory) if normalized else None,
    }
    ok = not failed_checks and not validation_errors
    return {
        "schema_version": RESOURCE_SOAK_SCHEMA_VERSION,
        "contract": RESOURCE_SOAK_CONTRACT,
        "contract_version": RESOURCE_SOAK_CONTRACT_VERSION,
        "session_id": session_id,
        "runtime_lane": runtime_lane,
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "status": "pass" if ok else "fail",
        "ok": ok,
        "stop_reason": stop_reason,
        "thresholds": thresholds,
        "checks": checks,
        "failed_checks": failed_checks,
        "validation_errors": validation_errors,
        "metrics": metrics,
        "runtime_binding": normalized_binding,
    }


def _capture_resource_soak_sample(
    inspect: Mapping[str, Any],
    runner: CommandRunner,
    cgroup_path: Path,
    *,
    elapsed_seconds: float,
    expected_init_pid: int,
) -> dict[str, object]:
    confirmation = confirm_container_gpu_ownership(
        inspect,
        runner,
        timeout_seconds=min(2.0, RESOURCE_SOAK_SAMPLE_INTERVAL_SECONDS),
    )
    if int(confirmation.get("container_init_pid") or 0) != expected_init_pid:
        raise RuntimeContainerError(
            "resource-soak container GPU owner PID changed during sampling"
        )
    return {
        "elapsed_seconds": elapsed_seconds,
        "captured_at_utc": utc_now(),
        **read_container_cgroup_v2_sample(cgroup_path),
        **characterize_confirmed_gpu_memory(confirmation),
        "gpu_owner_verified": True,
    }


def persist_resource_soak_evidence(
    launcher_evidence: Path,
    samples: Sequence[Mapping[str, object]],
    report: Mapping[str, object],
) -> dict[str, object]:
    session_id = str(report.get("session_id") or "")
    runtime_lane = str(report.get("runtime_lane") or "")
    runtime_instance_id = str(report.get("runtime_instance_id") or "")
    runtime_run_id = str(report.get("runtime_run_id") or "")
    runtime_binding = report.get("runtime_binding")
    if (
        SESSION_RE.fullmatch(session_id) is None
        or runtime_lane not in RESOURCE_SOAK_GPU_MEMORY_LIMIT_MIB
        or RUNTIME_ID_RE.fullmatch(runtime_instance_id) is None
        or RUNTIME_ID_RE.fullmatch(runtime_run_id) is None
        or not isinstance(runtime_binding, Mapping)
    ):
        raise RuntimeContainerError("resource-soak report lacks an exact session/lane binding")
    normalized_binding = _normalize_resource_runtime_binding(
        runtime_binding,
        runtime_lane=runtime_lane,
    )
    samples_path = launcher_evidence / RESOURCE_SOAK_SAMPLES_FILENAME
    report_path = launcher_evidence / RESOURCE_SOAK_REPORT_FILENAME
    for path in (samples_path, report_path):
        if path.exists() or path.is_symlink():
            raise RuntimeContainerError(
                f"refusing to replace immutable resource-soak evidence: {path}"
            )
    _write_private_json(
        samples_path,
        {
            "schema_version": RESOURCE_SOAK_SCHEMA_VERSION,
            "contract": f"{RESOURCE_SOAK_CONTRACT}.samples",
            "contract_version": RESOURCE_SOAK_CONTRACT_VERSION,
            "session_id": session_id,
            "runtime_lane": runtime_lane,
            "runtime_instance_id": runtime_instance_id,
            "runtime_run_id": runtime_run_id,
            "runtime_binding": normalized_binding,
            "sample_count": len(samples),
            "samples": list(samples),
        },
    )
    report_payload = dict(report)
    report_payload["samples_evidence"] = {
        "filename": samples_path.name,
        "sha256": _sha256_file(samples_path),
        "sample_count": len(samples),
    }
    _write_private_json(report_path, report_payload)
    return report_payload


def run_resource_soak(
    roots: HostRoots,
    runner: CommandRunner,
    container_name: str,
    inspect: Mapping[str, Any],
    launcher_evidence: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    duration_seconds: float,
    runtime_binding: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    """Sample one exact reviewed container until the fixed acceptance duration ends."""

    requested_stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal requested_stop
        requested_stop = True

    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous:
        signal.signal(signum, handle_signal)

    samples: list[dict[str, object]] = []
    sampling_error: str | None = None
    stop_reason = "resource_sampling_failure"
    cgroup_path: Path | None = None
    started_at_utc = utc_now()
    started = time.monotonic()
    try:
        _, expected_init_pid = _container_id_and_init_pid(inspect)
        cgroup_path = resolve_container_cgroup_v2_path(inspect)
        sample_index = 0
        while True:
            target = started + sample_index * RESOURCE_SOAK_SAMPLE_INTERVAL_SECONDS
            while True:
                remaining = target - time.monotonic()
                if remaining <= 0:
                    break
                if requested_stop:
                    break
                time.sleep(min(0.25, remaining))
            if requested_stop:
                stop_reason = "operator_signal"
                break
            state = runner.run(
                _docker_command(
                    roots,
                    "inspect",
                    container_name,
                    "--format",
                    "{{.State.Running}} {{.State.ExitCode}}",
                ),
                timeout=10.0,
            ).stdout.strip()
            if not state.startswith("true "):
                raise RuntimeContainerError(
                    f"DS9 container exited during resource soak: {state}"
                )
            elapsed = time.monotonic() - started
            samples.append(
                _capture_resource_soak_sample(
                    inspect,
                    runner,
                    cgroup_path,
                    elapsed_seconds=elapsed,
                    expected_init_pid=expected_init_pid,
                )
            )
            if elapsed >= duration_seconds:
                stop_reason = "duration_complete"
                break
            sample_index += 1
    except Exception as exc:
        sampling_error = f"{type(exc).__name__}: {exc}"
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    report = evaluate_resource_soak_samples(
        samples,
        session_id=session_id,
        runtime_lane=runtime_lane,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        requested_duration_seconds=duration_seconds,
        stop_reason=stop_reason,
        runtime_binding=runtime_binding,
        sampling_error=sampling_error,
    )
    report.update(
        {
            "started_at_utc": started_at_utc,
            "finished_at_utc": utc_now(),
            "container_cgroup_v2_path": str(cgroup_path) if cgroup_path else None,
        }
    )
    persisted = persist_resource_soak_evidence(
        launcher_evidence,
        samples,
        report,
    )
    return stop_reason, persisted


def _wait_active_duration(
    roots: HostRoots,
    runner: CommandRunner,
    container_name: str,
    duration_seconds: float,
) -> str:
    requested_stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal requested_stop
        requested_stop = True

    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous:
        signal.signal(signum, handle_signal)
    started = time.monotonic()
    try:
        while True:
            if requested_stop:
                return "operator_signal"
            if duration_seconds > 0 and time.monotonic() - started >= duration_seconds:
                return "duration_complete"
            state = runner.run(
                _docker_command(
                    roots,
                    "inspect",
                    container_name,
                    "--format",
                    "{{.State.Running}} {{.State.ExitCode}}",
                ),
                timeout=10.0,
            ).stdout.strip()
            if not state.startswith("true "):
                raise RuntimeContainerError(
                    f"DS9 container exited before orchestrated shutdown: {state}"
                )
            time.sleep(0.25)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _capture_container_artifacts(
    roots: HostRoots,
    runner: CommandRunner,
    container_name: str,
    launcher_evidence: Path,
) -> tuple[str, Mapping[str, Any]]:
    logs = runner.run(
        _docker_command(roots, "logs", "--timestamps", container_name),
        timeout=30.0,
        check=False,
    )
    log_text = (logs.stdout or "") + (logs.stderr or "")
    _write_private_text(launcher_evidence / "runtime.log", log_text)
    inspect_result = runner.run(
        _docker_command(roots, "inspect", container_name),
        timeout=15.0,
        check=False,
    )
    inspect_text = inspect_result.stdout or ""
    if inspect_text.strip():
        _write_private_text(launcher_evidence / "container-inspect.json", inspect_text)
        inspect = _json_object(inspect_text, label="container inspection")
    else:
        inspect = {}
    return log_text, inspect


def _container_absent(
    roots: HostRoots, runner: CommandRunner, container_name: str
) -> bool:
    output = runner.run(
        _docker_command(
            roots,
            "ps",
            "-a",
            "--filter",
            f"name=^/{container_name}$",
            "--format",
            "{{.ID}}",
        ),
        timeout=15.0,
    ).stdout
    return not output.strip()


def require_appliance_runtime_snapshot_current(binding: ApplianceBinding) -> None:
    admitted = compute_noesis_checkout_identity(REPO_ROOT)
    selected_checkout = binding.selector.noesis_checkout
    if (
        admitted.revision != selected_checkout.software_revision
        or admitted.snapshot_sha256 != selected_checkout.snapshot_sha256
    ):
        raise ApplianceConfigurationError(
            "Noesis runtime snapshot drifted immediately before DS9 exec"
        )


def _remove_container(
    roots: HostRoots,
    runner: CommandRunner,
    container_name: str,
    *,
    force: bool,
) -> bool:
    command = _docker_command(roots, "rm")
    if force:
        command.append("--force")
    command.append(container_name)
    result = runner.run(command, timeout=30.0, check=False)
    if result.returncode != 0 and not _container_absent(roots, runner, container_name):
        return False
    return _container_absent(roots, runner, container_name)


def _seal_evidence(directory: Path) -> str:
    rows: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file() or path.is_symlink():
            continue
        rows.append(f"{_sha256_file(path)}  {path.name}")
    body = "\n".join(rows) + ("\n" if rows else "")
    _write_private_text(directory / "SHA256SUMS", body)
    return _sha256_file(directory / "SHA256SUMS")


def build_plan(
    *,
    roots: HostRoots,
    secrets_: SecretFiles,
    session: SessionPaths,
    docker: DockerState,
    source: CheckoutSnapshot,
    repo_provenance: Mapping[str, str],
    artifact_readiness: Mapping[str, object],
    config: Mapping[str, object],
    runtime_containers: Sequence[str],
    runtime_processes: Sequence[str],
    gpu_owners: Sequence[str],
    unavailable_ports: Sequence[int],
    artifact_lock: Mapping[str, object],
    lane: RuntimeLane = BASELINE_LANE,
    appliance_binding: ApplianceBinding | None = None,
) -> dict[str, object]:
    lane = resolve_runtime_lane(lane)
    blockers: list[str] = []
    if runtime_containers:
        blockers.append("secondary daemon already owns a DS9 runtime container")
    if runtime_processes:
        blockers.append("host already owns a DS9 runtime process")
    if gpu_owners:
        blockers.append("GPU compute owner(s) are active")
    if unavailable_ports:
        blockers.append("canonical host-network port(s) are unavailable")
    if not bool(artifact_readiness.get("ok")):
        blockers.append(f"DS9 {lane.name} artifact/provenance gate is not ready")
    if not bool(artifact_lock.get("available")):
        blockers.append("DS9 artifact transaction lock is owned")
    command = build_container_command(
        roots,
        secrets_,
        session,
        lane,
        appliance_binding=appliance_binding,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "mode": "plan",
        "created_at_utc": utc_now(),
        "ready_for_explicit_run": not blockers,
        "blockers": blockers,
        "session_id": session.session_id,
        "runtime_lane": lane.name,
        "image": {
            "reference": IMAGE_REF,
            "id": docker.image_id,
            "base_digest": docker.base_digest,
            "parent_build_image_reference": docker.parent_build_image_reference,
            "parent_build_image_id": docker.parent_build_image_id,
            "parent_rootfs_layer_count": docker.parent_rootfs_layer_count,
            "runtime_rootfs_layer_count": docker.runtime_rootfs_layer_count,
            "parent_rootfs_sha256": docker.parent_rootfs_sha256,
            "runtime_rootfs_sha256": docker.runtime_rootfs_sha256,
        },
        "secondary_docker": {
            "daemon_id": docker.daemon_id,
            "root": docker.docker_root,
            "default_runtime": docker.default_runtime,
            "networks": dict(docker.networks),
        },
        "repository_provenance": dict(repo_provenance),
        "artifact_readiness": dict(artifact_readiness),
        "artifact_transaction_lock": dict(artifact_lock),
        "canonical_runtime": dict(config),
        "checkout_snapshot": source.summary(),
        "host_roots": {
            "artifacts": str(roots.artifacts),
            "runtime": str(roots.runtime),
        },
        "session_paths": {
            "build": str(
                session.build
                if appliance_binding is None
                else appliance_binding.state.build_directory
            ),
            "state": str(session.state),
            "depth": str(session.depth),
            "runtime_evidence": str(session.runtime_evidence),
            "launcher_evidence": str(session.launcher_evidence),
        },
        "analytics_state": analytics_state_contract(
            session,
            lane,
            appliance_binding,
        ),
        "secrets": {
            "camera_sources": {"validated": True, "mounted_individually": True},
            "mapanything_rpc": {"validated": True, "mounted_individually": True},
            "internal_auth": {"validated": True, "mounted_individually": True},
            "parent_tmpfs_mode": "0700",
        },
        "owners": {
            "runtime_containers": list(runtime_containers),
            "runtime_processes": list(runtime_processes),
            "gpu_compute": list(gpu_owners),
        },
        "unavailable_ports": list(unavailable_ports),
        "docker_run_invoked": False,
        "authorized_run_command": redacted_command(command, secrets_),
    }


def preflight(
    *,
    env: Mapping[str, str],
    session_id: str,
    runner: CommandRunner,
    lane: RuntimeLane = BASELINE_LANE,
    appliance_binding: ApplianceBinding | None = None,
) -> tuple[
    HostRoots,
    SecretFiles,
    SessionPaths,
    DockerState,
    CheckoutSnapshot,
    dict[str, object],
]:
    lane = resolve_runtime_lane(lane)
    roots = resolve_host_roots(env)
    secrets_ = resolve_secret_files(env, roots)
    session = SessionPaths.from_root(roots.runtime, session_id)
    docker = inspect_docker_state(roots, runner)
    repo_provenance = validate_repository_provenance()
    host_gpu = host_gpu_identity(runner)
    artifact_readiness = validate_runtime_artifacts(
        roots.artifacts,
        host_gpu=host_gpu,
        lane=lane,
    )
    config = validate_canonical_config(lane)
    if appliance_binding is not None:
        _read_analytics_state_pair(
            session,
            lane,
            appliance_binding,
        )
    source = snapshot_checkout()
    runtime_containers = existing_runtime_containers(roots, runner)
    runtime_processes = existing_host_runtime_processes()
    gpu_owners = gpu_compute_owners(runner)
    unavailable_ports = unavailable_canonical_ports()
    artifact_lock = artifact_transaction_lock_status(roots.artifacts)
    plan = build_plan(
        roots=roots,
        secrets_=secrets_,
        session=session,
        docker=docker,
        source=source,
        repo_provenance=repo_provenance,
        artifact_readiness=artifact_readiness,
        config=config,
        runtime_containers=runtime_containers,
        runtime_processes=runtime_processes,
        gpu_owners=gpu_owners,
        unavailable_ports=unavailable_ports,
        artifact_lock=artifact_lock,
        lane=lane,
        appliance_binding=appliance_binding,
    )
    return roots, secrets_, session, docker, source, plan


def execute(
    *,
    roots: HostRoots,
    secrets_: SecretFiles,
    session: SessionPaths,
    source_before: CheckoutSnapshot,
    plan: Mapping[str, object],
    runner: CommandRunner,
    duration_seconds: float,
    startup_timeout_seconds: float,
    shutdown_timeout_seconds: float,
    lane: RuntimeLane = BASELINE_LANE,
    resource_soak: bool = False,
    appliance_context_env: Mapping[str, str] | None = None,
    appliance_binding: ApplianceBinding | None = None,
) -> dict[str, object]:
    lane = resolve_runtime_lane(lane)
    if resource_soak:
        _resource_soak_lane(lane.name)
    if resource_soak and (
        not math.isfinite(duration_seconds)
        or duration_seconds < RESOURCE_SOAK_MIN_DURATION_SECONDS
    ):
        raise RuntimeContainerError(
            "resource-soak acceptance requires duration_seconds >= "
            f"{RESOURCE_SOAK_MIN_DURATION_SECONDS:g}"
        )
    if not bool(plan.get("ready_for_explicit_run")):
        raise RuntimeContainerError(
            "runtime plan has blockers: " + ", ".join(plan.get("blockers") or [])
        )
    if str(plan.get("runtime_lane") or "") != lane.name:
        raise RuntimeContainerError(
            f"runtime plan lane differs from requested execution lane: "
            f"plan={plan.get('runtime_lane')!r} requested={lane.name!r}"
        )
    if appliance_binding is not None:
        require_appliance_state_current(appliance_binding)
    _validate_plan_analytics_state(
        plan,
        session,
        lane,
        appliance_binding,
    )
    artifact_lock_descriptor, artifact_lock_path = acquire_artifact_transaction_lock(
        roots.artifacts
    )
    try:
        secret_before = assert_immediate_run_preconditions(
            roots=roots,
            secrets_=secrets_,
            source_before=source_before,
            planned_artifact_readiness=(plan.get("artifact_readiness") or {}),
            runner=runner,
            lane=lane,
        )
        prepare_session_paths(roots, session, appliance_binding)
        analytics_state = prepare_analytics_state(
            session,
            lane,
            appliance_binding,
        )
        launcher_dir = session.launcher_evidence
        _write_private_json(launcher_dir / "launch-plan.json", dict(plan))
        _write_private_json(
            launcher_dir / "checkout-before.json", source_before.summary()
        )
        analytics_before = capture_analytics_state_evidence(
            session,
            lane,
            phase="before",
            action=str(analytics_state["action"]),
            appliance_binding=appliance_binding,
        )

        if appliance_binding is not None:
            require_appliance_state_current(appliance_binding)

        command = build_container_command(
            roots,
            secrets_,
            session,
            lane,
            appliance_context_env=appliance_context_env,
            appliance_binding=appliance_binding,
        )
        if appliance_binding is not None:
            require_appliance_runtime_snapshot_current(appliance_binding)
        container_name = f"noesis-ds9-runtime-{session.session_id}"
        started_at = utc_now()
        run_attempted = False
        started = False
        term_sent = False
        forced_removal = False
        removed = False
        exit_code: int | None = None
        stop_reason = "startup_failure"
        runtime_log = ""
        inspect: Mapping[str, Any] = {}
        runtime_identity: dict[str, object] = {}
        gpu_owner_confirmation: Mapping[str, object] = {}
        resource_soak_report: dict[str, object] | None = None
        resource_runtime_binding: dict[str, str] | None = None
        error: str | None = None
    except Exception:
        release_artifact_transaction_lock(artifact_lock_descriptor)
        raise
    try:
        run_attempted = True
        result = runner.run(command, timeout=60.0)
        container_id = result.stdout.strip()
        started = True
        if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
            raise RuntimeContainerError(
                f"Docker did not return a container ID: {container_id!r}"
            )
        if secret_identities(secrets_) != secret_before:
            raise RuntimeContainerError(
                "runtime secret file changed while Docker was binding it"
            )
        inspect = _json_object(
            runner.run(
                _docker_command(roots, "inspect", container_name),
                timeout=15.0,
            ).stdout,
            label="container inspection",
        )
        validate_container_inspect(
            inspect,
            roots=roots,
            secrets_=secrets_,
            session=session,
            lane=lane,
            appliance_context_env=appliance_context_env,
            appliance_binding=appliance_binding,
        )
        _wait_for_readiness(
            roots,
            runner,
            container_name,
            startup_timeout_seconds,
        )
        runtime_identity = observe_runtime_identity(
            secrets_.internal_auth,
            session_id=session.session_id,
            runtime_lane=lane.name,
        )
        _write_private_json(
            launcher_dir / RUNTIME_IDENTITY_FILENAME,
            runtime_identity,
        )
        gpu_owner_confirmation = confirm_container_gpu_ownership(inspect, runner)
        release_artifact_transaction_lock(artifact_lock_descriptor)
        artifact_lock_descriptor = None
        if resource_soak:
            resource_runtime_binding = build_resource_runtime_binding(
                plan,
                inspect,
                lane=lane,
            )
            stop_reason, resource_soak_report = run_resource_soak(
                roots,
                runner,
                container_name,
                inspect,
                launcher_dir,
                session_id=session.session_id,
                runtime_lane=lane.name,
                runtime_instance_id=str(runtime_identity["runtime_instance_id"]),
                runtime_run_id=str(runtime_identity["runtime_run_id"]),
                duration_seconds=duration_seconds,
                runtime_binding=resource_runtime_binding,
            )
        else:
            stop_reason = _wait_active_duration(
                roots,
                runner,
                container_name,
                duration_seconds,
            )
        signal_result = runner.run(
            _docker_command(roots, "kill", "--signal=TERM", container_name),
            timeout=15.0,
            check=False,
        )
        if signal_result.returncode != 0:
            raise RuntimeContainerError(
                "failed to deliver SIGTERM to the DS9 container"
            )
        term_sent = True
        wait_result = runner.run(
            _docker_command(roots, "wait", container_name),
            timeout=shutdown_timeout_seconds,
            check=False,
        )
        if wait_result.returncode != 0:
            raise RuntimeContainerError(
                f"docker wait failed: {(wait_result.stderr or wait_result.stdout).strip()}"
            )
        try:
            exit_code = int(wait_result.stdout.strip())
        except ValueError as exc:
            raise RuntimeContainerError(
                f"docker wait returned an invalid exit code: {wait_result.stdout!r}"
            ) from exc
        runtime_log, inspect = _capture_container_artifacts(
            roots,
            runner,
            container_name,
            launcher_dir,
        )
        lifecycle = validate_shutdown_log(runtime_log)
        if exit_code != 0:
            raise RuntimeContainerError(f"DS9 container exited {exit_code}; expected 0")
        if not bool(lifecycle["ok"]):
            raise RuntimeContainerError(
                "DS9 container shutdown log did not prove the ordered lifecycle contract"
            )
        removed = _remove_container(
            roots,
            runner,
            container_name,
            force=False,
        )
        if not removed:
            raise RuntimeContainerError("stopped DS9 container could not be removed")
        if resource_soak_report is not None and not bool(
            resource_soak_report.get("ok")
        ):
            error = f"{lane.name} resource-soak acceptance failed: " + ", ".join(
                str(value) for value in resource_soak_report.get("failed_checks", [])
            )
    except Exception as exc:
        error = str(exc)
        if run_attempted and not started:
            try:
                started = not _container_absent(roots, runner, container_name)
            except Exception:
                pass
        if started:
            try:
                runtime_log, inspect = _capture_container_artifacts(
                    roots,
                    runner,
                    container_name,
                    launcher_dir,
                )
            except Exception as capture_exc:
                error = f"{error}; evidence capture failed: {capture_exc}"
            if not _container_absent(roots, runner, container_name):
                forced_removal = True
                removed = _remove_container(
                    roots,
                    runner,
                    container_name,
                    force=True,
                )
        if not isinstance(exc, RuntimeContainerError):
            error = f"{type(exc).__name__}: {error}"
    finally:
        release_artifact_transaction_lock(artifact_lock_descriptor)
        artifact_lock_descriptor = None

    analytics_after: Mapping[str, object]
    try:
        analytics_after = capture_analytics_state_evidence(
            session,
            lane,
            phase="after",
            action="validated",
            appliance_binding=appliance_binding,
        )
    except Exception as analytics_exc:
        analytics_after = {
            "phase": "after",
            "action": "validation_failed",
            "error": f"{type(analytics_exc).__name__}: {analytics_exc}",
        }
        analytics_error = f"analytics state validation failed after runtime: {analytics_exc}"
        error = f"{error}; {analytics_error}" if error else analytics_error

    ports_closed = False
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if all(_port_is_free(port) for port in CANONICAL_PORTS):
            ports_closed = True
            break
        time.sleep(0.2)
    try:
        source_after = snapshot_checkout()
        source_comparison = compare_snapshots(source_before, source_after)
    except Exception as exc:
        source_after = None
        source_comparison = {
            "unchanged": False,
            "error": str(exc),
            "before": source_before.summary(),
        }
    lifecycle = validate_shutdown_log(runtime_log)
    container_absent = _container_absent(roots, runner, container_name)
    ok = bool(
        started
        and term_sent
        and exit_code == 0
        and lifecycle["ok"]
        and removed
        and not forced_removal
        and container_absent
        and ports_closed
        and source_comparison.get("unchanged")
        and error is None
    )
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "mode": "run",
        "session_id": session.session_id,
        "runtime_lane": lane.name,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "ok": ok,
        "error": error,
        "container": {
            "name": container_name,
            "started": started,
            "term_sent": term_sent,
            "exit_code": exit_code,
            "stop_reason": stop_reason,
            "forced_removal": forced_removal,
            "removed": removed,
            "absent_after": container_absent,
        },
        "ports_closed_after": ports_closed,
        "artifact_transaction_lock": {
            "path": str(artifact_lock_path),
            "held_through_readiness_and_gpu_confirmation": bool(gpu_owner_confirmation),
        },
        "gpu_owner_confirmation": dict(gpu_owner_confirmation),
        "runtime_identity": dict(runtime_identity),
        "analytics_state": {
            "before": dict(analytics_before),
            "after": dict(analytics_after),
        },
        "shutdown_lifecycle": lifecycle,
        "checkout": source_comparison,
        "evidence_directory": str(launcher_dir),
    }
    if resource_soak:
        summary["resource_soak"] = dict(resource_soak_report or {"ok": False})
    _write_private_json(launcher_dir / "summary.json", summary)
    summary["evidence_manifest_sha256"] = _seal_evidence(launcher_dir)
    if not ok:
        raise RuntimeContainerError(
            f"DS9 runtime-container gate failed; evidence preserved at {launcher_dir}: {error or summary}"
        )
    return summary


def _load_appliance_supervisor_binding(
    *,
    selector_file: Path,
    selector_sha256: str,
    env: Mapping[str, str],
) -> tuple[ApplianceBinding, RuntimeLane]:
    try:
        binding = load_appliance_binding(
            selector_file=selector_file,
            selector_sha256=selector_sha256,
            repo_root=REPO_ROOT,
            env=env,
            expected_family="ds9",
        )
    except ApplianceConfigurationError:
        raise
    except Exception as exc:
        raise ApplianceConfigurationError(
            f"DS9 appliance selector validation failed: {exc}"
        ) from exc
    selected = binding.selector.runtime
    if not isinstance(selected, DS9RuntimeSelector):
        raise ApplianceConfigurationError("appliance selector did not select DS9")
    try:
        supervisor = Path(selected.supervisor_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ApplianceConfigurationError("selected DS9 supervisor is unavailable") from exc
    if supervisor != Path(__file__).resolve() or supervisor != Path(
        selected.supervisor_path
    ):
        raise ApplianceConfigurationError(
            "running DS9 supervisor does not match the exact selected executable"
        )
    supervisor_info = supervisor.lstat()
    if (
        not stat.S_ISREG(supervisor_info.st_mode)
        or stat.S_ISLNK(supervisor_info.st_mode)
        or supervisor_info.st_uid != os.getuid()
        or supervisor_info.st_nlink != 1
        or not os.access(supervisor, os.X_OK)
    ):
        raise ApplianceConfigurationError(
            "selected DS9 supervisor must be an owner-owned single-link executable file"
        )
    exact_env = {
        "NOESIS_DS9_DOCKER_ROOT": selected.docker_root,
        "NOESIS_DS9_ARTIFACT_ROOT": selected.artifact_root,
        "NOESIS_DS9_RUNTIME_ROOT": selected.runtime_root,
    }
    for name, expected in exact_env.items():
        if str(env.get(name, "")) != expected:
            raise ApplianceConfigurationError(
                f"selected DS9 root differs from inherited {name}"
            )
    if selected.runtime_image_id != IMAGE_ID:
        raise ApplianceConfigurationError("selected DS9 runtime image ID drifted")
    if selected.build_image_id != PARENT_BUILD_IMAGE_ID:
        raise ApplianceConfigurationError("selected DS9 build image ID drifted")
    try:
        realization_sha256 = _sha256_file(
            Path(selected.artifact_root) / REALIZATION_FILENAME
        )
        ownership_sha256 = _sha256_file(REPO_ROOT / OWNERSHIP_MATRIX_RELATIVE)
    except OSError as exc:
        raise ApplianceConfigurationError(
            "selected DS9 realization or ownership matrix is unavailable"
        ) from exc
    if realization_sha256 != selected.asset_realization_sha256:
        raise ApplianceConfigurationError(
            "selected DS9 asset realization digest drifted"
        )
    if ownership_sha256 != selected.ownership_matrix_sha256:
        raise ApplianceConfigurationError(
            "selected DS9 ownership matrix digest drifted"
        )
    return binding, resolve_runtime_lane(selected.lane)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run the fail-closed canonical DS9 runtime container."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("plan", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "--session-id",
            default=None,
            help="Safe evidence/session identifier; generated by default.",
        )
        sub.add_argument(
            "--lane",
            choices=tuple(RUNTIME_LANES),
            default=BASELINE_LANE.name,
            help="Reviewed runtime lane; arbitrary config/profile input is not accepted.",
        )
        if name == "run":
            sub.add_argument(
                "--authorize-gpu-runtime",
                action="store_true",
                help="Required acknowledgement for NVIDIA + host network/IPC runtime mode.",
            )
            sub.add_argument(
                "--duration-seconds",
                type=float,
                default=30.0,
                help="Active canary duration after readiness; 0 runs until SIGINT/SIGTERM.",
            )
            sub.add_argument(
                "--resource-soak",
                action="store_true",
                help=(
                    "Run the reviewed V3DT/Wholebody cgroup-v2 and GPU resource "
                    "acceptance gate; requires at least 300 seconds."
                ),
            )
            sub.add_argument("--startup-timeout-seconds", type=float, default=240.0)
            sub.add_argument("--shutdown-timeout-seconds", type=float, default=90.0)
    for name in ("appliance-check", "appliance-run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--deployment-selector", type=Path, required=True)
        sub.add_argument("--selector-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode in {"appliance-check", "appliance-run"}:
        try:
            binding, lane = _load_appliance_supervisor_binding(
                selector_file=args.deployment_selector,
                selector_sha256=args.selector_sha256,
                env=os.environ,
            )
        except ApplianceConfigurationError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 78
        session_id = new_session_id()
        runner = CommandRunner()
        try:
            roots, secret_files, session, _docker, source, plan = preflight(
                env=os.environ,
                session_id=session_id,
                runner=runner,
                lane=lane,
                appliance_binding=binding,
            )
        except RuntimeContainerError as exc:
            print(f"[FAIL] DS9 appliance preflight rejected immutable state: {exc}", file=sys.stderr)
            return 78
        context_env = runtime_context_environment(binding)
        plan = dict(plan)
        plan["appliance_deployment"] = {
            "deployment_id": binding.deployment_id,
            "selector_sha256": binding.selector_sha256,
            "state_release_id": binding.state_release_id,
            "runtime_family": binding.runtime_family,
            "runtime_variant": binding.runtime_variant,
            "software_revision": binding.software_revision,
            "boot_id": binding.boot_id,
            "runtime_snapshot_kind": binding.selector.noesis_checkout.snapshot_kind,
            "runtime_snapshot_sha256": binding.selector.noesis_checkout.snapshot_sha256,
        }
        plan["authorized_run_command"] = redacted_command(
            build_container_command(
                roots,
                secret_files,
                session,
                lane,
                appliance_context_env=context_env,
                appliance_binding=binding,
            ),
            secret_files,
        )
        if not bool(plan.get("ready_for_explicit_run")):
            print(
                "[FAIL] DS9 appliance preflight has transient blockers: "
                + ", ".join(str(value) for value in plan.get("blockers") or []),
                file=sys.stderr,
            )
            return 1
        if args.mode == "appliance-check":
            try:
                require_appliance_runtime_snapshot_current(binding)
            except ApplianceConfigurationError as exc:
                print(f"[FAIL] {exc}", file=sys.stderr)
                return 78
            print(
                json.dumps(
                    {
                        "contract": "noesis.ds9.appliance_check",
                        "contract_version": 1,
                        "deployment_id": binding.deployment_id,
                        "selector_sha256": binding.selector_sha256,
                        "runtime_variant": binding.runtime_variant,
                        "ready": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        try:
            summary = execute(
                roots=roots,
                secrets_=secret_files,
                session=session,
                source_before=source,
                plan=plan,
                runner=runner,
                duration_seconds=0.0,
                startup_timeout_seconds=240.0,
                shutdown_timeout_seconds=90.0,
                lane=lane,
                resource_soak=False,
                appliance_context_env=context_env,
                appliance_binding=binding,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        except ApplianceConfigurationError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 78
        except RuntimeContainerError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
    try:
        lane = resolve_runtime_lane(args.lane)
    except RuntimeContainerError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    session_id = str(args.session_id or new_session_id()).strip().lower()
    if not SESSION_RE.fullmatch(session_id):
        print(
            "[FAIL] session ID must match [a-z0-9][a-z0-9-]{5,47}",
            file=sys.stderr,
        )
        return 2
    if args.mode == "run":
        if not args.authorize_gpu_runtime:
            print(
                "[FAIL] run mode requires --authorize-gpu-runtime; use plan for a no-launch audit",
                file=sys.stderr,
            )
            return 2
        if args.duration_seconds < 0:
            print("[FAIL] --duration-seconds must be >= 0", file=sys.stderr)
            return 2
        if args.resource_soak and lane.name not in RESOURCE_SOAK_GPU_MEMORY_LIMIT_MIB:
            print(
                "[FAIL] --resource-soak requires --lane v3dt, wholebody49-s, "
                "or wholebody49-x",
                file=sys.stderr,
            )
            return 2
        if args.resource_soak and (
            not math.isfinite(args.duration_seconds)
            or args.duration_seconds < RESOURCE_SOAK_MIN_DURATION_SECONDS
        ):
            print(
                "[FAIL] --resource-soak requires --duration-seconds >= "
                f"{RESOURCE_SOAK_MIN_DURATION_SECONDS:g}",
                file=sys.stderr,
            )
            return 2
        if args.startup_timeout_seconds <= 0:
            print("[FAIL] --startup-timeout-seconds must be > 0", file=sys.stderr)
            return 2
        if args.shutdown_timeout_seconds < 90:
            print(
                "[FAIL] --shutdown-timeout-seconds must be >= 90 (75s runtime watchdog plus 15s margin)",
                file=sys.stderr,
            )
            return 2
    runner = CommandRunner()
    try:
        roots, secret_files, session, _docker, source, plan = preflight(
            env=os.environ,
            session_id=session_id,
            runner=runner,
            lane=lane,
        )
        if args.mode == "plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if bool(plan["ready_for_explicit_run"]) else 3
        summary = execute(
            roots=roots,
            secrets_=secret_files,
            session=session,
            source_before=source,
            plan=plan,
            runner=runner,
            duration_seconds=float(args.duration_seconds),
            startup_timeout_seconds=float(args.startup_timeout_seconds),
            shutdown_timeout_seconds=float(args.shutdown_timeout_seconds),
            lane=lane,
            resource_soak=bool(args.resource_soak),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except RuntimeContainerError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
