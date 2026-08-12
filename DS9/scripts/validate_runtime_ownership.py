#!/usr/bin/env python3
"""Validate the declared DS8/DS9 ownership and capability boundary."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from noesis_core.strict_json import StrictJSONError, strict_json_loads


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml"
EXPECTED_MATRIX_ID = "noesis-ds8-ds9-runtime-ownership"
REALIZATION_FILENAME = "asset_realization.json"
REALIZATION_CONTRACT = "noesis.ds9.asset_realization"
RUNTIME_CONTRACT = "noesis.ds9.canonical_runtime_container"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,47}$")
CHECKSUM_LINE_RE = re.compile(
    r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9_.-]{0,127})$"
)
MAX_PRIVATE_EVIDENCE_BYTES = 64 * 1024 * 1024
_OPENAT_TEST_HOOK: Callable[[str, Path, Path], None] | None = None
EVIDENCE_TYPES = frozenset(
    {"repository_source", "asset_realization", "runtime_session"}
)
ALLOWED_MODULE_CLASSIFICATIONS = frozenset(
    {"shared_single_source", "sdk_adapter", "duplicated_pending_convergence"}
)
ALLOWED_CAPABILITY_STATUSES = frozenset(
    {"parity", "shared", "adapter_specific", "known_gap", "blocked"}
)
STRICT_BLOCKING_STATUSES = frozenset({"known_gap", "blocked"})
REPOSITORY_SOURCE_KEYS = frozenset(
    {
        "path",
        "contains_all",
        "contains_none",
        "yaml_keys",
        "yaml_equals",
        "python_import_origin",
    }
)
ASSET_REALIZATION_KEYS = frozenset(
    {
        "profiles",
        "realization_sha256",
        "base_manifest_sha256",
        "source_contracts_sha256",
        "runtime_image_id",
        "artifact_ids",
        "output_sha256",
    }
)
RUNTIME_SESSION_KEYS = ASSET_REALIZATION_KEYS | frozenset(
    {
        "session_id",
        "lane",
        "checksum_manifest",
        "checksum_sha256",
        "behavior_documents",
    }
)
EVIDENCE_TIER = {
    "repository_source": 1,
    "asset_realization": 2,
    "runtime_session": 3,
}
MAX_LIVE_EVIDENCE_AGE = timedelta(hours=24)
MAX_PLAN_TO_SUMMARY_START = timedelta(seconds=120)
MAX_SUMMARY_TO_INSPECT_START = timedelta(seconds=10)
MAX_INSPECT_TO_SUMMARY_FINISH = timedelta(seconds=10)
EXPECTED_SECURITY_OPTIONS = ["no-new-privileges", "label=disable"]
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
EXPECTED_READONLY_PATHS = [
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger",
]
EXPECTED_MASKED_PATHS = [
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
]
RUNTIME_LANES: Mapping[str, Mapping[str, Any]] = {
    "baseline": {
        "profiles": ["canonical"],
        "pipeline": "DS9/config/infer.yaml",
        "cameras": "config/cameras.yaml",
        "pgie_profile": "yolo26",
        "model_size": "m",
        "tracking_mode": "baseline",
        "minimum_duration_seconds": 30.0,
    },
    "v3dt": {
        "profiles": ["v3dt"],
        "pipeline": "DS9/config/infer_v3dt.yaml",
        "cameras": "DS9/config/cameras_v3dt.yaml",
        "pgie_profile": "yolo26_seg",
        "model_size": "s",
        "tracking_mode": "v3dt",
        "minimum_duration_seconds": 30.0,
    },
    "wholebody49-s": {
        "profiles": [
            "runtime_common",
            "artifact:parser.wholebody49",
            "artifact:parser.yolo_detect",
            "artifact:engine.depth_tracking_dav2",
            "artifact:engine.yolo26_detect_m",
            "artifact:engine.wholebody49_s_masks",
        ],
        "pipeline": "DS9/config/infer.yaml",
        "cameras": "config/cameras.yaml",
        "pgie_profile": "wholebody49",
        "model_size": "s",
        "tracking_mode": "baseline",
        "minimum_duration_seconds": 30.0,
    },
    "wholebody49-x": {
        "profiles": [
            "runtime_common",
            "artifact:parser.wholebody49",
            "artifact:parser.yolo_detect",
            "artifact:engine.depth_tracking_dav2",
            "artifact:engine.yolo26_detect_m",
            "artifact:engine.wholebody49_x_boxes",
        ],
        "pipeline": "DS9/config/infer.yaml",
        "cameras": "config/cameras.yaml",
        "pgie_profile": "wholebody49",
        "model_size": "x",
        "tracking_mode": "baseline",
        "minimum_duration_seconds": 30.0,
    },
}
CANONICAL_PORTS = {"websocket": 6008, "rest": 8080, "rtsp": 8554}
CANONICAL_ENDPOINTS = {
    "websocket": "ws://127.0.0.1:6008",
    "rest": "http://127.0.0.1:8080",
    "rtsp": "rtsp://127.0.0.1:8554/mosaic",
}
RUNTIME_IDENTITY_CONTRACT = "noesis.ds9.supervisor_runtime_identity"
RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
RESOURCE_SOAK_SAMPLES_FILENAME = "runtime-resource-soak-samples.json"
EXPECTED_V3DT_CAMERAS = frozenset({"living-room", "kitchen", "family-room"})
EXPECTED_V3DT_MIN_BBOX3D_COVERAGE = 0.95
EXPECTED_FLOORPLAN_MAX_SNAPSHOT_AGE_S = 120.0
EXPECTED_FLOORPLAN_CONTRACT_VERSION = 10
GST_CLOCK_TIME_NONE = (1 << 64) - 1
MAX_CAPTURE_RGB_DIMENSION = 16_384
MAX_CAPTURE_RGB_FRAME_BYTES = 64 * 1024 * 1024
REID_ARTIFACT_ID = "engine.reid_swin_tiny"
REID_ENGINE_PATH = "DS9/models/engines/reid_swin_tiny_aicity156_dyn_b16_fp16.engine"
REID_MODEL_LAYER = "fc_pred"
REID_EMBEDDING_DIMENSION = 256
IDENTITY_SOURCE_FILENAMES = {
    "reid_open_set_occupied_v1": "identity-open-set-occupied-source.json",
    "v3dt_identity_gate_v1": "v3dt-identity-open-set-occupied-source.json",
}
SEMANTIC_IDENTITY_SNAPSHOT_FILENAME = "semantic-identity-evidence.jsonl"
WHOLEBODY_SOURCE_TRANSCRIPT_FILENAME = "wholebody49-occupied-scene-source.json"
WHOLEBODY_MEDIA_SOURCE_FILENAME = "wholebody49-media-decode-source.json"
FLOORPLAN_SOURCE_TRANSCRIPT_FILENAME = "mapanything-depth-quality-source.json"
V3DT_WORLD_SOURCE_TRANSCRIPT_FILENAME = "v3dt-world-contract-source.json"
WHOLEBODY_RENDER_EVIDENCE_POLICY = (
    "pair_with_the_same_lane_rtsp_decode_and_supervisor_runtime_log; "
    "this gate proves parser output consumption, not subjective overlay aesthetics"
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
    "double free",
    "Orderly pipeline EOS request failed",
    "Pipeline quiescence was not proven",
    "Native pipeline teardown failed",
    "wait timed out",
    "wait() timed out",
    "GStreamer teardown failed",
    "invalid pointer",
    "Aborted (core dumped)",
)
MANDATORY_RUNTIME_JSON = frozenset(
    {
        "launch-plan.json",
        "checkout-before.json",
        "summary.json",
        "container-inspect.json",
        "runtime-identity.json",
    }
)
CAPABILITY_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "cli.core_runtime_options": {"surface": "cli", "minimum_tier": "repository_source"},
    "cli.pgie_profile_underscore_alias": {"surface": "cli", "minimum_tier": "repository_source"},
    "config.canonical_model_roles": {"surface": "config", "minimum_tier": "repository_source"},
    "config.local_bev_frame": {"surface": "config", "minimum_tier": "repository_source"},
    "depth.dewarper_validity_masks": {"surface": "config", "minimum_tier": "repository_source"},
    "api.depth_refresh": {"surface": "api", "minimum_tier": "repository_source"},
    "api.internal_gateway_auth": {"surface": "security", "minimum_tier": "repository_source"},
    "api.analytics_roi": {"surface": "api", "minimum_tier": "repository_source"},
    "api.reid_aliases": {"surface": "api", "minimum_tier": "repository_source"},
    "api.household_identity": {"surface": "api", "minimum_tier": "repository_source"},
    "identity.camera_topology_validation": {"surface": "identity", "minimum_tier": "repository_source"},
    "api.virtual_twin": {"surface": "api", "minimum_tier": "repository_source"},
    "telemetry.core_message_types": {"surface": "telemetry", "minimum_tier": "repository_source"},
    "telemetry.canonical_world_snapshot": {"surface": "telemetry", "minimum_tier": "repository_source"},
    "telemetry.public_frame_time_identity": {"surface": "telemetry", "minimum_tier": "repository_source"},
    "health.canonical_capability_progress": {"surface": "api", "minimum_tier": "repository_source"},
    "api.scene_releases": {"surface": "api", "minimum_tier": "repository_source"},
    "telemetry.bev_contract": {"surface": "telemetry", "minimum_tier": "repository_source"},
    "telemetry.person_ground_state": {"surface": "telemetry", "minimum_tier": "repository_source"},
    "model.wholebody49_profile": {
        "surface": "cli",
        "minimum_tier": "runtime_session",
        "minimum_duration_seconds": 300.0,
        "runtime_requirements": {
            "wholebody49-s": [
                "runtime_resource_soak_v2",
                "wholebody49_media_decode_v1",
                "wholebody49_occupied_s_v2",
            ],
            "wholebody49-x": [
                "runtime_resource_soak_v2",
                "wholebody49_media_decode_v1",
                "wholebody49_occupied_x_v2",
            ],
        },
    },
    "model.reid_profile": {
        "surface": "config",
        "minimum_tier": "runtime_session",
        "runtime_requirements": {
            "baseline": ["reid_open_set_occupied_v1", "semantic_gate_v3"]
        },
    },
    "model.mapanything_validated_fp32_builder": {
        "surface": "artifacts",
        "minimum_tier": "runtime_session",
        "runtime_requirements": {"baseline": ["mapanything_depth_quality_v4"]},
    },
    "media.websocket_webrtc_server": {"surface": "media", "minimum_tier": "repository_source"},
    "media.mosaic_webrtc_readiness": {"surface": "media", "minimum_tier": "repository_source"},
    "metadata.ds9_pose_path": {"surface": "metadata", "minimum_tier": "repository_source"},
    "tracking.v3dt": {
        "surface": "tracking",
        "minimum_tier": "runtime_session",
        "minimum_duration_seconds": 300.0,
        "runtime_requirements": {
            "v3dt": [
                "v3dt_world_gate_v2",
                "semantic_gate_v3",
                "v3dt_identity_gate_v1",
                "runtime_resource_soak_v2",
            ]
        },
    },
    "artifacts.canonical_graph": {
        "surface": "artifacts",
        "minimum_tier": "asset_realization",
        "required_profiles": ["canonical"],
    },
}


def _equals(pointer: str, value: Any) -> Mapping[str, Any]:
    return {"pointer": pointer, "op": "equals", "value": value, "type": type(value)}


def _minimum(pointer: str, value: int | float, expected_type: type) -> Mapping[str, Any]:
    return {"pointer": pointer, "op": "minimum", "value": value, "type": expected_type}


BEHAVIOR_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "wholebody49_occupied_s_v2": {
        "filename": "wholebody49-occupied-scene.json",
        "contract": "noesis.ds9.wholebody49_occupied_scene_gate",
        "contract_version": 2,
        "lanes": {"wholebody49-s"},
        "assertions": (
            _equals("/ok", True),
            _equals("/status", "pass"),
            _equals("/scene_status", "occupied"),
            _equals("/mode", "masks"),
            _equals("/checks/tracking_messages", True),
            _equals("/checks/occupied_person_tracks", True),
            _equals("/checks/pipeline_ready", True),
            _equals("/checks/pipeline_errors_absent", True),
            _equals("/checks/core_cpu_copy_violations_absent", True),
            _equals("/checks/source_inventory_exact", True),
            _equals("/checks/source_frames_advancing", True),
            _equals("/checks/source_frame_rate_sufficient", True),
            _equals("/checks/tracking_cadence_within_limit", True),
            _equals("/checks/mask_path_active", True),
        ),
    },
    "wholebody49_occupied_x_v2": {
        "filename": "wholebody49-occupied-scene.json",
        "contract": "noesis.ds9.wholebody49_occupied_scene_gate",
        "contract_version": 2,
        "lanes": {"wholebody49-x"},
        "assertions": (
            _equals("/ok", True),
            _equals("/status", "pass"),
            _equals("/scene_status", "occupied"),
            _equals("/mode", "boxes"),
            _equals("/checks/tracking_messages", True),
            _equals("/checks/occupied_person_tracks", True),
            _equals("/checks/pipeline_ready", True),
            _equals("/checks/pipeline_errors_absent", True),
            _equals("/checks/core_cpu_copy_violations_absent", True),
            _equals("/checks/source_inventory_exact", True),
            _equals("/checks/source_frames_advancing", True),
            _equals("/checks/source_frame_rate_sufficient", True),
            _equals("/checks/tracking_cadence_within_limit", True),
            _equals("/checks/bbox_path_active", True),
        ),
    },
    "wholebody49_media_decode_v1": {
        "filename": "wholebody49-media-decode.json",
        "contract": "noesis.ds9.wholebody49_media_decode_gate",
        "contract_version": 1,
        "lanes": {"wholebody49-s", "wholebody49-x"},
        "assertions": (
            _equals("/ok", True),
            _equals("/status", "pass"),
            _equals("/checks/connection_states_accepted", True),
            _equals("/checks/rtp_threshold_met", True),
            _equals("/checks/decoded_frame_threshold_met", True),
            _equals("/checks/decoded_h264_pad_observed", True),
            _equals("/checks/direct_rtsp_decode_threshold_met", True),
            _minimum("/metrics/decoded_frames", 1, int),
            _minimum("/metrics/rtsp_decoded_frames", 1, int),
        ),
    },
    "reid_open_set_occupied_v1": {
        "filename": "identity-open-set-occupied.json",
        "contract": "noesis.ds9.identity-shadow-live-gate",
        "lanes": {"baseline"},
        "assertions": (
            _equals("/ok", True),
            _equals("/claims/runtime_shadow_health/status", "pass"),
            _equals("/claims/tracker_subject_continuity/status", "pass"),
            _equals("/claims/open_set_non_force/status", "observed"),
            _minimum("/counts/person_rows", 1, int),
            _minimum("/counts/fresh_embedding_rows", 1, int),
            _minimum("/counts/open_set_non_force_rows", 1, int),
        ),
    },
    "mapanything_depth_quality_v4": {
        "filename": "mapanything-depth-quality.json",
        "contract": "noesis.ds9.floorplan-live-gate",
        "schema_version": 4,
        "contract_version": 4,
        "lanes": {"baseline"},
        "assertions": (
            _equals("/ok", True),
            _equals("/all_configured_camera_floorplans_validated", True),
            _equals("/cache_only_zero_mutation", True),
            _equals("/bev_renderer_ready", True),
            _equals("/bev_failed_camera_count", 0),
            _equals("/all_configured_cameras_bev_ready", True),
            _minimum("/configured_camera_count", 1, int),
            _minimum("/validated_camera_count", 1, int),
        ),
    },
    "v3dt_world_gate_v2": {
        "filename": "v3dt-world-contract.json",
        "contract": "noesis.ds9.v3dt_global_world_live_gate",
        "schema_version": 2,
        "contract_version": 2,
        "lanes": {"v3dt"},
        "assertions": (
            _equals("/ok", True),
            _equals("/status", "pass"),
            _equals("/occupied_scene_observed", True),
            _equals("/locked_world_contract/world_frame", "backend_world_m"),
            _equals("/locked_world_contract/caminfo_world_axes", "xzy"),
            _equals("/locked_world_contract/mv3dt_overlap_promotion", "not_claimed"),
            _equals("/locked_world_contract/time_sync_promotion", "not_claimed"),
            _minimum("/tracks_seen", 1, int),
        ),
    },
    "semantic_gate_v3": {
        "filename": "semantic-observation.json",
        "contract": "noesis.ds9.semantic-observation-live-gate",
        "schema_version": 3,
        "contract_version": 3,
        "lanes": {"baseline", "v3dt"},
        "assertions": (
            _equals("/ok", True),
            _equals("/status", "pass"),
            _equals("/checks/occupied_scene_observed", True),
            _equals("/checks/persisted_embedding_anchor", True),
            _equals("/checks/pose_component_observed", True),
            _equals("/checks/usable_depth_component_observed", True),
            _equals("/checks/backend_world_m_component_observed", True),
            _equals("/checks/bounded_identity_cohort_accepted", True),
            _equals("/checks/capture_and_observation_bounds_enforced", True),
            _equals("/checks/acquisition_window_enforced", True),
            _equals("/checks/publication_clock_within_acquisition_window", True),
            _equals("/checks/artifact_fingerprint_continuity_enforced", True),
            _equals(
                "/checks/tracker_continuity_within_acquisition_window", True
            ),
            _equals(
                "/checks/tracking_publication_contiguous_within_window_unanchored_origin",
                True,
            ),
            _equals(
                "/checks/tracker_lifecycle_contiguous_within_window_unanchored_origin",
                True,
            ),
            _equals(
                "/checks/tombstone_last_published_presence_exact_after_unanchored_origin",
                True,
            ),
            _equals("/counts/acquisition_window_violations", 0),
            _equals("/counts/publication_clock_violations", 0),
            _minimum("/counts/accepted_semantic_cohorts", 1, int),
        ),
    },
    "v3dt_identity_gate_v1": {
        "filename": "v3dt-identity-open-set-occupied.json",
        "contract": "noesis.ds9.identity-shadow-live-gate",
        "lanes": {"v3dt"},
        "assertions": (
            _equals("/ok", True),
            _equals("/claims/runtime_shadow_health/status", "pass"),
            _equals("/claims/tracker_subject_continuity/status", "pass"),
            _equals("/claims/open_set_non_force/status", "observed"),
            _minimum("/counts/person_rows", 1, int),
            _minimum("/counts/fresh_embedding_rows", 1, int),
        ),
    },
    "runtime_resource_soak_v2": {
        "filename": "runtime-resource-soak-report.json",
        "contract": "noesis.ds9.runtime_resource_soak",
        "schema_version": 2,
        "contract_version": 2,
        "lanes": {"v3dt", "wholebody49-s", "wholebody49-x"},
        "assertions": (
            _equals("/ok", True),
            _equals("/status", "pass"),
            _equals("/checks/requested_duration_sufficient", True),
            _equals("/checks/observed_duration_sufficient", True),
            _equals("/checks/oom_increment_zero", True),
            _equals("/checks/oom_kill_increment_zero", True),
            _equals("/checks/gpu_process_memory_below_limit", True),
            _equals("/checks/pids_below_limit", True),
            _equals("/checks/runtime_binding_complete", True),
        ),
    },
}


class _OwnershipYamlLoader(yaml.SafeLoader):
    """Unique-key loader for the tracked ownership authority."""


def _construct_ownership_mapping(
    loader: _OwnershipYamlLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    keys: set[Any] = set()
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("ownership YAML merge keys are forbidden")
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in keys
        except TypeError as exc:
            raise ValueError("ownership YAML mapping key is not hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate ownership YAML key: {key!r}")
        keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_OwnershipYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_ownership_mapping,
)


class _SourceYamlLoader(yaml.SafeLoader):
    """Reject duplicate explicit keys while retaining YAML merge semantics."""


def _construct_source_mapping(
    loader: _SourceYamlLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    explicit_keys: set[Any] = set()
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in explicit_keys
        except TypeError as exc:
            raise ValueError("repository source YAML key is not hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate repository source YAML key: {key!r}")
        explicit_keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_SourceYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_source_mapping,
)


def _parse_source_yaml(raw: str, label: str) -> Mapping[str, Any]:
    try:
        payload = yaml.load(raw, Loader=_SourceYamlLoader) or {}
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"{label} is invalid unique-key YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be a mapping")
    return payload


def _parse_ownership_yaml(raw: str, label: str) -> Mapping[str, Any]:
    try:
        tokens = tuple(yaml.scan(raw))
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} is invalid YAML: {exc}") from exc
    if any(isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)) for token in tokens):
        raise ValueError(f"{label} YAML aliases and anchors are forbidden")
    try:
        payload = yaml.load(raw, Loader=_OwnershipYamlLoader) or {}
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"{label} is invalid unique-key YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} document root must be a mapping")
    return payload


def _load_yaml_with_raw(path: Path) -> tuple[Mapping[str, Any], bytes]:
    absolute = path.expanduser().absolute()
    if len(absolute.parts) < 2:
        raise ValueError("ownership matrix path must name an absolute regular file")
    raw = _anchored_file_content(
        Path("/"),
        Path(*absolute.parts[1:]),
        f"ownership matrix {absolute}",
        directory_policy="authority",
        file_policy="authority",
    )
    if not isinstance(raw, bytes):
        raise ValueError("ownership matrix read returned an invalid type")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("ownership matrix must be UTF-8") from exc
    return _parse_ownership_yaml(text, f"ownership matrix {absolute}"), raw


def _load_yaml(path: Path) -> Mapping[str, Any]:
    return _load_yaml_with_raw(path)[0]


def _repo_path(raw: Any) -> Path:
    text = str(raw or "").strip()
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"path must be non-empty and repository-relative: {raw!r}")
    return REPO_ROOT / candidate


def _get_dotted(payload: Any, dotted: str) -> tuple[bool, Any]:
    current = payload
    for token in str(dotted).split("."):
        if not isinstance(current, Mapping) or token not in current:
            return False, None
        current = current[token]
    return True, current


def _validate_evidence(label: str, evidence: Mapping[str, Any], errors: list[str]) -> None:
    try:
        path = _repo_path(evidence.get("path"))
    except ValueError as exc:
        errors.append(f"{label}: {exc}")
        return
    try:
        raw = _read_authority_bytes(path, f"repository source {label}")
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"{label}: unable to read repository evidence: {exc}")
        return
    payload: Any = None

    for needle in evidence.get("contains_all", []) or []:
        if str(needle) not in text:
            errors.append(f"{label}: {path.relative_to(REPO_ROOT)} is missing required text {needle!r}")
    for needle in evidence.get("contains_none", []) or []:
        if str(needle) in text:
            errors.append(f"{label}: {path.relative_to(REPO_ROOT)} unexpectedly contains {needle!r}")

    yaml_keys = evidence.get("yaml_keys", []) or []
    yaml_equals = evidence.get("yaml_equals", {}) or {}
    if yaml_keys or yaml_equals:
        try:
            payload = _parse_source_yaml(text, f"repository source {label}")
        except Exception as exc:
            errors.append(f"{label}: unable to parse YAML evidence {path.relative_to(REPO_ROOT)}: {exc}")
            return
        for dotted in yaml_keys:
            present, _ = _get_dotted(payload, str(dotted))
            if not present:
                errors.append(f"{label}: {path.relative_to(REPO_ROOT)} is missing YAML key {dotted}")
        if not isinstance(yaml_equals, Mapping):
            errors.append(f"{label}: yaml_equals must be a mapping")
        else:
            for dotted, expected in yaml_equals.items():
                present, actual = _get_dotted(payload, str(dotted))
                if not present:
                    errors.append(f"{label}: {path.relative_to(REPO_ROOT)} is missing YAML key {dotted}")
                elif actual != expected:
                    errors.append(
                        f"{label}: {path.relative_to(REPO_ROOT)} YAML {dotted}={actual!r}, expected {expected!r}"
                    )

    import_origin = evidence.get("python_import_origin")
    if import_origin is not None:
        _validate_python_import_origin(
            label,
            import_origin,
            evidence_path=path,
            errors=errors,
        )


def _validate_python_import_origin(
    label: str,
    spec: Any,
    *,
    evidence_path: Path,
    errors: list[str],
) -> None:
    if not isinstance(spec, Mapping):
        errors.append(f"{label}: python_import_origin must be a mapping")
        return
    expected_keys = {"module", "expected_path", "prepend_paths"}
    if set(spec) != expected_keys:
        errors.append(
            f"{label}: python_import_origin keys must be {sorted(expected_keys)}"
        )
        return
    module = str(spec.get("module") or "").strip()
    if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) is None:
        errors.append(f"{label}: python_import_origin.module is invalid")
        return
    try:
        expected_path = _repo_path(spec.get("expected_path")).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(f"{label}: invalid python import expected_path: {exc}")
        return
    try:
        if expected_path != evidence_path.resolve(strict=True):
            errors.append(
                f"{label}: python import expected_path must equal repository evidence path"
            )
            return
    except (OSError, RuntimeError) as exc:
        errors.append(f"{label}: unable to resolve repository evidence path: {exc}")
        return
    raw_prepend = spec.get("prepend_paths")
    if not isinstance(raw_prepend, list) or not raw_prepend:
        errors.append(f"{label}: python_import_origin.prepend_paths must be non-empty")
        return
    prepend: list[str] = []
    for index, raw in enumerate(raw_prepend):
        try:
            candidate = _repo_path(raw).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(
                f"{label}: invalid python import prepend_paths[{index}]: {exc}"
            )
            return
        if not candidate.is_dir():
            errors.append(
                f"{label}: python import prepend_paths[{index}] is not a directory"
            )
            return
        prepend.append(str(candidate))

    script = r'''
import importlib
import json
import sys
from pathlib import Path

module = sys.argv[1]
roots = json.loads(sys.argv[2])
sys.path = roots + [
    value
    for value in sys.path
    if value and str(Path(value).resolve()) not in set(roots)
]
loaded = importlib.import_module(module)
origin = getattr(loaded, "__file__", None)
if not origin:
    raise RuntimeError(f"module has no file origin: {module}")
print(json.dumps({"origin": str(Path(origin).resolve())}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONNOUSERSITE", None)
    try:
        result = subprocess.run(
            [sys.executable, "-P", "-c", script, module, json.dumps(prepend)],
            cwd="/",
            env=environment,
            text=True,
            capture_output=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"{label}: python import origin check failed: {exc}")
        return
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        errors.append(
            f"{label}: python import origin check exited {result.returncode}: "
            f"{detail[-1] if detail else 'no error detail'}"
        )
        return
    try:
        output = strict_json_loads(
            result.stdout.strip().splitlines()[-1],
            label="python import origin output",
        )
        observed_path = Path(str(output["origin"])).resolve(strict=True)
    except (IndexError, KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: python import origin output is invalid: {exc}")
        return
    if observed_path != expected_path:
        try:
            observed_display = observed_path.relative_to(REPO_ROOT)
        except ValueError:
            observed_display = observed_path
        errors.append(
            f"{label}: {module} resolved to {observed_display}, expected "
            f"{expected_path.relative_to(REPO_ROOT)}"
        )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_json(raw: bytes, label: str) -> Any:
    try:
        return strict_json_loads(raw, label=label)
    except StrictJSONError as exc:
        raise ValueError(f"{label} is not strict JSON: {exc.reason}") from exc


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _strict_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):  # noqa: E721
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _external_root(
    raw: str | Path | None,
    label: str,
    *,
    accepted_modes: frozenset[int],
) -> Path:
    text = str(raw or "").strip()
    candidate_raw = Path(text).expanduser()
    if not text or not candidate_raw.is_absolute():
        raise ValueError(f"{label} must be supplied as an explicit absolute path")
    candidate = Path(os.path.abspath(candidate_raw))
    for component in reversed((candidate, *candidate.parents)):
        try:
            component_info = component.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"{label} ancestor is missing: {component}") from exc
        if stat.S_ISLNK(component_info.st_mode):
            raise ValueError(f"{label} contains a symlink ancestor: {component}")
    try:
        info = candidate.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {candidate}") from exc
    mode = stat.S_IMODE(info.st_mode)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or mode not in accepted_modes
    ):
        rendered = ", ".join(f"{value:04o}" for value in sorted(accepted_modes))
        raise ValueError(
            f"{label} must be an owned non-symlink directory with mode in {{{rendered}}}"
        )
    if candidate in {Path("/"), REPO_ROOT.resolve(), (REPO_ROOT / "DS9").resolve()}:
        raise ValueError(f"{label} is an unsafe ownership root: {candidate}")
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"{label} must be outside the repository checkout")
    return candidate


def _private_relative_path(raw: Any, label: str) -> Path:
    text = str(raw or "").strip()
    relative = Path(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or any(token in text for token in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"{label} must be a safe non-empty relative path")
    return relative


def _private_path(root: Path, raw: Any, label: str) -> Path:
    relative = _private_relative_path(raw, label)
    candidate = root
    for index, part in enumerate(relative.parts):
        candidate /= part
        try:
            info = candidate.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"{label} is missing: {candidate}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} contains a symlink: {candidate}")
        if index < len(relative.parts) - 1:
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise ValueError(
                    f"{label} parent must be an owned mode-0700 directory: {candidate}"
                )
    return candidate


def _validate_anchored_directory(info: os.stat_result, policy: str, label: str) -> None:
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} path component is not a directory")
    if policy == "private" and (info.st_uid != os.getuid() or mode != 0o700):
        raise ValueError(f"{label} parent must be an owned mode-0700 directory")
    if policy == "artifact" and (
        info.st_uid != os.getuid() or mode & 0o022
    ):
        raise ValueError(f"{label} artifact parent is writable or not owned")


def _validate_anchored_file(
    info: os.stat_result,
    policy: str,
    label: str,
    *,
    allow_empty: bool,
    max_bytes: int | None,
) -> None:
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or (not allow_empty and info.st_size <= 0)
        or (max_bytes is not None and info.st_size > max_bytes)
    ):
        raise ValueError(f"{label} is not an owned single-link bounded regular file")
    if policy == "private" and mode != 0o600:
        raise ValueError(f"{label} must have mode 0600")
    if policy == "artifact" and mode & 0o022:
        raise ValueError(f"{label} artifact file is writable by group/other")


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _anchored_file_content(
    root: Path,
    relative: Path,
    label: str,
    *,
    directory_policy: str,
    file_policy: str,
    allow_empty: bool = False,
    max_bytes: int | None = MAX_PRIVATE_EVIDENCE_BYTES,
    hash_only: bool = False,
) -> bytes | str:
    relative = _private_relative_path(relative, label)
    if len(relative.parts) < 1:
        raise ValueError(f"{label} relative path is empty")
    root_before = root.lstat()
    root_fd = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptors = [root_fd]
    directory_records: list[tuple[int, str, int, os.stat_result]] = []
    try:
        root_open = os.fstat(root_fd)
        if not _same_inode(root_before, root_open):
            raise ValueError(f"{label} root changed while it was opened")
        current_fd = root_fd
        for part in relative.parts[:-1]:
            before = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            child_fd = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=current_fd,
            )
            descriptors.append(child_fd)
            opened = os.fstat(child_fd)
            if not _same_inode(before, opened) or stat.S_ISLNK(before.st_mode):
                raise ValueError(f"{label} directory changed while it was opened")
            _validate_anchored_directory(opened, directory_policy, label)
            directory_records.append((current_fd, part, child_fd, opened))
            current_fd = child_fd
        filename = relative.parts[-1]
        before_file = os.stat(filename, dir_fd=current_fd, follow_symlinks=False)
        file_fd = os.open(
            filename,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current_fd,
        )
        descriptors.append(file_fd)
        opened_file = os.fstat(file_fd)
        if not _same_inode(before_file, opened_file) or stat.S_ISLNK(before_file.st_mode):
            raise ValueError(f"{label} changed while it was opened")
        _validate_anchored_file(
            opened_file,
            file_policy,
            label,
            allow_empty=allow_empty,
            max_bytes=max_bytes,
        )
        if _OPENAT_TEST_HOOK is not None:
            _OPENAT_TEST_HOOK(label, root, relative)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        read_size = 0
        while True:
            block = os.read(file_fd, 4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if not hash_only:
                chunks.append(block)
            read_size += len(block)
        after_file = os.fstat(file_fd)
        if read_size != opened_file.st_size or (
            after_file.st_dev,
            after_file.st_ino,
            after_file.st_size,
            after_file.st_mtime_ns,
            after_file.st_ctime_ns,
        ) != (
            opened_file.st_dev,
            opened_file.st_ino,
            opened_file.st_size,
            opened_file.st_mtime_ns,
            opened_file.st_ctime_ns,
        ):
            raise ValueError(f"{label} changed while it was read")
        current_name = os.stat(filename, dir_fd=current_fd, follow_symlinks=False)
        if not _same_inode(current_name, opened_file):
            raise ValueError(f"{label} name was replaced during validation")
        for parent_fd, name, child_fd, opened in reversed(directory_records):
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not _same_inode(named, opened) or not _same_inode(os.fstat(child_fd), opened):
                raise ValueError(f"{label} parent was replaced during validation")
        root_after = root.lstat()
        if not _same_inode(root_after, root_open) or not _same_inode(os.fstat(root_fd), root_open):
            raise ValueError(f"{label} root was replaced during validation")
        return digest.hexdigest() if hash_only else b"".join(chunks)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _anchored_directory_entries(
    root: Path,
    relative: Path,
    label: str,
    *,
    directory_policy: str,
) -> dict[str, os.stat_result]:
    relative = _private_relative_path(relative, label)
    root_before = root.lstat()
    root_fd = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptors = [root_fd]
    records: list[tuple[int, str, int, os.stat_result]] = []
    try:
        root_open = os.fstat(root_fd)
        if not _same_inode(root_before, root_open):
            raise ValueError(f"{label} root changed while opening")
        current_fd = root_fd
        for part in relative.parts:
            before = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            child_fd = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=current_fd,
            )
            descriptors.append(child_fd)
            opened = os.fstat(child_fd)
            if not _same_inode(before, opened) or stat.S_ISLNK(before.st_mode):
                raise ValueError(f"{label} directory changed while opening")
            _validate_anchored_directory(opened, directory_policy, label)
            records.append((current_fd, part, child_fd, opened))
            current_fd = child_fd
        if _OPENAT_TEST_HOOK is not None:
            _OPENAT_TEST_HOOK(label, root, relative)
        entries = {
            name: os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            for name in os.listdir(current_fd)
        }
        for parent_fd, name, child_fd, opened in reversed(records):
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not _same_inode(named, opened) or not _same_inode(os.fstat(child_fd), opened):
                raise ValueError(f"{label} directory was replaced during validation")
        root_after = root.lstat()
        if not _same_inode(root_after, root_open) or not _same_inode(os.fstat(root_fd), root_open):
            raise ValueError(f"{label} root was replaced during validation")
        return entries
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_authority_bytes(path: Path, label: str) -> bytes:
    try:
        relative = path.absolute().relative_to(REPO_ROOT.absolute())
    except ValueError as exc:
        raise ValueError(f"{label} is outside the repository authority root") from exc
    raw = _anchored_file_content(
        REPO_ROOT,
        relative,
        label,
        directory_policy="authority",
        file_policy="authority",
    )
    if not isinstance(raw, bytes):  # pragma: no cover - defensive type contract
        raise ValueError(f"{label} read returned an invalid type")
    return raw


def _canonical_source_id(value: Any, label: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    if isinstance(value, int):
        source_id = value
    elif isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]{0,5})", value):
        source_id = int(value)
    else:
        raise ValueError(f"{label} must be a canonical non-negative decimal source ID")
    if source_id < 0 or source_id > 999999:
        raise ValueError(f"{label} is outside the supported source-ID range")
    return str(source_id)


def _reviewed_lane_source_ids(lane_name: str) -> tuple[str, ...]:
    lane = RUNTIME_LANES.get(lane_name)
    if not isinstance(lane, Mapping):
        raise ValueError(f"unsupported runtime lane source inventory: {lane_name!r}")
    pipeline = _parse_source_yaml(
        _read_authority_bytes(
            REPO_ROOT / str(lane["pipeline"]),
            f"{lane_name} source-inventory pipeline authority",
        ).decode("utf-8"),
        f"{lane_name} source-inventory pipeline authority",
    )
    cameras = _parse_source_yaml(
        _read_authority_bytes(
            REPO_ROOT / str(lane["cameras"]),
            f"{lane_name} source-inventory camera authority",
        ).decode("utf-8"),
        f"{lane_name} source-inventory camera authority",
    )
    raw_sources = pipeline.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{lane_name} pipeline sources must be a non-empty list")
    pipeline_ids = [
        _canonical_source_id(
            source.get("source-id", index),
            f"{lane_name} pipeline source {index}",
        )
        for index, source in enumerate(raw_sources)
        if isinstance(source, Mapping)
    ]
    if len(pipeline_ids) != len(raw_sources) or len(pipeline_ids) != len(set(pipeline_ids)):
        raise ValueError(f"{lane_name} pipeline source inventory is malformed or duplicated")
    raw_cameras = cameras.get("cameras")
    if not isinstance(raw_cameras, Mapping) or not raw_cameras:
        raise ValueError(f"{lane_name} camera source inventory is missing")
    camera_ids: list[str] = []
    for key, raw_camera in raw_cameras.items():
        if not isinstance(raw_camera, Mapping):
            raise ValueError(f"{lane_name} camera {key!r} must be a mapping")
        overrides = [
            name for name in ("source_id", "source", "sensor_id") if name in raw_camera
        ]
        if len(overrides) > 1:
            raise ValueError(f"{lane_name} camera {key!r} has ambiguous source IDs")
        camera_ids.append(
            _canonical_source_id(
                raw_camera[overrides[0]] if overrides else key,
                f"{lane_name} camera {key!r}",
            )
        )
    if len(camera_ids) != len(set(camera_ids)) or set(camera_ids) != set(pipeline_ids):
        raise ValueError(
            f"{lane_name} pipeline/camera source inventory differs: "
            f"pipeline={sorted(pipeline_ids, key=int)} cameras={sorted(camera_ids, key=int)}"
        )
    return tuple(sorted(pipeline_ids, key=int))


def _require_exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unexpected = sorted(set(payload) - expected)
        raise ValueError(
            f"{label} keys differ from the typed contract; "
            f"missing={missing} unexpected={unexpected}"
        )


def _sorted_unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized) or normalized != sorted(set(normalized)):
        raise ValueError(f"{label} must be sorted, unique, non-empty strings")
    return normalized


def _profiles(value: Any, label: str) -> list[str]:
    profiles = _sorted_unique_strings(value, label)
    for profile in profiles:
        if (
            len(profile) > 128
            or any(token in profile for token in ("\x00", "\n", "\r"))
        ):
            raise ValueError(f"{label} contains an unsafe profile name")
    return profiles


def _selected_engine_records(
    manifest: Mapping[str, Any], profiles: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    policy = manifest.get("policy")
    inheritance_raw = policy.get("profile_inheritance") if isinstance(policy, Mapping) else {}
    if not isinstance(inheritance_raw, Mapping):
        raise ValueError("asset manifest profile inheritance must be a mapping")
    inheritance: dict[str, tuple[str, ...]] = {}
    for child, parents in inheritance_raw.items():
        if not isinstance(parents, list):
            raise ValueError("asset manifest profile inheritance entry must be a list")
        inheritance[str(child)] = tuple(str(parent) for parent in parents)

    effective: dict[str, frozenset[str]] = {}

    def expand(profile: str, visiting: frozenset[str] = frozenset()) -> frozenset[str]:
        if profile in visiting:
            raise ValueError(f"asset manifest profile inheritance cycle: {profile}")
        if profile in effective:
            return effective[profile]
        values = {profile}
        for parent in inheritance.get(profile, ()):
            values.update(expand(parent, visiting | {profile}))
        result = frozenset(values)
        effective[profile] = result
        return result

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("asset manifest artifacts must be a list")
    selected: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or artifact.get("kind") != "tensorrt_engine":
            continue
        artifact_id = str(artifact.get("id") or "").strip()
        required = {str(item) for item in artifact.get("required_profiles", []) or []}
        for profile in profiles:
            if (
                profile.startswith("artifact:")
                and artifact_id == profile.removeprefix("artifact:")
            ) or (not profile.startswith("artifact:") and bool(required & expand(profile))):
                if not artifact_id or artifact_id in selected:
                    if artifact_id in selected:
                        break
                    raise ValueError("asset manifest contains an invalid engine ID")
                selected[artifact_id] = artifact
                break
    return selected


def _artifact_output_path(root: Path, artifact: Mapping[str, Any], label: str) -> Path:
    raw = str(artifact.get("output") or "").strip()
    relative = Path(raw)
    if (
        not raw
        or relative.is_absolute()
        or ".." in relative.parts
        or tuple(relative.parts[:2]) != ("DS9", "models")
        or any(token in raw for token in ("*", "?", "[", "\x00", "\n", "\r"))
    ):
        raise ValueError(f"{label} has an unsafe or non-exact external output path")
    candidate = root
    for index, part in enumerate(relative.parts[1:]):
        candidate /= part
        try:
            info = candidate.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"{label} output is missing: {candidate}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} output path contains a symlink: {candidate}")
        if index < len(relative.parts[1:]) - 1:
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o022
            ):
                raise ValueError(f"{label} output parent is not privately owned")
        elif (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o022
            or info.st_size <= 0
        ):
            raise ValueError(f"{label} output is not a safe owned single-link file")
    return candidate


def _load_asset_validator() -> ModuleType:
    path = REPO_ROOT / "DS9" / "scripts" / "validate_asset_manifest.py"
    name = "_noesis_ds9_asset_validator_for_ownership"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load DS9 asset validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _declared_output_hashes(
    evidence: Mapping[str, Any], artifact_ids: Sequence[str], label: str
) -> dict[str, str]:
    raw = evidence.get("output_sha256")
    if not isinstance(raw, Mapping) or set(raw) != set(artifact_ids):
        raise ValueError(f"{label}.output_sha256 must exactly cover artifact_ids")
    return {
        artifact_id: _require_sha256(raw[artifact_id], f"{label}.output_sha256.{artifact_id}")
        for artifact_id in artifact_ids
    }


def _manifest_runtime_image_id(manifest: Mapping[str, Any], label: str) -> str:
    runtime = manifest.get("runtime")
    image = runtime.get("image") if isinstance(runtime, Mapping) else None
    image_id = (
        str(image.get("image_id") or "").strip()
        if isinstance(image, Mapping)
        else ""
    )
    if (
        not image_id.startswith("sha256:")
        or SHA256_RE.fullmatch(image_id.removeprefix("sha256:")) is None
    ):
        raise ValueError(f"{label}: asset manifest runtime image authority is incomplete")
    return image_id


def _validate_asset_realization_evidence(
    evidence: Mapping[str, Any],
    *,
    artifact_root: str | Path | None,
    label: str,
) -> dict[str, Any]:
    _require_exact_keys(evidence, ASSET_REALIZATION_KEYS, label)
    root = _external_root(
        artifact_root,
        "artifact root",
        accepted_modes=frozenset({0o700, 0o750}),
    )
    profiles = _profiles(evidence.get("profiles"), f"{label}.profiles")
    artifact_ids = _sorted_unique_strings(
        evidence.get("artifact_ids"), f"{label}.artifact_ids"
    )
    expected_outputs = _declared_output_hashes(evidence, artifact_ids, label)
    expected_realization_sha256 = _require_sha256(
        evidence.get("realization_sha256"), f"{label}.realization_sha256"
    )
    expected_manifest_sha256 = _require_sha256(
        evidence.get("base_manifest_sha256"), f"{label}.base_manifest_sha256"
    )
    expected_source_contracts_sha256 = _require_sha256(
        evidence.get("source_contracts_sha256"),
        f"{label}.source_contracts_sha256",
    )
    expected_runtime_image_id = str(evidence.get("runtime_image_id") or "").strip()
    if (
        not expected_runtime_image_id.startswith("sha256:")
        or SHA256_RE.fullmatch(
            expected_runtime_image_id.removeprefix("sha256:")
        )
        is None
    ):
        raise ValueError(f"{label}.runtime_image_id must be a sha256 image ID")

    manifest_path = REPO_ROOT / "DS9" / "asset_manifest.yaml"
    source_contracts_path = (
        REPO_ROOT / "DS9" / "config" / "engine_source_contracts.json"
    )
    manifest_raw = _read_authority_bytes(manifest_path, "asset manifest authority")
    source_contracts_raw = _read_authority_bytes(
        source_contracts_path, "engine source-contract authority"
    )
    manifest_sha256 = _sha256_bytes(manifest_raw)
    source_contracts_sha256 = _sha256_bytes(source_contracts_raw)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(f"{label}: tracked asset-manifest digest drift")
    if source_contracts_sha256 != expected_source_contracts_sha256:
        raise ValueError(f"{label}: tracked source-contract digest drift")
    manifest = yaml.safe_load(manifest_raw.decode("utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{label}: asset manifest root must be a mapping")
    runtime_image_id = _manifest_runtime_image_id(manifest, label)
    if runtime_image_id != expected_runtime_image_id:
        raise ValueError(f"{label}: runtime image authority drift")
    selected = _selected_engine_records(manifest, profiles)
    if sorted(selected) != artifact_ids:
        raise ValueError(
            f"{label}: selected engine IDs drifted; "
            f"actual={sorted(selected)} expected={artifact_ids}"
        )

    realization_path = _private_path(root, REALIZATION_FILENAME, "asset realization")
    realization_raw = _anchored_file_content(
        root,
        Path(REALIZATION_FILENAME),
        "asset realization",
        directory_policy="artifact",
        file_policy="private",
    )
    if not isinstance(realization_raw, bytes):
        raise ValueError("asset realization read returned an invalid type")
    realization_sha256 = _sha256_bytes(realization_raw)
    if realization_sha256 != expected_realization_sha256:
        raise ValueError(f"{label}: asset-realization digest drift")
    realization = _parse_json(realization_raw, "asset realization")
    if not isinstance(realization, Mapping):
        raise ValueError("asset realization root must be a mapping")
    if (
        type(realization.get("schema_version")) is not int  # noqa: E721
        or realization.get("schema_version") != 1
        or realization.get("contract") != REALIZATION_CONTRACT
        or realization.get("base_manifest")
        != {"path": "DS9/asset_manifest.yaml", "sha256": manifest_sha256}
        or realization.get("source_contracts")
        != {
            "path": "DS9/config/engine_source_contracts.json",
            "sha256": source_contracts_sha256,
        }
    ):
        raise ValueError(f"{label}: realization authority binding drift")
    overlays = realization.get("artifacts")
    if not isinstance(overlays, Mapping):
        raise ValueError(f"{label}: realization artifacts must be a mapping")

    observed_outputs: dict[str, str] = {}
    for artifact_id in artifact_ids:
        overlay = overlays.get(artifact_id)
        provenance = overlay.get("provenance") if isinstance(overlay, Mapping) else None
        declared_hash = (
            str(provenance.get("output_sha256") or "").strip()
            if isinstance(provenance, Mapping)
            else ""
        )
        output_path = _artifact_output_path(root, selected[artifact_id], artifact_id)
        actual_hash = _anchored_file_content(
            root,
            output_path.relative_to(root),
            artifact_id,
            directory_policy="artifact",
            file_policy="artifact",
            max_bytes=None,
            hash_only=True,
        )
        if not isinstance(actual_hash, str):
            raise ValueError(f"{artifact_id} hash returned an invalid type")
        if declared_hash != actual_hash or expected_outputs[artifact_id] != actual_hash:
            raise ValueError(f"{label}: selected output digest drift: {artifact_id}")
        observed_outputs[artifact_id] = actual_hash

    validator = _load_asset_validator()
    for profile in profiles:
        result = validator.validate_asset_realization(
            manifest_path,
            realization_path,
            root,
            profile=profile,
            check_files=True,
            require_provenance=True,
        )
        if not isinstance(result, Mapping) or not bool(result.get("ok")):
            blockers = list(result.get("blockers", []) if isinstance(result, Mapping) else [])
            errors = list(result.get("errors", []) if isinstance(result, Mapping) else [])
            raise ValueError(
                f"{label}: authoritative asset profile {profile!r} is blocked; "
                f"blockers={blockers} errors={errors}"
            )
        if result.get("profile") != profile or result.get("blockers"):
            raise ValueError(
                f"{label}: authoritative asset profile result is incomplete: {profile}"
            )

    if _sha256_bytes(
        _read_authority_bytes(manifest_path, "asset manifest final CAS")
    ) != manifest_sha256:
        raise ValueError(f"{label}: asset manifest changed during validation")
    if _sha256_bytes(
        _read_authority_bytes(
            source_contracts_path, "engine source-contract final CAS"
        )
    ) != source_contracts_sha256:
        raise ValueError(f"{label}: source contracts changed during validation")
    if _sha256_bytes(
        _anchored_file_content(
            root,
            Path(REALIZATION_FILENAME),
            "asset realization final CAS",
            directory_policy="artifact",
            file_policy="private",
        )
    ) != realization_sha256:
        raise ValueError(f"{label}: asset realization changed during validation")
    for artifact_id, expected_hash in observed_outputs.items():
        output_path = _artifact_output_path(root, selected[artifact_id], artifact_id)
        actual_hash = _anchored_file_content(
            root,
            output_path.relative_to(root),
            artifact_id,
            directory_policy="artifact",
            file_policy="artifact",
            max_bytes=None,
            hash_only=True,
        )
        if actual_hash != expected_hash:
            raise ValueError(f"{label}: selected output changed during validation: {artifact_id}")

    return {
        "profiles": profiles,
        "realization_sha256": realization_sha256,
        "base_manifest_sha256": manifest_sha256,
        "source_contracts_sha256": source_contracts_sha256,
        "runtime_image_id": runtime_image_id,
        "artifact_ids": artifact_ids,
        "output_sha256": observed_outputs,
    }


def build_asset_realization_selector(
    *,
    profiles: Sequence[str],
    artifact_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _external_root(
        artifact_root,
        "artifact root",
        accepted_modes=frozenset({0o700, 0o750}),
    )
    normalized_profiles = _profiles(list(profiles), "promotion profiles")
    manifest_path = REPO_ROOT / "DS9" / "asset_manifest.yaml"
    source_contracts_path = (
        REPO_ROOT / "DS9" / "config" / "engine_source_contracts.json"
    )
    manifest_raw = _read_authority_bytes(
        manifest_path, "promotion asset manifest authority"
    )
    source_contracts_raw = _read_authority_bytes(
        source_contracts_path, "promotion source-contract authority"
    )
    manifest = yaml.safe_load(manifest_raw.decode("utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("promotion asset manifest root must be a mapping")
    runtime_image_id = _manifest_runtime_image_id(
        manifest, "promotion asset realization"
    )
    selected = _selected_engine_records(manifest, normalized_profiles)
    artifact_ids = sorted(selected)
    realization_raw = _anchored_file_content(
        root,
        Path(REALIZATION_FILENAME),
        "promotion asset realization",
        directory_policy="artifact",
        file_policy="private",
    )
    if not isinstance(realization_raw, bytes):
        raise ValueError("promotion asset realization read returned invalid bytes")
    output_sha256: dict[str, str] = {}
    for artifact_id in artifact_ids:
        output_path = _artifact_output_path(root, selected[artifact_id], artifact_id)
        digest = _anchored_file_content(
            root,
            output_path.relative_to(root),
            f"promotion output {artifact_id}",
            directory_policy="artifact",
            file_policy="artifact",
            max_bytes=None,
            hash_only=True,
        )
        if not isinstance(digest, str):
            raise ValueError(f"promotion output hash is invalid: {artifact_id}")
        output_sha256[artifact_id] = digest
    selector = {
        "profiles": normalized_profiles,
        "realization_sha256": _sha256_bytes(realization_raw),
        "base_manifest_sha256": _sha256_bytes(manifest_raw),
        "source_contracts_sha256": _sha256_bytes(source_contracts_raw),
        "runtime_image_id": runtime_image_id,
        "artifact_ids": artifact_ids,
        "output_sha256": output_sha256,
    }
    detail = _validate_asset_realization_evidence(
        selector,
        artifact_root=root,
        label="promotion asset realization",
    )
    return selector, detail


def build_runtime_session_selector(
    *,
    capability_id: str,
    lane: str,
    session_id: str,
    artifact_root: str | Path,
    runtime_root: str | Path,
    docker_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if SESSION_RE.fullmatch(str(session_id)) is None:
        raise ValueError("promotion session_id is unsafe")
    lane_policy = RUNTIME_LANES.get(lane)
    capability_policy = CAPABILITY_REGISTRY.get(capability_id)
    runtime_requirements = (
        capability_policy.get("runtime_requirements")
        if isinstance(capability_policy, Mapping)
        else None
    )
    if lane_policy is None or not isinstance(runtime_requirements, Mapping):
        raise ValueError("promotion capability/lane has no runtime requirement")
    behavior_documents = runtime_requirements.get(lane)
    if not isinstance(behavior_documents, list) or not behavior_documents:
        raise ValueError("promotion lane is not required by the capability")
    artifact_selector, _artifact_detail = build_asset_realization_selector(
        profiles=lane_policy["profiles"],
        artifact_root=artifact_root,
    )
    runtime = _external_root(
        runtime_root,
        "runtime root",
        accepted_modes=frozenset({0o700}),
    )
    launcher_relative = Path("evidence") / session_id / "launcher"
    checksum_raw = _anchored_file_content(
        runtime,
        launcher_relative / "SHA256SUMS",
        "promotion runtime checksum manifest",
        directory_policy="private",
        file_policy="private",
    )
    if not isinstance(checksum_raw, bytes):
        raise ValueError("promotion runtime checksum manifest is invalid")
    selector = {
        **artifact_selector,
        "session_id": session_id,
        "lane": lane,
        "checksum_manifest": "launcher/SHA256SUMS",
        "checksum_sha256": _sha256_bytes(checksum_raw),
        "behavior_documents": sorted(str(value) for value in behavior_documents),
    }
    detail = _validate_runtime_session_evidence(
        selector,
        artifact_root=artifact_root,
        runtime_root=runtime,
        docker_root=docker_root,
        label="promotion runtime session",
        capability_id=capability_id,
    )
    return selector, detail


def _require_private_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError(f"{label} must be an owned mode-0700 non-symlink directory")


def _load_checksum_covered_files(
    runtime_root: Path,
    launcher_relative: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, bytes]:
    checksum_raw = _anchored_file_content(
        runtime_root,
        launcher_relative / "SHA256SUMS",
        "runtime checksum manifest",
        directory_policy="private",
        file_policy="private",
    )
    if not isinstance(checksum_raw, bytes):
        raise ValueError("runtime checksum manifest read returned an invalid type")
    if _sha256_bytes(checksum_raw) != expected_manifest_sha256:
        raise ValueError("runtime checksum-manifest digest drift")
    try:
        checksum_text = checksum_raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("runtime checksum manifest must be ASCII") from exc
    if not checksum_text.endswith("\n"):
        raise ValueError("runtime checksum manifest must end with one newline")
    declared: dict[str, str] = {}
    for line in checksum_text.splitlines():
        match = CHECKSUM_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"runtime checksum manifest has an unsafe row: {line!r}")
        digest, name = match.groups()
        if name == "SHA256SUMS" or name in declared:
            raise ValueError(f"runtime checksum manifest has a duplicate/recursive row: {name}")
        declared[name] = digest
    if not declared:
        raise ValueError("runtime checksum manifest is empty")

    directory_entries = _anchored_directory_entries(
        runtime_root,
        launcher_relative,
        "runtime launcher evidence",
        directory_policy="private",
    )
    actual_names: set[str] = set()
    for name, info in directory_entries.items():
        if name == "SHA256SUMS":
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"runtime launcher evidence contains an unsafe entry: {name}")
        actual_names.add(name)
    if set(declared) != actual_names:
        raise ValueError(
            "runtime checksum manifest coverage drift; "
            f"missing={sorted(actual_names - set(declared))} "
            f"stale={sorted(set(declared) - actual_names)}"
        )
    if not MANDATORY_RUNTIME_JSON.issubset(declared) or "runtime.log" not in declared:
        raise ValueError("runtime checksum manifest omits mandatory lifecycle evidence")

    covered: dict[str, bytes] = {}
    for name, expected_digest in declared.items():
        raw = _anchored_file_content(
            runtime_root,
            launcher_relative / name,
            f"runtime evidence {name}",
            directory_policy="private",
            file_policy="private",
            allow_empty=True,
        )
        if not isinstance(raw, bytes):
            raise ValueError(f"runtime evidence {name} read returned an invalid type")
        if _sha256_bytes(raw) != expected_digest:
            raise ValueError(f"runtime checksum mismatch: {name}")
        covered[name] = raw
    return covered


def _json_pointer(payload: Any, pointer: str, label: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/") or len(pointer) > 1024:
        raise ValueError(f"{label} must be an RFC 6901 JSON pointer")
    current = payload
    for raw_token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw_token):
            raise ValueError(f"{label} contains an invalid JSON pointer escape")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise ValueError(f"{label} does not resolve at {token!r}")
            current = current[token]
        elif isinstance(current, list):
            if re.fullmatch(r"(?:0|[1-9][0-9]*)", token) is None:
                raise ValueError(f"{label} contains a non-canonical list index")
            index = int(token)
            if index >= len(current):
                raise ValueError(f"{label} list index is out of range")
            current = current[index]
        else:
            raise ValueError(f"{label} traverses a scalar value")
    return current


def _exact_object(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} keys differ from producer schema: {actual}")
    return value


def _exact_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _exact_nonnegative_int(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):  # noqa: E721
        raise ValueError(f"{label} must be an exact {'positive' if positive else 'nonnegative'} integer")
    return value


def _exact_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _exact_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_wholebody_behavior(
    document: Mapping[str, Any],
    label: str,
    *,
    covered: Mapping[str, bytes],
    session_id: str,
    lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    inspect_started: datetime,
    inspect_finished: datetime,
) -> None:
    _exact_object(
        document,
        {
            "schema_version", "contract", "contract_version", "session_id",
            "runtime_lane", "runtime_instance_id", "runtime_run_id", "ok",
            "status", "scene_status", "mode", "expected_source_ids",
            "observed_source_ids", "thresholds", "checks", "tracking_messages",
            "tracks_seen", "stats_samples", "tracking_window_seconds",
            "source_metrics", "counters", "pipeline_errors", "source_evidence",
            "render_evidence_policy",
        },
        label,
    )
    mode = document.get("mode")
    expected_mode = "masks" if lane == "wholebody49-s" else "boxes"
    if mode != expected_mode:
        raise ValueError(f"{label} Wholebody49 lane/mode drifted")
    canonical_source_ids = [int(value) for value in _reviewed_lane_source_ids(lane)]
    for field in ("expected_source_ids", "observed_source_ids"):
        source_ids = document.get(field)
        if (
            not isinstance(source_ids, list)
            or any(type(value) is not int for value in source_ids)  # noqa: E721
            or source_ids != canonical_source_ids
        ):
            raise ValueError(
                f"{label} Wholebody49 source inventory differs from reviewed lane {lane}"
            )
    if document.get("render_evidence_policy") != WHOLEBODY_RENDER_EVIDENCE_POLICY:
        raise ValueError(f"{label}.render_evidence_policy drifted")

    source_metadata = _exact_object(
        document.get("source_evidence"),
        {
            "filename",
            "sha256",
            "message_count",
            "first_observed_at_us",
            "last_observed_at_us",
        },
        f"{label}.source_evidence",
    )
    source_raw = covered.get(WHOLEBODY_SOURCE_TRANSCRIPT_FILENAME)
    if (
        source_raw is None
        or source_metadata.get("filename")
        != WHOLEBODY_SOURCE_TRANSCRIPT_FILENAME
        or source_metadata.get("sha256") != _sha256_bytes(source_raw)
    ):
        raise ValueError(f"{label} lacks its checksum-covered source transcript")
    if len(source_raw) > 8 * 1024 * 1024:
        raise ValueError(f"{label} source transcript exceeds its byte bound")
    source = _exact_object(
        _parse_json(source_raw, f"{label} source transcript"),
        {
            "schema_version",
            "contract",
            "contract_version",
            "session_id",
            "runtime_lane",
            "runtime_instance_id",
            "runtime_run_id",
            "mode",
            "expected_source_ids",
            "thresholds",
            "privacy",
            "message_count",
            "messages",
        },
        f"{label} source transcript",
    )
    messages = _exact_list(source.get("messages"), f"{label} source messages")
    if (
        type(source.get("schema_version")) is not int  # noqa: E721
        or source.get("schema_version") != 1
        or source.get("contract")
        != "noesis.ds9.wholebody49-source-transcript"
        or source.get("contract_version") != 2
        or source.get("session_id") != session_id
        or source.get("runtime_lane") != lane
        or source.get("runtime_instance_id") != runtime_instance_id
        or source.get("runtime_run_id") != runtime_run_id
        or source.get("mode") != mode
        or source.get("message_count") != len(messages)
        or source.get("message_count") != source_metadata.get("message_count")
    ):
        raise ValueError(f"{label} source transcript envelope drifted")

    module = _load_sibling_script_module(
        "wholebody49_occupied_scene_smoke_test.py",
        "_noesis_ds9_wholebody_source_for_ownership",
    )
    try:
        if source.get("privacy") != module.SOURCE_PRIVACY_POLICY:
            raise ValueError(f"{label} source transcript privacy policy drifted")
        expected_source_ids = source.get("expected_source_ids")
        if (
            not isinstance(expected_source_ids, list)
            or any(
                type(value) is not int  # noqa: E721
                for value in expected_source_ids
            )
            or expected_source_ids != canonical_source_ids
            or expected_source_ids != document.get("expected_source_ids")
        ):
            raise ValueError(f"{label} source inventory binding drifted")
        canonical_source = module._source_transcript_document(
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            mode=mode,
            expected_source_ids=expected_source_ids,
            events=messages,
        )
        if not _strict_json_equal(source, canonical_source):
            raise ValueError(f"{label} source transcript is not canonical")
        if source_raw != module._encoded_private_json(canonical_source):
            raise ValueError(
                f"{label} source transcript bytes differ from the producer encoding"
            )
        canonical_metadata = module._source_evidence_metadata(
            encoded=source_raw,
            document=source,
        )
        if not _strict_json_equal(source_metadata, canonical_metadata):
            raise ValueError(f"{label} source transcript metadata drifted")
        recomputed = module.analyze_evidence(
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            mode=mode,
            expected_source_ids=expected_source_ids,
            events=messages,
            source_evidence=canonical_metadata,
        )
    finally:
        sys.modules.pop(module.__name__, None)
    if not _strict_json_equal(document, recomputed):
        raise ValueError(f"{label} does not recompute from its source transcript")
    first_observed = _exact_nonnegative_int(
        source_metadata.get("first_observed_at_us"),
        f"{label}.source_evidence.first_observed_at_us",
        positive=True,
    )
    last_observed = _exact_nonnegative_int(
        source_metadata.get("last_observed_at_us"),
        f"{label}.source_evidence.last_observed_at_us",
        positive=True,
    )
    if (
        first_observed > last_observed
        or first_observed < int(inspect_started.timestamp() * 1_000_000)
        or last_observed > int(inspect_finished.timestamp() * 1_000_000)
    ):
        raise ValueError(f"{label} source transcript escapes the runtime window")


def _load_sibling_script_module(filename: str, module_name: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load canonical DS9 script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


CANONICAL_PRODUCER_BEHAVIOR_IDS = frozenset(BEHAVIOR_CONTRACTS)


def _canonical_behavior_report_bytes(
    behavior_id: str,
    document: Mapping[str, Any],
) -> bytes:
    """Encode a behavior report with its owning producer's exact encoder."""

    if behavior_id in {"wholebody49_occupied_s_v2", "wholebody49_occupied_x_v2"}:
        filename = "wholebody49_occupied_scene_smoke_test.py"
        module_name = "_noesis_ds9_wholebody_report_encoder_for_ownership"
        encoder_name = "_encoded_private_json"
    elif behavior_id == "wholebody49_media_decode_v1":
        filename = "wholebody49_media_decode_gate.py"
        module_name = "_noesis_ds9_wholebody_media_report_encoder_for_ownership"
        encoder_name = "_encoded"
    elif behavior_id in {"reid_open_set_occupied_v1", "v3dt_identity_gate_v1"}:
        filename = "ds9_identity_shadow_live_gate.py"
        module_name = "_noesis_ds9_identity_report_encoder_for_ownership"
        encoder_name = "_encoded_private_json"
    elif behavior_id == "mapanything_depth_quality_v4":
        filename = "ds9_floorplan_live_gate.py"
        module_name = "_noesis_ds9_floorplan_report_encoder_for_ownership"
        encoder_name = "_encoded_private_json"
    elif behavior_id == "v3dt_world_gate_v2":
        filename = "v3dt_world_contract_smoke_test.py"
        module_name = "_noesis_ds9_world_report_encoder_for_ownership"
        encoder_name = "_encoded_private_json"
    elif behavior_id == "semantic_gate_v3":
        filename = "ds9_semantic_observation_smoke_test.py"
        module_name = "_noesis_ds9_semantic_report_encoder_for_ownership"
        encoder_name = "_encoded_report"
    elif behavior_id == "runtime_resource_soak_v2":
        supervisor = _load_runtime_supervisor_module()
        try:
            encoded = supervisor._encoded_private_json(document)
        finally:
            sys.modules.pop(supervisor.__name__, None)
        if not isinstance(encoded, bytes):
            raise ValueError("resource-soak producer encoder returned invalid bytes")
        return encoded
    else:
        raise ValueError(f"no canonical producer encoder registered for {behavior_id}")

    module = _load_sibling_script_module(filename, module_name)
    try:
        encoder = getattr(module, encoder_name, None)
        if not callable(encoder):
            raise ValueError(
                f"behavior {behavior_id} producer lacks its canonical report encoder"
            )
        encoded = encoder(document)
    finally:
        sys.modules.pop(module.__name__, None)
    if not isinstance(encoded, bytes):
        raise ValueError(f"behavior {behavior_id} producer encoder returned invalid bytes")
    return encoded


def _validate_wholebody_media_behavior(
    document: Mapping[str, Any],
    label: str,
    *,
    covered: Mapping[str, bytes],
    session_id: str,
    lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    inspect_started: datetime,
    inspect_finished: datetime,
) -> None:
    _exact_object(
        document,
        {
            "schema_version", "contract", "contract_version", "session_id",
            "runtime_lane", "runtime_instance_id", "runtime_run_id", "ok",
            "status", "checks", "metrics", "thresholds", "source_evidence",
            "errors",
        },
        label,
    )
    source_metadata = _exact_object(
        document.get("source_evidence"),
        {"filename", "sha256", "started_at_us", "finished_at_us"},
        f"{label}.source_evidence",
    )
    source_raw = covered.get(WHOLEBODY_MEDIA_SOURCE_FILENAME)
    if (
        source_raw is None
        or source_metadata.get("filename") != WHOLEBODY_MEDIA_SOURCE_FILENAME
        or source_metadata.get("sha256") != _sha256_bytes(source_raw)
    ):
        raise ValueError(f"{label} lacks checksum-covered decoded-media source")
    if len(source_raw) > 256 * 1024:
        raise ValueError(f"{label} decoded-media source exceeds its byte bound")
    source = _parse_json(source_raw, f"{label} decoded-media source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{label} decoded-media source root must be an object")
    module = _load_sibling_script_module(
        "wholebody49_media_decode_gate.py",
        "_noesis_ds9_wholebody_media_for_ownership",
    )
    try:
        if (
            source.get("session_id") != session_id
            or source.get("runtime_lane") != lane
            or source.get("runtime_instance_id") != runtime_instance_id
            or source.get("runtime_run_id") != runtime_run_id
        ):
            raise ValueError(f"{label} decoded-media source identity drifted")
        if source_raw != module._encoded(source):
            raise ValueError(
                f"{label} decoded-media source bytes differ from the producer encoding"
            )
        canonical_metadata = module._source_evidence(source_raw, source)
        if not _strict_json_equal(source_metadata, canonical_metadata):
            raise ValueError(f"{label} decoded-media source metadata drifted")
        recomputed = module.analyze_source(
            source,
            source_evidence=canonical_metadata,
        )
    finally:
        sys.modules.pop(module.__name__, None)
    if not _strict_json_equal(document, recomputed):
        raise ValueError(f"{label} decoded-media report does not exactly replay")
    started_at_us = _exact_nonnegative_int(
        source_metadata.get("started_at_us"),
        f"{label}.source_evidence.started_at_us",
        positive=True,
    )
    finished_at_us = _exact_nonnegative_int(
        source_metadata.get("finished_at_us"),
        f"{label}.source_evidence.finished_at_us",
        positive=True,
    )
    if (
        started_at_us > finished_at_us
        or started_at_us < int(inspect_started.timestamp() * 1_000_000)
        or finished_at_us > int(inspect_finished.timestamp() * 1_000_000)
    ):
        raise ValueError(f"{label} decoded-media window escapes container lifetime")


def _reviewed_reid_model_binding(
    lane: str,
    artifact_binding: Mapping[str, Any],
) -> dict[str, object]:
    lane_policy = RUNTIME_LANES.get(lane)
    if not isinstance(lane_policy, Mapping):
        raise ValueError(f"unsupported ReID runtime lane: {lane}")
    pipeline_path = _repo_path(lane_policy.get("pipeline"))
    pipeline_raw = _read_authority_bytes(
        pipeline_path, f"{lane} ReID pipeline authority"
    )
    pipeline = _parse_source_yaml(
        pipeline_raw.decode("utf-8"), f"{lane} ReID pipeline authority"
    )
    models = pipeline.get("models")
    reid = models.get("reid") if isinstance(models, Mapping) else None
    expected_config = {
        "enable": True,
        "engine": REID_ENGINE_PATH,
        "layer": REID_MODEL_LAYER,
        "embedding_dim": REID_EMBEDDING_DIMENSION,
    }
    if not isinstance(reid, Mapping) or any(
        reid.get(key) != value for key, value in expected_config.items()
    ):
        raise ValueError(f"{lane} reviewed ReID config drifted")
    outputs = artifact_binding.get("output_sha256")
    model_sha256 = outputs.get(REID_ARTIFACT_ID) if isinstance(outputs, Mapping) else None
    if SHA256_RE.fullmatch(str(model_sha256 or "")) is None:
        raise ValueError("selected artifact realization lacks the ReID engine digest")
    return {
        "model_sha256": str(model_sha256),
        "model_layer": REID_MODEL_LAYER,
        "embedding_dimension": REID_EMBEDDING_DIMENSION,
    }


def _validate_identity_behavior(
    document: Mapping[str, Any],
    label: str,
    *,
    behavior_id: str,
    covered: Mapping[str, bytes],
    artifact_binding: Mapping[str, Any],
    session_id: str,
    lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    inspect_started: datetime,
    inspect_finished: datetime,
) -> None:
    _exact_object(
        document,
        {
            "schema_version",
            "contract",
            "contract_version",
            "session_id",
            "runtime_lane",
            "runtime_instance_id",
            "runtime_run_id",
            "ok",
            "claims",
            "counts",
            "health_before",
            "health_after",
            "source_evidence",
            "errors",
        },
        label,
    )
    claims = _exact_object(
        document.get("claims"),
        {
            "runtime_shadow_health",
            "tracker_subject_continuity",
            "cross_camera_assignment_continuity",
            "open_set_non_force",
            "semantic_accuracy",
            "public_authority",
        },
        f"{label}.claims",
    )
    for name in (
        "runtime_shadow_health",
        "tracker_subject_continuity",
        "cross_camera_assignment_continuity",
        "open_set_non_force",
    ):
        claim = _exact_object(
            claims.get(name), {"status", "required", "evidence_count"}, f"{label}.claims.{name}"
        )
        _exact_nonempty_string(claim.get("status"), f"{label}.claims.{name}.status")
        if type(claim.get("required")) is not bool:  # noqa: E721
            raise ValueError(f"{label}.claims.{name}.required must be boolean")
        _exact_nonnegative_int(claim.get("evidence_count"), f"{label}.claims.{name}.evidence_count")
    runtime_claim = claims["runtime_shadow_health"]
    continuity_claim = claims["tracker_subject_continuity"]
    cross_camera_claim = claims["cross_camera_assignment_continuity"]
    open_set_claim = claims["open_set_non_force"]
    if runtime_claim != {"status": "pass", "required": True, "evidence_count": 2}:
        raise ValueError(f"{label} runtime shadow-health claim drifted")
    if (
        continuity_claim.get("status") != "pass"
        or continuity_claim.get("required") is not True
        or int(continuity_claim["evidence_count"]) < 1
    ):
        raise ValueError(f"{label} tracker-subject continuity claim drifted")
    if cross_camera_claim.get("status") not in {"observed", "not_observed"}:
        raise ValueError(f"{label} cross-camera claim status drifted")
    if (
        cross_camera_claim.get("required") is True
        and cross_camera_claim.get("status") != "observed"
    ):
        raise ValueError(f"{label} required cross-camera evidence is absent")
    if (
        open_set_claim.get("status") != "observed"
        or open_set_claim.get("required") is not True
        or int(open_set_claim["evidence_count"]) < 1
    ):
        raise ValueError(f"{label} open-set non-force claim drifted")
    for name in ("semantic_accuracy", "public_authority"):
        claim = _exact_object(
            claims.get(name), {"status", "required", "reason"}, f"{label}.claims.{name}"
        )
        _exact_nonempty_string(claim.get("status"), f"{label}.claims.{name}.status")
        _exact_nonempty_string(claim.get("reason"), f"{label}.claims.{name}.reason")
        if type(claim.get("required")) is not bool:  # noqa: E721
            raise ValueError(f"{label}.claims.{name}.required must be boolean")
    if (
        claims["semantic_accuracy"].get("status") != "not_evaluated"
        or claims["semantic_accuracy"].get("required") is not False
        or claims["public_authority"].get("status") != "blocked"
        or claims["public_authority"].get("required") is not True
    ):
        raise ValueError(f"{label} accuracy/authority claims drifted")
    counts = _exact_object(
        document.get("counts"),
        {
            "tracking_messages",
            "person_rows",
            "fresh_embedding_rows",
            "held_rows",
            "continuity_subjects",
            "cross_camera_subjects",
            "cross_camera_pairs",
            "open_set_non_force_rows",
            "overlap_permit_rows",
            "run_id_count",
        },
        f"{label}.counts",
    )
    for key, value in counts.items():
        _exact_nonnegative_int(value, f"{label}.counts.{key}")
    if (
        counts["run_id_count"] != 1
        or counts["tracking_messages"] < 1
        or counts["person_rows"] < 1
        or counts["fresh_embedding_rows"] < 2
        or counts["continuity_subjects"] < 1
        or counts["open_set_non_force_rows"] < 1
    ):
        raise ValueError(f"{label} occupied identity counts do not prove the claims")
    if (
        continuity_claim["evidence_count"] != counts["continuity_subjects"]
        or open_set_claim["evidence_count"] != counts["open_set_non_force_rows"]
        or cross_camera_claim["evidence_count"] != counts["cross_camera_subjects"]
        or (cross_camera_claim["status"] == "observed")
        != (counts["cross_camera_subjects"] > 0)
    ):
        raise ValueError(f"{label} identity claim/count relationships drifted")
    health_keys = {
        "runtime_mode",
        "public_authority_cutover_status",
        "runtime_model_fingerprint",
        "runtime_model_layer",
        "runtime_embedding_dim",
        "scoring_calibration_status",
        "scoring_authority_scope",
        "resident_count",
        "observation_cache_entries",
        "observation_cache_max_entries",
        "observation_cache_ttl_s",
    }
    for phase in ("health_before", "health_after"):
        health = _exact_object(document.get(phase), health_keys, f"{label}.{phase}")
        if (
            health.get("runtime_mode") != "shadow"
            or health.get("public_authority_cutover_status") != "blocked"
        ):
            raise ValueError(f"{label}.{phase} authority mode drifted")
        if SHA256_RE.fullmatch(str(health.get("runtime_model_fingerprint") or "")) is None:
            raise ValueError(f"{label}.{phase} model fingerprint is invalid")
        _exact_nonempty_string(
            health.get("runtime_model_layer"), f"{label}.{phase}.runtime_model_layer"
        )
        _exact_nonempty_string(
            health.get("scoring_calibration_status"),
            f"{label}.{phase}.scoring_calibration_status",
        )
        if health.get("scoring_authority_scope") is not None and not isinstance(
            health.get("scoring_authority_scope"), str
        ):
            raise ValueError(f"{label}.{phase}.scoring_authority_scope is invalid")
        _exact_nonnegative_int(
            health.get("runtime_embedding_dim"),
            f"{label}.{phase}.runtime_embedding_dim",
            positive=True,
        )
        for key in ("resident_count", "observation_cache_entries"):
            _exact_nonnegative_int(health.get(key), f"{label}.{phase}.{key}")
        _exact_nonnegative_int(
            health.get("observation_cache_max_entries"),
            f"{label}.{phase}.observation_cache_max_entries",
            positive=True,
        )
        if _exact_finite_number(
            health.get("observation_cache_ttl_s"),
            f"{label}.{phase}.observation_cache_ttl_s",
        ) <= 0:
            raise ValueError(f"{label}.{phase}.observation_cache_ttl_s must be positive")
        if health["observation_cache_entries"] > health["observation_cache_max_entries"]:
            raise ValueError(f"{label}.{phase} identity cache count exceeds its bound")
    before = document["health_before"]
    after = document["health_after"]
    for key in (
        "runtime_mode",
        "public_authority_cutover_status",
        "runtime_model_fingerprint",
        "runtime_model_layer",
        "runtime_embedding_dim",
        "scoring_calibration_status",
        "scoring_authority_scope",
        "resident_count",
        "observation_cache_max_entries",
        "observation_cache_ttl_s",
    ):
        if before.get(key) != after.get(key):
            raise ValueError(f"{label} identity health changed during the gate at {key}")
    if int(after["observation_cache_entries"]) < 1:
        raise ValueError(f"{label} identity cache remained empty after the gate")
    if _exact_list(document.get("errors"), f"{label}.errors"):
        raise ValueError(f"{label}.errors must be empty for a pass")

    reviewed_model = _reviewed_reid_model_binding(lane, artifact_binding)
    for phase in ("health_before", "health_after"):
        health = document[phase]
        if (
            health["runtime_model_fingerprint"] != reviewed_model["model_sha256"]
            or health["runtime_model_layer"] != reviewed_model["model_layer"]
            or health["runtime_embedding_dim"]
            != reviewed_model["embedding_dimension"]
        ):
            raise ValueError(f"{label}.{phase} is not bound to the realized ReID model")

    expected_source_filename = IDENTITY_SOURCE_FILENAMES[behavior_id]
    source_metadata = _exact_object(
        document.get("source_evidence"),
        {
            "filename",
            "sha256",
            "message_count",
            "row_count",
            "first_observed_at_us",
            "last_observed_at_us",
            "observed_run_id",
        },
        f"{label}.source_evidence",
    )
    source_raw = covered.get(expected_source_filename)
    if (
        source_raw is None
        or source_metadata.get("filename") != expected_source_filename
        or source_metadata.get("sha256") != _sha256_bytes(source_raw)
    ):
        raise ValueError(f"{label} lacks its checksum-covered identity transcript")
    source_document = _parse_json(source_raw, f"{label} source transcript")
    source = _exact_object(
        source_document,
        {
            "schema_version",
            "contract",
            "contract_version",
            "session_id",
            "runtime_lane",
            "runtime_instance_id",
            "runtime_run_id",
            "policy",
            "health_before",
            "health_after",
            "message_count",
            "row_count",
            "messages",
        },
        f"{label} source transcript",
    )
    if (
        type(source.get("schema_version")) is not int  # noqa: E721
        or source.get("schema_version") != 2
        or source.get("contract")
        != "noesis.ds9.identity-shadow-source-transcript"
        or source.get("contract_version") != 2
        or source.get("session_id") != session_id
        or source.get("runtime_lane") != lane
        or source.get("runtime_instance_id") != runtime_instance_id
        or source.get("runtime_run_id") != runtime_run_id
    ):
        raise ValueError(f"{label} identity source envelope drifted")
    source_policy = _exact_object(
        source.get("policy"),
        {
            "continuity_gap_s",
            "require_cross_camera",
            "require_open_set",
            "min_fresh_embeddings",
        },
        f"{label} source policy",
    )
    if (
        source_policy.get("continuity_gap_s") != 2.0
        or source_policy.get("require_cross_camera")
        is not bool(claims["cross_camera_assignment_continuity"]["required"])
        or source_policy.get("require_open_set") is not True
        or source_policy.get("min_fresh_embeddings") != 2
    ):
        raise ValueError(f"{label} identity source policy drifted")
    messages = _exact_list(source.get("messages"), f"{label} source messages")
    _exact_nonnegative_int(
        source.get("message_count"), f"{label} source message_count", positive=True
    )
    _exact_nonnegative_int(
        source.get("row_count"), f"{label} source row_count", positive=True
    )
    if source["message_count"] != len(messages):
        raise ValueError(f"{label} source message count does not reconcile")

    identity_module = _load_sibling_script_module(
        "ds9_identity_shadow_live_gate.py",
        "_noesis_ds9_identity_source_for_ownership",
    )
    try:
        source_health_before = _exact_object(
            source.get("health_before"),
            set(identity_module.HEALTH_SOURCE_FIELDS),
            f"{label} source health_before",
        )
        source_health_after = _exact_object(
            source.get("health_after"),
            set(identity_module.HEALTH_SOURCE_FIELDS),
            f"{label} source health_after",
        )
        replay_before, replay_after = identity_module._validate_health_pair(
            source_health_before,
            source_health_after,
            expected_layer=str(reviewed_model["model_layer"]),
            expected_dimension=int(reviewed_model["embedding_dimension"]),
        )
        if not _strict_json_equal(before, replay_before) or not _strict_json_equal(
            after, replay_after
        ):
            raise ValueError(f"{label} report health differs from sealed source health")
        collector = identity_module.IdentityEvidenceCollector(continuity_gap_s=2.0)
        for message in messages:
            collector.observe_payload(message)
        if collector.errors:
            raise ValueError(f"{label} source transcript replay failed: {collector.errors}")
        canonical_source = identity_module._source_transcript_document(
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            collector=collector,
            health_before=source_health_before,
            health_after=source_health_after,
            require_cross_camera=bool(
                claims["cross_camera_assignment_continuity"]["required"]
            ),
            require_open_set=True,
            min_fresh_embeddings=2,
        )
        if not _strict_json_equal(source, canonical_source):
            raise ValueError(f"{label} source transcript is not canonical")
        if source_raw != identity_module._encoded_private_json(canonical_source):
            raise ValueError(
                f"{label} source transcript bytes differ from the producer encoding"
            )
        canonical_metadata = identity_module._source_evidence_metadata(
            filename=expected_source_filename,
            encoded=source_raw,
            document=source,
            collector=collector,
        )
        if not _strict_json_equal(source_metadata, canonical_metadata):
            raise ValueError(f"{label} source transcript metadata drifted")
        recomputed_report = identity_module._build_report(
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            collector=collector,
            health_before=replay_before,
            health_after=replay_after,
            require_cross_camera=bool(
                claims["cross_camera_assignment_continuity"]["required"]
            ),
            require_open_set=True,
            min_fresh_embeddings=2,
            source_evidence=canonical_metadata,
            errors=[],
        )
    finally:
        sys.modules.pop(identity_module.__name__, None)
    if not _strict_json_equal(document, recomputed_report):
        raise ValueError(f"{label} does not exactly recompute from its source transcript")
    first_observed = _exact_nonnegative_int(
        source_metadata.get("first_observed_at_us"),
        f"{label}.source_evidence.first_observed_at_us",
        positive=True,
    )
    last_observed = _exact_nonnegative_int(
        source_metadata.get("last_observed_at_us"),
        f"{label}.source_evidence.last_observed_at_us",
        positive=True,
    )
    if (
        source_metadata.get("observed_run_id") != runtime_run_id
        or first_observed > last_observed
        or first_observed < int(inspect_started.timestamp() * 1_000_000)
        or last_observed > int(inspect_finished.timestamp() * 1_000_000)
    ):
        raise ValueError(f"{label} source transcript window/run binding drifted")


def _reviewed_floorplan_bindings() -> dict[str, str]:
    floorplan_module = _load_sibling_script_module(
        "ds9_floorplan_live_gate.py",
        "_noesis_ds9_floorplan_contract_for_ownership",
    )
    try:
        camera_ids = floorplan_module._active_camera_ids(
            REPO_ROOT / RUNTIME_LANES["baseline"]["pipeline"],
            REPO_ROOT / RUNTIME_LANES["baseline"]["cameras"],
        )
    finally:
        sys.modules.pop(floorplan_module.__name__, None)
    registration_raw = _read_authority_bytes(
        REPO_ROOT / "DS9" / "config" / "depth_registration.json",
        "DS9 depth-registration authority",
    )
    registration = _parse_json(
        registration_raw, "DS9 depth-registration authority"
    )
    cameras = registration.get("cameras") if isinstance(registration, Mapping) else None
    if not isinstance(cameras, Mapping):
        raise ValueError("DS9 depth-registration camera authority is missing")
    result: dict[str, str] = {}
    for camera_id in camera_ids:
        row = cameras.get(camera_id)
        fingerprint = (
            row.get("calibration_fingerprint")
            if isinstance(row, Mapping)
            else None
        )
        digest = (
            fingerprint.get("fingerprint_sha256")
            if isinstance(fingerprint, Mapping)
            else None
        )
        if SHA256_RE.fullmatch(str(digest or "")) is None:
            raise ValueError(
                f"DS9 depth-registration calibration fingerprint is missing: {camera_id}"
            )
        result[str(camera_id)] = str(digest)
    return result


def _validate_capture_rgb_evidence(value: object, label: str) -> None:
    rgb = _exact_object(
        value,
        {
            "status",
            "provider_configured",
            "source_id",
            "batch_id",
            "captured_at_us",
            "frame_id",
            "source_media_pts_ns",
            "width",
            "height",
            "color_space",
            "content_sha256",
        },
        label,
    )
    source_pts = _exact_nonnegative_int(
        rgb.get("source_media_pts_ns"),
        f"{label}.source_media_pts_ns",
    )
    width = _exact_nonnegative_int(
        rgb.get("width"),
        f"{label}.width",
        positive=True,
    )
    height = _exact_nonnegative_int(
        rgb.get("height"),
        f"{label}.height",
        positive=True,
    )
    if (
        rgb.get("status") != "available"
        or rgb.get("provider_configured") is not True
        or rgb.get("color_space") != "rgb8"
        or source_pts == GST_CLOCK_TIME_NONE
        or width > MAX_CAPTURE_RGB_DIMENSION
        or height > MAX_CAPTURE_RGB_DIMENSION
        or width * height * 3 > MAX_CAPTURE_RGB_FRAME_BYTES
        or SHA256_RE.fullmatch(str(rgb.get("content_sha256") or "")) is None
    ):
        raise ValueError(f"{label} exact RGB contract drifted")
    _exact_nonnegative_int(rgb.get("source_id"), f"{label}.source_id")
    _exact_nonnegative_int(rgb.get("batch_id"), f"{label}.batch_id")
    _exact_nonnegative_int(
        rgb.get("captured_at_us"),
        f"{label}.captured_at_us",
        positive=True,
    )
    _exact_nonnegative_int(rgb.get("frame_id"), f"{label}.frame_id")


def _validate_floorplan_behavior(
    document: Mapping[str, Any],
    label: str,
    *,
    covered: Mapping[str, bytes],
    session_id: str,
    lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    inspect_started: datetime,
    inspect_finished: datetime,
) -> None:
    report = _exact_object(
        document,
        {
            "schema_version",
            "contract",
            "contract_version",
            "session_id",
            "runtime_lane",
            "runtime_instance_id",
            "runtime_run_id",
            "ok",
            "max_snapshot_age_s",
            "configured_camera_count",
            "validated_camera_count",
            "all_configured_camera_floorplans_validated",
            "exact_capture_event_camera_count",
            "cache_only_validated_camera_count",
            "cache_only_zero_mutation",
            "bev_renderer_ready",
            "bev_active_camera_count",
            "bev_inactive_ready_camera_count",
            "bev_failed_camera_count",
            "all_configured_cameras_bev_ready",
            "cameras",
            "cache_only_cameras",
            "runtime_health",
            "source_evidence",
            "errors",
        },
        label,
    )
    configured = _exact_nonnegative_int(
        report.get("configured_camera_count"),
        f"{label}.configured_camera_count",
        positive=True,
    )
    validated = _exact_nonnegative_int(
        report.get("validated_camera_count"),
        f"{label}.validated_camera_count",
        positive=True,
    )
    freshness_bound = _exact_finite_number(
        report.get("max_snapshot_age_s"), f"{label}.max_snapshot_age_s"
    )
    if freshness_bound != EXPECTED_FLOORPLAN_MAX_SNAPSHOT_AGE_S:
        raise ValueError(f"{label}.max_snapshot_age_s is not the canonical bound")
    cameras = _exact_list(report.get("cameras"), f"{label}.cameras")
    cache_cameras = _exact_list(
        report.get("cache_only_cameras"), f"{label}.cache_only_cameras"
    )
    bev_active = _exact_nonnegative_int(
        report.get("bev_active_camera_count"),
        f"{label}.bev_active_camera_count",
    )
    bev_inactive = _exact_nonnegative_int(
        report.get("bev_inactive_ready_camera_count"),
        f"{label}.bev_inactive_ready_camera_count",
    )
    bev_failed = _exact_nonnegative_int(
        report.get("bev_failed_camera_count"),
        f"{label}.bev_failed_camera_count",
    )
    if (
        validated != configured
        or len(cameras) != configured
        or len(cache_cameras) != configured
        or report.get("exact_capture_event_camera_count") != configured
        or report.get("cache_only_validated_camera_count") != configured
        or report.get("cache_only_zero_mutation") is not True
        or report.get("all_configured_camera_floorplans_validated") is not True
        or report.get("bev_renderer_ready") is not True
        or bev_active + bev_inactive != configured
        or bev_failed != 0
        or report.get("all_configured_cameras_bev_ready") is not True
    ):
        raise ValueError(f"{label} camera counts do not reconcile")
    reviewed_cameras = _reviewed_floorplan_bindings()
    if configured != len(reviewed_cameras):
        raise ValueError(
            f"{label} configured camera count differs from reviewed config"
        )
    names: set[str] = set()
    for index, raw in enumerate(cameras):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}.cameras[{index}] must be an object")
        camera = raw
        name = _exact_nonempty_string(
            camera.get("camera_id"), f"{label}.cameras[{index}].camera_id"
        )
        if name in names:
            raise ValueError(f"{label} contains a duplicate camera")
        names.add(name)
        snapshot_age = _exact_finite_number(
            camera.get("snapshot_age_s"),
            f"{label}.cameras[{index}].snapshot_age_s",
        )
        if snapshot_age < 0 or snapshot_age > freshness_bound:
            raise ValueError(
                f"{label}.cameras[{index}].snapshot_age_s exceeds the producer bound"
            )
        _exact_nonnegative_int(
            camera.get("point_count"),
            f"{label}.cameras[{index}].point_count",
            positive=True,
        )
        shape = camera.get("grid_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)  # noqa: E721
            or shape == [1, 1]
        ):
            raise ValueError(f"{label}.cameras[{index}].grid_shape is invalid")
        if (
            camera.get("served_from_cache") is not False
            or camera.get("floorplan_contract_version")
            != EXPECTED_FLOORPLAN_CONTRACT_VERSION
            or camera.get("capture_event_rgb_status") != "available"
        ):
            raise ValueError(f"{label}.cameras[{index}] floorplan contract drifted")
        _validate_capture_rgb_evidence(
            camera.get("capture_event_rgb_evidence"),
            f"{label}.cameras[{index}].capture_event_rgb_evidence",
        )
        observation = _exact_object(
            camera.get("observation_meta"),
            {
                "contract",
                "observed_definition",
                "unknown_definition",
                "observed_cells",
                "unknown_cells",
                "total_cells",
            },
            f"{label}.cameras[{index}].observation_meta",
        )
        observed_cells = _exact_nonnegative_int(
            observation.get("observed_cells"),
            f"{label}.cameras[{index}].observation_meta.observed_cells",
        )
        unknown_cells = _exact_nonnegative_int(
            observation.get("unknown_cells"),
            f"{label}.cameras[{index}].observation_meta.unknown_cells",
        )
        total_cells = _exact_nonnegative_int(
            observation.get("total_cells"),
            f"{label}.cameras[{index}].observation_meta.total_cells",
            positive=True,
        )
        if (
            observation.get("contract") != "noesis.floorplan.observation.v1"
            or observation.get("observed_definition")
            != "one_or_more_valid_projected_depth_points"
            or observation.get("unknown_definition")
            != "zero_valid_projected_depth_points_within_grid_bounds"
            or observed_cells <= 0
            or total_cells != shape[0] * shape[1]
            or observed_cells + unknown_cells != total_cells
        ):
            raise ValueError(
                f"{label}.cameras[{index}] observation contract drifted"
            )
        inferred_walkable_present = camera.get("inferred_walkable_present")
        if type(inferred_walkable_present) is not bool:  # noqa: E721
            raise ValueError(
                f"{label}.cameras[{index}].inferred_walkable_present is invalid"
            )
        layer_sha256s = camera.get("layer_sha256s")
        required_layer_digests = {
            "density",
            "observed",
            "unknown",
            "height",
            "height_agl",
            "distance",
        }
        allowed_layer_digests = required_layer_digests | {
            "gradient",
            "obstacle_height",
            "walkable",
            "inferred_walkable",
        }
        if (
            not isinstance(layer_sha256s, Mapping)
            or not required_layer_digests <= set(layer_sha256s)
            or not set(layer_sha256s) <= allowed_layer_digests
            or any(
                SHA256_RE.fullmatch(str(value or "")) is None
                for value in layer_sha256s.values()
            )
            or (
                inferred_walkable_present
                and not {"walkable", "inferred_walkable"} <= set(layer_sha256s)
            )
            or (
                not inferred_walkable_present
                and (
                    "walkable" in layer_sha256s
                    or "inferred_walkable" in layer_sha256s
                )
            )
        ):
            raise ValueError(
                f"{label}.cameras[{index}] floorplan layer evidence drifted"
            )
        for digest_key in (
            "calibration_fingerprint",
            "snapshot_content_sha256",
            "floorplan_payload_sha256",
            "capture_event_evidence_sha256",
            "fusion_evidence_sha256",
        ):
            if SHA256_RE.fullmatch(str(camera.get(digest_key) or "")) is None:
                raise ValueError(
                    f"{label}.cameras[{index}].{digest_key} is invalid"
                )
        if camera.get("calibration_fingerprint") != reviewed_cameras.get(name):
            raise ValueError(
                f"{label}.cameras[{index}] calibration fingerprint is not reviewed"
            )
    if names != set(reviewed_cameras):
        raise ValueError(f"{label} camera inventory differs from reviewed config")
    if _exact_list(report.get("errors"), f"{label}.errors"):
        raise ValueError(f"{label}.errors must be empty for a pass")

    source_metadata = _exact_object(
        report.get("source_evidence"),
        {
            "filename",
            "sha256",
            "message_count",
            "first_observed_at_us",
            "last_observed_at_us",
        },
        f"{label}.source_evidence",
    )
    source_raw = covered.get(FLOORPLAN_SOURCE_TRANSCRIPT_FILENAME)
    if (
        source_raw is None
        or source_metadata.get("filename") != FLOORPLAN_SOURCE_TRANSCRIPT_FILENAME
        or source_metadata.get("sha256") != _sha256_bytes(source_raw)
    ):
        raise ValueError(f"{label} lacks its checksum-covered source transcript")
    if len(source_raw) > 2 * 1024 * 1024:
        raise ValueError(f"{label} source transcript exceeds its byte bound")
    source = _exact_object(
        _parse_json(source_raw, f"{label} source transcript"),
        {
            "schema_version",
            "contract",
            "contract_version",
            "session_id",
            "runtime_lane",
            "runtime_instance_id",
            "runtime_run_id",
            "camera_ids",
            "max_snapshot_age_s",
            "privacy",
            "message_count",
            "messages",
        },
        f"{label} source transcript",
    )
    source_messages = _exact_list(
        source.get("messages"), f"{label} source messages"
    )
    source_camera_ids = _exact_list(
        source.get("camera_ids"), f"{label} source camera_ids"
    )
    if (
        type(source.get("schema_version")) is not int  # noqa: E721
        or source.get("schema_version") != 4
        or source.get("contract") != "noesis.ds9.floorplan-source-transcript"
        or source.get("contract_version") != 4
        or source.get("session_id") != session_id
        or source.get("runtime_lane") != lane
        or source.get("runtime_instance_id") != runtime_instance_id
        or source.get("runtime_run_id") != runtime_run_id
        or source.get("max_snapshot_age_s")
        != EXPECTED_FLOORPLAN_MAX_SNAPSHOT_AGE_S
        or source.get("message_count") != len(source_messages)
        or source.get("message_count") != source_metadata.get("message_count")
        or any(not isinstance(value, str) or not value for value in source_camera_ids)
        or source_camera_ids != [camera["camera_id"] for camera in cameras]
    ):
        raise ValueError(f"{label} source transcript envelope drifted")

    module = _load_sibling_script_module(
        "ds9_floorplan_live_gate.py",
        "_noesis_ds9_floorplan_source_for_ownership",
    )
    try:
        if source.get("privacy") != module.SOURCE_PRIVACY_POLICY:
            raise ValueError(f"{label} source transcript privacy policy drifted")
        replayed_results = module._replay_source_events(
            source_messages,
            camera_ids=source_camera_ids,
        )
        canonical_source = module._source_transcript_document(
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            camera_ids=source_camera_ids,
            max_snapshot_age_s=EXPECTED_FLOORPLAN_MAX_SNAPSHOT_AGE_S,
            events=source_messages,
        )
        if not _strict_json_equal(source, canonical_source):
            raise ValueError(f"{label} source transcript is not canonical")
        if source_raw != module._encoded_private_json(canonical_source):
            raise ValueError(
                f"{label} source transcript bytes differ from the producer encoding"
            )
        canonical_metadata = module._source_evidence_metadata(
            encoded=source_raw,
            document=source,
        )
        if not _strict_json_equal(source_metadata, canonical_metadata):
            raise ValueError(f"{label} source transcript metadata drifted")
        recomputed = module._build_report(
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            camera_ids=tuple(source_camera_ids),
            results=replayed_results["fresh_results"],
            cache_results=replayed_results["cache_results"],
            after_fresh_health=replayed_results["after_fresh_health"],
            after_cache_only_health=replayed_results[
                "after_cache_only_health"
            ],
            max_snapshot_age_s=EXPECTED_FLOORPLAN_MAX_SNAPSHOT_AGE_S,
            errors=[],
            source_evidence=canonical_metadata,
        )
    finally:
        sys.modules.pop(module.__name__, None)
    if not _strict_json_equal(report, recomputed):
        raise ValueError(f"{label} does not recompute from its source transcript")
    first_observed = _exact_nonnegative_int(
        source_metadata.get("first_observed_at_us"),
        f"{label}.source_evidence.first_observed_at_us",
        positive=True,
    )
    last_observed = _exact_nonnegative_int(
        source_metadata.get("last_observed_at_us"),
        f"{label}.source_evidence.last_observed_at_us",
        positive=True,
    )
    if (
        first_observed > last_observed
        or first_observed < int(inspect_started.timestamp() * 1_000_000)
        or last_observed > int(inspect_finished.timestamp() * 1_000_000)
    ):
        raise ValueError(f"{label} source transcript escapes the runtime window")


def _validate_world_behavior(
    document: Mapping[str, Any],
    label: str,
    *,
    covered: Mapping[str, bytes],
    launcher_dir: Path,
    session_id: str,
    lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    inspect_started: datetime,
    inspect_finished: datetime,
    expected_build_root: Path | None = None,
) -> None:
    source_metadata = _exact_object(
        document.get("source_evidence"),
        {
            "filename",
            "sha256",
            "message_count",
            "track_count",
            "first_observed_at_us",
            "last_observed_at_us",
        },
        f"{label}.source_evidence",
    )
    source_raw = covered.get(V3DT_WORLD_SOURCE_TRANSCRIPT_FILENAME)
    if (
        source_raw is None
        or source_metadata.get("filename") != V3DT_WORLD_SOURCE_TRANSCRIPT_FILENAME
        or source_metadata.get("sha256") != _sha256_bytes(source_raw)
    ):
        raise ValueError(f"{label} lacks its checksum-covered source transcript")
    if len(source_raw) > 32 * 1024 * 1024:
        raise ValueError(f"{label} source transcript exceeds its byte bound")
    source = _exact_object(
        _parse_json(source_raw, f"{label} source transcript"),
        {
            "schema_version",
            "contract",
            "contract_version",
            "session_id",
            "runtime_lane",
            "runtime_instance_id",
            "runtime_run_id",
            "privacy",
            "config_binding",
            "policy",
            "message_count",
            "track_count",
            "messages",
        },
        f"{label} source transcript",
    )
    messages = _exact_list(source.get("messages"), f"{label} source messages")
    if (
        type(source.get("schema_version")) is not int  # noqa: E721
        or source.get("schema_version") != 2
        or source.get("contract") != "noesis.ds9.v3dt-world-source-transcript"
        or source.get("contract_version") != 2
        or source.get("session_id") != session_id
        or source.get("runtime_lane") != lane
        or source.get("runtime_instance_id") != runtime_instance_id
        or source.get("runtime_run_id") != runtime_run_id
        or source.get("message_count") != len(messages)
        or source.get("message_count") != source_metadata.get("message_count")
        or source.get("track_count") != source_metadata.get("track_count")
    ):
        raise ValueError(f"{label} source transcript envelope drifted")

    module = _load_sibling_script_module(
        "v3dt_world_contract_smoke_test.py",
        "_noesis_ds9_world_source_for_ownership",
    )
    try:
        if source.get("privacy") != module.SOURCE_PRIVACY_POLICY:
            raise ValueError(f"{label} source transcript privacy policy drifted")
        if source.get("policy") != module._policy(
            EXPECTED_V3DT_MIN_BBOX3D_COVERAGE
        ):
            raise ValueError(f"{label} source transcript policy drifted")
        source_binding = module._canonical_config_binding(
            source.get("config_binding")
        )
        live_binding = module.build_config_binding(
            pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
            cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
            calibration_config=REPO_ROOT / "config/camera_calibration.json",
            alignment_config=REPO_ROOT / "config/ply_alignment.json",
            launcher_dir=launcher_dir,
            session_id=session_id,
            runtime_lane=lane,
            expected_build_root=expected_build_root,
        )
        if not _strict_json_equal(source_binding, live_binding):
            raise ValueError(f"{label} config binding differs from the sealed session")
        launch_plan_binding = source_binding["files"]["launch_plan"]
        covered_launch_plan = covered.get("launch-plan.json")
        if (
            covered_launch_plan is None
            or launch_plan_binding.get("sha256")
            != _sha256_bytes(covered_launch_plan)
        ):
            raise ValueError(f"{label} config binding does not cover launch-plan.json")
        replayed_tracks, replayed_messages = module._replay_source_messages(
            messages,
            session_id=session_id,
        )
        if source.get("track_count") != len(replayed_tracks):
            raise ValueError(f"{label} source transcript track count drifted")
        canonical_source = module._source_transcript_document(
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            config_binding=source_binding,
            min_bbox3d_coverage=EXPECTED_V3DT_MIN_BBOX3D_COVERAGE,
            messages=messages,
        )
        if not _strict_json_equal(source, canonical_source):
            raise ValueError(f"{label} source transcript is not canonical")
        if source_raw != module._encoded_private_json(canonical_source):
            raise ValueError(
                f"{label} source transcript bytes differ from the producer encoding"
            )
        canonical_metadata = module._source_evidence_metadata(
            encoded=source_raw,
            document=source,
        )
        if not _strict_json_equal(source_metadata, canonical_metadata):
            raise ValueError(f"{label} source transcript metadata drifted")
        recomputed = module.analyze_tracks(
            replayed_tracks,
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            tracking_messages=replayed_messages,
            config_binding=source_binding,
            min_bbox3d_coverage=EXPECTED_V3DT_MIN_BBOX3D_COVERAGE,
            source_evidence=canonical_metadata,
        )
    finally:
        sys.modules.pop(module.__name__, None)
    if not _strict_json_equal(document, recomputed):
        raise ValueError(f"{label} does not recompute from its source transcript")
    first_observed = _exact_nonnegative_int(
        source_metadata.get("first_observed_at_us"),
        f"{label}.source_evidence.first_observed_at_us",
        positive=True,
    )
    last_observed = _exact_nonnegative_int(
        source_metadata.get("last_observed_at_us"),
        f"{label}.source_evidence.last_observed_at_us",
        positive=True,
    )
    if (
        first_observed > last_observed
        or first_observed < int(inspect_started.timestamp() * 1_000_000)
        or last_observed > int(inspect_finished.timestamp() * 1_000_000)
    ):
        raise ValueError(f"{label} source transcript escapes the runtime window")


def _validate_semantic_behavior(
    document: Mapping[str, Any],
    label: str,
    *,
    covered: Mapping[str, bytes],
    artifact_binding: Mapping[str, Any],
    session_id: str,
    lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    inspect_started: datetime,
    inspect_finished: datetime,
    expected_identity_evidence_path: Path,
) -> None:
    _exact_object(
        document,
        {
            "schema_version", "contract", "contract_version", "session_id",
            "runtime_lane", "runtime_instance_id", "runtime_run_id", "ok",
            "status", "checks", "counts", "evidence_snapshot",
            "component_availability", "sample_cohort", "source_evidence", "errors",
        },
        label,
    )
    checks = _exact_object(
        document.get("checks"),
        {
            "occupied_scene_observed", "pipeline_stats_observed",
            "pipeline_errors_absent", "raw_embedding_vectors_absent",
            "public_track_observation_association",
            "one_to_one_track_observation_association",
            "persisted_embedding_anchor",
            "pose_component_observed", "usable_depth_component_observed",
            "backend_world_m_component_observed",
            "bounded_identity_cohort_accepted",
            "capture_and_observation_bounds_enforced",
            "acquisition_window_enforced",
            "publication_clock_within_acquisition_window",
            "artifact_fingerprint_continuity_enforced",
            "tracker_continuity_within_acquisition_window",
            "tracking_publication_contiguous_within_window_unanchored_origin",
            "tracker_lifecycle_contiguous_within_window_unanchored_origin",
            "tombstone_last_published_presence_exact_after_unanchored_origin",
        },
        f"{label}.checks",
    )
    if any(type(value) is not bool for value in checks.values()):  # noqa: E721
        raise ValueError(f"{label}.checks must contain booleans")
    if any(value is not True for value in checks.values()):
        raise ValueError(f"{label}.checks must all pass")
    counts = _exact_object(
        document.get("counts"),
        {
            "tracking_messages", "stats_messages", "person_tracks",
            "canonical_observations", "associated_samples",
            "public_provenance_links", "persisted_anchors",
            "pose_component_samples", "usable_depth_component_samples",
            "backend_world_m_component_samples", "accepted_semantic_cohorts",
            "identity_discontinuities", "fingerprint_discontinuities",
            "tracker_reuse_discontinuities", "temporal_discontinuities",
            "pipeline_error_count",
            "raw_vector_field_count",
            "acquisition_window_violations",
            "publication_clock_violations",
            "unanchored_origin_tombstones",
        },
        f"{label}.counts",
    )
    for key, value in counts.items():
        _exact_nonnegative_int(value, f"{label}.counts.{key}")
    if (
        counts["tracking_messages"] < 1
        or counts["stats_messages"] < 1
        or counts["person_tracks"] < 1
        or counts["canonical_observations"] < 1
        or counts["associated_samples"] < 1
        or counts["public_provenance_links"] < 1
        or counts["persisted_anchors"] < 1
        or counts["pose_component_samples"] < 1
        or counts["usable_depth_component_samples"] < 1
        or counts["backend_world_m_component_samples"] < 1
        or counts["accepted_semantic_cohorts"] < 1
        or counts["pipeline_error_count"] != 0
        or counts["raw_vector_field_count"] != 0
        or counts["acquisition_window_violations"] != 0
        or counts["publication_clock_violations"] != 0
    ):
        raise ValueError(f"{label} semantic counts do not prove the registered claims")
    if not (
        counts["accepted_semantic_cohorts"]
        <= counts["persisted_anchors"]
        <= counts["public_provenance_links"]
        <= counts["associated_samples"]
        and counts["pose_component_samples"] <= counts["associated_samples"]
        and counts["usable_depth_component_samples"] <= counts["associated_samples"]
        and counts["backend_world_m_component_samples"] <= counts["associated_samples"]
        and counts["canonical_observations"]
        == counts["associated_samples"]
        == counts["person_tracks"]
    ):
        raise ValueError(f"{label} semantic link counts are incoherent")
    availability = _exact_object(
        document.get("component_availability"),
        {"pose", "usable_depth", "backend_world_m"},
        f"{label}.component_availability",
    )
    if any(type(value) is not bool for value in availability.values()):  # noqa: E721
        raise ValueError(f"{label}.component_availability must contain booleans")
    if any(value is not True for value in availability.values()):
        raise ValueError(f"{label}.component_availability must all be true")
    snapshot = _exact_object(
        document.get("evidence_snapshot"),
        {
            "bytes", "rows", "sha256", "source_path", "sealed_filename",
            "session_id", "runtime_run_id", "model_sha256",
            "model_semantic_profile_sha256", "model_layer", "embedding_dimension",
            "first_sequence", "last_sequence",
        },
        f"{label}.evidence_snapshot",
    )
    _exact_nonnegative_int(snapshot.get("bytes"), f"{label}.evidence_snapshot.bytes", positive=True)
    _exact_nonnegative_int(snapshot.get("rows"), f"{label}.evidence_snapshot.rows", positive=True)
    for key in ("first_sequence", "last_sequence"):
        _exact_nonnegative_int(snapshot.get(key), f"{label}.evidence_snapshot.{key}")
    if (
        SHA256_RE.fullmatch(str(snapshot.get("sha256") or "")) is None
        or SHA256_RE.fullmatch(str(snapshot.get("model_sha256") or "")) is None
        or SHA256_RE.fullmatch(
            str(snapshot.get("model_semantic_profile_sha256") or "")
        ) is None
        or snapshot.get("source_path") != str(expected_identity_evidence_path)
        or snapshot.get("sealed_filename") != SEMANTIC_IDENTITY_SNAPSHOT_FILENAME
        or snapshot.get("session_id") != session_id
        or snapshot.get("runtime_run_id") != runtime_run_id
        or snapshot.get("model_layer") != REID_MODEL_LAYER
        or snapshot.get("embedding_dimension") != REID_EMBEDDING_DIMENSION
        or snapshot["first_sequence"] > snapshot["last_sequence"]
    ):
        raise ValueError(f"{label}.evidence_snapshot binding drifted")
    cohort = _exact_object(
        document.get("sample_cohort"),
        {
            "source_id", "camera_id", "tracker_id",
            "tracker_lifecycle_generation", "run_id", "started_at_us",
            "finished_at_us", "span_us", "capture_started_at_us",
            "capture_finished_at_us", "capture_span_us", "media_pts_span_ns",
            "maximum_span_us", "identity",
            "fingerprints", "anchor", "components",
        },
        f"{label}.sample_cohort",
    )
    for key in (
        "source_id", "tracker_id", "tracker_lifecycle_generation",
        "started_at_us", "finished_at_us", "capture_started_at_us",
        "capture_finished_at_us",
    ):
        _exact_nonnegative_int(
            cohort.get(key),
            f"{label}.sample_cohort.{key}",
            positive=key not in {"source_id", "tracker_id"},
        )
    span_us = _exact_nonnegative_int(
        cohort.get("span_us"), f"{label}.sample_cohort.span_us"
    )
    capture_span_us = _exact_nonnegative_int(
        cohort.get("capture_span_us"),
        f"{label}.sample_cohort.capture_span_us",
    )
    media_pts_span_ns = _exact_nonnegative_int(
        cohort.get("media_pts_span_ns"),
        f"{label}.sample_cohort.media_pts_span_ns",
    )
    maximum_span_us = _exact_nonnegative_int(
        cohort.get("maximum_span_us"),
        f"{label}.sample_cohort.maximum_span_us",
        positive=True,
    )
    _exact_nonempty_string(cohort.get("camera_id"), f"{label}.sample_cohort.camera_id")
    _exact_nonempty_string(cohort.get("run_id"), f"{label}.sample_cohort.run_id")
    if (
        cohort["finished_at_us"] < cohort["started_at_us"]
        or span_us != cohort["finished_at_us"] - cohort["started_at_us"]
        or cohort["capture_finished_at_us"] < cohort["capture_started_at_us"]
        or capture_span_us
        != cohort["capture_finished_at_us"] - cohort["capture_started_at_us"]
        or maximum_span_us != 1_500_000
        or span_us > maximum_span_us
        or capture_span_us > maximum_span_us
        or media_pts_span_ns > maximum_span_us * 1000
        or cohort.get("run_id") != runtime_run_id
    ):
        raise ValueError(f"{label}.sample_cohort temporal/run binding drifted")
    identity = _exact_object(
        cohort.get("identity"),
        {"state", "subject_id", "compatibility_sid", "resident_uuid", "visitor_generation"},
        f"{label}.sample_cohort.identity",
    )
    state = _exact_nonempty_string(
        identity.get("state"), f"{label}.sample_cohort.identity.state"
    )
    subject_id = _exact_nonempty_string(
        identity.get("subject_id"), f"{label}.sample_cohort.identity.subject_id"
    )
    _exact_nonnegative_int(
        identity.get("compatibility_sid"),
        f"{label}.sample_cohort.identity.compatibility_sid",
        positive=True,
    )
    if state == "resident":
        resident_uuid = _exact_nonempty_string(
            identity.get("resident_uuid"),
            f"{label}.sample_cohort.identity.resident_uuid",
        )
        if subject_id != f"resident:{resident_uuid}" or identity.get("visitor_generation") is not None:
            raise ValueError(f"{label}.sample_cohort resident identity drifted")
    elif state == "visitor":
        generation = _exact_nonnegative_int(
            identity.get("visitor_generation"),
            f"{label}.sample_cohort.identity.visitor_generation",
        )
        if (
            identity.get("resident_uuid") is not None
            or not subject_id.startswith("visitor:")
            or not subject_id.endswith(f":generation:{generation}")
        ):
            raise ValueError(f"{label}.sample_cohort visitor identity drifted")
    else:
        raise ValueError(f"{label}.sample_cohort identity must be resident or visitor")
    fingerprints = _exact_object(
        cohort.get("fingerprints"),
        {"calibration_sha256", "model_sha256", "config_sha256"},
        f"{label}.sample_cohort.fingerprints",
    )
    if any(SHA256_RE.fullmatch(str(value or "")) is None for value in fingerprints.values()):
        raise ValueError(f"{label}.sample_cohort fingerprints are invalid")
    anchor = _exact_object(
        cohort.get("anchor"),
        {
            "frame_id", "canonical_observation_id", "identity_observation_id",
            "captured_at_us", "observed_at_us", "media_pts_ns", "event_id",
            "published_at_us",
            "embedding_sequence",
            "embedding_model_sha256", "embedding_model_semantic_profile_sha256",
            "embedding_model_layer", "embedding_dimension",
        },
        f"{label}.sample_cohort.anchor",
    )
    for key in ("frame_id", "embedding_sequence"):
        _exact_nonnegative_int(anchor.get(key), f"{label}.sample_cohort.anchor.{key}")
    _exact_nonnegative_int(
        anchor.get("captured_at_us"),
        f"{label}.sample_cohort.anchor.captured_at_us",
        positive=True,
    )
    _exact_nonnegative_int(
        anchor.get("observed_at_us"),
        f"{label}.sample_cohort.anchor.observed_at_us",
        positive=True,
    )
    _exact_nonnegative_int(
        anchor.get("published_at_us"),
        f"{label}.sample_cohort.anchor.published_at_us",
        positive=True,
    )
    _exact_nonnegative_int(
        anchor.get("embedding_dimension"),
        f"{label}.sample_cohort.anchor.embedding_dimension",
        positive=True,
    )
    for key in (
        "canonical_observation_id", "identity_observation_id", "event_id",
        "embedding_model_layer",
    ):
        _exact_nonempty_string(anchor.get(key), f"{label}.sample_cohort.anchor.{key}")
    for key in (
        "event_id", "embedding_model_sha256",
        "embedding_model_semantic_profile_sha256",
    ):
        if SHA256_RE.fullmatch(str(anchor.get(key) or "")) is None:
            raise ValueError(f"{label}.sample_cohort.anchor.{key} is invalid")
    _exact_nonnegative_int(
        anchor.get("media_pts_ns"),
        f"{label}.sample_cohort.anchor.media_pts_ns",
    )
    if (
        not cohort["started_at_us"]
        <= anchor["observed_at_us"]
        <= cohort["finished_at_us"]
        or not cohort["capture_started_at_us"]
        <= anchor["captured_at_us"]
        <= cohort["capture_finished_at_us"]
    ):
        raise ValueError(f"{label}.sample_cohort anchor escapes its temporal bound")
    components = _exact_object(
        cohort.get("components"),
        {"pose", "usable_depth", "backend_world_m"},
        f"{label}.sample_cohort.components",
    )
    for name, fields in (
        (
            "pose",
            {
                "frame_id", "canonical_observation_id", "captured_at_us",
                "observed_at_us", "published_at_us", "media_pts_ns",
            },
        ),
        (
            "usable_depth",
            {
                "frame_id", "canonical_observation_id", "captured_at_us",
                "observed_at_us", "published_at_us", "media_pts_ns",
                "depth_metric",
            },
        ),
        (
            "backend_world_m",
            {
                "frame_id", "canonical_observation_id", "captured_at_us",
                "observed_at_us", "published_at_us", "media_pts_ns",
            },
        ),
    ):
        component = _exact_object(
            components.get(name), fields, f"{label}.sample_cohort.components.{name}"
        )
        _exact_nonnegative_int(
            component.get("frame_id"),
            f"{label}.sample_cohort.components.{name}.frame_id",
        )
        observed_at_us = _exact_nonnegative_int(
            component.get("observed_at_us"),
            f"{label}.sample_cohort.components.{name}.observed_at_us",
            positive=True,
        )
        captured_at_us = _exact_nonnegative_int(
            component.get("captured_at_us"),
            f"{label}.sample_cohort.components.{name}.captured_at_us",
            positive=True,
        )
        published_at_us = _exact_nonnegative_int(
            component.get("published_at_us"),
            f"{label}.sample_cohort.components.{name}.published_at_us",
            positive=True,
        )
        _exact_nonnegative_int(
            component.get("media_pts_ns"),
            f"{label}.sample_cohort.components.{name}.media_pts_ns",
        )
        _exact_nonempty_string(
            component.get("canonical_observation_id"),
            f"{label}.sample_cohort.components.{name}.canonical_observation_id",
        )
        if name == "usable_depth":
            _exact_nonempty_string(
                component.get("depth_metric"),
                f"{label}.sample_cohort.components.usable_depth.depth_metric",
            )
        if not cohort["started_at_us"] <= observed_at_us <= cohort["finished_at_us"]:
            raise ValueError(f"{label}.sample_cohort component escapes its temporal bound")
        if not (
            cohort["capture_started_at_us"]
            <= captured_at_us
            <= cohort["capture_finished_at_us"]
        ):
            raise ValueError(
                f"{label}.sample_cohort component escapes its capture bound"
            )
        if published_at_us < observed_at_us:
            raise ValueError(
                f"{label}.sample_cohort component publication precedes observation"
            )
    reviewed_model = _reviewed_reid_model_binding(lane, artifact_binding)
    if (
        anchor.get("embedding_model_sha256") != reviewed_model["model_sha256"]
        or anchor.get("embedding_model_layer") != reviewed_model["model_layer"]
        or anchor.get("embedding_dimension") != reviewed_model["embedding_dimension"]
        or snapshot.get("model_sha256") != reviewed_model["model_sha256"]
        or anchor.get("embedding_model_semantic_profile_sha256")
        != snapshot.get("model_semantic_profile_sha256")
        or anchor.get("embedding_sequence") > snapshot["last_sequence"]
    ):
        raise ValueError(f"{label}.sample_cohort model/semantic binding drifted")
    if _exact_list(document.get("errors"), f"{label}.errors"):
        raise ValueError(f"{label}.errors must be empty for a pass")

    snapshot_raw = covered.get(SEMANTIC_IDENTITY_SNAPSHOT_FILENAME)
    if (
        snapshot_raw is None
        or len(snapshot_raw) != snapshot["bytes"]
        or _sha256_bytes(snapshot_raw) != snapshot["sha256"]
    ):
        raise ValueError(f"{label} lacks its checksum-covered identity snapshot")
    source_metadata = _exact_object(
        document.get("source_evidence"),
        {
            "filename", "sha256", "message_count",
            "first_observed_at_us", "last_observed_at_us",
            "first_captured_at_us", "last_captured_at_us",
            "first_published_at_us", "last_published_at_us",
            "acquisition_started_at_us", "acquisition_finished_at_us",
            "capture_pre_window_leeway_us",
        },
        f"{label}.source_evidence",
    )
    source_raw = covered.get("semantic-observation-source.json")
    if (
        source_raw is None
        or source_metadata.get("filename") != "semantic-observation-source.json"
        or source_metadata.get("sha256") != _sha256_bytes(source_raw)
    ):
        raise ValueError(f"{label} lacks its checksum-covered semantic transcript")
    source_document = _parse_json(source_raw, f"{label} semantic transcript")
    source = _exact_object(
        source_document,
        {
            "schema_version", "contract", "contract_version", "session_id",
            "runtime_lane", "runtime_instance_id", "runtime_run_id",
            "acquisition_window", "privacy", "policy", "message_count",
            "messages",
        },
        f"{label} semantic transcript",
    )
    messages = _exact_list(source.get("messages"), f"{label} semantic messages")
    if (
        type(source.get("schema_version")) is not int  # noqa: E721
        or source.get("schema_version") != 3
        or source.get("contract")
        != "noesis.ds9.semantic-observation-source-transcript"
        or source.get("contract_version") != 3
        or source.get("session_id") != session_id
        or source.get("runtime_lane") != lane
        or source.get("runtime_instance_id") != runtime_instance_id
        or source.get("runtime_run_id") != runtime_run_id
        or source.get("message_count") != len(messages)
        or source.get("message_count") != source_metadata.get("message_count")
    ):
        raise ValueError(f"{label} semantic source envelope drifted")

    semantic_module = _load_sibling_script_module(
        "ds9_semantic_observation_smoke_test.py",
        "_noesis_ds9_semantic_source_for_ownership",
    )
    try:
        acquisition = _exact_object(
            source.get("acquisition_window"),
            {
                "started_at_us", "finished_at_us",
                "capture_pre_window_leeway_us", "observed_time_bounds",
                "capture_time_bounds",
            },
            f"{label} semantic source acquisition window",
        )
        acquisition_started_at_us = _exact_nonnegative_int(
            acquisition.get("started_at_us"),
            f"{label}.acquisition_window.started_at_us",
            positive=True,
        )
        acquisition_finished_at_us = _exact_nonnegative_int(
            acquisition.get("finished_at_us"),
            f"{label}.acquisition_window.finished_at_us",
            positive=True,
        )
        canonical_acquisition = semantic_module._acquisition_window_document(
            started_at_us=acquisition_started_at_us,
            finished_at_us=acquisition_finished_at_us,
        )
        if not _strict_json_equal(acquisition, canonical_acquisition):
            raise ValueError(
                f"{label} semantic source acquisition window drifted"
            )
        if (
            source_metadata.get("acquisition_started_at_us")
            != acquisition_started_at_us
            or source_metadata.get("acquisition_finished_at_us")
            != acquisition_finished_at_us
            or source_metadata.get("capture_pre_window_leeway_us")
            != semantic_module.SEMANTIC_CAPTURE_PRE_WINDOW_LEEWAY_US
        ):
            raise ValueError(f"{label} semantic acquisition metadata drifted")
        if source.get("privacy") != semantic_module.SOURCE_PRIVACY_POLICY:
            raise ValueError(f"{label} semantic source privacy policy drifted")
        if source.get("policy") != semantic_module._source_transcript_document(
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            acquisition_started_at_us=acquisition_started_at_us,
            acquisition_finished_at_us=acquisition_finished_at_us,
            collector=semantic_module.SemanticObservationCollector(),
        )["policy"]:
            raise ValueError(f"{label} semantic source cohort policy drifted")
        if len(source_raw) > semantic_module.MAX_SOURCE_TRANSCRIPT_BYTES:
            raise ValueError(f"{label} semantic source exceeds its byte bound")
        parsed_snapshot = semantic_module._parse_evidence_snapshot_bytes(
            snapshot_raw,
            source_path=str(expected_identity_evidence_path),
        )
        if (
            parsed_snapshot.row_count != snapshot["rows"]
            or min(parsed_snapshot.rows_by_sequence) != snapshot["first_sequence"]
            or max(parsed_snapshot.rows_by_sequence) != snapshot["last_sequence"]
        ):
            raise ValueError(f"{label} identity snapshot row/sequence binding drifted")
        collector = semantic_module.SemanticObservationCollector()
        for message in messages:
            collector.observe_payload(message)
        canonical_source = semantic_module._source_transcript_document(
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            acquisition_started_at_us=acquisition_started_at_us,
            acquisition_finished_at_us=acquisition_finished_at_us,
            collector=collector,
        )
        if not _strict_json_equal(source, canonical_source):
            raise ValueError(f"{label} semantic transcript is not canonical")
        if source_raw != semantic_module._encoded_source_transcript(canonical_source):
            raise ValueError(
                f"{label} semantic transcript bytes differ from the producer encoding"
            )
        canonical_source_metadata = semantic_module._source_evidence_metadata(
            filename="semantic-observation-source.json",
            encoded=source_raw,
            document=source,
        )
        if not _strict_json_equal(source_metadata, canonical_source_metadata):
            raise ValueError(f"{label} semantic transcript metadata drifted")
        recomputed_report = semantic_module._evaluate(
            collector,
            parsed_snapshot,
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            acquisition_started_at_us=acquisition_started_at_us,
            acquisition_finished_at_us=acquisition_finished_at_us,
            expected_model_layer=REID_MODEL_LAYER,
            expected_embedding_dimension=REID_EMBEDDING_DIMENSION,
            sealed_snapshot_filename=SEMANTIC_IDENTITY_SNAPSHOT_FILENAME,
            source_evidence=canonical_source_metadata,
        )
    finally:
        sys.modules.pop(semantic_module.__name__, None)
    if not _strict_json_equal(document, recomputed_report):
        raise ValueError(f"{label} does not exactly recompute from sealed source evidence")
    first_observed = _exact_nonnegative_int(
        source_metadata.get("first_observed_at_us"),
        f"{label}.source_evidence.first_observed_at_us",
        positive=True,
    )
    last_observed = _exact_nonnegative_int(
        source_metadata.get("last_observed_at_us"),
        f"{label}.source_evidence.last_observed_at_us",
        positive=True,
    )
    first_captured = _exact_nonnegative_int(
        source_metadata.get("first_captured_at_us"),
        f"{label}.source_evidence.first_captured_at_us",
        positive=True,
    )
    last_captured = _exact_nonnegative_int(
        source_metadata.get("last_captured_at_us"),
        f"{label}.source_evidence.last_captured_at_us",
        positive=True,
    )
    first_published = _exact_nonnegative_int(
        source_metadata.get("first_published_at_us"),
        f"{label}.source_evidence.first_published_at_us",
        positive=True,
    )
    last_published = _exact_nonnegative_int(
        source_metadata.get("last_published_at_us"),
        f"{label}.source_evidence.last_published_at_us",
        positive=True,
    )
    inspect_started_us = int(inspect_started.timestamp() * 1_000_000)
    inspect_finished_us = int(inspect_finished.timestamp() * 1_000_000)
    capture_lower_bound = max(
        1,
        acquisition_started_at_us
        - semantic_module.SEMANTIC_CAPTURE_PRE_WINDOW_LEEWAY_US,
    )
    if (
        first_observed > last_observed
        or first_captured > last_captured
        or first_published > last_published
        or not inspect_started_us
        <= acquisition_started_at_us
        <= acquisition_finished_at_us
        <= inspect_finished_us
        or first_observed < acquisition_started_at_us
        or last_observed > acquisition_finished_at_us
        or first_captured < capture_lower_bound
        or last_captured > acquisition_finished_at_us
        or last_published > acquisition_finished_at_us
        or anchor["published_at_us"] > acquisition_finished_at_us
        or cohort["started_at_us"] < acquisition_started_at_us
        or cohort["finished_at_us"] > acquisition_finished_at_us
        or cohort["capture_started_at_us"] < capture_lower_bound
        or cohort["capture_finished_at_us"] > acquisition_finished_at_us
        or any(
            row.observed_at_us < inspect_started_us
            or row.observed_at_us > inspect_finished_us
            for row in parsed_snapshot.rows_by_sequence.values()
        )
    ):
        raise ValueError(f"{label} sealed semantic evidence escapes the runtime window")


def _load_runtime_supervisor_module() -> ModuleType:
    path = Path(__file__).resolve().with_name("run_canonical_runtime_container.py")
    name = "_noesis_ds9_runtime_supervisor_for_ownership"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load canonical runtime supervisor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _validate_resource_behavior(
    document: Mapping[str, Any],
    *,
    covered: Mapping[str, bytes],
    artifact_binding: Mapping[str, Any],
    label: str,
    session_id: str,
    lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    inspect_started: datetime,
    inspect_finished: datetime,
) -> None:
    _exact_object(
        document,
        {
            "schema_version", "contract", "contract_version", "session_id", "runtime_lane",
            "runtime_instance_id", "runtime_run_id", "status", "ok", "stop_reason",
            "thresholds", "checks", "failed_checks", "validation_errors", "metrics",
            "started_at_utc", "finished_at_utc", "container_cgroup_v2_path",
            "samples_evidence", "runtime_binding",
        },
        label,
    )
    threshold_keys = {
        "minimum_duration_seconds", "sample_interval_seconds", "warmup_seconds",
        "maximum_sample_gap_seconds", "maximum_initial_sample_delay_seconds",
        "minimum_sample_count", "minimum_post_warmup_sample_count",
        "maximum_memory_slope_bytes_per_second", "maximum_memory_growth_bytes",
        "maximum_absolute_memory_bytes_exclusive",
        "maximum_gpu_process_memory_mib_exclusive", "maximum_pids_exclusive",
    }
    check_keys = {
        "samples_well_formed", "sampling_completed_without_error", "duration_stop_completed",
        "requested_duration_sufficient", "observed_duration_sufficient",
        "sample_count_sufficient", "post_warmup_sample_count_sufficient",
        "sampling_cadence_sufficient", "gpu_ownership_continuous",
        "memory_event_counters_monotonic", "oom_increment_zero", "oom_kill_increment_zero",
        "post_warmup_slope_within_limit", "post_warmup_growth_within_limit",
        "absolute_memory_below_limit",
        "gpu_process_memory_below_limit", "pids_below_limit",
        "runtime_binding_complete",
    }
    metric_keys = {
        "requested_duration_seconds", "observed_duration_seconds",
        "first_sample_elapsed_seconds", "last_sample_elapsed_seconds",
        "maximum_sample_gap_seconds", "sample_count", "post_warmup_sample_count",
        "memory_current_start_bytes", "memory_current_end_bytes", "memory_current_max_bytes",
        "memory_peak_max_bytes", "post_warmup_slope_bytes_per_second",
        "post_warmup_growth_bytes", "oom_increment", "oom_kill_increment",
        "pids_current_max", "gpu_used_memory_mib_start", "gpu_used_memory_mib_end",
        "gpu_used_memory_mib_max",
    }
    thresholds = _exact_object(document.get("thresholds"), threshold_keys, f"{label}.thresholds")
    checks = _exact_object(document.get("checks"), check_keys, f"{label}.checks")
    metrics = _exact_object(document.get("metrics"), metric_keys, f"{label}.metrics")
    if any(type(value) is not bool for value in checks.values()):  # noqa: E721
        raise ValueError(f"{label}.checks must contain exact booleans")
    if _exact_list(document.get("failed_checks"), f"{label}.failed_checks"):
        raise ValueError(f"{label}.failed_checks must be empty")
    if _exact_list(document.get("validation_errors"), f"{label}.validation_errors"):
        raise ValueError(f"{label}.validation_errors must be empty")
    for key, value in thresholds.items():
        _exact_finite_number(value, f"{label}.thresholds.{key}")
    requested_duration = _exact_finite_number(
        metrics.get("requested_duration_seconds"),
        f"{label}.metrics.requested_duration_seconds",
    )
    samples_raw = covered.get(RESOURCE_SOAK_SAMPLES_FILENAME)
    if samples_raw is None:
        raise ValueError(f"{label} lacks checksum-covered raw resource samples")
    samples_sha256 = _sha256_bytes(samples_raw)
    samples_document = _parse_json(samples_raw, f"{label} raw samples")
    samples_root = _exact_object(
        samples_document,
        {
            "schema_version", "contract", "contract_version", "session_id", "runtime_lane",
            "runtime_instance_id", "runtime_run_id", "sample_count", "samples",
            "runtime_binding",
        },
        f"{label}.samples",
    )
    if (
        samples_root.get("schema_version") != 2
        or type(samples_root.get("schema_version")) is not int  # noqa: E721
        or samples_root.get("contract") != "noesis.ds9.runtime_resource_soak.samples"
        or samples_root.get("contract_version") != 2
        or samples_root.get("session_id") != session_id
        or samples_root.get("runtime_lane") != lane
        or samples_root.get("runtime_instance_id") != runtime_instance_id
        or samples_root.get("runtime_run_id") != runtime_run_id
    ):
        raise ValueError(f"{label} raw sample envelope drifted")
    samples = _exact_list(samples_root.get("samples"), f"{label}.samples.samples")
    sample_count = _exact_nonnegative_int(
        samples_root.get("sample_count"), f"{label}.samples.sample_count", positive=True
    )
    if sample_count != len(samples):
        raise ValueError(f"{label} raw sample count does not reconcile")
    supervisor = _load_runtime_supervisor_module()
    try:
        canonical_samples_raw = supervisor._encoded_private_json(samples_root)
    finally:
        sys.modules.pop(supervisor.__name__, None)
    if samples_raw != canonical_samples_raw:
        raise ValueError(
            f"{label} raw sample bytes differ from the producer encoding"
        )
    runtime_binding = _exact_object(
        document.get("runtime_binding"),
        {
            "container_id", "runtime_image_id", "checkout_sha256",
            "realization_sha256", "primary_engine_artifact_id",
            "primary_engine_sha256", "pipeline_config",
            "pipeline_config_sha256", "cameras_config", "cameras_config_sha256",
        },
        f"{label}.runtime_binding",
    )
    if not _strict_json_equal(samples_root.get("runtime_binding"), runtime_binding):
        raise ValueError(f"{label} raw sample runtime binding drifted")
    evidence = _exact_object(
        document.get("samples_evidence"),
        {"filename", "sha256", "sample_count"},
        f"{label}.samples_evidence",
    )
    if evidence != {
        "filename": RESOURCE_SOAK_SAMPLES_FILENAME,
        "sha256": samples_sha256,
        "sample_count": sample_count,
    }:
        raise ValueError(f"{label} raw sample hash/count binding drifted")

    sample_keys = {
        "elapsed_seconds", "captured_at_utc", "memory_current_bytes", "memory_peak_bytes",
        "memory_events_low", "memory_events_high", "memory_events_max", "memory_events_oom",
        "memory_events_oom_kill", "pids_current", "gpu_compute_owner_count",
        "gpu_used_memory_mib", "gpu_largest_process_memory_mib", "gpu_owner_verified",
    }
    captured: list[datetime] = []
    for index, raw_sample in enumerate(samples):
        sample = _exact_object(raw_sample, sample_keys, f"{label}.samples[{index}]")
        _exact_finite_number(sample.get("elapsed_seconds"), f"{label}.samples[{index}].elapsed_seconds")
        captured.append(
            _parse_rfc3339_utc(
                sample.get("captured_at_utc"), f"{label}.samples[{index}].captured_at_utc"
            )
        )
        for key in sample_keys - {"elapsed_seconds", "captured_at_utc", "gpu_owner_verified"}:
            _exact_nonnegative_int(sample.get(key), f"{label}.samples[{index}].{key}")
        if type(sample.get("gpu_owner_verified")) is not bool:  # noqa: E721
            raise ValueError(f"{label}.samples[{index}].gpu_owner_verified must be boolean")
    report_started = _parse_rfc3339_utc(document.get("started_at_utc"), f"{label}.started_at_utc")
    report_finished = _parse_rfc3339_utc(document.get("finished_at_utc"), f"{label}.finished_at_utc")
    if any(right <= left for left, right in zip(captured, captured[1:])):
        raise ValueError(f"{label} sample capture timestamps do not advance")
    if not (
        inspect_started <= report_started <= captured[0] <= captured[-1] <= report_finished <= inspect_finished
    ):
        raise ValueError(f"{label} resource window escapes inspected container lifetime")
    elapsed_first = float(samples[0]["elapsed_seconds"])
    elapsed_last = float(samples[-1]["elapsed_seconds"])
    wall_span = (captured[-1] - captured[0]).total_seconds()
    elapsed_span = elapsed_last - elapsed_first
    maximum_gap = float(thresholds["maximum_sample_gap_seconds"])
    if abs(wall_span - elapsed_span) > maximum_gap:
        raise ValueError(f"{label} sample wall-clock and monotonic windows diverge")
    cgroup_path = _exact_nonempty_string(
        document.get("container_cgroup_v2_path"), f"{label}.container_cgroup_v2_path"
    )
    cgroup = Path(cgroup_path)
    if (
        not cgroup.is_absolute()
        or ".." in cgroup.parts
        or Path(os.path.abspath(cgroup_path)) != cgroup
        or cgroup == Path("/sys/fs/cgroup")
        or Path("/sys/fs/cgroup") not in cgroup.parents
    ):
        raise ValueError(f"{label} cgroup path is not canonical")

    plan = _parse_json(covered["launch-plan.json"], f"{label} launch plan")
    checkout = _parse_json(
        covered["checkout-before.json"], f"{label} checkout binding"
    )
    inspect_raw = _parse_json(
        covered["container-inspect.json"], f"{label} container binding"
    )
    if (
        not isinstance(plan, Mapping)
        or not isinstance(checkout, Mapping)
        or not isinstance(inspect_raw, list)
        or len(inspect_raw) != 1
        or not isinstance(inspect_raw[0], Mapping)
    ):
        raise ValueError(f"{label} runtime binding authorities are malformed")
    inspect = inspect_raw[0]
    container_id = str(inspect.get("Id") or "").strip().lower()
    lane_policy = RUNTIME_LANES[lane]
    engine_id = {
        "v3dt": "engine.v3dt_tracker_reid",
        "wholebody49-s": "engine.wholebody49_s_masks",
        "wholebody49-x": "engine.wholebody49_x_boxes",
    }.get(lane)
    output_sha256 = artifact_binding.get("output_sha256")
    if engine_id is None or not isinstance(output_sha256, Mapping):
        raise ValueError(f"{label} has no reviewed primary engine binding")
    image = plan.get("image")
    expected_binding = {
        "container_id": container_id,
        "runtime_image_id": str(image.get("id") if isinstance(image, Mapping) else ""),
        "checkout_sha256": str(checkout.get("sha256") or ""),
        "realization_sha256": str(artifact_binding.get("realization_sha256") or ""),
        "primary_engine_artifact_id": engine_id,
        "primary_engine_sha256": str(output_sha256.get(engine_id) or ""),
        "pipeline_config": str(lane_policy["pipeline"]),
        "pipeline_config_sha256": _sha256_bytes(
            _read_authority_bytes(
                REPO_ROOT / str(lane_policy["pipeline"]),
                f"{label} pipeline config",
            )
        ),
        "cameras_config": str(lane_policy["cameras"]),
        "cameras_config_sha256": _sha256_bytes(
            _read_authority_bytes(
                REPO_ROOT / str(lane_policy["cameras"]),
                f"{label} cameras config",
            )
        ),
    }
    if not _strict_json_equal(runtime_binding, expected_binding):
        raise ValueError(f"{label} container/checkout/artifact/engine/config binding drifted")
    accepted_cgroup_components = {container_id, f"docker-{container_id}.scope"}
    if not accepted_cgroup_components.intersection(cgroup.parts):
        raise ValueError(f"{label} cgroup path does not bind the inspected container")

    supervisor = _load_runtime_supervisor_module()
    try:
        recomputed = supervisor.evaluate_resource_soak_samples(
            samples,
            session_id=session_id,
            runtime_lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            requested_duration_seconds=requested_duration,
            stop_reason=str(document.get("stop_reason") or ""),
            runtime_binding=runtime_binding,
        )
    finally:
        sys.modules.pop(supervisor.__name__, None)
    for key in (
        "schema_version", "contract", "contract_version", "session_id", "runtime_lane",
        "runtime_instance_id", "runtime_run_id", "status", "ok", "stop_reason",
        "thresholds", "checks", "failed_checks", "validation_errors", "metrics",
        "runtime_binding",
    ):
        if not _strict_json_equal(document.get(key), recomputed.get(key)):
            raise ValueError(f"{label} resource report does not recompute exactly at {key}")


def _validate_behavior_structure(
    behavior_id: str,
    document: Mapping[str, Any],
    *,
    covered: Mapping[str, bytes],
    launcher_dir: Path | None = None,
    artifact_binding: Mapping[str, Any],
    label: str,
    session_id: str,
    lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    inspect_started: datetime,
    inspect_finished: datetime,
    expected_identity_evidence_path: Path,
    expected_build_root: Path | None = None,
) -> None:
    if behavior_id in {"wholebody49_occupied_s_v2", "wholebody49_occupied_x_v2"}:
        _validate_wholebody_behavior(
            document,
            label,
            covered=covered,
            session_id=session_id,
            lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
        )
    elif behavior_id == "wholebody49_media_decode_v1":
        _validate_wholebody_media_behavior(
            document,
            label,
            covered=covered,
            session_id=session_id,
            lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
        )
    elif behavior_id in {"reid_open_set_occupied_v1", "v3dt_identity_gate_v1"}:
        _validate_identity_behavior(
            document,
            label,
            behavior_id=behavior_id,
            covered=covered,
            artifact_binding=artifact_binding,
            session_id=session_id,
            lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
        )
    elif behavior_id == "mapanything_depth_quality_v4":
        _validate_floorplan_behavior(
            document,
            label,
            covered=covered,
            session_id=session_id,
            lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
        )
    elif behavior_id == "v3dt_world_gate_v2":
        if launcher_dir is None:
            raise ValueError(f"{label} V3DT v2 producer schema lacks launcher binding")
        _validate_world_behavior(
            document,
            label,
            covered=covered,
            launcher_dir=launcher_dir,
            session_id=session_id,
            lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
            expected_build_root=expected_build_root,
        )
    elif behavior_id == "semantic_gate_v3":
        _validate_semantic_behavior(
            document,
            label,
            covered=covered,
            artifact_binding=artifact_binding,
            session_id=session_id,
            lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
            expected_identity_evidence_path=expected_identity_evidence_path,
        )
    elif behavior_id == "runtime_resource_soak_v2":
        _validate_resource_behavior(
            document,
            covered=covered,
            artifact_binding=artifact_binding,
            label=label,
            session_id=session_id,
            lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
        )
    else:
        raise ValueError(f"no closed producer schema registered for {behavior_id}")


def _validate_behavior_documents(
    behavior_ids_raw: Any,
    *,
    covered: Mapping[str, bytes],
    launcher_dir: Path | None = None,
    artifact_binding: Mapping[str, Any],
    label: str,
    capability_id: str,
    session_id: str,
    lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    inspect_started: datetime,
    inspect_finished: datetime,
    expected_identity_evidence_path: Path,
    expected_build_root: Path | None = None,
) -> list[str]:
    behavior_ids = _sorted_unique_strings(
        behavior_ids_raw, f"{label}.behavior_documents"
    )
    capability_policy = CAPABILITY_REGISTRY.get(capability_id)
    runtime_requirements = (
        capability_policy.get("runtime_requirements")
        if isinstance(capability_policy, Mapping)
        else None
    )
    expected_ids = (
        sorted(runtime_requirements.get(lane, []))
        if isinstance(runtime_requirements, Mapping)
        else []
    )
    if behavior_ids != expected_ids:
        raise ValueError(
            f"{label}.behavior_documents differ from validator registry; "
            f"actual={behavior_ids} expected={expected_ids}"
        )
    filenames: set[str] = set()
    for behavior_id in behavior_ids:
        spec = BEHAVIOR_CONTRACTS.get(behavior_id)
        if not isinstance(spec, Mapping):
            raise ValueError(f"unregistered behavior contract: {behavior_id}")
        if lane not in spec.get("lanes", set()):
            raise ValueError(f"behavior {behavior_id} is not registered for lane {lane}")
        filename = str(spec.get("filename") or "")
        if filename in filenames or filename not in covered:
            raise ValueError(
                f"behavior {behavior_id} lacks one unique checksum-covered exact filename"
            )
        filenames.add(filename)
        document = _parse_json(covered[filename], f"behavior {behavior_id}")
        if not isinstance(document, Mapping):
            raise ValueError(f"behavior {behavior_id} root must be a mapping")
        if (
            behavior_id in CANONICAL_PRODUCER_BEHAVIOR_IDS
            and covered[filename]
            != _canonical_behavior_report_bytes(behavior_id, document)
        ):
            raise ValueError(
                f"behavior {behavior_id} report bytes differ from the producer encoding"
            )
        if (
            type(document.get("schema_version")) is not int  # noqa: E721
            or document.get("schema_version")
            != int(spec.get("schema_version", 1))
            or document.get("contract") != spec.get("contract")
            or type(document.get("contract_version")) is not int  # noqa: E721
            or document.get("contract_version")
            != int(spec.get("contract_version", 1))
            or document.get("session_id") != session_id
            or document.get("runtime_lane") != lane
            or document.get("runtime_instance_id") != runtime_instance_id
            or document.get("runtime_run_id") != runtime_run_id
        ):
            raise ValueError(
                f"behavior {behavior_id} schema/contract/session/lane binding drift"
            )
        _validate_behavior_structure(
            behavior_id,
            document,
            covered=covered,
            launcher_dir=launcher_dir,
            artifact_binding=artifact_binding,
            label=f"behavior {behavior_id}",
            session_id=session_id,
            lane=lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
            expected_identity_evidence_path=expected_identity_evidence_path,
            expected_build_root=expected_build_root,
        )
        for assertion in spec.get("assertions", ()):
            if not isinstance(assertion, Mapping):
                raise ValueError(f"behavior registry assertion is invalid: {behavior_id}")
            pointer = str(assertion.get("pointer") or "")
            actual = _json_pointer(
                document, pointer, f"behavior {behavior_id} pointer {pointer}"
            )
            expected_type = assertion.get("type")
            if not isinstance(expected_type, type) or type(actual) is not expected_type:  # noqa: E721
                raise ValueError(
                    f"behavior {behavior_id} pointer {pointer} has wrong JSON type"
                )
            operation = assertion.get("op")
            expected = assertion.get("value")
            if operation == "equals":
                passed = actual == expected
            elif operation == "minimum":
                passed = actual >= expected
            else:
                raise ValueError(
                    f"behavior registry assertion uses unsupported operation: {operation}"
                )
            if not passed:
                raise ValueError(
                    f"behavior {behavior_id} pointer {pointer} failed {operation}: "
                    f"actual={actual!r} expected={expected!r}"
                )
    return behavior_ids


def _current_checkout_summary() -> Mapping[str, Any]:
    module = _load_runtime_supervisor_module()
    try:
        snapshot = module.snapshot_checkout(REPO_ROOT).summary()
    finally:
        sys.modules.pop(module.__name__, None)
    if not isinstance(snapshot, Mapping):
        raise ValueError("canonical checkout snapshot did not return a mapping")
    return dict(snapshot)


def _parse_rfc3339_utc(value: Any, label: str) -> datetime:
    text = str(value or "")
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z",
        text,
    )
    if match is None:
        raise ValueError(f"{label} must be explicit RFC3339 UTC")
    base, fraction = match.groups()
    normalized = base + (f".{(fraction or '').ljust(6, '0')[:6]}" if fraction else "")
    try:
        parsed = datetime.fromisoformat(normalized).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid calendar timestamp") from exc
    return parsed


def _explicit_docker_root(raw: str | Path | None) -> Path:
    text = str(raw or "").strip()
    candidate_raw = Path(text).expanduser()
    if not text or not candidate_raw.is_absolute():
        raise ValueError("docker root must be supplied as an explicit absolute path")
    candidate = Path(os.path.abspath(candidate_raw))
    for component in reversed((candidate, *candidate.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"docker root ancestor is missing: {component}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"docker root contains a symlink ancestor: {component}")
    info = candidate.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid not in {0, os.getuid()}
        or stat.S_IMODE(info.st_mode) & 0o007
        or stat.S_IMODE(info.st_mode) & 0o020
    ):
        raise ValueError("docker root must be a root/current-user owned non-writable-private directory")
    return candidate


def _current_host_fingerprint(
    *,
    docker_root: Path,
    artifact_root: Path,
    runtime_root: Path,
) -> Mapping[str, Any]:
    path = REPO_ROOT / "DS9" / "scripts" / "run_canonical_runtime_container.py"
    name = "_noesis_ds9_runtime_host_fingerprint_for_ownership"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load canonical host fingerprint implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        roots = module.HostRoots(
            docker=docker_root,
            artifacts=artifact_root,
            runtime=runtime_root,
        )
        runner = module.CommandRunner()
        docker = module.inspect_docker_state(roots, runner)
        gpu = module.host_gpu_identity(runner)
        image_environment_raw = runner.run(
            module._docker_command(
                roots,
                "image",
                "inspect",
                module.IMAGE_ID,
                "--format",
                "{{json .Config.Env}}",
            ),
            timeout=15.0,
        ).stdout
        image_environment = strict_json_loads(
            image_environment_raw,
            label="runtime image Config.Env fingerprint",
        )
        if not isinstance(image_environment, list):
            raise ValueError("runtime image Config.Env fingerprint is not a list")
    finally:
        sys.modules.pop(name, None)
    return {
        "gpu": dict(gpu),
        "docker": {
            "daemon_id": docker.daemon_id,
            "root": docker.docker_root,
            "default_runtime": docker.default_runtime,
            "networks": dict(docker.networks),
        },
        "image": {
            "reference": module.IMAGE_REF,
            "id": docker.image_id,
            "base_digest": docker.base_digest,
            "parent_build_image_reference": docker.parent_build_image_reference,
            "parent_build_image_id": docker.parent_build_image_id,
            "parent_rootfs_layer_count": docker.parent_rootfs_layer_count,
            "runtime_rootfs_layer_count": docker.runtime_rootfs_layer_count,
            "parent_rootfs_sha256": docker.parent_rootfs_sha256,
            "runtime_rootfs_sha256": docker.runtime_rootfs_sha256,
        },
        "image_environment": [str(value) for value in image_environment],
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _runtime_arguments(lane: Mapping[str, Any]) -> list[str]:
    return [
        "DS9/noesis/ds9_runtime.py",
        "--pipeline-config",
        str(lane["pipeline"]),
        "--cameras-config",
        str(lane["cameras"]),
        "--pgie-profile",
        str(lane["pgie_profile"]),
        "--size",
        str(lane["model_size"]),
        "--tracking-mode",
        str(lane["tracking_mode"]),
        "--ws-host",
        "127.0.0.1",
        "--ws-port",
        "6008",
        "--rest-host",
        "127.0.0.1",
        "--rest-port",
        "8080",
        "--enable-rest",
        "--storage-base",
        "/var/lib/noesis/depth",
        "--log-level",
        "INFO",
    ]


def _validate_secret_mount_source(path: Path, label: str) -> None:
    if not path.is_absolute() or len(path.parts) < 2:
        raise ValueError(f"{label} secret source must be an absolute file")
    _anchored_file_content(
        Path("/"),
        Path(*path.parts[1:]),
        f"{label} secret source",
        directory_policy="authority",
        file_policy="private",
        max_bytes=64 * 1024,
    )


def _normalized_hostconfig_mounts(rows: Any) -> dict[str, tuple[Path, bool]]:
    if not isinstance(rows, list):
        raise ValueError("container HostConfig.Mounts must be a list")
    normalized: dict[str, tuple[Path, bool]] = {}
    sources: set[Path] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("container HostConfig.Mounts contains a non-object")
        allowed_keys = {"Type", "Source", "Target", "BindOptions", "ReadOnly"}
        target = str(row.get("Target") or "")
        source_raw = str(row.get("Source") or "")
        source = Path(source_raw)
        bind_options = row.get("BindOptions")
        if (
            row.get("Type") != "bind"
            or not target.startswith("/")
            or not source.is_absolute()
            or not isinstance(bind_options, Mapping)
            or bind_options.get("Propagation") != "rprivate"
            or target in normalized
            or source in sources
            or type(row.get("ReadOnly", False)) is not bool  # noqa: E721
            or not set(row).issubset(allowed_keys)
            or not {"Type", "Source", "Target", "BindOptions"}.issubset(row)
        ):
            raise ValueError("container HostConfig.Mounts has an unsafe/duplicate bind")
        normalized[target] = (Path(os.path.abspath(source)), bool(row.get("ReadOnly", False)))
        sources.add(Path(os.path.abspath(source)))
    return normalized


def _normalized_inspect_mounts(rows: Any) -> dict[str, tuple[Path, bool]]:
    if not isinstance(rows, list):
        raise ValueError("container Mounts must be a list")
    normalized: dict[str, tuple[Path, bool]] = {}
    sources: set[Path] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("container Mounts contains a non-object")
        target = str(row.get("Destination") or "")
        source = Path(str(row.get("Source") or ""))
        if (
            row.get("Type") != "bind"
            or not target.startswith("/")
            or not source.is_absolute()
            or row.get("Propagation") != "rprivate"
            or target in normalized
            or source in sources
            or type(row.get("RW")) is not bool  # noqa: E721
            or set(row)
            != {"Type", "Source", "Destination", "Mode", "RW", "Propagation"}
            or row.get("Mode") != ""
        ):
            raise ValueError("container Mounts has an unsafe/duplicate bind")
        normalized[target] = (Path(os.path.abspath(source)), not bool(row.get("RW")))
        sources.add(Path(os.path.abspath(source)))
    return normalized


def _required_runtime_environment(
    *,
    session_id: str,
    lane: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "HOME": "/var/lib/noesis/state/home",
        "XDG_CACHE_HOME": "/var/lib/noesis/build/cache",
        "XDG_RUNTIME_DIR": "/var/lib/noesis/build/xdg-runtime",
        "CUDA_CACHE_PATH": "/var/lib/noesis/build/cuda-cache",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NOESIS_DS9_ARTIFACT_ROOT": "/opt/noesis/ds9-artifacts",
        "NOESIS_BUILD_DIR": "/var/lib/noesis/build",
        "NOESIS_DEV_CONSOLE_LAUNCH_DIR": "/var/lib/noesis/build",
        "NOESIS_CAMERA_SECRETS_FILE": "/run/noesis-secrets/camera_sources.json",
        "NOESIS_MAPANYTHING_API_KEY_FILE": "/run/noesis-secrets/mapanything_rpc.key",
        "NOESIS_INTERNAL_AUTH_MODE": "required",
        "NOESIS_INTERNAL_AUTH_TOKEN_FILE": "/run/noesis-secrets/gateway-token",
        "NOESIS_WORLD_JOURNAL_PATH": "/var/lib/noesis/state/world_ds9.sqlite3",
        "NOESIS_IDENTITY_V2_STORE": "/var/lib/noesis/state/household/identity_v2.sqlite3",
        "NOESIS_IDENTITY_V2_EVIDENCE_PATH": "/var/lib/noesis/evidence/identity_v2.jsonl",
        "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": session_id,
        "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME": "ds9",
        "NOESIS_REID_ALIAS_FILE": "/var/lib/noesis/state/reid_aliases.json",
        "NOESIS_REID_SID_POOL_FILE": "/var/lib/noesis/state/sid_pool.json",
        "NOESIS_REID_GALLERY_FILE": "/var/lib/noesis/state/reid_gallery.npz",
        "NOESIS_ANALYTICS_CONFIG": "/var/lib/noesis/state/analytics/nvdsanalytics.yaml",
        "NOESIS_ANALYTICS_EXCLUDE_CONFIG": "/var/lib/noesis/state/analytics/config_nvdsanalytics_exclude.ini",
        "NOESIS_SCENE_STORE_PATH": "/var/lib/noesis/state/scene_releases.sqlite3",
        "NOESIS_VIRTUAL_TWIN_ROOT": "/var/lib/noesis/state/virtual_twin",
        "NOESIS_CALIBRATION_AUDIT_DIR": "/var/lib/noesis/evidence/calibration",
        "NOESIS_V3DT_DIAG_DIR": "/var/lib/noesis/evidence/v3dt",
        "NOESIS_V3DT_DIAG_SESSION": session_id,
        "NOESIS_PGIE_PROFILE": str(lane["pgie_profile"]),
        "NOESIS_TRACKING_MODE": str(lane["tracking_mode"]),
        "NOESIS_MOSAIC_RTSP_ENABLED": "0",
        "NOESIS_MOSAIC_WEBRTC_ENABLED": "1",
        "NOESIS_REID_ENABLED": "1",
        "NOESIS_SHUTDOWN_GRACE_SECONDS": "75",
    }


def _validate_container_environment(
    env_rows: Any,
    *,
    session_id: str,
    lane: Mapping[str, Any],
    image_environment: Any,
) -> None:
    if not isinstance(env_rows, list):
        raise ValueError("container Config.Env must be a list")
    env: dict[str, str] = {}
    for raw in env_rows:
        key, separator, value = str(raw).partition("=")
        if not separator or not key or key in env:
            raise ValueError("container Config.Env contains a malformed or duplicate key")
        env[key] = value
    required = _required_runtime_environment(session_id=session_id, lane=lane)
    for key, expected in required.items():
        if env.get(key) != expected:
            raise ValueError(f"container environment contract drift: {key}")
    forbidden_exact = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MENON_OWNER_PASSWORD",
        "MENON_OWNER_BOOTSTRAP_TOKEN",
        "HS_PASS",
        "NOESIS_REST_CORS_ALLOW_ALL",
        "NOESIS_DS9_ALLOW_PYDS_COMPAT",
        "NOESIS_NATIVE_EXT_DIR",
    }
    forbidden = sorted(
        key
        for key in env
        if key in forbidden_exact
        or key.startswith("NOESIS_DS8")
        or (
            key.startswith("NOESIS_")
            and any(token in key for token in ("FALLBACK", "LEGACY", "DEGRADED"))
        )
    )
    if forbidden:
        raise ValueError(f"container inherited forbidden override/fallback env: {forbidden}")
    if not isinstance(image_environment, list):
        raise ValueError("current runtime image environment fingerprint is missing")
    base: dict[str, str] = {}
    for raw in image_environment:
        key, separator, value = str(raw).partition("=")
        if not separator or not key or key in base:
            raise ValueError("runtime image environment fingerprint is malformed")
        base[key] = value
    expected = {**base, **required}
    if env != expected:
        raise ValueError(
            "container Config.Env differs from the exact runtime-image plus canonical override set; "
            f"missing={sorted(set(expected) - set(env))} "
            f"extra={sorted(set(env) - set(expected))}"
        )


def _validate_container_mounts(
    inspect: Mapping[str, Any],
    inspect_host: Mapping[str, Any],
    *,
    artifact_root: Path,
    runtime_root: Path,
    session_id: str,
) -> None:
    host_mounts = _normalized_hostconfig_mounts(inspect_host.get("Mounts"))
    inspect_mounts = _normalized_inspect_mounts(inspect.get("Mounts"))
    if host_mounts != inspect_mounts:
        raise ValueError("container HostConfig/inspect mount replays disagree")
    expected_known = {
        "/workspace": (REPO_ROOT.absolute(), True),
        "/opt/noesis/ds9-artifacts": (artifact_root, True),
        "/var/lib/noesis/build": (runtime_root / "build" / session_id, False),
        "/var/lib/noesis/state": (runtime_root / "state" / session_id, False),
        "/var/lib/noesis/depth": (runtime_root / "depth" / session_id, False),
        "/var/lib/noesis/evidence": (
            runtime_root / "evidence" / session_id / "runtime",
            False,
        ),
        "/var/lib/noesis/state/analytics": (
            runtime_root / "persistent" / "analytics",
            False,
        ),
    }
    secret_destinations = {
        "/run/noesis-secrets/camera_sources.json",
        "/run/noesis-secrets/mapanything_rpc.key",
        "/run/noesis-secrets/gateway-token",
    }
    if set(host_mounts) != set(expected_known) | secret_destinations:
        raise ValueError("container has missing or extra bind-mount destinations")
    for destination, expected in expected_known.items():
        if host_mounts.get(destination) != expected:
            raise ValueError(f"container bind mount drift: {destination}")
        source, read_only = expected
        if not read_only and destination != "/workspace":
            relative = source.relative_to(runtime_root)
            _anchored_directory_entries(
                runtime_root,
                relative,
                f"container writable mount {destination}",
                directory_policy="private",
            )
    secret_sources: list[Path] = []
    for destination in sorted(secret_destinations):
        source, read_only = host_mounts[destination]
        if not read_only:
            raise ValueError(f"container secret mount is writable: {destination}")
        if any(
            source == root or source in root.parents or root in source.parents
            for root in (REPO_ROOT.absolute(), artifact_root, runtime_root)
        ):
            raise ValueError(f"container secret source overlaps an owned runtime root: {destination}")
        _validate_secret_mount_source(source, destination)
        secret_sources.append(source)
    for index, source in enumerate(secret_sources):
        for other in secret_sources[index + 1 :]:
            if source == other or source in other.parents or other in source.parents:
                raise ValueError("container secret mount sources overlap")


def _validate_ordered_shutdown_log(raw: bytes) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("runtime log must be UTF-8") from exc
    folded = text.casefold()
    found_failures = [
        signature
        for signature in SHUTDOWN_FAILURE_SIGNATURES
        if signature.casefold() in folded
    ]
    severity = [
        token
        for token in ("ERROR", "CRITICAL")
        if re.search(rf"(^|[\s:]){token}([\s:])", text, flags=re.MULTILINE)
    ]
    positions = [text.rfind(marker) for marker in SHUTDOWN_MARKERS]
    ordered = bool(
        min(positions) >= 0
        and positions[0] < positions[1]
        and positions[0] < positions[2]
        and max(positions[1], positions[2]) < positions[3] < positions[4]
    )
    if found_failures or severity or not ordered:
        raise ValueError(
            "runtime log does not prove exact ordered EOS shutdown; "
            f"failures={found_failures} severity={severity} positions={positions}"
        )


def _mapping_at(payload: Any, dotted: str, label: str) -> Mapping[str, Any]:
    present, value = _get_dotted(payload, dotted)
    if not present or not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing mapping {dotted}")
    return value


def _value_at(payload: Any, dotted: str, label: str) -> Any:
    present, value = _get_dotted(payload, dotted)
    if not present:
        raise ValueError(f"{label} is missing {dotted}")
    return value


def _validate_runtime_session_evidence(
    evidence: Mapping[str, Any],
    *,
    artifact_root: str | Path | None,
    runtime_root: str | Path | None,
    docker_root: str | Path | None,
    label: str,
    capability_id: str,
) -> dict[str, Any]:
    _require_exact_keys(evidence, RUNTIME_SESSION_KEYS, label)
    lane_name = str(evidence.get("lane") or "").strip()
    lane = RUNTIME_LANES.get(lane_name)
    if lane is None:
        raise ValueError(f"{label}.lane is not a reviewed canonical lane: {lane_name!r}")
    canonical_source_ids = list(_reviewed_lane_source_ids(lane_name))
    profiles = _profiles(evidence.get("profiles"), f"{label}.profiles")
    if profiles != sorted(lane["profiles"]):
        raise ValueError(
            f"{label}.profiles differ from reviewed lane {lane_name}: {lane['profiles']}"
        )
    session_id = str(evidence.get("session_id") or "").strip()
    if SESSION_RE.fullmatch(session_id) is None:
        raise ValueError(f"{label}.session_id is unsafe")
    if evidence.get("checksum_manifest") != "launcher/SHA256SUMS":
        raise ValueError(f"{label}.checksum_manifest must be launcher/SHA256SUMS")
    checksum_sha256 = _require_sha256(
        evidence.get("checksum_sha256"), f"{label}.checksum_sha256"
    )

    binding_evidence = {key: evidence[key] for key in ASSET_REALIZATION_KEYS}
    binding = _validate_asset_realization_evidence(
        binding_evidence,
        artifact_root=artifact_root,
        label=f"{label}.artifacts",
    )
    runtime = _external_root(
        runtime_root,
        "runtime root",
        accepted_modes=frozenset({0o700}),
    )
    artifact = _external_root(
        artifact_root,
        "artifact root",
        accepted_modes=frozenset({0o700, 0o750}),
    )
    if runtime == artifact or runtime in artifact.parents or artifact in runtime.parents:
        raise ValueError("artifact and runtime roots must be disjoint")
    docker = _explicit_docker_root(docker_root)
    if any(
        left == right or left in right.parents or right in left.parents
        for left, right in ((docker, runtime), (docker, artifact))
    ):
        raise ValueError("docker, artifact, and runtime roots must be disjoint")
    launcher = _private_path(
        runtime,
        Path("evidence") / session_id / "launcher",
        "runtime launcher evidence",
    )
    _require_private_directory(launcher, "runtime launcher evidence")
    launcher_relative = Path("evidence") / session_id / "launcher"
    covered = _load_checksum_covered_files(
        runtime,
        launcher_relative,
        expected_manifest_sha256=checksum_sha256,
    )

    plan = _parse_json(covered["launch-plan.json"], "runtime launch plan")
    checkout_before = _parse_json(
        covered["checkout-before.json"], "runtime checkout-before"
    )
    summary = _parse_json(covered["summary.json"], "runtime summary")
    runtime_identity = _parse_json(
        covered["runtime-identity.json"], "supervisor runtime identity"
    )
    inspect_raw = _parse_json(
        covered["container-inspect.json"], "runtime container inspection"
    )
    if not isinstance(plan, Mapping) or not isinstance(checkout_before, Mapping):
        raise ValueError("runtime launch-plan/checkout evidence roots must be mappings")
    if not isinstance(summary, Mapping) or not isinstance(runtime_identity, Mapping):
        raise ValueError("runtime summary root must be a mapping")
    if "appliance_deployment" in plan:
        raise ValueError(
            "runtime-session promotion accepts only an isolated ephemeral canary; "
            "appliance-run lifecycle evidence is a distinct, non-promotable contract"
        )
    if (
        not isinstance(inspect_raw, list)
        or len(inspect_raw) != 1
        or not isinstance(inspect_raw[0], Mapping)
    ):
        raise ValueError("runtime container inspection must contain exactly one object")
    inspect = inspect_raw[0]

    if (
        type(plan.get("schema_version")) is not int  # noqa: E721
        or plan.get("schema_version") != 1
        or plan.get("contract") != RUNTIME_CONTRACT
        or plan.get("mode") != "plan"
        or plan.get("session_id") != session_id
        or plan.get("runtime_lane") != lane_name
        or plan.get("ready_for_explicit_run") is not True
        or plan.get("blockers") != []
        or plan.get("unavailable_ports") != []
        or plan.get("docker_run_invoked") is not False
    ):
        raise ValueError("runtime launch plan does not prove an exact blocker-free lane")
    plan_created = _parse_rfc3339_utc(
        plan.get("created_at_utc"), "runtime launch-plan created_at_utc"
    )
    host_roots = _mapping_at(plan, "host_roots", "runtime launch plan")
    if host_roots != {"artifacts": str(artifact), "runtime": str(runtime)}:
        raise ValueError("runtime launch plan host roots differ from explicit roots")
    if plan.get("owners") != {
        "runtime_containers": [],
        "runtime_processes": [],
        "gpu_compute": [],
    }:
        raise ValueError("runtime launch plan does not prove exclusive initial ownership")
    plan_artifact_lock = _mapping_at(
        plan, "artifact_transaction_lock", "runtime launch plan"
    )
    if (
        plan_artifact_lock.get("available") is not True
        or plan_artifact_lock.get("path")
        != str(artifact / ".noesis-ds9-artifact-transaction.lock")
    ):
        raise ValueError("runtime launch plan artifact lock was unavailable or unbound")
    canonical_runtime = _mapping_at(plan, "canonical_runtime", "runtime launch plan")
    session_paths = _mapping_at(plan, "session_paths", "runtime launch plan")
    expected_session_paths = {
        "build": str(runtime / "build" / session_id),
        "state": str(runtime / "state" / session_id),
        "depth": str(runtime / "depth" / session_id),
        "runtime_evidence": str(runtime / "evidence" / session_id / "runtime"),
        "launcher_evidence": str(launcher),
    }
    if not _strict_json_equal(session_paths, expected_session_paths):
        raise ValueError("runtime launch plan session paths are not exact")
    exact_runtime_fields = {
        "lane": lane_name,
        "pipeline": lane["pipeline"],
        "cameras": lane["cameras"],
        "pgie_profile": lane["pgie_profile"],
        "model_size": lane["model_size"],
        "tracking_mode": lane["tracking_mode"],
        "artifact_profiles": lane["profiles"],
        "required_engine_ids": binding["artifact_ids"],
        "source_ids": canonical_source_ids,
        "ports": CANONICAL_PORTS,
        "endpoints": CANONICAL_ENDPOINTS,
    }
    for key, expected in exact_runtime_fields.items():
        if canonical_runtime.get(key) != expected:
            raise ValueError(f"runtime launch plan canonical_runtime.{key} drift")

    identity = _exact_object(
        runtime_identity,
        {
            "schema_version", "contract", "contract_version", "session_id", "runtime_lane",
            "runtime_instance_id", "runtime_run_id", "health_generated_at_us",
            "observed_at_utc", "endpoints",
        },
        "supervisor runtime identity",
    )
    runtime_instance_id = str(identity.get("runtime_instance_id") or "")
    runtime_run_id = str(identity.get("runtime_run_id") or "")
    if (
        type(identity.get("schema_version")) is not int  # noqa: E721
        or identity.get("schema_version") != 1
        or identity.get("contract") != RUNTIME_IDENTITY_CONTRACT
        or identity.get("contract_version") != 1
        or identity.get("session_id") != session_id
        or identity.get("runtime_lane") != lane_name
        or identity.get("endpoints") != CANONICAL_ENDPOINTS
        or RUNTIME_ID_RE.fullmatch(runtime_instance_id) is None
        or RUNTIME_ID_RE.fullmatch(runtime_run_id) is None
        or type(identity.get("health_generated_at_us")) is not int  # noqa: E721
        or identity.get("health_generated_at_us") <= 0
    ):
        raise ValueError("supervisor runtime identity binding drift")

    readiness = _mapping_at(plan, "artifact_readiness", "runtime launch plan")
    exact_readiness = {
        "ok": True,
        "lane": lane_name,
        "profiles": lane["profiles"],
        "realization_sha256": binding["realization_sha256"],
        "base_manifest_sha256": binding["base_manifest_sha256"],
        "engine_source_contracts_sha256": binding["source_contracts_sha256"],
        "required_engine_ids": binding["artifact_ids"],
    }
    for key, expected in exact_readiness.items():
        if readiness.get(key) != expected:
            raise ValueError(f"runtime launch-plan artifact binding drift: {key}")
    profile_results = readiness.get("profile_results")
    if not isinstance(profile_results, Mapping) or set(profile_results) != set(profiles):
        raise ValueError("runtime launch-plan profile result coverage drift")
    for profile in profiles:
        result = profile_results[profile]
        if (
            not isinstance(result, Mapping)
            or result.get("ok") is not True
            or result.get("profile") != profile
            or result.get("errors") != []
            or result.get("blockers") != []
            or result.get("realization_sha256") != binding["realization_sha256"]
        ):
            raise ValueError(f"runtime launch-plan profile is incomplete: {profile}")

    recorded_fingerprint = {
        "gpu": dict(
            _mapping_at(
                readiness,
                "host_compatibility.host_gpu",
                "runtime launch-plan artifact readiness",
            )
        ),
        "docker": dict(
            _mapping_at(plan, "secondary_docker", "runtime launch plan")
        ),
        "image": dict(_mapping_at(plan, "image", "runtime launch plan")),
    }
    if recorded_fingerprint["docker"].get("root") != str(docker / "data"):
        raise ValueError("runtime launch-plan Docker data root differs from explicit root")
    observed_host = dict(
        _current_host_fingerprint(
            docker_root=docker,
            artifact_root=artifact,
            runtime_root=runtime,
        )
    )
    observed_fingerprint = {
        key: observed_host.get(key) for key in ("gpu", "docker", "image")
    }

    recorded_snapshot = plan.get("checkout_snapshot")
    summary_checkout = _mapping_at(summary, "checkout", "runtime summary")
    if (
        not isinstance(recorded_snapshot, Mapping)
        or dict(recorded_snapshot) != dict(checkout_before)
        or summary_checkout.get("unchanged") is not True
        or summary_checkout.get("added") != []
        or summary_checkout.get("changed") != []
        or summary_checkout.get("removed") != []
        or summary_checkout.get("differences_truncated") is not False
        or summary_checkout.get("before") != recorded_snapshot
        or summary_checkout.get("after") != recorded_snapshot
    ):
        raise ValueError("runtime checkout before/after evidence is not exact and unchanged")
    live_checkout = dict(_current_checkout_summary())
    if dict(recorded_snapshot) != live_checkout:
        raise ValueError(
            "runtime session checkout/native snapshot is historical and cannot prove current behavior"
        )

    if (
        type(summary.get("schema_version")) is not int  # noqa: E721
        or summary.get("schema_version") != 1
        or summary.get("contract") != RUNTIME_CONTRACT
        or summary.get("mode") != "run"
        or summary.get("session_id") != session_id
        or summary.get("runtime_lane") != lane_name
        or summary.get("ok") is not True
        or summary.get("error") is not None
        or summary.get("ports_closed_after") is not True
        or summary.get("evidence_directory") != str(launcher)
        or not _strict_json_equal(summary.get("runtime_identity"), identity)
    ):
        raise ValueError("runtime summary does not prove an exact successful lane")
    summary_started = _parse_rfc3339_utc(
        summary.get("started_at_utc"), "runtime summary started_at_utc"
    )
    summary_finished = _parse_rfc3339_utc(
        summary.get("finished_at_utc"), "runtime summary finished_at_utc"
    )
    summary_artifact_lock = _mapping_at(
        summary, "artifact_transaction_lock", "runtime summary"
    )
    if (
        summary_artifact_lock.get("path")
        != str(artifact / ".noesis-ds9-artifact-transaction.lock")
        or summary_artifact_lock.get("held_through_readiness_and_gpu_confirmation")
        is not True
    ):
        raise ValueError("runtime summary does not prove artifact-lock ownership")
    gpu_confirmation = _mapping_at(
        summary, "gpu_owner_confirmation", "runtime summary"
    )
    compute_owners = gpu_confirmation.get("compute_owners")
    if (
        not isinstance(compute_owners, list)
        or not compute_owners
        or any(not str(owner).strip() for owner in compute_owners)
        or not isinstance(gpu_confirmation.get("container_init_pid"), int)
        or gpu_confirmation.get("container_init_pid") <= 1
    ):
        raise ValueError("runtime summary does not prove exact container GPU ownership")
    container = _mapping_at(summary, "container", "runtime summary")
    expected_container_name = f"noesis-ds9-runtime-{session_id}"
    exact_container = {
        "name": expected_container_name,
        "started": True,
        "term_sent": True,
        "exit_code": 0,
        "forced_removal": False,
        "removed": True,
        "absent_after": True,
    }
    for key, expected in exact_container.items():
        actual = container.get(key)
        if type(actual) is not type(expected) or actual != expected:  # noqa: E721
            raise ValueError(f"runtime summary container.{key} drift")
    if container.get("stop_reason") != "duration_complete":
        raise ValueError("runtime summary did not complete its authorized duration")
    expected_lifecycle = {
        "ok": True,
        "missing_markers": [],
        "ordered": True,
        "failure_signatures": [],
        "severity_signatures": [],
    }
    if not _strict_json_equal(summary.get("shutdown_lifecycle"), expected_lifecycle):
        raise ValueError("runtime summary does not prove exact ordered EOS lifecycle")
    _validate_ordered_shutdown_log(covered["runtime.log"])

    container_id = str(inspect.get("Id") or "").strip().lower()
    image_id = str(_value_at(plan, "image.id", "runtime launch plan") or "")
    if (
        not image_id.startswith("sha256:")
        or SHA256_RE.fullmatch(image_id.removeprefix("sha256:")) is None
        or image_id != binding["runtime_image_id"]
    ):
        raise ValueError("runtime launch-plan image ID is invalid")
    inspect_state = _mapping_at(inspect, "State", "container inspection")
    inspect_config = _mapping_at(inspect, "Config", "container inspection")
    inspect_host = _mapping_at(inspect, "HostConfig", "container inspection")
    _validate_container_environment(
        inspect_config.get("Env"),
        session_id=session_id,
        lane=lane,
        image_environment=observed_host.get("image_environment"),
    )
    _validate_container_mounts(
        inspect,
        inspect_host,
        artifact_root=artifact,
        runtime_root=runtime,
        session_id=session_id,
    )
    labels = inspect_config.get("Labels")
    expected_labels = {
        "com.noesis.role": "ds9-runtime",
        "com.noesis.session": session_id,
        "com.noesis.runtime-lane": lane_name,
    }
    device_requests = inspect_host.get("DeviceRequests")
    exact_gpu_request = bool(
        isinstance(device_requests, list)
        and len(device_requests) == 1
        and isinstance(device_requests[0], Mapping)
        and set(device_requests[0])
        == {"Driver", "Count", "DeviceIDs", "Capabilities", "Options"}
        and device_requests[0].get("Driver") == ""
        and type(device_requests[0].get("Count")) is int  # noqa: E721
        and device_requests[0].get("Count") == 0
        and device_requests[0].get("DeviceIDs") == ["0"]
        and device_requests[0].get("Capabilities") == [["gpu"]]
        and device_requests[0].get("Options") == {}
    )
    expected_tmpfs = {
        "/tmp": (
            "rw,exec,nosuid,nodev,size=2147483648,"
            f"uid={os.geteuid()},gid={os.getegid()},mode=0700"
        ),
        "/run/noesis-secrets": (
            "rw,noexec,nosuid,nodev,size=65536,"
            f"uid={os.geteuid()},gid={os.getegid()},mode=0700"
        ),
    }
    restart_policy = inspect_host.get("RestartPolicy")
    restart_exact = bool(
        isinstance(restart_policy, Mapping)
        and set(restart_policy) == {"Name", "MaximumRetryCount"}
        and restart_policy.get("Name") == "no"
        and type(restart_policy.get("MaximumRetryCount")) is int  # noqa: E721
        and restart_policy.get("MaximumRetryCount") == 0
    )
    ulimits = inspect_host.get("Ulimits")
    ulimit_exact = bool(
        isinstance(ulimits, list)
        and len(ulimits) == 1
        and isinstance(ulimits[0], Mapping)
        and set(ulimits[0]) == {"Name", "Hard", "Soft"}
        and ulimits[0].get("Name") == "nofile"
        and type(ulimits[0].get("Hard")) is int  # noqa: E721
        and type(ulimits[0].get("Soft")) is int  # noqa: E721
        and ulimits[0].get("Hard") == 65536
        and ulimits[0].get("Soft") == 65536
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or inspect.get("Image") != image_id
        or inspect.get("Name") != f"/{expected_container_name}"
        or inspect_state.get("Status") != "exited"
        or inspect_state.get("Running") is not False
        or type(inspect_state.get("ExitCode")) is not int  # noqa: E721
        or inspect_state.get("ExitCode") != 0
        or inspect_state.get("OOMKilled") is not False
        or inspect_state.get("Error") != ""
        or inspect_config.get("User") != f"{os.geteuid()}:{os.getegid()}"
        or inspect_config.get("WorkingDir") != "/workspace"
        or inspect_config.get("Entrypoint") != ["python3"]
        or inspect_config.get("Cmd") != _runtime_arguments(lane)
        or not isinstance(labels, Mapping)
        or any(labels.get(key) != value for key, value in expected_labels.items())
        or inspect_host.get("ReadonlyRootfs") is not True
        or inspect_host.get("Privileged") is not False
        or inspect_host.get("CapAdd") not in (None, [])
        or {str(value).upper() for value in inspect_host.get("CapDrop") or []}
        != {"ALL"}
        or inspect_host.get("SecurityOpt") != EXPECTED_SECURITY_OPTIONS
        or inspect.get("AppArmorProfile") != EXPECTED_APPARMOR_PROFILE
        or any(
            field not in inspect_host or inspect_host[field] != expected
            for field, expected in EXPECTED_SENSITIVE_HOST_DEFAULTS.items()
        )
        or any(field in inspect_host for field in EXPECTED_ABSENT_SENSITIVE_HOST_FIELDS)
        or inspect_host.get("Devices") != []
        or inspect_host.get("Binds") is not None
        or inspect_host.get("PidMode") != ""
        or inspect_host.get("UsernsMode") != ""
        or inspect_host.get("ReadonlyPaths") != EXPECTED_READONLY_PATHS
        or inspect_host.get("MaskedPaths") != EXPECTED_MASKED_PATHS
        or type(inspect_host.get("Memory")) is not int  # noqa: E721
        or inspect_host.get("Memory") != 26 * 1024 * 1024 * 1024
        or type(inspect_host.get("MemorySwap")) is not int  # noqa: E721
        or inspect_host.get("MemorySwap") != 26 * 1024 * 1024 * 1024
        or inspect_host.get("MemorySwappiness") not in (None, 0)
        or type(inspect_host.get("PidsLimit")) is not int  # noqa: E721
        or inspect_host.get("PidsLimit") != 4096
        or inspect_host.get("Tmpfs") != expected_tmpfs
        or inspect_host.get("Init") is not True
        or not restart_exact
        or not ulimit_exact
        or inspect_host.get("LogConfig")
        != {
            "Type": "json-file",
            "Config": {"max-file": "2", "max-size": "50m"},
        }
        or inspect_host.get("NetworkMode") != "host"
        or inspect_host.get("IpcMode") != "host"
        or inspect_host.get("Runtime") != "nvidia"
        or not exact_gpu_request
    ):
        raise ValueError("container inspection identity/lane/exit/port command drift")

    inspect_started = _parse_rfc3339_utc(
        inspect_state.get("StartedAt"), "container inspection State.StartedAt"
    )
    inspect_finished = _parse_rfc3339_utc(
        inspect_state.get("FinishedAt"), "container inspection State.FinishedAt"
    )
    identity_observed = _parse_rfc3339_utc(
        identity.get("observed_at_utc"), "supervisor runtime identity observed_at_utc"
    )
    if not (
        plan_created <= summary_started <= inspect_started < inspect_finished <= summary_finished
    ):
        raise ValueError("runtime plan/summary/container timestamps are not monotonically bound")
    if summary_started - plan_created > MAX_PLAN_TO_SUMMARY_START:
        raise ValueError("runtime plan-to-summary-start slack exceeds the exact bound")
    if inspect_started - summary_started > MAX_SUMMARY_TO_INSPECT_START:
        raise ValueError("runtime summary-to-container-start slack exceeds the exact bound")
    if summary_finished - inspect_finished > MAX_INSPECT_TO_SUMMARY_FINISH:
        raise ValueError("runtime container-finish-to-summary slack exceeds the exact bound")
    if not inspect_started <= identity_observed <= inspect_finished:
        raise ValueError("supervisor runtime identity was not observed during the container run")
    duration_seconds = (inspect_finished - inspect_started).total_seconds()
    capability_policy = CAPABILITY_REGISTRY[capability_id]
    minimum_duration = max(
        float(lane["minimum_duration_seconds"]),
        float(capability_policy.get("minimum_duration_seconds", 0.0)),
    )
    if duration_seconds < minimum_duration:
        raise ValueError(
            f"runtime lane duration is insufficient: {duration_seconds:.3f} < {minimum_duration:.3f}"
        )

    current_time = _utc_now()
    if current_time.tzinfo is None or current_time.utcoffset() != timedelta(0):
        raise ValueError("current evidence comparison time must be explicit UTC")
    live_reasons: list[str] = []
    age = current_time - inspect_finished
    if age < timedelta(minutes=-5):
        raise ValueError("runtime evidence finished implausibly in the future")
    if age > MAX_LIVE_EVIDENCE_AGE:
        live_reasons.append("runtime evidence exceeds the live-readiness maximum age")
    if not _strict_json_equal(observed_fingerprint, recorded_fingerprint):
        live_reasons.append("current GPU/driver/Docker/image fingerprint differs")

    behavior_ids = _validate_behavior_documents(
        evidence.get("behavior_documents"),
        covered=covered,
        launcher_dir=launcher,
        artifact_binding=binding,
        label=label,
        capability_id=capability_id,
        session_id=session_id,
        lane=lane_name,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        inspect_started=inspect_started,
        inspect_finished=inspect_finished,
        expected_identity_evidence_path=(
            runtime / "evidence" / session_id / "runtime" / "identity_v2.jsonl"
        ),
        expected_build_root=runtime / "build" / session_id,
    )
    supervisor = _load_runtime_supervisor_module()
    artifact_lock_descriptor: int | None = None
    try:
        artifact_lock_descriptor, artifact_lock_path = (
            supervisor.acquire_artifact_transaction_lock(artifact)
        )
        if artifact_lock_path != (
            artifact / supervisor.ARTIFACT_TRANSACTION_LOCK_FILENAME
        ):
            raise ValueError("runtime final validation acquired the wrong artifact lock")
        final_covered = _load_checksum_covered_files(
            runtime,
            launcher_relative,
            expected_manifest_sha256=checksum_sha256,
        )
        if final_covered != covered:
            raise ValueError("runtime launcher evidence changed during validation")
        pre_checkout_binding = _validate_asset_realization_evidence(
            binding_evidence,
            artifact_root=artifact,
            label=f"{label}.artifacts final CAS before checkout",
        )
        if not _strict_json_equal(pre_checkout_binding, binding):
            raise ValueError("runtime artifact binding changed during final validation")
        terminal_checkout = dict(_current_checkout_summary())
        if terminal_checkout != dict(recorded_snapshot):
            raise ValueError("runtime checkout changed during final validation")
        post_checkout_binding = _validate_asset_realization_evidence(
            binding_evidence,
            artifact_root=artifact,
            label=f"{label}.artifacts final CAS after checkout",
        )
        if not _strict_json_equal(post_checkout_binding, pre_checkout_binding):
            raise ValueError(
                "runtime artifact binding changed during final checkout validation"
            )
        post_artifact_covered = _load_checksum_covered_files(
            runtime,
            launcher_relative,
            expected_manifest_sha256=checksum_sha256,
        )
        if post_artifact_covered != final_covered:
            raise ValueError(
                "runtime launcher evidence changed during terminal artifact validation"
            )
        post_artifact_checkout = dict(_current_checkout_summary())
        if post_artifact_checkout != terminal_checkout:
            raise ValueError(
                "runtime checkout changed during terminal artifact validation"
            )
        result = {
            **post_checkout_binding,
            "session_id": session_id,
            "lane": lane_name,
            "runtime_instance_id": runtime_instance_id,
            "runtime_run_id": runtime_run_id,
            "checksum_sha256": checksum_sha256,
            "container_id": container_id,
            "checkout_sha256": post_artifact_checkout.get("sha256"),
            "behavior_documents": behavior_ids,
            "duration_seconds": duration_seconds,
            "finished_at_utc": summary.get("finished_at_utc"),
            "live_ready": not live_reasons,
            "live_readiness_blockers": live_reasons,
        }
        return result
    finally:
        try:
            supervisor.release_artifact_transaction_lock(
                artifact_lock_descriptor
            )
        finally:
            sys.modules.pop(supervisor.__name__, None)


def _iter_source_files(scan_roots: Iterable[Any], suffixes: set[str], errors: list[str]) -> Iterable[Path]:
    for raw in scan_roots:
        try:
            root = _repo_path(raw)
        except ValueError as exc:
            errors.append(f"policy.scan_roots: {exc}")
            continue
        if not root.exists():
            errors.append(f"policy.scan_roots: path is missing: {root.relative_to(REPO_ROOT)}")
            continue
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in suffixes:
                yield path


def _actual_duplicates() -> set[str]:
    root_dir = REPO_ROOT / "noesis"
    ds9_dir = REPO_ROOT / "DS9" / "noesis"

    def relative_files(base: Path) -> set[str]:
        return {
            str(path.relative_to(base))
            for path in base.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }

    return relative_files(root_dir) & relative_files(ds9_dir)


def _absolute_imports(path: Path, errors: list[str]) -> set[str]:
    """Return absolute import targets, including ``from package import child``."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as exc:
        errors.append(f"unable to parse scanned Python source {path.relative_to(REPO_ROOT)}: {exc}")
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            base = str(node.module or "").strip()
            if base:
                imports.add(base)
            for alias in node.names:
                imports.add(f"{base}.{alias.name}" if base else alias.name)
    return imports


def _registry_readiness_blocker(
    capability_id: str,
    evidence: Mapping[str, Mapping[str, Any]],
) -> str | None:
    policy = CAPABILITY_REGISTRY[capability_id]
    minimum_tier = str(policy["minimum_tier"])
    available_tier = max(
        (
            EVIDENCE_TIER[evidence_type]
            for evidence_type, subjects in evidence.items()
            if subjects
        ),
        default=0,
    )
    if available_tier < EVIDENCE_TIER[minimum_tier]:
        return f"validator requires {minimum_tier} evidence"
    required_profiles = set(policy.get("required_profiles", []) or [])
    if required_profiles:
        observed_profiles = {
            profile
            for detail in evidence.get("asset_realization", {}).values()
            for profile in detail.get("profiles", [])
        }
        if not required_profiles.issubset(observed_profiles):
            return (
                "validator-required realized profiles are missing: "
                + ", ".join(sorted(required_profiles - observed_profiles))
            )
    runtime_requirements = policy.get("runtime_requirements")
    if isinstance(runtime_requirements, Mapping):
        runtime_details = list(evidence.get("runtime_session", {}).values())
        for lane, behavior_ids in runtime_requirements.items():
            expected = sorted(str(value) for value in behavior_ids)
            matching = [
                detail
                for detail in runtime_details
                if detail.get("lane") == lane
                and detail.get("behavior_documents") == expected
                and detail.get("live_ready") is True
            ]
            if not matching:
                return (
                    f"validator requires fresh live lane {lane} with registered behavior "
                    + ", ".join(expected)
                )
    return None


def _validate_promotion_event_binding(
    event: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    detail: Mapping[str, Any],
    capability_id: str,
    evidence_type: str,
    subject: str,
    checkout_sha256: str,
) -> None:
    if (
        event.get("capability_id") != capability_id
        or event.get("evidence_type") != evidence_type
        or event.get("subject") != subject
        or not _strict_json_equal(event.get("selector"), selector)
        or event.get("checkout_sha256") != checkout_sha256
    ):
        raise ValueError("promotion event selector/key/checkout binding drifted")
    expected_artifact_binding = {
        key: detail[key] for key in ASSET_REALIZATION_KEYS
    }
    if not _strict_json_equal(
        event.get("artifact_binding"), expected_artifact_binding
    ):
        raise ValueError("promotion event artifact binding drifted")
    if evidence_type == "runtime_session":
        expected_runtime_binding = {
            "session_id": detail.get("session_id"),
            "lane": detail.get("lane"),
            "runtime_instance_id": detail.get("runtime_instance_id"),
            "runtime_run_id": detail.get("runtime_run_id"),
        }
        if (
            not _strict_json_equal(
                event.get("runtime_binding"), expected_runtime_binding
            )
            or event.get("evidence_sha256s_digest")
            != detail.get("checksum_sha256")
        ):
            raise ValueError("promotion event runtime/evidence binding drifted")
    elif (
        event.get("runtime_binding") is not None
        or event.get("evidence_sha256s_digest") is not None
    ):
        raise ValueError("asset promotion carries runtime-only bindings")


def validate_matrix(
    matrix: Mapping[str, Any],
    *,
    require_parity: bool = False,
    artifact_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    docker_root: str | Path | None = None,
    promoted_evidence: Mapping[str, Any] | None = None,
    promotion_events: Mapping[str, Any] | None = None,
    allow_inline_dynamic_evidence: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    evidence_counts = {evidence_type: 0 for evidence_type in sorted(EVIDENCE_TYPES)}
    evidence_details: dict[str, dict[str, dict[str, Any]]] = {}
    external_evidence = promoted_evidence or {}
    external_events = promotion_events or {}
    if not isinstance(external_evidence, Mapping) or not isinstance(
        external_events, Mapping
    ):
        return {
            "ok": False,
            "errors": ["external promotion evidence/events must be mappings"],
            "warnings": [],
            "blockers": [],
        }

    schema_version = matrix.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:  # noqa: E721
        errors.append("schema_version must be the exact integer 2")
    if matrix.get("matrix_id") != EXPECTED_MATRIX_ID:
        errors.append(f"matrix_id must be {EXPECTED_MATRIX_ID!r}")
    policy = matrix.get("policy")
    if not isinstance(policy, Mapping):
        return {"ok": False, "errors": ["policy must be a mapping"], "warnings": [], "blockers": []}

    declared_classes = set(policy.get("allowed_module_classifications", []) or [])
    declared_statuses = set(policy.get("allowed_capability_statuses", []) or [])
    declared_strict = set(policy.get("strict_blocking_statuses", []) or [])
    if declared_classes != ALLOWED_MODULE_CLASSIFICATIONS:
        errors.append("policy.allowed_module_classifications differs from validator authority")
    if declared_statuses != ALLOWED_CAPABILITY_STATUSES:
        errors.append("policy.allowed_capability_statuses differs from validator authority")
    if declared_strict != STRICT_BLOCKING_STATUSES:
        errors.append("policy.strict_blocking_statuses differs from validator authority")
    allowed_classes = ALLOWED_MODULE_CLASSIFICATIONS
    allowed_statuses = ALLOWED_CAPABILITY_STATUSES
    strict_statuses = STRICT_BLOCKING_STATUSES

    modules = matrix.get("modules")
    if not isinstance(modules, list) or not modules:
        errors.append("modules must be a non-empty list")
        modules = []
    module_ids: set[str] = set()
    declared_duplicates: set[str] = set()
    for index, module in enumerate(modules):
        label = f"modules[{index}]"
        if not isinstance(module, Mapping):
            errors.append(f"{label} must be a mapping")
            continue
        module_id = str(module.get("id", "")).strip()
        if not module_id:
            errors.append(f"{label}.id is required")
        elif module_id in module_ids:
            errors.append(f"duplicate module id: {module_id}")
        module_ids.add(module_id)
        classification = str(module.get("classification", "")).strip()
        if classification not in allowed_classes:
            errors.append(f"{label} has unsupported classification {classification!r}")
            continue
        if not str(module.get("rationale", "") or module.get("reason", "")).strip():
            errors.append(f"{label} must explain its ownership rationale")

        required_paths: list[str] = []
        if classification == "shared_single_source":
            required_paths = ["owner_path"]
            if module.get("ds8_path") or module.get("ds9_path"):
                errors.append(f"{label}: shared_single_source must declare only owner_path")
        elif classification == "sdk_adapter":
            adapter_paths = [key for key in ("ds8_path", "ds9_path") if module.get(key)]
            if len(adapter_paths) != 1:
                errors.append(f"{label}: sdk_adapter requires exactly one of ds8_path or ds9_path")
            else:
                required_paths = adapter_paths
        elif classification == "duplicated_pending_convergence":
            required_paths = ["ds8_path", "ds9_path"]
            if module.get("owner_path"):
                errors.append(f"{label}: duplicated modules must not declare owner_path")
            if not str(module.get("convergence_target", "")).strip():
                errors.append(f"{label}: duplicated modules require convergence_target")
            ds8_raw = str(module.get("ds8_path", ""))
            ds9_raw = str(module.get("ds9_path", ""))
            prefix_a = "noesis/"
            prefix_b = "DS9/noesis/"
            if ds8_raw.startswith(prefix_a) and ds9_raw.startswith(prefix_b):
                rel_a = ds8_raw[len(prefix_a) :]
                rel_b = ds9_raw[len(prefix_b) :]
                if rel_a != rel_b:
                    errors.append(f"{label}: duplicated paths do not describe the same relative module")
                else:
                    declared_duplicates.add(rel_a)
            else:
                errors.append(f"{label}: duplicated paths must live below noesis/ and DS9/noesis/")

        for key in required_paths:
            try:
                path = _repo_path(module.get(key))
            except ValueError as exc:
                errors.append(f"{label}.{key}: {exc}")
                continue
            if not path.exists():
                errors.append(f"{label}.{key} is missing: {path.relative_to(REPO_ROOT)}")

    actual_duplicates = _actual_duplicates()
    for relative in sorted(actual_duplicates - declared_duplicates):
        errors.append(f"unclassified DS8/DS9 duplicate: {relative}")
    for relative in sorted(declared_duplicates - actual_duplicates):
        errors.append(f"declared duplicate no longer exists on both sides: {relative}")

    forbidden = policy.get("forbidden_ds9_imports", []) or []
    suffixes = {str(value) for value in policy.get("scan_suffixes", [".py"]) or [".py"]}
    source_files = list(_iter_source_files(policy.get("scan_roots", []), suffixes, errors))
    source_text = {path: path.read_text(encoding="utf-8", errors="replace") for path in source_files}
    source_imports = {path: _absolute_imports(path, errors) for path in source_files if path.suffix == ".py"}
    for rule in forbidden:
        if not isinstance(rule, Mapping):
            errors.append("policy.forbidden_ds9_imports entries must be mappings")
            continue
        rule_id = str(rule.get("id", "<unnamed>"))
        module = str(rule.get("module", "") or "").strip()
        regex_raw = str(rule.get("regex", "") or "").strip()
        pattern = None
        if regex_raw:
            try:
                pattern = re.compile(regex_raw)
            except re.error as exc:
                errors.append(f"forbidden import rule {rule_id} has invalid regex: {exc}")
                continue
        if not module and pattern is None:
            errors.append(f"forbidden import rule {rule_id} requires module or regex")
            continue
        for path in source_files:
            imported = source_imports.get(path, set())
            module_match = bool(module) and any(
                candidate == module or candidate.startswith(f"{module}.") for candidate in imported
            )
            regex_match = pattern.search(source_text[path]) is not None if pattern is not None else False
            if module_match or regex_match:
                errors.append(f"forbidden DS9 dependency {rule_id}: {path.relative_to(REPO_ROOT)}")

    capabilities = matrix.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        errors.append("capabilities must be a non-empty list")
        capabilities = []
    capability_ids: set[str] = set()
    status_counts: dict[str, int] = {}
    capability_rows: dict[str, Mapping[str, Any]] = {}
    declared_blocker_reasons: dict[str, str] = {}
    for index, capability in enumerate(capabilities):
        label = f"capabilities[{index}]"
        if not isinstance(capability, Mapping):
            errors.append(f"{label} must be a mapping")
            continue
        capability_id = str(capability.get("id", "")).strip()
        if not capability_id:
            errors.append(f"{label}.id is required")
        elif capability_id in capability_ids:
            errors.append(f"duplicate capability id: {capability_id}")
        capability_ids.add(capability_id)
        if capability_id:
            capability_rows[capability_id] = capability
        status = str(capability.get("status", "")).strip()
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in allowed_statuses:
            errors.append(f"{label} has unsupported status {status!r}")
        if status in strict_statuses:
            gap = capability.get("gap")
            if not isinstance(gap, Mapping):
                errors.append(f"{label}: {status} capability requires a gap record")
            else:
                for key in ("owner", "reason", "exit_criteria"):
                    if not str(gap.get(key, "")).strip():
                        errors.append(f"{label}.gap.{key} is required")
            declared_blocker_reasons[capability_id] = (
                str(gap.get("reason")) if isinstance(gap, Mapping) else status
            )

        declared_evidence = capability.get("evidence")
        if not isinstance(declared_evidence, Mapping) or not declared_evidence:
            errors.append(f"{label}.evidence must be a non-empty mapping")
        else:
            unknown_types = set(declared_evidence) - EVIDENCE_TYPES
            if unknown_types:
                errors.append(
                    f"{label}.evidence has unsupported typed evidence: {sorted(unknown_types)}"
                )
            inline_dynamic = (
                set(declared_evidence) & EVIDENCE_TYPES
            ) - {"repository_source"}
            if inline_dynamic and not allow_inline_dynamic_evidence:
                errors.append(
                    f"{label}.evidence must keep dynamic selectors in the external registry: "
                    f"{sorted(inline_dynamic)}"
                )
            evidence: dict[str, Any] = {
                key: copy.deepcopy(value)
                for key, value in declared_evidence.items()
                if allow_inline_dynamic_evidence or key == "repository_source"
            }
            promoted_for_capability = external_evidence.get(capability_id, {})
            if promoted_for_capability and not isinstance(
                promoted_for_capability, Mapping
            ):
                errors.append(
                    f"external promotions for {capability_id} must be a mapping"
                )
                promoted_for_capability = {}
            for evidence_type, promoted_subjects in promoted_for_capability.items():
                if evidence_type not in {"asset_realization", "runtime_session"}:
                    errors.append(
                        f"external promotion for {capability_id} has unsupported type "
                        f"{evidence_type!r}"
                    )
                    continue
                if not isinstance(promoted_subjects, Mapping) or not promoted_subjects:
                    errors.append(
                        f"external promotion {capability_id}.{evidence_type} "
                        "must contain subjects"
                    )
                    continue
                destination = evidence.setdefault(evidence_type, {})
                if not isinstance(destination, dict):
                    errors.append(
                        f"external promotion conflicts with {capability_id}.{evidence_type}"
                    )
                    continue
                for subject, selector in promoted_subjects.items():
                    if subject in destination:
                        errors.append(
                            f"external promotion conflicts with inline selector: "
                            f"{capability_id}.{evidence_type}.{subject}"
                        )
                        continue
                    destination[subject] = copy.deepcopy(selector)
            for evidence_type in sorted(set(evidence) & EVIDENCE_TYPES):
                subjects = evidence[evidence_type]
                type_label = f"{label}.evidence.{evidence_type}"
                if not isinstance(subjects, Mapping) or not subjects:
                    errors.append(f"{type_label} must be a non-empty subject mapping")
                    continue
                for subject, subject_evidence in subjects.items():
                    subject_label = f"{capability_id}.{evidence_type}.{subject}"
                    if not str(subject).strip():
                        errors.append(f"{type_label} contains an empty subject")
                        continue
                    if not isinstance(subject_evidence, Mapping):
                        errors.append(f"{subject_label} must be a mapping")
                        continue
                    evidence_counts[evidence_type] += 1
                    try:
                        if evidence_type == "repository_source":
                            unexpected = set(subject_evidence) - REPOSITORY_SOURCE_KEYS
                            if unexpected or "path" not in subject_evidence:
                                raise ValueError(
                                    "repository_source keys differ from the typed contract; "
                                    f"unexpected={sorted(unexpected)} path_required=True"
                                )
                            source_errors: list[str] = []
                            _validate_evidence(subject_label, subject_evidence, source_errors)
                            if source_errors:
                                raise ValueError("; ".join(source_errors))
                            detail = {
                                "path": str(subject_evidence.get("path")),
                                "validated": True,
                            }
                        elif evidence_type == "asset_realization":
                            detail = _validate_asset_realization_evidence(
                                subject_evidence,
                                artifact_root=artifact_root,
                                label=subject_label,
                            )
                        else:
                            detail = _validate_runtime_session_evidence(
                                subject_evidence,
                                artifact_root=artifact_root,
                                runtime_root=runtime_root,
                                docker_root=docker_root,
                                label=subject_label,
                                capability_id=capability_id,
                            )
                    except Exception as exc:
                        errors.append(f"{subject_label}: {type(exc).__name__}: {exc}")
                        continue
                    evidence_details.setdefault(capability_id, {}).setdefault(
                        evidence_type, {}
                    )[str(subject)] = detail
                    if (
                        evidence_type == "runtime_session"
                        and detail.get("live_ready") is not True
                    ):
                        errors.append(
                            f"{subject_label}: runtime selector is not current "
                            "live-readiness evidence: "
                            + ", ".join(
                                str(value)
                                for value in detail.get(
                                    "live_readiness_blockers", []
                                )
                            )
                        )

    expected_capability_ids = set(CAPABILITY_REGISTRY)
    if capability_ids != expected_capability_ids:
        errors.append(
            "capability IDs differ from validator registry; "
            f"missing={sorted(expected_capability_ids - capability_ids)} "
            f"unexpected={sorted(capability_ids - expected_capability_ids)}"
        )
    unknown_promoted_capabilities = set(external_evidence) - expected_capability_ids
    unknown_event_capabilities = set(external_events) - expected_capability_ids
    if unknown_promoted_capabilities or unknown_event_capabilities:
        errors.append(
            "external promotion capability IDs differ from the validator registry; "
            f"evidence={sorted(unknown_promoted_capabilities)} "
            f"events={sorted(unknown_event_capabilities)}"
        )

    promoted_keys: set[tuple[str, str, str]] = set()
    event_keys: set[tuple[str, str, str]] = set()
    for capability_id, typed in external_evidence.items():
        if not isinstance(typed, Mapping):
            continue
        for evidence_type, subjects in typed.items():
            if not isinstance(subjects, Mapping):
                continue
            promoted_keys.update(
                (str(capability_id), str(evidence_type), str(subject))
                for subject in subjects
            )
    for capability_id, typed in external_events.items():
        if not isinstance(typed, Mapping):
            continue
        for evidence_type, subjects in typed.items():
            if not isinstance(subjects, Mapping):
                continue
            event_keys.update(
                (str(capability_id), str(evidence_type), str(subject))
                for subject in subjects
            )
    if promoted_keys != event_keys:
        errors.append(
            "external promotion selectors/events differ; "
            f"missing_events={sorted(promoted_keys - event_keys)} "
            f"orphan_events={sorted(event_keys - promoted_keys)}"
        )
    if promoted_keys == event_keys and promoted_keys:
        try:
            current_checkout = dict(_current_checkout_summary())
            checkout_sha256 = _require_sha256(
                current_checkout.get("sha256"), "current checkout sha256"
            )
        except Exception as exc:
            errors.append(f"external promotion checkout binding failed: {exc}")
        else:
            for capability_id, evidence_type, subject in sorted(promoted_keys):
                selector = external_evidence[capability_id][evidence_type][subject]
                event = external_events[capability_id][evidence_type][subject]
                detail = (
                    evidence_details.get(capability_id, {})
                    .get(evidence_type, {})
                    .get(subject)
                )
                if not isinstance(selector, Mapping) or not isinstance(event, Mapping):
                    errors.append(
                        f"external promotion {capability_id}.{evidence_type}.{subject} "
                        "must use object selector/event records"
                    )
                    continue
                if not isinstance(detail, Mapping):
                    errors.append(
                        f"external promotion {capability_id}.{evidence_type}.{subject} "
                        "did not produce validated typed evidence"
                    )
                    continue
                try:
                    _validate_promotion_event_binding(
                        event,
                        selector=selector,
                        detail=detail,
                        capability_id=capability_id,
                        evidence_type=evidence_type,
                        subject=subject,
                        checkout_sha256=checkout_sha256,
                    )
                except Exception as exc:
                    errors.append(
                        f"external promotion {capability_id}.{evidence_type}.{subject}: "
                        f"{type(exc).__name__}: {exc}"
                    )
    registry_blockers: dict[str, str] = {}
    for capability_id in sorted(expected_capability_ids & capability_ids):
        row = capability_rows[capability_id]
        expected_surface = CAPABILITY_REGISTRY[capability_id]["surface"]
        if row.get("surface") != expected_surface:
            errors.append(
                f"{capability_id}: surface differs from validator registry: {expected_surface}"
            )
        reason = _registry_readiness_blocker(
            capability_id, evidence_details.get(capability_id, {})
        )
        if reason is not None:
            registry_blockers[capability_id] = reason

    blocker_ids = set(declared_blocker_reasons) | set(registry_blockers)
    blockers = []
    for capability_id in sorted(blocker_ids):
        reason = (
            declared_blocker_reasons[capability_id]
            if capability_id in declared_blocker_reasons
            else registry_blockers[capability_id]
        )
        blockers.append(f"{capability_id}: {reason}")

    if require_parity and blockers:
        errors.extend(f"strict parity blocker: {blocker}" for blocker in blockers)
    elif blockers:
        warnings.append(
            f"{len(blockers)} effective parity blocker(s); "
            "rerun with --require-parity to fail on them"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "blockers": blockers,
        "module_count": len(modules),
        "duplicate_count": len(actual_duplicates),
        "capability_count": len(capabilities),
        "status_counts": status_counts,
        "evidence_counts": evidence_counts,
        "evidence_details": evidence_details,
    }


def _active_registry_promotions(
    snapshot: Any,
    *,
    matrix_id: str,
    matrix_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    promoted: dict[str, Any] = {}
    events: dict[str, Any] = {}
    for (capability_id, evidence_type, subject), event in snapshot.active_for_matrix(
        matrix_id, matrix_sha256
    ).items():
        selector = event.get("selector")
        if not isinstance(selector, Mapping):
            raise ValueError("active registry promotion lacks an object selector")
        promoted.setdefault(capability_id, {}).setdefault(evidence_type, {})[
            subject
        ] = dict(selector)
        events.setdefault(capability_id, {}).setdefault(evidence_type, {})[
            subject
        ] = dict(event)
    return promoted, events


def _load_active_registry_promotions(
    runtime_root: Path,
    *,
    matrix_id: str,
    matrix_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    module = _load_sibling_script_module(
        "runtime_ownership_registry.py",
        "_noesis_ds9_runtime_ownership_registry_for_validator",
    )
    try:
        snapshot = module.read_registry(runtime_root)
        return _active_registry_promotions(
            snapshot,
            matrix_id=matrix_id,
            matrix_sha256=matrix_sha256,
        )
    finally:
        sys.modules.pop(module.__name__, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--require-parity",
        action="store_true",
        help="Fail on every static contract blocker or missing current promotion.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Explicit external DS9 artifact root required by typed realized/session evidence.",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help="Explicit private DS9 runtime root required by typed session evidence.",
    )
    parser.add_argument(
        "--docker-root",
        type=Path,
        help="Explicit isolated DS9 Docker root required by typed session evidence.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the complete result as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        matrix, matrix_raw = _load_yaml_with_raw(args.matrix.absolute())
        matrix_sha256 = _sha256_bytes(matrix_raw)
        promoted_evidence: Mapping[str, Any] = {}
        promotion_events: Mapping[str, Any] = {}
        if args.runtime_root is not None:
            promoted_evidence, promotion_events = _load_active_registry_promotions(
                args.runtime_root.expanduser().absolute(),
                matrix_id=str(matrix.get("matrix_id") or ""),
                matrix_sha256=matrix_sha256,
            )
        result = validate_matrix(
            matrix,
            require_parity=bool(args.require_parity),
            artifact_root=args.artifact_root,
            runtime_root=args.runtime_root,
            docker_root=args.docker_root,
            promoted_evidence=promoted_evidence,
            promotion_events=promotion_events,
        )
    except Exception as exc:
        result = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"], "warnings": [], "blockers": []}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for message in result.get("errors", []):
            print(f"[FAIL] {message}", file=sys.stderr)
        for message in result.get("warnings", []):
            print(f"[WARN] {message}")
        if result.get("ok"):
            print(
                "[OK] DS8/DS9 ownership matrix is structurally valid "
                f"({result.get('module_count')} modules, {result.get('capability_count')} capabilities, "
                f"{result.get('duplicate_count')} classified duplicates)"
            )
        if result.get("blockers"):
            print("[INFO] Effective parity blockers:")
            for blocker in result["blockers"]:
                print(f"  - {blocker}")
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
