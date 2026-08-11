#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.contracts.health import (  # noqa: E402
    CapabilityHealth,
    CapabilityStatus,
)
from noesis_core.private_paths import read_private_file  # noqa: E402
from noesis_core.strict_json import strict_json_loads  # noqa: E402
from noesis_core.v3dt_validation import (  # noqa: E402
    V3DTBBoxTimeoutContract,
    V3DT_BBOX_DEFAULT_ATTEMPTS,
)
from DS9.scripts import ds9_floorplan_live_gate as floorplan_gate  # noqa: E402
from DS9.scripts import ds9_identity_shadow_live_gate as identity_gate  # noqa: E402
from DS9.scripts import ds9_semantic_observation_smoke_test as semantic_gate  # noqa: E402
from DS9.scripts import v3dt_world_contract_smoke_test as v3dt_world_gate  # noqa: E402
from DS9.scripts import (  # noqa: E402
    wholebody49_media_decode_gate as wholebody_media_gate,
)
from DS9.scripts import (  # noqa: E402
    wholebody49_occupied_scene_smoke_test as wholebody_gate,
)

CANONICAL_PGIE_PROFILE = "yolo26"
CANONICAL_MODEL_SIZE = "m"
CANONICAL_TRACKING_MODE = "baseline"
V3DT_PGIE_PROFILE = "yolo26_seg"
V3DT_MODEL_SIZE = "s"
V3DT_TRACKING_MODE = "v3dt"
PRODUCTION_SHUTDOWN_GRACE_S = 75.0
PRODUCTION_SHUTDOWN_TIMEOUT_S = 90.0
MAX_HEALTH_RESPONSE_BYTES = 64 * 1024
MAX_SUPERVISOR_JSON_BYTES = 4 * 1024 * 1024
REQUIRED_CAPABILITIES = ("tracking_observations", "global_world")
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,47}$")
RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
CANONICAL_ENDPOINTS = {
    "websocket": "ws://127.0.0.1:6008",
    "rest": "http://127.0.0.1:8080",
    "rtsp": "rtsp://127.0.0.1:8554/mosaic",
}
CANONICAL_PORTS = {"websocket": 6008, "rest": 8080, "rtsp": 8554}
RUNTIME_IDENTITY_CONTRACT = "noesis.ds9.supervisor_runtime_identity"
IDENTITY_REPORT_FILENAMES = {
    "baseline": "identity-open-set-occupied.json",
    "v3dt": "v3dt-identity-open-set-occupied.json",
}
IDENTITY_SOURCE_FILENAMES = {
    "baseline": "identity-open-set-occupied-source.json",
    "v3dt": "v3dt-identity-open-set-occupied-source.json",
}
SEMANTIC_REPORT_FILENAME = "semantic-observation.json"
SEMANTIC_IDENTITY_SNAPSHOT_FILENAME = "semantic-identity-evidence.jsonl"
SEMANTIC_SOURCE_TRANSCRIPT_FILENAME = "semantic-observation-source.json"
FLOORPLAN_REPORT_FILENAME = "mapanything-depth-quality.json"
FLOORPLAN_SOURCE_TRANSCRIPT_FILENAME = "mapanything-depth-quality-source.json"
V3DT_WORLD_REPORT_FILENAME = "v3dt-world-contract.json"
V3DT_WORLD_SOURCE_TRANSCRIPT_FILENAME = "v3dt-world-contract-source.json"
WHOLEBODY49_REPORT_FILENAME = "wholebody49-occupied-scene.json"
WHOLEBODY49_SOURCE_TRANSCRIPT_FILENAME = "wholebody49-occupied-scene-source.json"
WHOLEBODY49_MEDIA_REPORT_FILENAME = "wholebody49-media-decode.json"
WHOLEBODY49_MEDIA_SOURCE_FILENAME = "wholebody49-media-decode-source.json"

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
    "NOESIS_IDENTITY_V2_MODE",
    "NOESIS_IDENTITY_V2_EVIDENCE_PATH",
    "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID",
    "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME",
    "NOESIS_SKIP_CUDA_PREFLIGHT",
    "NOESIS_OBJECT_DEPTH_ALLOW_HOST_ROI_COPY",
    "NOESIS_REST_CORS_ALLOW_ALL",
    "NOESIS_REST_CORS_ORIGIN_REGEX",
    "NOESIS_REST_CORS_ORIGINS",
    "NOESIS_INFLUX_TOKEN",
    "NOESIS_MQTT_PASSWORD",
    "NOESIS_MOSAIC_WEBRTC_STUN_SERVER",
    "NOESIS_MOSAIC_WEBRTC_TURN_SERVER",
    "NOESIS_WS_PORT_FALLBACK_TRIES",
    "NOESIS_WS_BIND_RETRY_TRIES",
    "NOESIS_DEEPSTREAM_MAJOR",
    "NOESIS_DEEPSTREAM_HOME",
    "NOESIS_MODEL_DIR",
    "NOESIS_ONNX_DIR",
    "NOESIS_ENGINE_DIR",
    "NOESIS_PIPELINE_DIR",
    "NOESIS_BUILD_DIR",
    "NOESIS_NATIVE_EXT_DIR",
    "NOESIS_NATIVE_BUILD_SCRIPT_DIR",
    "NOESIS_GST_PLUGIN_DIR",
    "NOESIS_RFDETR_TRT_PLUGIN_LIB",
)

PROTECTED_EXTRA_ENV_KEYS = frozenset(
    (
        *SANITIZED_ENV_KEYS,
        "NOESIS_INTERNAL_AUTH_MODE",
        "NOESIS_INTERNAL_AUTH_TOKEN_FILE",
        "NOESIS_PGIE_PROFILE",
        "NOESIS_TRACKING_MODE",
        "NOESIS_MOSAIC_RTSP_ENABLED",
        "NOESIS_MOSAIC_WEBRTC_ENABLED",
        "NOESIS_REID_ENABLED",
        "NOESIS_SHUTDOWN_GRACE_SECONDS",
    )
)

GATE_NAMES = (
    "rtsp",
    "webrtc",
    "reid",
    "semantic-observation",
    "floorplan",
    "bev",
    "bridge",
    "ma-depth",
    "zero-copy-stats",
    "zero-copy-rest",
    "v3dt-bbox3d",
    "v3dt-world",
    "wholebody-occupied",
)
V3DT_PROMOTION_REQUIRED_GATES = frozenset(
    {
        "rtsp",
        "webrtc",
        "reid",
        "semantic-observation",
        "v3dt-bbox3d",
        "v3dt-world",
    }
)

NATIVE_CRASH_SIGNATURES = (
    "Fatal Python error",
    "Segmentation fault",
    "SIGSEGV",
    "malloc():",
    "double free",
    "corrupted size",
    "corrupted double-linked list",
    "terminate called without an active exception",
    "Aborted (core dumped)",
)

SHUTDOWN_SUCCESS_MARKERS = {
    "orderly_eos_accepted": "Orderly pipeline EOS accepted:",
    "shutdown_eos_callback": "EOS received on pipeline (reason=shutdown_requested)",
    "servicemaker_wait_returned": "pyservicemaker wait() returned (pipeline stopped)",
    "shutdown_complete": "Shutdown complete",
}

SHUTDOWN_FAILURE_SIGNATURES = (
    *NATIVE_CRASH_SIGNATURES,
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
)

OWNED_LOG_SEVERITY_SIGNATURES = ("ERROR", "CRITICAL")

RTSP_ERROR_SIGNATURES = (
    "ERROR:",
    "Could not",
    "not-linked",
    "Internal data stream error",
    "streaming stopped",
    "Connection refused",
    "No route to host",
    "Service Unavailable",
)


@dataclass
class StepResult:
    name: str
    ok: bool
    returncode: int | None
    duration_s: float
    log_path: str
    command: str
    notes: str = ""
    timed_out: bool = False
    signatures: list[str] | None = None
    forced_termination: bool = False
    signal_sent: bool | None = None


@dataclass(frozen=True)
class CapabilityEvidence:
    instance_id: str
    run_id: str
    generated_at_us: int
    sequences: dict[str, int]

    def public_payload(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "run_id": self.run_id,
            "generated_at_us": self.generated_at_us,
            "sequences": dict(self.sequences),
        }


@dataclass(frozen=True)
class ValidationLane:
    name: str
    pipeline_config: Path
    cameras_config: Path
    pgie_profile: str
    model_size: str
    tracking_mode: str
    source_ids: tuple[int, ...]


BASELINE_LANE = ValidationLane(
    name="baseline",
    pipeline_config=Path("DS9/config/infer.yaml"),
    cameras_config=Path("config/cameras.yaml"),
    pgie_profile=CANONICAL_PGIE_PROFILE,
    model_size=CANONICAL_MODEL_SIZE,
    tracking_mode=CANONICAL_TRACKING_MODE,
    source_ids=(0, 1, 2),
)
V3DT_LANE = ValidationLane(
    name="v3dt",
    pipeline_config=Path("DS9/config/infer_v3dt.yaml"),
    cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
    pgie_profile=V3DT_PGIE_PROFILE,
    model_size=V3DT_MODEL_SIZE,
    tracking_mode=V3DT_TRACKING_MODE,
    source_ids=(0, 1, 2),
)
WHOLEBODY49_S_LANE = ValidationLane(
    name="wholebody49-s",
    pipeline_config=Path("DS9/config/infer.yaml"),
    cameras_config=Path("config/cameras.yaml"),
    pgie_profile="wholebody49",
    model_size="s",
    tracking_mode="baseline",
    source_ids=(0, 1, 2),
)
WHOLEBODY49_X_LANE = ValidationLane(
    name="wholebody49-x",
    pipeline_config=Path("DS9/config/infer.yaml"),
    cameras_config=Path("config/cameras.yaml"),
    pgie_profile="wholebody49",
    model_size="x",
    tracking_mode="baseline",
    source_ids=(0, 1, 2),
)
VALIDATION_LANES = {
    lane.name: lane
    for lane in (
        BASELINE_LANE,
        V3DT_LANE,
        WHOLEBODY49_S_LANE,
        WHOLEBODY49_X_LANE,
    )
}


def _validation_lane(args: argparse.Namespace) -> ValidationLane:
    raw = str(getattr(args, "lane", BASELINE_LANE.name) or "").strip().lower()
    lane = VALIDATION_LANES.get(raw)
    if lane is None:
        raise ValueError(
            f"unsupported DS9 validation lane {raw!r}; expected one of {sorted(VALIDATION_LANES)}"
        )
    return lane


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _runtime_session_id(args: argparse.Namespace) -> str:
    session_id = str(getattr(args, "session_id", "") or "").strip().lower()
    if SESSION_RE.fullmatch(session_id) is None:
        raise ValueError("session ID must match [a-z0-9][a-z0-9-]{5,47}")
    return session_id


def _prepare_private_output_dir(path: Path) -> Path:
    """Create or validate the runner-owned behavior evidence directory."""

    output_dir = Path(path).expanduser().absolute()
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = output_dir.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("behavior output directory must be a regular directory")
    if info.st_uid != os.geteuid():
        raise ValueError("behavior output directory owner mismatch")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("behavior output directory must be owner-only mode 0700")
    return output_dir


def _behavior_evidence_dir(args: argparse.Namespace) -> Path:
    configured = getattr(args, "ownership_evidence_dir", None)
    return Path(configured if configured is not None else args.output_dir)


def _runtime_identity(args: argparse.Namespace) -> tuple[str, str]:
    instance_id = str(getattr(args, "runtime_instance_id", "") or "")
    run_id = str(getattr(args, "runtime_run_id", "") or "")
    if (
        RUNTIME_ID_RE.fullmatch(instance_id) is None
        or RUNTIME_ID_RE.fullmatch(run_id) is None
    ):
        raise ValueError("runtime instance/run identity is unavailable")
    return instance_id, run_id


def _bind_runtime_identity(
    args: argparse.Namespace,
    observed: CapabilityEvidence,
) -> None:
    expected_instance = getattr(args, "supervisor_runtime_instance_id", None)
    expected_run = getattr(args, "supervisor_runtime_run_id", None)
    if expected_instance is not None and (
        observed.instance_id != expected_instance or observed.run_id != expected_run
    ):
        raise ValueError(
            "authenticated runner identity differs from supervisor-observed runtime"
        )
    args.runtime_instance_id = observed.instance_id
    args.runtime_run_id = observed.run_id


def _identity_report_path(args: argparse.Namespace) -> Path:
    lane = _validation_lane(args)
    filename = IDENTITY_REPORT_FILENAMES.get(lane.name)
    if filename is None:
        return Path(args.output_dir) / "identity-shadow.json"
    return _behavior_evidence_dir(args) / filename


def _identity_source_path(args: argparse.Namespace) -> Path:
    lane = _validation_lane(args)
    filename = IDENTITY_SOURCE_FILENAMES.get(lane.name)
    if filename is None:
        return Path(args.output_dir) / "identity-shadow-source.json"
    return _behavior_evidence_dir(args) / filename


def _semantic_report_path(args: argparse.Namespace) -> Path:
    return _behavior_evidence_dir(args) / SEMANTIC_REPORT_FILENAME


def _semantic_snapshot_path(args: argparse.Namespace) -> Path:
    return _semantic_report_path(args).parent / SEMANTIC_IDENTITY_SNAPSHOT_FILENAME


def _semantic_source_path(args: argparse.Namespace) -> Path:
    return _semantic_report_path(args).parent / SEMANTIC_SOURCE_TRANSCRIPT_FILENAME


def _cmd_text(cmd: Sequence[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def _normalize_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _find_signatures(text: str, signatures: Iterable[str]) -> list[str]:
    found = []
    for signature in signatures:
        if signature in text:
            found.append(signature)
    return found


def _find_signatures_casefold(text: str, signatures: Iterable[str]) -> list[str]:
    folded_text = text.casefold()
    return [
        signature for signature in signatures if signature.casefold() in folded_text
    ]


def _validate_owned_shutdown_log(text: str) -> tuple[list[str], list[str]]:
    """Return missing lifecycle evidence and owned-runtime failure signatures."""

    missing = [
        name for name, marker in SHUTDOWN_SUCCESS_MARKERS.items() if marker not in text
    ]
    failures = _find_signatures_casefold(text, SHUTDOWN_FAILURE_SIGNATURES)
    failures.extend(
        signature
        for signature in _find_signatures(text, OWNED_LOG_SEVERITY_SIGNATURES)
        if signature not in failures
    )
    if missing:
        return missing, failures

    positions = {
        name: text.rfind(marker) for name, marker in SHUTDOWN_SUCCESS_MARKERS.items()
    }
    ordered_names = tuple(SHUTDOWN_SUCCESS_MARKERS)
    for previous, current in zip(ordered_names, ordered_names[1:]):
        if positions[previous] > positions[current]:
            missing.append(f"{previous}_before_{current}")
    return missing, failures


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _append_step_note(result: StepResult, note: str) -> None:
    clean = " ".join(str(note or "").split())
    if not clean:
        return
    result.notes = f"{result.notes}; {clean}" if result.notes else clean


def _apply_identity_gate_report(
    result: StepResult,
    report_path: Path,
    *,
    source_path: Path,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    pipeline_config: Path,
    require_cross_camera: bool,
    require_open_set: bool,
    min_fresh_embeddings: int = 2,
) -> None:
    try:
        payload = identity_gate.validate_sealed_identity_shadow_report(
            report_path,
            source_path,
            pipeline_config=pipeline_config,
            session_id=session_id,
            runtime_lane=runtime_lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            require_cross_camera=require_cross_camera,
            require_open_set=require_open_set,
            min_fresh_embeddings=min_fresh_embeddings,
        )
        claims = payload.get("claims")
        if not isinstance(claims, Mapping):
            raise ValueError("identity gate report claims are missing")

        def status(name: str) -> tuple[str, bool]:
            row = claims.get(name)
            if not isinstance(row, Mapping):
                raise ValueError(f"identity gate claim is missing: {name}")
            value = str(row.get("status") or "")
            required = row.get("required")
            if not isinstance(required, bool):
                raise ValueError(f"identity gate claim required flag is invalid: {name}")
            return value, required

        runtime_status, runtime_required = status("runtime_shadow_health")
        continuity_status, continuity_required = status(
            "tracker_subject_continuity"
        )
        cross_status, cross_required = status(
            "cross_camera_assignment_continuity"
        )
        open_status, open_required = status("open_set_non_force")
        accuracy_status, accuracy_required = status("semantic_accuracy")
        authority_status, authority_required = status("public_authority")
        if (runtime_status, runtime_required) != ("pass", True):
            raise ValueError("identity shadow runtime health claim did not pass")
        if (continuity_status, continuity_required) != ("pass", True):
            raise ValueError("identity tracker continuity claim did not pass")
        if cross_status not in {"observed", "not_observed"}:
            raise ValueError("identity cross-camera claim status is invalid")
        if cross_required and cross_status != "observed":
            raise ValueError("required identity cross-camera evidence is absent")
        if open_status not in {"observed", "not_observed"}:
            raise ValueError("identity open-set claim status is invalid")
        if open_required and open_status != "observed":
            raise ValueError("required identity open-set evidence is absent")
        if (accuracy_status, accuracy_required) != ("not_evaluated", False):
            raise ValueError("identity gate must not claim unlabeled semantic accuracy")
        if (authority_status, authority_required) != ("blocked", True):
            raise ValueError("identity public authority was not blocked")
        _append_step_note(
            result,
            "identity_shadow=pass; tracker_continuity=pass; "
            f"cross_camera={cross_status}; open_set_non_force={open_status}; "
            "semantic_accuracy=not_evaluated; public_authority=blocked; "
            "exact_source_replay=pass",
        )
    except Exception as exc:
        result.ok = False
        _append_step_note(result, f"identity_report_invalid={exc}")


def _apply_floorplan_gate_report(
    result: StepResult,
    report_path: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    source_path: Path | None = None,
) -> None:
    try:
        source = (
            Path(source_path)
            if source_path is not None
            else report_path.with_name(FLOORPLAN_SOURCE_TRANSCRIPT_FILENAME)
        )
        payload = floorplan_gate.load_and_validate_sealed_authority(
            report_path,
            source,
            session_id=session_id,
            runtime_lane=runtime_lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
        )
        configured = payload.get("configured_camera_count")
        bev_active = payload.get("bev_active_camera_count")
        bev_inactive = payload.get("bev_inactive_ready_camera_count")
        if (
            isinstance(configured, bool)
            or not isinstance(configured, int)
            or configured <= 0
            or isinstance(bev_active, bool)
            or not isinstance(bev_active, int)
            or bev_active < 0
            or isinstance(bev_inactive, bool)
            or not isinstance(bev_inactive, int)
            or bev_inactive < 0
        ):
            raise ValueError("floorplan exact replay returned invalid activity counts")
        _append_step_note(
            result,
            f"all_configured_floorplans=pass; validated_camera_count={configured}; "
            "exact_capture_events=pass; rgb=not_requested; "
            f"bev_ready=pass(active={bev_active},inactive={bev_inactive}); "
            "cache_only_zero_mutation=pass; exact_source_replay=pass",
        )
    except Exception as exc:
        result.ok = False
        _append_step_note(result, f"floorplan_report_invalid={exc}")


def _apply_semantic_observation_gate_report(
    result: StepResult,
    report_path: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    pipeline_config: Path,
    identity_evidence_path: Path,
) -> None:
    try:
        expected_layer, expected_dimension = (
            semantic_gate._load_reviewed_reid_contract(pipeline_config)
        )
        payload = semantic_gate.validate_sealed_semantic_report(
            report_path,
            report_path.parent / SEMANTIC_IDENTITY_SNAPSHOT_FILENAME,
            report_path.parent / SEMANTIC_SOURCE_TRANSCRIPT_FILENAME,
            session_id=session_id,
            runtime_lane=runtime_lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            expected_identity_evidence_path=identity_evidence_path,
            expected_model_layer=expected_layer,
            expected_embedding_dimension=expected_dimension,
        )
        counts = payload.get("counts")
        complete = (
            counts.get("accepted_semantic_cohorts")
            if isinstance(counts, Mapping)
            else None
        )
        if isinstance(complete, bool) or not isinstance(complete, int) or complete < 1:
            raise ValueError("semantic observation has no accepted bounded cohort")
        _append_step_note(
            result,
            f"occupied_semantics=pass; bounded_identity_cohorts={complete}; "
            "pose_depth_world_components=pass; pipeline_errors=absent; "
            "exact_source_replay=pass",
        )
    except Exception as exc:
        result.ok = False
        _append_step_note(result, f"semantic_observation_report_invalid={exc}")


def _apply_v3dt_world_gate_report(
    result: StepResult,
    report_path: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    pipeline_config: Path,
    cameras_config: Path,
    launcher_dir: Path,
) -> None:
    try:
        payload = v3dt_world_gate.validate_sealed_v3dt_world_report(
            report_path,
            report_path.parent / V3DT_WORLD_SOURCE_TRANSCRIPT_FILENAME,
            pipeline_config=pipeline_config,
            cameras_config=cameras_config,
            calibration_config=Path("config/camera_calibration.json"),
            alignment_config=Path("config/ply_alignment.json"),
            launcher_dir=launcher_dir,
            session_id=session_id,
            runtime_lane=runtime_lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
        )
        _append_step_note(
            result,
            f"v3dt_world=pass; tracks={payload.get('tracks_seen')}; "
            "config_launcher_binding=pass; exact_source_replay=pass",
        )
    except Exception as exc:
        result.ok = False
        _append_step_note(result, f"v3dt_world_report_invalid={exc}")


def _apply_wholebody_occupied_gate_report(
    result: StepResult,
    report_path: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    mode: str,
    expected_source_ids: Sequence[int],
) -> None:
    try:
        payload = wholebody_gate.validate_sealed_wholebody49_report(
            report_path,
            report_path.parent / WHOLEBODY49_SOURCE_TRANSCRIPT_FILENAME,
            session_id=session_id,
            runtime_lane=runtime_lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
            mode=mode,
            expected_source_ids=expected_source_ids,
        )
        _append_step_note(
            result,
            f"wholebody_occupied=pass; tracks={payload.get('tracks_seen')}; "
            "all_sources_advancing=pass; exact_source_replay=pass",
        )
    except Exception as exc:
        result.ok = False
        _append_step_note(result, f"wholebody_occupied_report_invalid={exc}")


def _apply_wholebody_media_gate_report(
    result: StepResult,
    report_path: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
) -> None:
    try:
        payload = wholebody_media_gate.validate_sealed_wholebody49_media_report(
            report_path,
            report_path.parent / WHOLEBODY49_MEDIA_SOURCE_FILENAME,
            session_id=session_id,
            runtime_lane=runtime_lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
        )
        metrics = payload.get("metrics")
        decoded = metrics.get("decoded_frames") if isinstance(metrics, Mapping) else None
        _append_step_note(
            result,
            f"wholebody_media=pass; decoded_frames={decoded}; "
            "direct_rtsp_decode=pass; exact_source_replay=pass",
        )
    except Exception as exc:
        result.ok = False
        _append_step_note(result, f"wholebody_media_report_invalid={exc}")


def _apply_completed_behavior_gate_report(
    name: str,
    result: StepResult,
    *,
    args: argparse.Namespace,
    lane: ValidationLane,
) -> None:
    """Replay the exact sealed report owned by a completed behavior gate."""

    if not result.ok or result.timed_out or result.returncode != 0:
        return
    supported = {
        "reid",
        "semantic-observation",
        "floorplan",
        "v3dt-world",
        "wholebody-occupied",
    }
    if lane in {WHOLEBODY49_S_LANE, WHOLEBODY49_X_LANE}:
        supported.add("webrtc")
    if name not in supported:
        return
    runtime_instance_id, runtime_run_id = _runtime_identity(args)
    identity = {
        "session_id": _runtime_session_id(args),
        "runtime_lane": lane.name,
        "runtime_instance_id": runtime_instance_id,
        "runtime_run_id": runtime_run_id,
    }
    if name == "reid":
        _apply_identity_gate_report(
            result,
            _identity_report_path(args),
            **identity,
            source_path=_identity_source_path(args),
            pipeline_config=Path(args.pipeline_config),
            require_cross_camera=bool(
                getattr(args, "identity_require_cross_camera", False)
            ),
            require_open_set=bool(
                lane.name in {"baseline", "v3dt"}
                or getattr(args, "identity_require_open_set", False)
            ),
        )
    elif name == "semantic-observation":
        _apply_semantic_observation_gate_report(
            result,
            _semantic_report_path(args),
            **identity,
            pipeline_config=Path(args.pipeline_config),
            identity_evidence_path=Path(args.identity_evidence),
        )
    elif name == "floorplan":
        _apply_floorplan_gate_report(
            result,
            _behavior_evidence_dir(args) / FLOORPLAN_REPORT_FILENAME,
            **identity,
        )
    elif name == "v3dt-world":
        _apply_v3dt_world_gate_report(
            result,
            _behavior_evidence_dir(args) / V3DT_WORLD_REPORT_FILENAME,
            **identity,
            pipeline_config=Path(args.pipeline_config),
            cameras_config=Path(args.cameras_config),
            launcher_dir=_behavior_evidence_dir(args),
        )
    elif name == "wholebody-occupied":
        _apply_wholebody_occupied_gate_report(
            result,
            _behavior_evidence_dir(args) / WHOLEBODY49_REPORT_FILENAME,
            **identity,
            mode="masks" if lane is WHOLEBODY49_S_LANE else "boxes",
            expected_source_ids=lane.source_ids,
        )
    else:
        _apply_wholebody_media_gate_report(
            result,
            _behavior_evidence_dir(args) / WHOLEBODY49_MEDIA_REPORT_FILENAME,
            **identity,
        )


def _run_command(
    *,
    name: str,
    cmd: Sequence[str],
    output_dir: Path,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    timeout_ok: bool = False,
    timeout_ok_note: str = "bounded timeout reached",
    fail_signatures: Sequence[str] = (),
) -> StepResult:
    start = time.monotonic()
    log_path = output_dir / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _cmd_text(cmd)
    text = ""
    timed_out = False
    returncode: int | None = None
    notes = ""

    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=REPO_ROOT,
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        returncode = proc.returncode
        text = proc.stdout or ""
        ok = returncode == 0
        notes = "exit_0" if ok else f"exit_{returncode}"
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        text = _normalize_text(exc.stdout) + _normalize_text(exc.stderr)
        ok = bool(timeout_ok)
        notes = timeout_ok_note if ok else f"timed out after {timeout_s:.1f}s"
    except FileNotFoundError as exc:
        returncode = None
        text = f"{type(exc).__name__}: {exc}\n"
        ok = False
        notes = "command not found"
    except Exception as exc:
        returncode = None
        text = f"{type(exc).__name__}: {exc}\n"
        ok = False
        notes = "command failed before completion"

    log_path.write_text(text, encoding="utf-8", errors="replace")

    signatures = _find_signatures(text, fail_signatures)
    if signatures:
        ok = False
        if notes:
            notes = f"{notes}; signatures={','.join(signatures)}"
        else:
            notes = f"signatures={','.join(signatures)}"

    return StepResult(
        name=name,
        ok=ok,
        returncode=returncode,
        duration_s=round(time.monotonic() - start, 3),
        log_path=_repo_rel(log_path),
        command=command,
        notes=notes,
        timed_out=timed_out,
        signatures=signatures,
    )


def _parse_host_port(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    return host, int(port)


def _resolve_auth_token_file(
    env: Mapping[str, str], configured: str | Path | None = None
) -> Path:
    if configured is not None and str(configured).strip():
        return Path(configured).expanduser().absolute()
    configured = str(env.get("NOESIS_INTERNAL_AUTH_TOKEN_FILE", "") or "").strip()
    if configured:
        return Path(configured).expanduser().absolute()
    return (Path.home() / ".local" / "state" / "noesis" / "gateway-token").absolute()


def _load_auth_token(token_file: Path) -> str:
    from noesis.server.internal_auth import load_internal_token

    return load_internal_token(token_file)


def _runtime_env(
    extra_env: Sequence[str],
    *,
    token_file: Path,
    session_id: str,
    identity_evidence_path: Path,
    lane: ValidationLane = BASELINE_LANE,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if VALIDATION_LANES.get(lane.name) != lane:
        raise SystemExit(f"unreviewed DS9 validation lane: {lane.name!r}")
    session_id = str(session_id).strip().lower()
    if SESSION_RE.fullmatch(session_id) is None:
        raise SystemExit("runtime session ID is invalid")
    env = dict(os.environ if base_env is None else base_env)
    for key in SANITIZED_ENV_KEYS:
        env.pop(key, None)
    for item in extra_env:
        if "=" not in item:
            raise SystemExit(f"--env values must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--env values must include a non-empty key, got: {item}")
        if key in PROTECTED_EXTRA_ENV_KEYS:
            raise SystemExit(
                f"--env cannot override the canonical runtime contract: {key}"
            )
        env[key] = value
    env.update(
        {
            "NOESIS_INTERNAL_AUTH_MODE": "required",
            "NOESIS_INTERNAL_AUTH_TOKEN_FILE": str(token_file),
            "NOESIS_PGIE_PROFILE": lane.pgie_profile,
            "NOESIS_TRACKING_MODE": lane.tracking_mode,
            "NOESIS_MOSAIC_RTSP_ENABLED": "1",
            "NOESIS_MOSAIC_WEBRTC_ENABLED": "1",
            "NOESIS_REID_ENABLED": "1",
            "NOESIS_IDENTITY_V2_MODE": "shadow",
            "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(identity_evidence_path),
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": session_id,
            "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME": "ds9",
            "NOESIS_SHUTDOWN_GRACE_SECONDS": str(int(PRODUCTION_SHUTDOWN_GRACE_S)),
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def _build_runtime_command(args: argparse.Namespace) -> list[str]:
    lane = _validation_lane(args)
    ws_host, ws_port = _parse_host_port(str(args.ws), 6008)
    rest_host, rest_port = _parse_host_port(str(args.rest), 8080)
    return [
        sys.executable,
        "DS9/noesis/ds9_runtime.py",
        "--pipeline-config",
        str(args.pipeline_config),
        "--cameras-config",
        str(args.cameras_config),
        "--pgie-profile",
        lane.pgie_profile,
        "--size",
        lane.model_size,
        "--tracking-mode",
        lane.tracking_mode,
        "--ws-host",
        str(ws_host),
        "--ws-port",
        str(ws_port),
        "--rest-host",
        str(rest_host),
        "--rest-port",
        str(rest_port),
        "--enable-rest",
        "--log-level",
        "INFO",
    ]


def _spawn_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    runtime_env: Mapping[str, str],
) -> tuple[subprocess.Popen[None], Path]:
    runtime_log = output_dir / "runtime.log"
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    runtime_cmd = _build_runtime_command(args)
    runtime_log.write_text(f"$ {_cmd_text(runtime_cmd)}\n\n", encoding="utf-8")
    log_fh = runtime_log.open("a", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            runtime_cmd,
            cwd=REPO_ROOT,
            env=dict(runtime_env),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log_fh.close()
    return proc, runtime_log


def _validate_capability_payload(payload: object) -> CapabilityEvidence:
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
            raise ValueError(f"Noesis capability is not healthy: {capability}")
        sequence = row.evidence.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError(f"Noesis capability sequence is invalid: {capability}")
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
    rest_url: str, token: str, *, timeout_s: float
) -> CapabilityEvidence:
    host, port = _parse_host_port(rest_url, 8080)
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
            payload = strict_json_loads(raw, label="capability health response")
        except Exception as exc:
            raise RuntimeError("capability endpoint returned invalid JSON") from exc
        return _validate_capability_payload(payload)
    finally:
        connection.close()


def _probe_websocket_health(ws_url: str, token: str, *, timeout_s: float) -> None:
    from websockets.sync.client import connect

    parsed = urlparse(ws_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6008
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
        payload = strict_json_loads(raw, label="WebSocket health response")
    except Exception as exc:
        raise RuntimeError("WebSocket health response was invalid JSON") from exc
    if payload != expected:
        raise RuntimeError("WebSocket health contract mismatch")


def _probe_rtsp_describe(rtsp_url: str, *, timeout_s: float) -> None:
    parsed = urlparse(rtsp_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 554
    request = (
        f"DESCRIBE {rtsp_url} RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "Accept: application/sdp\r\n"
        "User-Agent: NoesisDS9CanonicalGate/1.0\r\n"
        "Connection: close\r\n"
        "\r\n"
    )
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(request.encode("ascii"))
        data = b""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if len(data) > 64 * 1024:
                raise RuntimeError("RTSP response exceeded size bound")
    status_line = data.decode("iso-8859-1", errors="replace").splitlines()[:1]
    parts = status_line[0].split() if status_line else []
    try:
        status_code = int(parts[1])
    except Exception as exc:
        raise RuntimeError("RTSP endpoint returned an invalid response") from exc
    if not 200 <= status_code < 300:
        raise RuntimeError(f"RTSP DESCRIBE returned {status_code}")


def _runtime_ready(
    args: argparse.Namespace,
    proc: subprocess.Popen[None] | None,
    *,
    token: str,
) -> tuple[bool, str, CapabilityEvidence | None]:
    deadline = time.monotonic() + float(args.startup_timeout_s)
    previous: CapabilityEvidence | None = None
    last_error = "runtime did not become ready"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return (
                False,
                f"runtime exited before readiness (rc={proc.returncode})",
                None,
            )
        try:
            current = _probe_capability_health(str(args.rest), token, timeout_s=2.0)
            if previous is not None:
                _require_advancement(previous, current)
                _probe_websocket_health(str(args.ws), token, timeout_s=2.0)
                _probe_rtsp_describe(str(args.rtsp_url), timeout_s=2.0)
                note = json.dumps(
                    current.public_payload(), separators=(",", ":"), sort_keys=True
                )
                return (
                    True,
                    f"authenticated contracts ready and advancing: {note}",
                    current,
                )
            previous = current
            last_error = "waiting for a second advancing capability sample"
        except Exception as exc:
            last_error = str(exc) or type(exc).__name__
        time.sleep(0.25)
    return False, f"not ready: {last_error}", previous


def _shutdown_runtime(
    proc: subprocess.Popen[None] | None,
    runtime_log: Path | None,
    timeout_s: float,
) -> StepResult:
    start = time.monotonic()
    command = "SIGTERM DS9 runtime"
    if proc is None:
        return StepResult(
            name="shutdown",
            ok=True,
            returncode=None,
            duration_s=0.0,
            log_path=_repo_rel(runtime_log) if runtime_log else "",
            command=command,
            notes="skipped: --no-spawn",
        )

    notes = ""
    forced_termination = False
    was_running = proc.poll() is None
    signal_failed = False
    signal_sent = False
    if was_running:
        try:
            proc.terminate()
            signal_sent = True
        except ProcessLookupError:
            signal_failed = True
            notes = "runtime disappeared before SIGTERM could be delivered"
        except Exception as exc:
            signal_failed = True
            notes = f"SIGTERM failed: {exc}"
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            forced_termination = True
            notes = (
                f"{notes}; shutdown exceeded production timeout, sent SIGKILL".strip(
                    "; "
                )
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
                notes = f"{notes}; process group did not exit after SIGKILL"
    else:
        notes = f"runtime exited before orchestrated shutdown (rc={proc.returncode})"

    text = ""
    if runtime_log and runtime_log.exists():
        text = runtime_log.read_text(encoding="utf-8", errors="replace")
    missing_evidence, signatures = _validate_owned_shutdown_log(text)
    if missing_evidence:
        notes = (
            f"{notes}; missing shutdown evidence={','.join(missing_evidence)}"
        ).strip("; ")
    if signatures:
        notes = f"{notes}; failure signatures={','.join(signatures)}".strip("; ")
    if not notes:
        notes = (
            "exit_0; orderly_eos_accepted; shutdown_eos_callback; "
            "servicemaker_wait_returned; shutdown_complete"
            if proc.returncode == 0
            else f"exit_{proc.returncode}"
        )
    ok = bool(
        was_running
        and not signal_failed
        and signal_sent
        and proc.returncode == 0
        and not forced_termination
        and not signatures
        and not missing_evidence
    )

    return StepResult(
        name="shutdown",
        ok=ok,
        returncode=proc.returncode,
        duration_s=round(time.monotonic() - start, 3),
        log_path=_repo_rel(runtime_log) if runtime_log else "",
        command=command,
        notes=notes,
        signatures=signatures,
        forced_termination=forced_termination,
        signal_sent=signal_sent,
    )


def _build_gates(
    args: argparse.Namespace, *, auth_token_file: Path
) -> list[tuple[str, list[str], float | None, bool, Sequence[str]]]:
    lane = _validation_lane(args)
    py = sys.executable
    pcfg = str(args.pipeline_config)
    ccfg = str(args.cameras_config)
    rest_refresh_url = args.rest.rstrip("/") + "/api/v1/depth/refresh"
    session_id = _runtime_session_id(args)
    runtime_instance_id, runtime_run_id = _runtime_identity(args)
    evidence_dir = _behavior_evidence_dir(args)
    identity_report = _identity_report_path(args)
    if lane in {WHOLEBODY49_S_LANE, WHOLEBODY49_X_LANE}:
        webrtc_command = [
            py,
            "DS9/scripts/wholebody49_media_decode_gate.py",
            "--ws",
            args.ws,
            "--rtsp-url",
            args.rtsp_url,
            "--duration",
            str(args.webrtc_duration_s),
            "--rtsp-duration",
            str(args.webrtc_duration_s),
            "--pt",
            str(args.webrtc_pt),
            "--min-rtp",
            str(args.webrtc_min_rtp),
            "--min-decoded",
            str(args.webrtc_min_decoded),
            "--rtsp-min-decoded",
            "1",
            "--session-id",
            session_id,
            "--runtime-lane",
            lane.name,
            "--runtime-instance-id",
            runtime_instance_id,
            "--runtime-run-id",
            runtime_run_id,
            "--source-out",
            str(evidence_dir / WHOLEBODY49_MEDIA_SOURCE_FILENAME),
            "--out",
            str(evidence_dir / WHOLEBODY49_MEDIA_REPORT_FILENAME),
            "--auth-token-file",
            str(auth_token_file),
        ]
    else:
        webrtc_command = [
            py,
            "scripts/webrtc_gateway_smoke_test.py",
            "--ws",
            args.ws,
            "--duration",
            str(args.webrtc_duration_s),
            "--pt",
            str(args.webrtc_pt),
            "--min-rtp",
            str(args.webrtc_min_rtp),
            "--min-decoded",
            str(args.webrtc_min_decoded),
            "--auth-token-file",
            str(auth_token_file),
        ]
    baseline = [
        (
            "rtsp",
            [
                "gst-launch-1.0",
                "-e",
                "rtspsrc",
                f"location={args.rtsp_url}",
                "latency=100",
                "!",
                "rtph264depay",
                "!",
                "h264parse",
                "!",
                "avdec_h264",
                "!",
                "fakesink",
                "sync=false",
            ],
            args.rtsp_duration_s,
            True,
            RTSP_ERROR_SIGNATURES,
        ),
        (
            "webrtc",
            webrtc_command,
            (2.0 * args.webrtc_duration_s) + 35.0,
            False,
            (),
        ),
        (
            "reid",
            [
                py,
                "DS9/scripts/ds9_identity_shadow_live_gate.py",
                "--ws",
                args.ws,
                "--rest",
                args.rest,
                "--pipeline-config",
                pcfg,
                "--duration",
                str(args.reid_duration_s),
                "--session-id",
                session_id,
                "--runtime-lane",
                lane.name,
                "--runtime-instance-id",
                runtime_instance_id,
                "--runtime-run-id",
                runtime_run_id,
                "--source-out",
                str(_identity_source_path(args)),
                "--out",
                str(identity_report),
                "--auth-token-file",
                str(auth_token_file),
                *(
                    ["--require-cross-camera"]
                    if bool(getattr(args, "identity_require_cross_camera", False))
                    else []
                ),
                *(
                    ["--require-open-set"]
                    if lane.name in {"baseline", "v3dt"}
                    or bool(getattr(args, "identity_require_open_set", False))
                    else []
                ),
            ],
            args.reid_duration_s + 25.0,
            False,
            (),
        ),
        (
            "semantic-observation",
            [
                py,
                "DS9/scripts/ds9_semantic_observation_smoke_test.py",
                "--ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--duration",
                str(args.semantic_observation_duration_s),
                "--session-id",
                session_id,
                "--runtime-lane",
                lane.name,
                "--runtime-instance-id",
                runtime_instance_id,
                "--runtime-run-id",
                runtime_run_id,
                "--identity-evidence",
                str(args.identity_evidence),
                "--snapshot-out",
                str(_semantic_snapshot_path(args)),
                "--source-out",
                str(_semantic_source_path(args)),
                "--out",
                str(_semantic_report_path(args)),
                "--auth-token-file",
                str(auth_token_file),
            ],
            args.semantic_observation_duration_s + 25.0,
            False,
            (),
        ),
        (
            "bev",
            [
                py,
                "scripts/menon_bev_track_parity_smoke_test.py",
                "--no-spawn",
                "--ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--calibration-config",
                "config/camera_calibration.json",
                "--alignment-config",
                "config/ply_alignment.json",
                "--floorplan-authority",
                str(evidence_dir / FLOORPLAN_REPORT_FILENAME),
                "--floorplan-source",
                str(evidence_dir / FLOORPLAN_SOURCE_TRANSCRIPT_FILENAME),
                "--session-id",
                session_id,
                "--runtime-lane",
                lane.name,
                "--runtime-instance-id",
                runtime_instance_id,
                "--runtime-run-id",
                runtime_run_id,
                "--duration",
                str(args.bev_duration_s),
                "--auth-token-file",
                str(auth_token_file),
            ],
            args.bev_duration_s + 25.0,
            False,
            (),
        ),
        (
            "bridge",
            [
                py,
                "DS9/scripts/ds9_bridge_contract_smoke_test.py",
                "--ws",
                args.ws,
                "--duration",
                str(args.bridge_duration_s),
                "--require-embedding-track",
                "--auth-token-file",
                str(auth_token_file),
            ],
            args.bridge_duration_s + 30.0,
            False,
            (),
        ),
        (
            "floorplan",
            [
                py,
                "DS9/scripts/ds9_floorplan_live_gate.py",
                "--ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--max-age-sec",
                str(args.floorplan_max_age_sec),
                "--timeout-s",
                str(args.floorplan_timeout_s),
                "--session-id",
                session_id,
                "--runtime-lane",
                lane.name,
                "--runtime-instance-id",
                runtime_instance_id,
                "--runtime-run-id",
                runtime_run_id,
                "--out",
                str(evidence_dir / FLOORPLAN_REPORT_FILENAME),
                "--source-out",
                str(evidence_dir / FLOORPLAN_SOURCE_TRANSCRIPT_FILENAME),
                "--auth-token-file",
                str(auth_token_file),
            ],
            args.floorplan_timeout_s + 15.0,
            False,
            (),
        ),
        (
            "ma-depth",
            [
                py,
                "scripts/ma_depth_rpc_smoke_test.py",
                "--no-spawn",
                "--ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--camera",
                args.ma_camera,
                "--auth-token-file",
                str(auth_token_file),
            ],
            90.0,
            False,
            (),
        ),
        (
            "zero-copy-stats",
            [
                py,
                "scripts/zero_copy_stats_smoke_test.py",
                "--no-spawn",
                "--stats-ws",
                args.ws,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--duration-s",
                str(args.zero_copy_stats_duration_s),
                "--auth-token-file",
                str(auth_token_file),
            ],
            args.zero_copy_stats_duration_s + 30.0,
            False,
            (),
        ),
        (
            "zero-copy-rest",
            [
                py,
                "scripts/zero_copy_smoke_test.py",
                "--no-spawn",
                "--stats-ws",
                args.ws,
                "--rest-url",
                rest_refresh_url,
                "--pipeline-config",
                pcfg,
                "--cameras-config",
                ccfg,
                "--duration-s",
                str(args.zero_copy_rest_duration_s),
                "--auth-token-file",
                str(auth_token_file),
            ],
            args.zero_copy_rest_duration_s + 40.0,
            False,
            (),
        ),
    ]
    if lane is BASELINE_LANE:
        baseline_by_name = {row[0]: row for row in baseline}
        ordered_names = (
            "rtsp",
            "webrtc",
            "reid",
            "semantic-observation",
            "floorplan",
            "bev",
            "bridge",
            "ma-depth",
            "zero-copy-stats",
            "zero-copy-rest",
        )
        return [baseline_by_name[name] for name in ordered_names]
    common_names = {"rtsp", "webrtc", "reid"}
    if lane is V3DT_LANE:
        common_names.add("semantic-observation")
    common = [row for row in baseline if row[0] in common_names]
    if lane is V3DT_LANE:
        bbox_timeout = V3DTBBoxTimeoutContract(
            duration_seconds=args.v3dt_bbox_duration_s,
            attempts=args.v3dt_bbox_attempts,
        )
        return [
            *common,
            (
                "v3dt-bbox3d",
                [
                    py,
                    "DS9/scripts/sv3dt_meta_smoke_test.py",
                    "--no-spawn",
                    "--ws",
                    args.ws,
                    "--duration",
                    str(args.v3dt_bbox_duration_s),
                    "--attempts",
                    str(args.v3dt_bbox_attempts),
                    "--auth-token-file",
                    str(auth_token_file),
                ],
                bbox_timeout.runner_timeout_seconds,
                False,
                (),
            ),
            (
                "v3dt-world",
                [
                    py,
                    "DS9/scripts/v3dt_world_contract_smoke_test.py",
                    "--ws",
                    args.ws,
                    "--duration",
                    str(args.v3dt_world_duration_s),
                    "--session-id",
                    session_id,
                    "--runtime-lane",
                    lane.name,
                    "--runtime-instance-id",
                    runtime_instance_id,
                    "--runtime-run-id",
                    runtime_run_id,
                    "--pipeline-config",
                    pcfg,
                    "--cameras-config",
                    ccfg,
                    "--launcher-evidence-dir",
                    str(evidence_dir),
                    "--out",
                    str(evidence_dir / V3DT_WORLD_REPORT_FILENAME),
                    "--source-out",
                    str(evidence_dir / V3DT_WORLD_SOURCE_TRANSCRIPT_FILENAME),
                    "--auth-token-file",
                    str(auth_token_file),
                ],
                args.v3dt_world_duration_s + 20.0,
                False,
                (),
            ),
        ]
    mode = "masks" if lane is WHOLEBODY49_S_LANE else "boxes"
    return [
        *common,
        (
            "wholebody-occupied",
            [
                py,
                "DS9/scripts/wholebody49_occupied_scene_smoke_test.py",
                "--mode",
                mode,
                "--ws",
                args.ws,
                "--duration",
                str(args.wholebody_duration_s),
                "--session-id",
                session_id,
                "--runtime-lane",
                lane.name,
                "--runtime-instance-id",
                runtime_instance_id,
                "--runtime-run-id",
                runtime_run_id,
                "--out",
                str(evidence_dir / WHOLEBODY49_REPORT_FILENAME),
                "--source-out",
                str(evidence_dir / WHOLEBODY49_SOURCE_TRANSCRIPT_FILENAME),
                *[
                    token
                    for source_id in lane.source_ids
                    for token in ("--expected-source-id", str(source_id))
                ],
                "--auth-token-file",
                str(auth_token_file),
            ],
            args.wholebody_duration_s + 20.0,
            False,
            (),
        ),
    ]


def _write_report(output_dir: Path, summary: Mapping[str, object]) -> Path:
    report_path = output_dir / "summary.md"
    lines = [
        "# DS9 Live Validation Summary",
        "",
        f"- ok: `{summary.get('ok')}`",
        f"- started_at_utc: `{summary.get('started_at_utc')}`",
        f"- output_dir: `{summary.get('output_dir')}`",
        f"- pipeline_config: `{summary.get('pipeline_config')}`",
        f"- cameras_config: `{summary.get('cameras_config')}`",
        f"- runtime_log: `{summary.get('runtime_log')}`",
        "",
        "| Step | OK | RC | Seconds | Notes | Log |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for result in summary.get("results", []):
        if not isinstance(result, Mapping):
            continue
        lines.append(
            "| {name} | `{ok}` | `{returncode}` | `{duration_s}` | {notes} | `{log_path}` |".format(
                name=result.get("name"),
                ok=result.get("ok"),
                returncode=result.get("returncode"),
                duration_s=result.get("duration_s"),
                notes=str(result.get("notes") or "").replace("|", "\\|"),
                log_path=result.get("log_path"),
            )
        )
    lines.append("")
    lines.append("Use the per-step logs above as the evidence bundle for doc updates.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DS9 live-RTSP production validation bundle."
    )
    parser.add_argument(
        "--lane",
        choices=tuple(VALIDATION_LANES),
        default=BASELINE_LANE.name,
        help="Reviewed validation lane; arbitrary profile/config combinations are forbidden.",
    )
    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument("--cameras-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Evidence session identifier. Required when attaching; generated once "
            "for a runner-owned runtime."
        ),
    )
    parser.add_argument(
        "--ownership-evidence-dir",
        type=Path,
        default=None,
        help=(
            "Canonical supervisor launcher directory for checksum-covered behavior "
            "reports. Requires --no-spawn and an exact matching launch-plan.json."
        ),
    )
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--rest", default="http://127.0.0.1:8080")
    parser.add_argument("--rtsp-url", default="rtsp://127.0.0.1:8554/mosaic")
    parser.add_argument(
        "--auth-token-file",
        type=Path,
        default=None,
        help=(
            "Existing owner-only internal bearer file. Defaults to "
            "NOESIS_INTERNAL_AUTH_TOKEN_FILE, then the appliance gateway-token path."
        ),
    )
    parser.add_argument("--startup-timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--shutdown-timeout-s", type=float, default=PRODUCTION_SHUTDOWN_TIMEOUT_S
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Attach to an already-running DS9 runtime.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--skip",
        action="append",
        choices=GATE_NAMES,
        default=[],
        help="Skip a gate; may be repeated.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra runtime env KEY=VALUE; may be repeated.",
    )
    parser.add_argument("--rtsp-duration-s", type=float, default=20.0)
    parser.add_argument("--webrtc-duration-s", type=float, default=6.0)
    parser.add_argument("--webrtc-pt", type=int, default=103)
    parser.add_argument("--webrtc-min-rtp", type=int, default=10)
    parser.add_argument("--webrtc-min-decoded", type=int, default=1)
    parser.add_argument("--reid-duration-s", type=float, default=35.0)
    parser.add_argument(
        "--semantic-observation-duration-s", type=float, default=45.0
    )
    parser.add_argument(
        "--identity-evidence",
        type=Path,
        default=None,
        help=(
            "Exact owner-only DS9 identity-v2 JSONL. Required for the baseline "
            "occupied-scene semantic gate."
        ),
    )
    parser.add_argument(
        "--identity-require-cross-camera",
        action="store_true",
        help=(
            "Fail unless one shadow subject is observed on fresh ReID evidence "
            "from at least two cameras. This proves assignment-path exercise, "
            "not person-level accuracy."
        ),
    )
    parser.add_argument(
        "--identity-require-open-set",
        action="store_true",
        help=(
            "Fail unless a fresh shadow decision conservatively exposes no "
            "subject/SID. This proves non-force behavior, not unknown-person accuracy."
        ),
    )
    parser.add_argument("--bev-duration-s", type=float, default=45.0)
    parser.add_argument("--bridge-duration-s", type=float, default=75.0)
    parser.add_argument("--floorplan-max-age-sec", type=float, default=120.0)
    parser.add_argument("--floorplan-timeout-s", type=float, default=75.0)
    parser.add_argument("--ma-camera", default="family-room")
    parser.add_argument("--zero-copy-stats-duration-s", type=float, default=60.0)
    parser.add_argument("--zero-copy-rest-duration-s", type=float, default=90.0)
    parser.add_argument("--v3dt-bbox-duration-s", type=float, default=20.0)
    parser.add_argument(
        "--v3dt-bbox-attempts",
        type=int,
        default=V3DT_BBOX_DEFAULT_ATTEMPTS,
    )
    parser.add_argument("--v3dt-world-duration-s", type=float, default=45.0)
    parser.add_argument("--wholebody-duration-s", type=float, default=45.0)
    args = parser.parse_args()
    if args.session_id is None and not args.no_spawn:
        args.session_id = f"live-{_timestamp().lower()}"
    lane = _validation_lane(args)
    if args.pipeline_config is None:
        args.pipeline_config = lane.pipeline_config
    if args.cameras_config is None:
        args.cameras_config = lane.cameras_config
    if args.ownership_evidence_dir is not None:
        args.ownership_evidence_dir = (
            args.ownership_evidence_dir.expanduser().absolute()
        )
    if args.identity_evidence is None and args.ownership_evidence_dir is not None:
        args.identity_evidence = (
            args.ownership_evidence_dir.expanduser().absolute().parent
            / "runtime"
            / "identity_v2.jsonl"
        )
    return args


def _validate_supervisor_evidence_binding(args: argparse.Namespace) -> None:
    configured = getattr(args, "ownership_evidence_dir", None)
    if configured is None:
        return
    if not bool(args.no_spawn):
        raise ValueError("ownership evidence output requires --no-spawn")
    directory = Path(configured).expanduser().absolute()
    info = directory.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("ownership evidence directory is not a regular directory")
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("ownership evidence directory must be owner-only mode 0700")
    expected_identity_evidence = directory.parent / "runtime" / "identity_v2.jsonl"
    actual_identity_evidence = Path(args.identity_evidence).expanduser().absolute()
    if actual_identity_evidence != expected_identity_evidence:
        raise ValueError(
            "ownership-attached identity evidence must be the exact sibling runtime file"
        )
    args.identity_evidence = actual_identity_evidence
    plan_path = directory / "launch-plan.json"
    plan = strict_json_loads(
        read_private_file(
            plan_path,
            label="supervisor launch-plan.json",
            max_bytes=MAX_SUPERVISOR_JSON_BYTES,
        ),
        label="supervisor launch-plan.json",
    )
    lane = _validation_lane(args)
    session_id = _runtime_session_id(args)
    canonical_runtime = plan.get("canonical_runtime") if isinstance(plan, Mapping) else None
    session_paths = plan.get("session_paths") if isinstance(plan, Mapping) else None
    if (
        not isinstance(plan, Mapping)
        or type(plan.get("schema_version")) is not int
        or plan.get("schema_version") != 1
        or plan.get("contract") != "noesis.ds9.canonical_runtime_container"
        or plan.get("mode") != "plan"
        or plan.get("session_id") != session_id
        or plan.get("runtime_lane") != lane.name
        or plan.get("ready_for_explicit_run") is not True
        or not isinstance(canonical_runtime, Mapping)
        or canonical_runtime.get("ports") != CANONICAL_PORTS
        or canonical_runtime.get("endpoints") != CANONICAL_ENDPOINTS
        or canonical_runtime.get("source_ids")
        != [str(value) for value in lane.source_ids]
        or not isinstance(session_paths, Mapping)
        or session_paths.get("launcher_evidence") != str(directory)
    ):
        raise ValueError("supervisor launch plan session/lane contract mismatch")
    identity_path = directory / "runtime-identity.json"
    identity = strict_json_loads(
        read_private_file(
            identity_path,
            label="supervisor runtime-identity.json",
            max_bytes=MAX_SUPERVISOR_JSON_BYTES,
        ),
        label="supervisor runtime-identity.json",
    )
    if (
        not isinstance(identity, Mapping)
        or type(identity.get("schema_version")) is not int
        or identity.get("schema_version") != 1
        or identity.get("contract") != RUNTIME_IDENTITY_CONTRACT
        or identity.get("contract_version") != 1
        or identity.get("session_id") != session_id
        or identity.get("runtime_lane") != lane.name
        or identity.get("endpoints") != CANONICAL_ENDPOINTS
        or RUNTIME_ID_RE.fullmatch(str(identity.get("runtime_instance_id") or ""))
        is None
        or RUNTIME_ID_RE.fullmatch(str(identity.get("runtime_run_id") or "")) is None
    ):
        raise ValueError("supervisor runtime identity contract mismatch")
    args.supervisor_runtime_instance_id = str(identity["runtime_instance_id"])
    args.supervisor_runtime_run_id = str(identity["runtime_run_id"])


def _validate_runtime_contract_args(args: argparse.Namespace) -> None:
    lane = _validation_lane(args)
    _runtime_session_id(args)
    if lane is V3DT_LANE and args.ownership_evidence_dir is not None:
        skipped_required = V3DT_PROMOTION_REQUIRED_GATES.intersection(
            set(getattr(args, "skip", ()) or ())
        )
        if skipped_required:
            raise ValueError(
                "ownership-attached V3DT validation cannot skip promotion-required "
                f"gates: {sorted(skipped_required)}"
            )
    _validate_supervisor_evidence_binding(args)
    if getattr(args, "ownership_evidence_dir", None) is not None:
        observed_endpoints = {
            "websocket": str(args.ws),
            "rest": str(args.rest),
            "rtsp": str(args.rtsp_url),
        }
        if observed_endpoints != CANONICAL_ENDPOINTS:
            raise ValueError(
                "ownership-attached validation requires exact canonical endpoints"
            )
    if float(args.startup_timeout_s) <= 0:
        raise ValueError("startup timeout must be > 0")
    if float(args.shutdown_timeout_s) < PRODUCTION_SHUTDOWN_GRACE_S + 15.0:
        raise ValueError(
            "shutdown timeout must exceed the production shutdown grace by at least 15 seconds"
        )
    for label, value in (
        ("ReID duration", args.reid_duration_s),
        (
            "semantic observation duration",
            args.semantic_observation_duration_s,
        ),
        ("floorplan maximum age", args.floorplan_max_age_sec),
        ("floorplan timeout", args.floorplan_timeout_s),
        ("Wholebody occupied duration", args.wholebody_duration_s),
    ):
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    if lane in {WHOLEBODY49_S_LANE, WHOLEBODY49_X_LANE}:
        if float(args.wholebody_duration_s) < 30.0:
            raise ValueError("Wholebody occupied duration must be at least 30 seconds")
        if args.ownership_evidence_dir is not None and {
            "rtsp",
            "webrtc",
            "wholebody-occupied",
        }.intersection(set(args.skip or [])):
            raise ValueError(
                "ownership-attached Wholebody validation cannot skip RTSP, decoded "
                "media, or occupied-scene gates"
            )
    if lane in {BASELINE_LANE, V3DT_LANE} and getattr(args, "identity_evidence", None) is None:
        raise ValueError(
            f"{lane.name} validation requires an explicit --identity-evidence JSONL"
        )
    if lane is V3DT_LANE:
        V3DTBBoxTimeoutContract(
            duration_seconds=args.v3dt_bbox_duration_s,
            attempts=args.v3dt_bbox_attempts,
        )
    for label, actual, expected in (
        ("pipeline config", Path(args.pipeline_config), lane.pipeline_config),
        ("camera config", Path(args.cameras_config), lane.cameras_config),
    ):
        if actual.expanduser().resolve() != (REPO_ROOT / expected).resolve():
            raise ValueError(
                f"{lane.name} {label} must be hard-pinned to {expected}; got {actual}"
            )
    for label, value, scheme, default_port in (
        ("WebSocket", args.ws, "ws", 6008),
        ("REST", args.rest, "http", 8080),
        ("RTSP", args.rtsp_url, "rtsp", 8554),
    ):
        parsed = urlparse(str(value))
        if parsed.scheme != scheme:
            raise ValueError(f"{label} endpoint must use {scheme}://")
        host, port = _parse_host_port(str(value), default_port)
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(f"{label} endpoint must use an explicit loopback host")
        if not 1 <= port <= 65535:
            raise ValueError(f"{label} endpoint port is invalid")


def _validate_identity_evidence_source(args: argparse.Namespace) -> None:
    path = Path(args.identity_evidence).expanduser().absolute()
    payload = read_private_file(
        path,
        label="DS9 semantic identity evidence source",
        max_bytes=256 * 1024 * 1024,
    )
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("semantic identity evidence source is empty or incomplete")


def main() -> int:
    args = parse_args()
    lane = _validation_lane(args)
    output_dir = _prepare_private_output_dir(
        args.output_dir
        or (DS9_ROOT / "build" / "live_validation" / _timestamp())
    )
    args.output_dir = output_dir
    if args.identity_evidence is None:
        args.identity_evidence = output_dir / "identity_v2.jsonl"

    results: list[StepResult] = []
    runtime_proc: subprocess.Popen[None] | None = None
    runtime_log: Path | None = None
    readiness_evidence: CapabilityEvidence | None = None
    final_readiness_evidence: CapabilityEvidence | None = None
    started_at = _timestamp()
    token = ""
    token_file: Path | None = None
    runtime_env: dict[str, str] = {}

    try:
        _validate_runtime_contract_args(args)
        token_file = _resolve_auth_token_file(os.environ, args.auth_token_file)
        token = _load_auth_token(token_file)
        runtime_env = _runtime_env(
            args.env,
            token_file=token_file,
            session_id=_runtime_session_id(args),
            identity_evidence_path=Path(args.identity_evidence),
            lane=lane,
        )
    except (Exception, SystemExit) as exc:
        results.append(
            StepResult(
                name="runtime-contract",
                ok=False,
                returncode=None,
                duration_s=0.0,
                log_path="",
                command="validate canonical DS9 launch and internal auth",
                notes=str(exc) or type(exc).__name__,
            )
        )

    try:
        if results:
            raise SystemExit(1)
        if not args.skip_preflight:
            result = _run_command(
                name="preflight",
                cmd=[
                    sys.executable,
                    "DS9/scripts/ds9_preflight.py",
                    "--config",
                    str(args.pipeline_config),
                    "--cameras-config",
                    str(args.cameras_config),
                ],
                output_dir=output_dir,
                env=runtime_env,
                timeout_s=60.0,
            )
            results.append(result)
            if not result.ok:
                raise SystemExit(1)

        if args.no_spawn:
            runtime_log = None
        else:
            runtime_proc, runtime_log = _spawn_runtime(args, output_dir, runtime_env)

        readiness_started = time.monotonic()
        ready, ready_note, readiness_evidence = _runtime_ready(
            args, runtime_proc, token=token
        )
        if ready and readiness_evidence is not None:
            try:
                _bind_runtime_identity(args, readiness_evidence)
            except ValueError as exc:
                ready = False
                ready_note = str(exc)
        results.append(
            StepResult(
                name="runtime-ready",
                ok=ready,
                returncode=(
                    runtime_proc.returncode
                    if runtime_proc and runtime_proc.poll() is not None
                    else None
                ),
                duration_s=round(time.monotonic() - readiness_started, 3),
                log_path=_repo_rel(runtime_log) if runtime_log else "",
                command="authenticated advancing capability, WebSocket health, and RTSP DESCRIBE checks",
                notes=ready_note,
            )
        )
        if not ready:
            raise SystemExit(1)

        skips = set(args.skip or [])
        assert token_file is not None
        for name, cmd, timeout_s, timeout_ok, signatures in _build_gates(
            args, auth_token_file=token_file
        ):
            if name in skips:
                results.append(
                    StepResult(
                        name=name,
                        ok=True,
                        returncode=None,
                        duration_s=0.0,
                        log_path="",
                        command=_cmd_text(cmd),
                        notes="skipped",
                    )
                )
                continue
            if name == "semantic-observation":
                try:
                    _validate_identity_evidence_source(args)
                except Exception as exc:
                    results.append(
                        StepResult(
                            name=name,
                            ok=False,
                            returncode=None,
                            duration_s=0.0,
                            log_path="",
                            command=_cmd_text(cmd),
                            notes=f"identity_evidence_invalid={exc}",
                        )
                    )
                    if args.fail_fast:
                        break
                    continue
            result = _run_command(
                name=name,
                cmd=cmd,
                output_dir=output_dir,
                env=runtime_env,
                timeout_s=timeout_s,
                timeout_ok=timeout_ok,
                fail_signatures=signatures,
            )
            _apply_completed_behavior_gate_report(
                name,
                result,
                args=args,
                lane=lane,
            )
            results.append(result)
            if args.fail_fast and not result.ok:
                break
        final_ready_started = time.monotonic()
        final_ready, final_ready_note, final_readiness_evidence = _runtime_ready(
            args,
            runtime_proc,
            token=token,
        )
        results.append(
            StepResult(
                name="runtime-final-ready",
                ok=final_ready,
                returncode=(
                    runtime_proc.returncode
                    if runtime_proc is not None and runtime_proc.poll() is not None
                    else None
                ),
                duration_s=round(time.monotonic() - final_ready_started, 3),
                log_path=_repo_rel(runtime_log) if runtime_log else "",
                command="repeat authenticated advancing runtime contracts",
                notes=final_ready_note,
            )
        )
    except SystemExit:
        pass
    except KeyboardInterrupt:
        results.append(
            StepResult(
                name="orchestrator",
                ok=False,
                returncode=None,
                duration_s=0.0,
                log_path=_repo_rel(runtime_log) if runtime_log else "",
                command="DS9 live validation orchestration",
                notes="interrupted",
            )
        )
    except Exception as exc:
        results.append(
            StepResult(
                name="orchestrator",
                ok=False,
                returncode=None,
                duration_s=0.0,
                log_path=_repo_rel(runtime_log) if runtime_log else "",
                command="DS9 live validation orchestration",
                notes=f"{type(exc).__name__}: {exc}",
            )
        )
    finally:
        results.append(
            _shutdown_runtime(runtime_proc, runtime_log, args.shutdown_timeout_s)
        )

    summary = {
        "ok": all(result.ok for result in results),
        "started_at_utc": started_at,
        "output_dir": _repo_rel(output_dir),
        "pipeline_config": str(args.pipeline_config),
        "cameras_config": str(args.cameras_config),
        "launch_contract": (
            f"owned_reviewed_{lane.name}"
            if not args.no_spawn
            else "attached_runtime_launch_unverified"
        ),
        "runtime_lane": lane.name,
        "session_id": _runtime_session_id(args),
        "ownership_evidence_dir": (
            str(_behavior_evidence_dir(args))
            if args.ownership_evidence_dir is not None
            else None
        ),
        "pgie_profile": lane.pgie_profile if not args.no_spawn else None,
        "model_size": lane.model_size if not args.no_spawn else None,
        "tracking_mode": lane.tracking_mode if not args.no_spawn else None,
        "internal_auth_mode": "required",
        "ws": args.ws,
        "rest": args.rest,
        "rtsp_url": args.rtsp_url,
        "runtime_log": _repo_rel(runtime_log) if runtime_log else "",
        "readiness": (
            readiness_evidence.public_payload() if readiness_evidence else None
        ),
        "final_readiness": (
            final_readiness_evidence.public_payload()
            if final_readiness_evidence
            else None
        ),
        "results": [asdict(result) for result in results],
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    report_path = _write_report(output_dir, summary)
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "summary": _repo_rel(summary_path),
                "report": _repo_rel(report_path),
            },
            indent=2,
        )
    )
    return 0 if bool(summary["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
