"""Shared identity-v2 service used by the canonical DS9 runtime.

This module is deliberately independent of DeepStream metadata wrappers.  Each
runtime extracts immutable primitive evidence while walking a source frame once,
then hands the complete Python-only batch to :class:`IdentityV2Service`.
"""

from __future__ import annotations

import configparser
import hashlib
import importlib
import json
import math
import os
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml
from pydantic import ValidationError

from noesis_core.contracts.identity_calibration import (
    IdentityAuthorityCutoverArtifact,
    IdentityOpenSetCalibrationArtifact,
    ShadowIdentityEvidenceRecord,
)
from noesis_core.private_paths import PrivatePathError, read_private_file
from noesis_core.strict_json import StrictJSONError, strict_json_loads

from reid.identity_v2 import (
    CameraOverlapEdge,
    FrameBatchResult,
    IdentityFrameCoordinator,
    IdentityStore,
    IdentityV2Runtime,
    IdentityEvidenceRecorder,
    identity_tracklet_id,
    load_migration_review,
    OpenSetPolicy,
    OverlapSharePermit,
    PrimitiveFrameObservation,
    PublicIdentityOverlay,
)
from reid.identity_v2.evidence import (
    DEFAULT_MAX_AGE_S,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_RECORDS,
)


class IdentityV2ConfigurationError(RuntimeError):
    """Raised when the canonical identity-v2 path is not fully specified."""


class IdentityV2Mode(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    AUTHORITATIVE = "authoritative"


_EMBEDDING_PROVENANCE_FIELDS = (
    "embedding_sequence",
    "embedding_model_sha256",
    "embedding_dimension",
)


@dataclass(frozen=True)
class OverlapProofPolicy:
    camera_a: str
    camera_b: str
    max_world_dist_m: float
    max_time_delta_s: float
    require_appearance_sim: float

    @property
    def pair(self) -> Tuple[str, str]:
        return tuple(sorted((self.camera_a, self.camera_b)))  # type: ignore[return-value]


@dataclass
class IdentityFramePrimitive:
    """A server-produced person primitive detached from DeepStream metadata."""

    camera_id: str
    tracker_id: str
    frame_id: int
    public_track: MutableMapping[str, Any]
    embedding: Optional[Sequence[float]]
    diagnostic_track: Optional[MutableMapping[str, Any]] = None
    bbox: Optional[Sequence[float]] = None
    frame_size: Optional[Sequence[int]] = None
    detection_confidence: Optional[float] = None
    tracker_confidence: Optional[float] = None
    world_xyz: Optional[Sequence[float]] = None
    world_valid: bool = False


@dataclass(frozen=True)
class IdentityFrameServiceResult:
    batch: FrameBatchResult
    primitive_count: int
    evidence_count: int
    overlap_permit_count: int


@dataclass(frozen=True)
class IdentityOsdDecision:
    camera_id: str
    frame_id: int
    tracker_id: str
    identity_state: str
    compatibility_sid: Optional[int]
    display_name: Optional[str]


@dataclass(frozen=True)
class _RecentProof:
    tracklet_id: str
    camera_id: str
    observed_at: float
    embedding: Tuple[float, ...]
    world_xyz: Tuple[float, float, float]
    subject_id: Optional[str]


@dataclass(frozen=True)
class _HeldOverlay:
    tracklet_id: str
    camera_id: str
    evidence_at: float
    overlay: PublicIdentityOverlay


def _require_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise IdentityV2ConfigurationError(f"{field} must be non-empty")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loaded_reid_transform_component() -> Mapping[str, str]:
    """Describe the exact native bridge Python will use for ReID extraction."""

    try:
        module = importlib.import_module("noesis_reid_meta_ext")
        raw_path = getattr(module, "__file__", None)
        path = Path(str(raw_path or "")).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError("loaded module has no non-empty regular file")
    except Exception as exc:
        return {
            "module": "noesis_reid_meta_ext",
            "status": "unavailable",
            "reason": type(exc).__name__,
        }
    return {
        "module": "noesis_reid_meta_ext",
        "status": "loaded",
        "basename": path.name,
        "sha256": _sha256_file(path),
    }


def _python_scoring_component_fingerprints() -> Mapping[str, Mapping[str, str]]:
    """Hash loaded Python components that transform or score embeddings."""

    components: dict[str, Mapping[str, str]] = {}
    for role, module_name in (
        ("frame_quality_and_normalization", "noesis.identity_v2_service"),
        ("gallery_similarity", "reid.identity_v2.runtime"),
        ("open_set_scoring", "reid.identity_v2.scoring"),
    ):
        module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        path = Path(str(raw_path or "")).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise IdentityV2ConfigurationError(
                f"identity-v2 executable scoring component is missing: {module_name}"
            )
        components[role] = {
            "module": module_name,
            "basename": path.name,
            "sha256": _sha256_file(path),
        }
    return components


def _canonical_profile_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_profile_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_profile_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentityV2ConfigurationError(
                "identity-v2 model semantic profile contains a non-finite value"
            )
        return value
    return str(value).strip()


def _nvinfer_semantic_sections(
    path: Path,
    *,
    model_fingerprint: str,
    model_engine_path: Path,
) -> Mapping[str, Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
        parser = configparser.ConfigParser(
            interpolation=None,
            inline_comment_prefixes=("#",),
            strict=False,
        )
        parser.read_string(raw.decode("utf-8"))
    except Exception as exc:
        raise IdentityV2ConfigurationError(
            f"identity-v2 ReID nvinfer config is invalid: {path}: {exc}"
        ) from exc
    if "property" not in parser:
        raise IdentityV2ConfigurationError(
            f"identity-v2 ReID nvinfer config has no [property] section: {path}"
        )
    sections: dict[str, dict[str, Any]] = {}
    for section in sorted(parser.sections()):
        values: dict[str, Any] = {}
        for key, raw_value in sorted(parser[section].items()):
            value = str(raw_value).strip()
            if "file" in key or "path" in key or key.endswith("-lib"):
                reference: dict[str, str] = {"basename": Path(value).name}
                if key == "model-engine-file":
                    configured_engine = Path(value).expanduser()
                    if not configured_engine.is_absolute():
                        configured_engine = path.parent / configured_engine
                    configured_engine = configured_engine.resolve()
                    if configured_engine != model_engine_path.resolve():
                        raise IdentityV2ConfigurationError(
                            "models.reid.engine and nvinfer model-engine-file must "
                            "resolve to the same active engine"
                        )
                    reference["sha256"] = model_fingerprint
                elif key in {"custom-lib-path", "custom-lib"}:
                    candidate = Path(value).expanduser()
                    if not candidate.is_absolute():
                        candidate = path.parent / candidate
                    candidate = candidate.resolve()
                    if not candidate.is_file() or candidate.stat().st_size <= 0:
                        raise IdentityV2ConfigurationError(
                            f"identity-v2 ReID custom library is missing: {candidate}"
                        )
                    reference["sha256"] = _sha256_file(candidate)
                values[key] = reference
            else:
                values[key] = value
        sections[section] = values
    return sections


def identity_model_semantic_profile_sha256(
    *,
    reid_config: Mapping[str, Any],
    model_fingerprint: str,
    model_layer: str,
    embedding_dim: int,
    repo_root: Path,
    pipeline_yaml_path: Path,
    reid_transform_component: Optional[Mapping[str, str]] = None,
) -> str:
    """Hash the active crop, preprocessing, tensor, and gallery score semantics.

    Path spellings are intentionally excluded. Runtime-affecting YAML values,
    parsed nvinfer properties, active engine bytes, and fixed scorer/crop
    algorithms are represented in one canonical document.
    """

    nvinfer_path = _resolve_required_file(
        reid_config.get("config-file-path"),
        field="models.reid.config-file-path",
        repo_root=repo_root,
        pipeline_yaml_path=pipeline_yaml_path,
    )
    model_engine_path = _resolve_required_file(
        reid_config.get("engine"),
        field="models.reid.engine",
        repo_root=repo_root,
        pipeline_yaml_path=pipeline_yaml_path,
    )
    if _sha256_file(model_engine_path) != str(model_fingerprint):
        raise IdentityV2ConfigurationError(
            "identity-v2 semantic profile engine bytes changed during startup"
        )
    model_config = {
        str(key): _canonical_profile_value(value)
        for key, value in sorted(reid_config.items(), key=lambda row: str(row[0]))
        if str(key) not in {"engine", "config-file-path"}
    }
    transform_component = dict(
        _loaded_reid_transform_component()
        if reid_transform_component is None
        else reid_transform_component
    )
    profile = {
        "contract": "noesis.identity.model_semantic_profile",
        "contract_version": 1,
        "model": {
            "engine_sha256": str(model_fingerprint),
            "output_layer": str(model_layer),
            "embedding_dim": int(embedding_dim),
        },
        "model_config": model_config,
        "nvinfer_config_basename": nvinfer_path.name,
        "nvinfer_sections": _nvinfer_semantic_sections(
            nvinfer_path,
            model_fingerprint=model_fingerprint,
            model_engine_path=model_engine_path,
        ),
        "executable_components": {
            "reid_metadata_extension": transform_component,
            "python": _python_scoring_component_fingerprints(),
        },
        "runtime_semantics": {
            "crop": "deepstream_secondary_object_bbox_direct_v1",
            "quality": "bbox_height_detection_tracker_blend_v1",
            "embedding_normalization": "l2_norm_v1",
            "candidate_similarity": "maximum_cosine_over_gallery_exemplars_v1",
            "resolution_scope": "complete_source_frame_batch_v1",
        },
    }
    canonical = json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"noesis-identity-model-semantic-profile-v1\0" + canonical
    ).hexdigest()


def _resolve_required_file(
    raw_path: Any,
    *,
    field: str,
    repo_root: Path,
    pipeline_yaml_path: Path,
) -> Path:
    text = _require_text(raw_path, field)
    candidate = Path(text).expanduser()
    candidates = (
        (candidate,)
        if candidate.is_absolute()
        else (
            repo_root / candidate,
            pipeline_yaml_path.parent / candidate,
        )
    )
    for item in candidates:
        resolved = item.resolve()
        if resolved.is_file() and resolved.stat().st_size > 0:
            return resolved
    rendered = ", ".join(str(item.resolve()) for item in candidates)
    raise IdentityV2ConfigurationError(
        f"{field} does not resolve to a non-empty regular file: {rendered}"
    )


def _read_mode(environ: Mapping[str, str]) -> IdentityV2Mode:
    raw = (
        str(environ.get("NOESIS_IDENTITY_V2_MODE", "shadow") or "shadow")
        .strip()
        .lower()
    )
    try:
        return IdentityV2Mode(raw)
    except ValueError as exc:
        choices = "|".join(item.value for item in IdentityV2Mode)
        raise IdentityV2ConfigurationError(
            f"NOESIS_IDENTITY_V2_MODE must be one of {choices}; got {raw!r}"
        ) from exc


def _bounded_env_number(
    environ: Mapping[str, str],
    name: str,
    default: int | float,
    *,
    minimum: int | float,
    maximum: int | float,
    integer: bool,
) -> int | float:
    raw = str(environ.get(name, default) or default).strip()
    try:
        value = int(raw) if integer else float(raw)
    except (TypeError, ValueError) as exc:
        raise IdentityV2ConfigurationError(f"{name} must be numeric") from exc
    if not math.isfinite(float(value)) or value < minimum or value > maximum:
        raise IdentityV2ConfigurationError(
            f"{name} must be within [{minimum}, {maximum}]"
        )
    return value


def _load_open_set_policy(
    *,
    environ: Mapping[str, str],
    repo_root: Path,
    model_fingerprint: str,
    model_layer: str,
    embedding_dim: int,
    model_semantic_profile_sha256: str,
) -> OpenSetPolicy:
    raw_path = str(environ.get("NOESIS_IDENTITY_V2_SCORING_ARTIFACT", "") or "").strip()
    if not raw_path:
        return OpenSetPolicy(
            model_semantic_profile_sha256=model_semantic_profile_sha256
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.absolute()
    if not path.is_file() or path.stat().st_size <= 0:
        raise IdentityV2ConfigurationError(
            f"identity-v2 scoring artifact is missing or empty: {path}"
        )
    try:
        raw = read_private_file(
            path,
            label="identity-v2 scoring artifact",
            max_bytes=4 * 1024 * 1024,
        )
        payload = strict_json_loads(
            raw,
            label="identity-v2 scoring artifact",
        )
    except PrivatePathError as exc:
        raise IdentityV2ConfigurationError(str(exc)) from exc
    except Exception as exc:
        raise IdentityV2ConfigurationError(
            f"identity-v2 scoring artifact is not valid JSON: {path}: {exc}"
        ) from exc
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    expected_sha256 = (
        str(environ.get("NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256", "") or "")
        .strip()
        .lower()
    )
    if not expected_sha256:
        raise IdentityV2ConfigurationError(
            "identity-v2 scoring artifact requires an independently approved "
            "NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256 pin"
        )
    if expected_sha256 != artifact_sha256:
        raise IdentityV2ConfigurationError(
            "identity-v2 scoring artifact SHA-256 does not match the approved pin"
        )
    try:
        artifact = IdentityOpenSetCalibrationArtifact.model_validate(payload)
    except ValidationError as exc:
        raise IdentityV2ConfigurationError(
            f"identity-v2 scoring artifact contract is invalid: {exc}"
        ) from exc
    if artifact.model_sha256 != model_fingerprint:
        raise IdentityV2ConfigurationError(
            "identity-v2 scoring artifact model_sha256 does not match active engine bytes"
        )
    if artifact.model_layer != model_layer:
        raise IdentityV2ConfigurationError(
            "identity-v2 scoring artifact model_layer does not match the active output"
        )
    if artifact.embedding_dim != int(embedding_dim):
        raise IdentityV2ConfigurationError(
            "identity-v2 scoring artifact embedding_dim does not match the active output"
        )
    approved_semantic_profile_sha256 = (
        str(environ.get("NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256", "") or "")
        .strip()
        .lower()
    )
    if not approved_semantic_profile_sha256:
        raise IdentityV2ConfigurationError(
            "identity-v2 scoring artifact requires an approved "
            "NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256"
        )
    if approved_semantic_profile_sha256 != model_semantic_profile_sha256:
        raise IdentityV2ConfigurationError(
            "identity-v2 approved semantic profile does not match the profile "
            "derived from the active ReID and nvinfer configuration"
        )
    if artifact.model_semantic_profile_sha256 != model_semantic_profile_sha256:
        raise IdentityV2ConfigurationError(
            "identity-v2 scoring artifact semantic profile does not match the active runtime"
        )
    artifact_id = f"sha256:{artifact_sha256}"
    try:
        return OpenSetPolicy(
            **artifact.policy.model_dump(mode="python"),
            calibration_artifact_id=artifact_id,
            model_semantic_profile_sha256=artifact.model_semantic_profile_sha256,
            maximum_resident_candidates=(
                artifact.gallery_limits.maximum_resident_candidate_count
            ),
            maximum_visitor_candidates=(
                artifact.gallery_limits.maximum_visitor_candidate_count
            ),
            maximum_total_candidates=(
                artifact.gallery_limits.maximum_total_candidate_count
            ),
            maximum_exemplars_per_candidate=(
                artifact.gallery_limits.maximum_exemplars_per_candidate
            ),
        )
    except (TypeError, ValueError) as exc:
        raise IdentityV2ConfigurationError(
            f"identity-v2 scoring artifact policy is invalid: {exc}"
        ) from exc


def _identity_runtime_name(environ: Mapping[str, str], pipeline_yaml_path: Path) -> str:
    _ = pipeline_yaml_path
    configured = str(environ.get("NOESIS_DEEPSTREAM_MAJOR", "") or "").strip()
    if configured and configured != "9":
        raise IdentityV2ConfigurationError(
            "NOESIS_DEEPSTREAM_MAJOR must be 9 for identity authority"
        )
    return "ds9"


def identity_authority_runtime_profile_sha256(*, runtime: str) -> str:
    """Hash the executable whole-frame identity surface for cutover evidence."""

    normalized_runtime = str(runtime or "").strip().lower()
    if normalized_runtime != "ds9":
        raise IdentityV2ConfigurationError(
            "identity authority runtime profile requires runtime=ds9"
        )
    source_root = Path(__file__).resolve().parents[1]
    shared = {
        "identity_service": source_root / "noesis" / "identity_v2_service.py",
        "identity_osd": source_root / "noesis" / "identity_v2_osd.py",
        "coordinator": source_root / "reid" / "identity_v2" / "coordinator.py",
        "models": source_root / "reid" / "identity_v2" / "models.py",
        "resolver": source_root / "reid" / "identity_v2" / "resolver.py",
        "runtime": source_root / "reid" / "identity_v2" / "runtime.py",
        "scoring": source_root / "reid" / "identity_v2" / "scoring.py",
    }
    runtime_components = {
        "runtime_entry": source_root / "DS9" / "noesis" / "ds9_runtime_core.py",
        "analytics_hook": source_root / "DS9" / "noesis" / "pipelines" / "hooks.py",
    }
    components = {**shared, **runtime_components}
    fingerprints = {}
    for role, path in sorted(components.items()):
        resolved = path.resolve()
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise IdentityV2ConfigurationError(
                f"identity authority runtime component is missing: {role}"
            )
        fingerprints[role] = {
            "basename": resolved.name,
            "sha256": _sha256_file(resolved),
        }
    body = json.dumps(
        {
            "contract": "noesis.identity.authority_runtime_profile",
            "contract_version": 1,
            "runtime": normalized_runtime,
            "components": fingerprints,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"noesis-identity-authority-runtime-profile-v1\0" + body
    ).hexdigest()


def _load_authority_cutover_artifact(
    *,
    environ: Mapping[str, str],
    repo_root: Path,
    runtime: str,
    model_fingerprint: str,
    model_layer: str,
    embedding_dim: int,
    model_semantic_profile_sha256: str,
    scoring_artifact_id: Optional[str],
    authority_runtime_profile_sha256: str,
    topology_path: Path,
    camera_ids: Sequence[str],
) -> str:
    raw_path = str(
        environ.get("NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT", "") or ""
    ).strip()
    if not raw_path:
        raise IdentityV2ConfigurationError(
            "authoritative identity-v2 requires a separate scorer-independent "
            "NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT with coordinator replay "
            "and occupied-scene evidence"
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.absolute()
    try:
        raw = read_private_file(
            path,
            label="identity-v2 authority cutover artifact",
            max_bytes=4 * 1024 * 1024,
        )
    except PrivatePathError as exc:
        raise IdentityV2ConfigurationError(str(exc)) from exc
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    expected_sha256 = (
        str(
            environ.get("NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT_SHA256", "")
            or ""
        )
        .strip()
        .lower()
    )
    if not expected_sha256:
        raise IdentityV2ConfigurationError(
            "identity-v2 authority cutover artifact requires an independently "
            "approved NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT_SHA256 pin"
        )
    if expected_sha256 != artifact_sha256:
        raise IdentityV2ConfigurationError(
            "identity-v2 authority cutover artifact SHA-256 does not match the approved pin"
        )
    try:
        artifact = IdentityAuthorityCutoverArtifact.model_validate(
            strict_json_loads(
                raw,
                label="identity-v2 authority cutover artifact",
            )
        )
    except (StrictJSONError, ValidationError) as exc:
        raise IdentityV2ConfigurationError(
            f"identity-v2 authority cutover artifact contract is invalid: {exc}"
        ) from exc

    scoring_sha256 = str(scoring_artifact_id or "")
    if not scoring_sha256.startswith("sha256:"):
        raise IdentityV2ConfigurationError(
            "identity-v2 authority cutover requires artifact-backed scorer identity"
        )
    scoring_sha256 = scoring_sha256.removeprefix("sha256:")
    expected_camera_ids = tuple(sorted({str(value).strip() for value in camera_ids}))
    topology_sha256 = _sha256_file(topology_path)
    comparisons = (
        ("runtime", artifact.runtime, runtime),
        ("model SHA-256", artifact.model_sha256, model_fingerprint),
        ("model layer", artifact.model_layer, model_layer),
        ("embedding dimension", artifact.embedding_dim, int(embedding_dim)),
        (
            "model semantic profile",
            artifact.model_semantic_profile_sha256,
            model_semantic_profile_sha256,
        ),
        ("scoring artifact", artifact.scoring_artifact_sha256, scoring_sha256),
        (
            "authority runtime profile",
            artifact.authority_runtime_profile_sha256,
            authority_runtime_profile_sha256,
        ),
        ("camera topology", artifact.camera_topology_sha256, topology_sha256),
        ("camera IDs", artifact.camera_ids, expected_camera_ids),
    )
    for label, declared, active in comparisons:
        if declared != active:
            raise IdentityV2ConfigurationError(
                f"identity-v2 authority cutover {label} does not match the active runtime"
            )

    evidence_paths: set[Path] = set()
    for evidence in (artifact.coordinator_replay, artifact.occupied_scene):
        evidence_path = Path(evidence.evidence_path).expanduser()
        if not evidence_path.is_absolute():
            evidence_path = path.parent / evidence_path
        evidence_path = evidence_path.absolute()
        if evidence_path == path or evidence_path in evidence_paths:
            raise IdentityV2ConfigurationError(
                "identity-v2 authority gates require distinct evidence files"
            )
        evidence_paths.add(evidence_path)
        try:
            evidence_bytes = read_private_file(
                evidence_path,
                label=f"identity-v2 {evidence.evidence_kind} evidence",
                max_bytes=64 * 1024 * 1024,
            )
        except PrivatePathError as exc:
            raise IdentityV2ConfigurationError(str(exc)) from exc
        if len(evidence_bytes) != evidence.evidence_size_bytes:
            raise IdentityV2ConfigurationError(
                f"identity-v2 {evidence.evidence_kind} evidence size changed"
            )
        if hashlib.sha256(evidence_bytes).hexdigest() != evidence.evidence_sha256:
            raise IdentityV2ConfigurationError(
                f"identity-v2 {evidence.evidence_kind} evidence SHA-256 changed"
            )
    return f"sha256:{artifact_sha256}"


def _load_strict_topology(
    *,
    topology_path: Path,
    camera_labels: Mapping[int, str],
) -> Tuple[Tuple[CameraOverlapEdge, ...], Tuple[OverlapProofPolicy, ...]]:
    if not topology_path.is_file():
        raise IdentityV2ConfigurationError(
            f"identity-v2 camera topology is required: {topology_path}"
        )
    try:
        raw = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IdentityV2ConfigurationError(
            f"identity-v2 camera topology is invalid YAML: {topology_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping) or int(raw.get("version", 0) or 0) != 1:
        raise IdentityV2ConfigurationError("camera topology must be a version 1 object")
    if bool(raw.get("overlap_allow_appearance_only", False)):
        raise IdentityV2ConfigurationError(
            "identity-v2 forbids appearance-only overlap permits"
        )
    cameras_raw = raw.get("cameras")
    if not isinstance(cameras_raw, Mapping) or not cameras_raw:
        raise IdentityV2ConfigurationError(
            "camera topology requires a non-empty cameras map"
        )
    configured_by_name: dict[str, int] = {}
    configured_by_source: dict[int, str] = {}
    for name_raw, spec in cameras_raw.items():
        name = _require_text(name_raw, "topology camera name")
        if not isinstance(spec, Mapping) or "source_id" not in spec:
            raise IdentityV2ConfigurationError(
                f"topology camera {name!r} requires source_id"
            )
        source_id = int(spec["source_id"])
        if name in configured_by_name or source_id in configured_by_source:
            raise IdentityV2ConfigurationError(
                "camera topology contains duplicate names or source IDs"
            )
        configured_by_name[name] = source_id
        configured_by_source[source_id] = name

    runtime_by_source = {
        int(source): _require_text(name, "runtime camera name")
        for source, name in camera_labels.items()
    }
    if runtime_by_source != configured_by_source:
        raise IdentityV2ConfigurationError(
            "camera topology does not exactly match the active runtime camera map: "
            f"configured={configured_by_source!r} runtime={runtime_by_source!r}"
        )

    overlap_raw = raw.get("overlaps", ())
    if not isinstance(overlap_raw, Sequence) or isinstance(overlap_raw, (str, bytes)):
        raise IdentityV2ConfigurationError("camera topology overlaps must be a list")
    coordinator_edges: list[CameraOverlapEdge] = []
    proof_policies: list[OverlapProofPolicy] = []
    seen_pairs: set[Tuple[str, str]] = set()
    for index, item in enumerate(overlap_raw):
        if not isinstance(item, Mapping):
            raise IdentityV2ConfigurationError(
                f"topology overlap {index} must be an object"
            )
        if not bool(item.get("enabled", True)):
            continue
        names = item.get("cameras")
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or len(names) != 2
        ):
            raise IdentityV2ConfigurationError(
                f"topology overlap {index} requires exactly two cameras"
            )
        camera_a = _require_text(names[0], f"overlap {index} camera_a")
        camera_b = _require_text(names[1], f"overlap {index} camera_b")
        if camera_a not in configured_by_name or camera_b not in configured_by_name:
            raise IdentityV2ConfigurationError(
                f"topology overlap {index} references an unknown camera"
            )
        pair = tuple(sorted((camera_a, camera_b)))
        if pair in seen_pairs:
            raise IdentityV2ConfigurationError(
                f"duplicate topology overlap pair: {pair!r}"
            )
        seen_pairs.add(pair)
        max_time = float(item.get("max_time_delta_s"))
        max_distance = float(item.get("max_world_dist_m"))
        min_similarity = float(item.get("require_appearance_sim"))
        if not math.isfinite(max_time) or max_time <= 0.0:
            raise IdentityV2ConfigurationError(
                "overlap max_time_delta_s must be positive"
            )
        if not math.isfinite(max_distance) or max_distance <= 0.0:
            raise IdentityV2ConfigurationError(
                "overlap max_world_dist_m must be positive"
            )
        if not math.isfinite(min_similarity) or not -1.0 <= min_similarity <= 1.0:
            raise IdentityV2ConfigurationError(
                "overlap require_appearance_sim must be within [-1, 1]"
            )
        coordinator_edges.append(
            CameraOverlapEdge(camera_a, camera_b, max_batch_gap_s=max_time)
        )
        proof_policies.append(
            OverlapProofPolicy(
                camera_a=pair[0],
                camera_b=pair[1],
                max_world_dist_m=max_distance,
                max_time_delta_s=max_time,
                require_appearance_sim=min_similarity,
            )
        )
    return tuple(coordinator_edges), tuple(proof_policies)


class IdentityV2Service:
    """Own one store/runtime/coordinator and adapt complete source frames."""

    def __init__(
        self,
        *,
        mode: IdentityV2Mode,
        run_id: str,
        model_engine_path: Path,
        model_fingerprint: str,
        model_layer: str,
        embedding_dim: int,
        model_semantic_profile_sha256: str,
        authority_runtime_profile_sha256: str,
        authority_cutover_artifact_id: Optional[str],
        store: IdentityStore,
        runtime: IdentityV2Runtime,
        coordinator: IdentityFrameCoordinator,
        overlap_policies: Sequence[OverlapProofPolicy],
        evidence_recorder: Optional[IdentityEvidenceRecorder] = None,
        migration_review: Optional[Mapping[str, Any]] = None,
        osd_cache_frames: int = 8,
    ) -> None:
        self.mode = IdentityV2Mode(mode)
        self.run_id = _require_text(run_id, "identity-v2 run_id")
        self.model_engine_path = model_engine_path.resolve()
        self.model_fingerprint = _require_text(model_fingerprint, "model fingerprint")
        self.model_layer = _require_text(model_layer, "model layer")
        self.embedding_dim = int(embedding_dim)
        self.model_semantic_profile_sha256 = _require_text(
            model_semantic_profile_sha256,
            "model semantic profile SHA-256",
        )
        self.authority_runtime_profile_sha256 = _require_text(
            authority_runtime_profile_sha256,
            "identity authority runtime profile SHA-256",
        )
        self.authority_cutover_artifact_id = (
            None
            if authority_cutover_artifact_id is None
            else _require_text(
                authority_cutover_artifact_id,
                "identity authority cutover artifact ID",
            )
        )
        if self.mode is IdentityV2Mode.AUTHORITATIVE and (
            self.authority_cutover_artifact_id is None
        ):
            raise IdentityV2ConfigurationError(
                "authoritative identity-v2 requires a verified authority cutover artifact"
            )
        self.store = store
        self.runtime = runtime
        self.coordinator = coordinator
        self.overlap_policies = tuple(overlap_policies)
        self.evidence_recorder = evidence_recorder
        self.migration_review = (
            None if migration_review is None else dict(migration_review)
        )
        self._osd_cache_frames = max(2, min(64, int(osd_cache_frames)))
        self._osd_decisions: dict[Tuple[str, int, str], IdentityOsdDecision] = {}
        self._osd_frame_order: list[Tuple[str, int]] = []
        self._overlap_by_pair = {
            policy.pair: policy for policy in self.overlap_policies
        }
        self._max_overlap_age_s = max(
            (policy.max_time_delta_s for policy in self.overlap_policies),
            default=0.0,
        )
        self._recent_proofs: dict[str, _RecentProof] = {}
        self._held_overlays: dict[str, _HeldOverlay] = {}
        self._identity_hold_ttl_s = float(self.coordinator.config.active_claim_ttl_s)
        self._lock = threading.RLock()
        self._closed = False

    @property
    def authoritative(self) -> bool:
        return self.mode is IdentityV2Mode.AUTHORITATIVE

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._recent_proofs.clear()
            self._held_overlays.clear()
            self._osd_decisions.clear()
            self._osd_frame_order.clear()
            recorder = self.evidence_recorder
        try:
            if recorder is not None:
                recorder.close()
        finally:
            self.store.close()

    def lookup_osd_decision(
        self,
        *,
        camera_id: str,
        frame_id: int,
        tracker_id: str,
    ) -> Optional[IdentityOsdDecision]:
        """Return only an exact frame-local decision; never a continuity guess."""

        key = (
            str(camera_id or "").strip(),
            int(frame_id),
            str(tracker_id or "").strip(),
        )
        with self._lock:
            return self._osd_decisions.get(key)

    def process_source_frame(
        self,
        *,
        camera_id: str,
        frame_id: int,
        primitives: Sequence[IdentityFramePrimitive],
        observed_at: Optional[float] = None,
    ) -> IdentityFrameServiceResult:
        """Resolve one complete source frame in exactly one coordinator call."""

        with self._lock:
            if self._closed:
                raise RuntimeError("identity-v2 service is closed")
            camera = _require_text(camera_id, "frame camera_id")
            frame = int(frame_id)
            if frame < 0:
                raise ValueError("identity-v2 frame_id must be non-negative")
            timestamp = float(time.time() if observed_at is None else observed_at)
            if not math.isfinite(timestamp):
                raise ValueError("identity-v2 observed_at must be finite")
            rows = tuple(primitives)
            seen_trackers: set[str] = set()
            observations: list[PrimitiveFrameObservation] = []
            primitives_by_tracklet: dict[str, IdentityFramePrimitive] = {}
            observation_by_tracklet: dict[str, PrimitiveFrameObservation] = {}
            for primitive in rows:
                # The legacy StableID diagnostics may remember that a tracker
                # has an embedding in its gallery.  That is not evidence that
                # this exact public frame has a server-extracted identity-v2
                # observation.  Clear both detached surfaces first and let the
                # process-owned v2 service stamp exact frame truth below.
                self._set_frame_embedding_presence(primitive, present=False)
                primitive_camera = _require_text(
                    primitive.camera_id, "primitive camera_id"
                )
                tracker_id = _require_text(primitive.tracker_id, "primitive tracker_id")
                if primitive_camera != camera or int(primitive.frame_id) != frame:
                    raise ValueError(
                        "identity-v2 primitive does not belong to the source frame"
                    )
                if tracker_id in seen_trackers:
                    raise ValueError(
                        "identity-v2 source frame contains a duplicate tracker"
                    )
                seen_trackers.add(tracker_id)
                if primitive.embedding is None:
                    continue
                embedding = self._validated_embedding(primitive.embedding)
                self._set_frame_embedding_presence(primitive, present=True)
                quality = self._quality(primitive)
                observation_id = self._observation_id(
                    camera_id=camera,
                    tracker_id=tracker_id,
                    frame_id=frame,
                    embedding=embedding,
                )
                observation = PrimitiveFrameObservation(
                    run_id=self.run_id,
                    camera_id=camera,
                    tracker_id=tracker_id,
                    frame_id=frame,
                    observation_id=observation_id,
                    quality=quality,
                    embedding=embedding,
                    evidence=(
                        "server_extracted_embedding",
                        f"model_sha256={self.model_fingerprint}",
                        f"model_layer={self.model_layer}",
                    ),
                )
                tracklet_id = identity_tracklet_id(
                    self.run_id,
                    camera,
                    tracker_id,
                )
                observations.append(observation)
                primitives_by_tracklet[tracklet_id] = primitive
                observation_by_tracklet[tracklet_id] = observation
                primitive.public_track["identity_observation_key"] = {
                    "run_id": self.run_id,
                    "camera_id": camera,
                    "tracker_id": tracker_id,
                    "frame_id": frame,
                    "observation_id": observation_id,
                }

            self._expire_recent_proofs(timestamp)
            permits = self._derive_overlap_permits(
                observations=observations,
                primitives_by_tracklet=primitives_by_tracklet,
                observed_at=timestamp,
            )
            # This is the sole coordinator invocation for the complete source frame.
            batch = self.coordinator.process_frame(
                tuple(observations),
                overlap_permits=permits,
            )
            persisted_records: Sequence[ShadowIdentityEvidenceRecord] = ()
            if self.evidence_recorder is not None:
                persisted_records = self.evidence_recorder.append_frame(
                    run_id=self.run_id,
                    model_sha256=self.model_fingerprint,
                    model_semantic_profile_sha256=(self.model_semantic_profile_sha256),
                    model_layer=self.model_layer,
                    embedding_dim=self.embedding_dim,
                    observed_at_us=max(1, int(timestamp * 1_000_000)),
                    observations=tuple(observations),
                    # These are the exact post-hard-mask rows consumed by the
                    # resolver, not a second gallery snapshot that could omit
                    # continuity/copresence constraints or race retention.
                    candidate_rows=batch.candidate_rows,
                    batch=batch,
                    scorer=self.runtime.resolver.scorer,
                    runtime_mode=self.mode.value,
                )
                # DS9/replay recorders enqueue rows and fsync them on their
                # owner thread.  Do not label a media-side row as durable
                # before that worker has completed; the evidence health
                # surface exposes the pending/dropped/error state instead.
                if not getattr(self.evidence_recorder, "async_mode", False):
                    self._stamp_persisted_embedding_provenance(
                        observation_by_tracklet=observation_by_tracklet,
                        primitives_by_tracklet=primitives_by_tracklet,
                        persisted_records=persisted_records,
                    )
            overlay_by_tracklet = {
                overlay.key.tracklet_id: overlay for overlay in batch.overlays
            }
            current_tracklets = set(primitives_by_tracklet)
            for tracklet_id, primitive in primitives_by_tracklet.items():
                overlay = overlay_by_tracklet[tracklet_id]
                self._apply_overlay(primitive.public_track, overlay)
                if getattr(self.evidence_recorder, "async_mode", False):
                    self._mark_queued_embedding_evidence(
                        primitive,
                        accepted=bool(persisted_records),
                    )
                elif persisted_records:
                    identity = primitive.public_track.get("identity_v2")
                    if isinstance(identity, Mapping):
                        updated_identity = dict(identity)
                        updated_identity["evidence_persistence"] = "durable"
                        primitive.public_track["identity_v2"] = updated_identity
                self._sync_diagnostic_track(primitive)
                if overlay.subject_id is not None and overlay.identity_state in (
                    "resident",
                    "visitor",
                ):
                    self._held_overlays[tracklet_id] = _HeldOverlay(
                        tracklet_id=tracklet_id,
                        camera_id=camera,
                        evidence_at=timestamp,
                        overlay=overlay,
                    )
                else:
                    self._held_overlays.pop(tracklet_id, None)
                observation = observation_by_tracklet[tracklet_id]
                world = self._valid_world(primitive)
                self._recent_proofs[tracklet_id] = _RecentProof(
                    tracklet_id=tracklet_id,
                    camera_id=camera,
                    observed_at=timestamp,
                    embedding=tuple(observation.embedding),
                    world_xyz=(
                        world if world is not None else (math.nan, math.nan, math.nan)
                    ),
                    subject_id=overlay.subject_id,
                )

            for primitive in rows:
                tracker_id = _require_text(primitive.tracker_id, "primitive tracker_id")
                tracklet_id = identity_tracklet_id(
                    self.run_id,
                    camera,
                    tracker_id,
                )
                if tracklet_id in current_tracklets:
                    continue
                primitive.public_track.pop("identity_observation_key", None)
                self._clear_embedding_provenance(primitive.public_track)
                held = self._held_overlays.get(tracklet_id)
                if (
                    held is not None
                    and timestamp - held.evidence_at <= self._identity_hold_ttl_s
                ):
                    self._apply_overlay(primitive.public_track, held.overlay)
                    primitive.public_track["identity_v2"].update(
                        {
                            "reason": "tracker_continuity_hold_no_fresh_embedding",
                            "fresh_embedding": False,
                        }
                    )
                else:
                    self._held_overlays.pop(tracklet_id, None)
                    self._apply_missing_embedding(primitive.public_track)
                self._sync_diagnostic_track(primitive)
                self._recent_proofs.pop(tracklet_id, None)

            # A camera-local tracker that vanished must not leave proof/label state.
            for tracklet_id, recent in tuple(self._recent_proofs.items()):
                if recent.camera_id == camera and tracklet_id not in current_tracklets:
                    self._recent_proofs.pop(tracklet_id, None)
            visible_tracklets = {
                identity_tracklet_id(
                    self.run_id,
                    camera,
                    _require_text(row.tracker_id, "primitive tracker_id"),
                )
                for row in rows
            }
            for tracklet_id, held in tuple(self._held_overlays.items()):
                if held.camera_id == camera and tracklet_id not in visible_tracklets:
                    self._held_overlays.pop(tracklet_id, None)

            self._record_osd_frame(
                camera_id=camera,
                frame_id=frame,
                primitives=rows,
            )

            return IdentityFrameServiceResult(
                batch=batch,
                primitive_count=len(rows),
                evidence_count=len(observations),
                overlap_permit_count=len(permits),
            )

    def _stamp_persisted_embedding_provenance(
        self,
        *,
        observation_by_tracklet: Mapping[str, PrimitiveFrameObservation],
        primitives_by_tracklet: Mapping[str, IdentityFramePrimitive],
        persisted_records: Sequence[ShadowIdentityEvidenceRecord],
    ) -> None:
        """Link fresh public rows only to evidence that was durably persisted."""

        sequence_by_observation: dict[str, int] = {}
        for record in persisted_records:
            observation_id = str(record.observation_id)
            if observation_id in sequence_by_observation:
                raise RuntimeError(
                    "identity evidence recorder returned duplicate observation IDs"
                )
            if (
                str(record.model_sha256) != self.model_fingerprint
                or int(record.embedding_dim) != self.embedding_dim
            ):
                raise RuntimeError(
                    "identity evidence recorder returned mismatched model provenance"
                )
            sequence_by_observation[observation_id] = int(record.sequence)

        expected_observations = {
            observation.observation_id
            for observation in observation_by_tracklet.values()
        }
        if set(sequence_by_observation) != expected_observations:
            raise RuntimeError(
                "identity evidence recorder returned incomplete observation linkage"
            )

        for tracklet_id, observation in observation_by_tracklet.items():
            primitive = primitives_by_tracklet[tracklet_id]
            primitive.public_track.update(
                {
                    "embedding_sequence": sequence_by_observation[
                        observation.observation_id
                    ],
                    "embedding_model_sha256": self.model_fingerprint,
                    "embedding_dimension": self.embedding_dim,
                }
            )

    @staticmethod
    def _clear_embedding_provenance(track: MutableMapping[str, Any]) -> None:
        for field in _EMBEDDING_PROVENANCE_FIELDS:
            track.pop(field, None)

    @classmethod
    def _mark_queued_embedding_evidence(
        cls,
        primitive: IdentityFramePrimitive,
        *,
        accepted: bool,
    ) -> None:
        """Keep live extraction/identity truth separate from durability."""

        track = primitive.public_track
        # The key, embedding_present, and fresh_embedding fields describe the
        # live frame.  Only the durable triad is withheld until the writer
        # commits; evidence_persistence makes that distinction explicit.
        cls._clear_embedding_provenance(track)
        identity = track.get("identity_v2")
        if isinstance(identity, Mapping):
            updated = dict(identity)
            updated["evidence_persistence"] = "queued" if accepted else "dropped"
            updated["reason"] = (
                "identity_evidence_queued"
                if accepted
                else "identity_evidence_dropped_queue_full"
            )
            track["identity_v2"] = updated

    @classmethod
    def _set_frame_embedding_presence(
        cls,
        primitive: IdentityFramePrimitive,
        *,
        present: bool,
    ) -> None:
        """Stamp exact current-frame embedding truth on detached track surfaces."""

        value = bool(present)
        primitive.public_track.pop("identity_observation_key", None)
        cls._clear_embedding_provenance(primitive.public_track)
        primitive.public_track["embedding_present"] = value
        diagnostic = primitive.diagnostic_track
        if diagnostic is not None:
            diagnostic.pop("identity_observation_key", None)
            cls._clear_embedding_provenance(diagnostic)
            diagnostic["embedding_present"] = value

    def _record_osd_frame(
        self,
        *,
        camera_id: str,
        frame_id: int,
        primitives: Sequence[IdentityFramePrimitive],
    ) -> None:
        frame_key = (str(camera_id), int(frame_id))
        # Replacing an exact frame is forbidden by coordinator replay checks, but
        # clear defensively so cache state can never combine two memberships.
        for key in tuple(self._osd_decisions):
            if key[:2] == frame_key:
                self._osd_decisions.pop(key, None)
        for primitive in primitives:
            track = primitive.public_track
            identity = track.get("identity_v2")
            if not isinstance(identity, Mapping):
                state = "provisional"
                sid = None
                display_name = None
            else:
                state = str(identity.get("state") or "provisional")
                try:
                    parsed_sid = int(identity.get("compatibility_sid"))
                except (TypeError, ValueError):
                    parsed_sid = 0
                sid = parsed_sid if parsed_sid > 0 else None
                raw_name = " ".join(str(identity.get("display_name") or "").split())
                display_name = raw_name or None
            # The coordinator tracker key may be source-epoch scoped after a
            # reconnect.  The downstream SDK metadata still carries the raw
            # numeric tracker ID, so keep the exact-frame OSD join on that
            # public value while identity resolution uses the scoped key.
            tracker_id = str(
                track.get("tracker_id", primitive.tracker_id)
            ).strip()
            if not tracker_id:
                raise ValueError("identity-v2 public OSD tracker ID is empty")
            self._osd_decisions[(frame_key[0], frame_key[1], tracker_id)] = (
                IdentityOsdDecision(
                    camera_id=frame_key[0],
                    frame_id=frame_key[1],
                    tracker_id=tracker_id,
                    identity_state=state,
                    compatibility_sid=sid,
                    display_name=display_name,
                )
            )
        if frame_key in self._osd_frame_order:
            self._osd_frame_order.remove(frame_key)
        self._osd_frame_order.append(frame_key)
        while len(self._osd_frame_order) > self._osd_cache_frames:
            expired = self._osd_frame_order.pop(0)
            for key in tuple(self._osd_decisions):
                if key[:2] == expired:
                    self._osd_decisions.pop(key, None)

    def _validated_embedding(self, raw: Sequence[float]) -> Tuple[float, ...]:
        values = tuple(float(value) for value in raw)
        if len(values) != self.embedding_dim:
            raise ValueError(
                f"identity-v2 embedding dimension {len(values)} does not match {self.embedding_dim}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("identity-v2 embedding contains non-finite values")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1e-12:
            raise ValueError("identity-v2 embedding has zero norm")
        return tuple(value / norm for value in values)

    def _quality(self, primitive: IdentityFramePrimitive) -> float:
        confidence_values = []
        for raw in (primitive.detection_confidence, primitive.tracker_confidence):
            try:
                value = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                confidence_values.append(max(0.0, min(1.0, value)))
        confidence = max(confidence_values, default=0.5)
        crop_quality = 0.5
        if primitive.bbox is not None and len(primitive.bbox) >= 4:
            try:
                height = max(0.0, float(primitive.bbox[3]))
                frame_height = 0.0
                if primitive.frame_size is not None and len(primitive.frame_size) >= 2:
                    frame_height = max(0.0, float(primitive.frame_size[1]))
                if frame_height > 0.0:
                    crop_quality = max(0.0, min(1.0, height / (0.20 * frame_height)))
                else:
                    crop_quality = max(0.0, min(1.0, height / 192.0))
            except (TypeError, ValueError):
                crop_quality = 0.5
        return float(max(0.0, min(1.0, 0.55 * crop_quality + 0.45 * confidence)))

    def _observation_id(
        self,
        *,
        camera_id: str,
        tracker_id: str,
        frame_id: int,
        embedding: Sequence[float],
    ) -> str:
        digest = hashlib.sha256(b"noesis-identity-observation-v1\0")
        metadata = json.dumps(
            {
                "run_id": self.run_id,
                "camera_id": camera_id,
                "tracker_id": tracker_id,
                "frame_id": int(frame_id),
                "model_sha256": self.model_fingerprint,
                "model_layer": self.model_layer,
                "embedding_dim": self.embedding_dim,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(metadata)
        digest.update(b"\0")
        digest.update(struct.pack(f"<{self.embedding_dim}f", *embedding))
        return f"obs1:{digest.hexdigest()}"

    def _expire_recent_proofs(self, now: float) -> None:
        if self._max_overlap_age_s <= 0.0:
            self._recent_proofs.clear()
            return
        for tracklet_id, proof in tuple(self._recent_proofs.items()):
            if now - proof.observed_at > self._max_overlap_age_s:
                self._recent_proofs.pop(tracklet_id, None)

    def _derive_overlap_permits(
        self,
        *,
        observations: Sequence[PrimitiveFrameObservation],
        primitives_by_tracklet: Mapping[str, IdentityFramePrimitive],
        observed_at: float,
    ) -> Tuple[OverlapSharePermit, ...]:
        permits: dict[Tuple[str, str, str], OverlapSharePermit] = {}
        for observation in observations:
            current_key = observation.to_runtime().key
            current_tracklet = current_key.tracklet_id
            primitive = primitives_by_tracklet[current_tracklet]
            current_world = self._valid_world(primitive)
            if current_world is None:
                continue
            for recent in self._recent_proofs.values():
                if recent.subject_id is None or recent.tracklet_id == current_tracklet:
                    continue
                policy = self._overlap_by_pair.get(
                    tuple(sorted((current_key.camera_id, recent.camera_id)))
                )
                if policy is None:
                    continue
                age = observed_at - recent.observed_at
                if age < 0.0 or age > policy.max_time_delta_s:
                    continue
                if not all(math.isfinite(value) for value in recent.world_xyz):
                    continue
                distance = math.sqrt(
                    sum(
                        (left - right) ** 2
                        for left, right in zip(current_world, recent.world_xyz)
                    )
                )
                if distance > policy.max_world_dist_m:
                    continue
                similarity = sum(
                    left * right
                    for left, right in zip(observation.embedding, recent.embedding)
                )
                if similarity < policy.require_appearance_sim:
                    continue
                permit = OverlapSharePermit(
                    recent.subject_id,
                    current_tracklet,
                    recent.tracklet_id,
                    reason=(
                        "topology_world_appearance_proof:"
                        f"dt={age:.6f}:dist={distance:.6f}:sim={similarity:.6f}"
                    ),
                )
                permits[(permit.identity_id, permit.tracklet_a, permit.tracklet_b)] = (
                    permit
                )
        return tuple(permits[key] for key in sorted(permits))

    @staticmethod
    def _valid_world(
        primitive: IdentityFramePrimitive,
    ) -> Optional[Tuple[float, float, float]]:
        if (
            not primitive.world_valid
            or primitive.world_xyz is None
            or len(primitive.world_xyz) < 3
        ):
            return None
        try:
            world = tuple(float(value) for value in primitive.world_xyz[:3])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in world):
            return None
        return world  # type: ignore[return-value]

    def _apply_missing_embedding(self, track: MutableMapping[str, Any]) -> None:
        track.pop("identity_observation_key", None)
        self._clear_embedding_provenance(track)
        track["embedding_present"] = False
        track["identity_v2"] = {
            "mode": self.mode.value,
            "state": "provisional",
            "reason": "server_embedding_unavailable",
            "subject_id": None,
            "compatibility_sid": None,
            "display_name": None,
            "resident_uuid": None,
            "visitor_generation": None,
            "fresh_embedding": False,
        }
        if self.authoritative:
            self._clear_public_identity(track, state="provisional")

    def _apply_overlay(
        self,
        track: MutableMapping[str, Any],
        overlay: PublicIdentityOverlay,
    ) -> None:
        resident_uuid = None
        if overlay.subject_id and overlay.subject_id.startswith("resident:"):
            resident_uuid = overlay.subject_id.split(":", 1)[1]
        overlap_permit = any(
            token in overlay.resolver_decision.evidence
            for token in (
                "overlap_share_permitted",
                "cross_batch_overlap_share_permitted",
            )
        )
        track["identity_v2"] = {
            "mode": self.mode.value,
            "state": overlay.identity_state,
            "reason": overlay.reason,
            "subject_id": (
                overlay.subject_id if self.mode is IdentityV2Mode.SHADOW else None
            ),
            "compatibility_sid": overlay.compatibility_sid,
            "display_name": overlay.display_name,
            "resident_uuid": resident_uuid,
            "visitor_generation": overlay.visitor_generation,
            "calibrated_confidence": overlay.calibrated_confidence,
            "provisional_evidence_count": overlay.provisional_evidence_count,
            "overlap_permit": overlap_permit,
            "fresh_embedding": True,
        }
        if not self.authoritative:
            return
        state = str(overlay.identity_state)
        if state not in ("resident", "visitor") or overlay.compatibility_sid is None:
            self._clear_public_identity(track, state=state)
            return
        self._clear_public_identity(track, state=state)
        track["stable_id"] = int(overlay.compatibility_sid)
        track["identity_state"] = state
        track["identity_kind"] = state
        track["reid_confidence"] = float(overlay.calibrated_confidence)
        track["overlap_permit"] = bool(overlap_permit)
        if state == "resident":
            track["resident_uuid"] = resident_uuid
            track["display_name"] = overlay.display_name
            track["visitor_generation"] = None
        else:
            track["resident_uuid"] = None
            track["display_name"] = None
            track["visitor_generation"] = (
                int(overlay.visitor_generation)
                if overlay.visitor_generation is not None
                else None
            )

    def _sync_diagnostic_track(self, primitive: IdentityFramePrimitive) -> None:
        diagnostic = primitive.diagnostic_track
        if diagnostic is None:
            return
        public = primitive.public_track
        if "identity_observation_key" in public:
            diagnostic["identity_observation_key"] = dict(
                public["identity_observation_key"]
            )
        else:
            diagnostic.pop("identity_observation_key", None)
        identity_v2 = public.get("identity_v2")
        if isinstance(identity_v2, Mapping):
            diagnostic["identity_v2"] = dict(identity_v2)
        else:
            diagnostic.pop("identity_v2", None)
        diagnostic["embedding_present"] = public.get("embedding_present") is True
        for field in _EMBEDDING_PROVENANCE_FIELDS:
            if field in public:
                diagnostic[field] = public[field]
            else:
                diagnostic.pop(field, None)
        if not self.authoritative:
            return
        for field in (
            "stable_id",
            "identity_state",
            "identity_kind",
            "resident_uuid",
            "display_name",
            "visitor_generation",
            "reid_confidence",
            "reid_required",
            "reid_identity",
            "appearance_id",
            "sid_candidate",
            "id_event",
            "id_reject_reason",
            "id_display",
            "overlap_permit",
        ):
            diagnostic[field] = public.get(field)

    @staticmethod
    def _clear_public_identity(track: MutableMapping[str, Any], *, state: str) -> None:
        for field in (
            "stable_id",
            "resident_uuid",
            "display_name",
            "visitor_generation",
            "reid_confidence",
            "reid_required",
            "reid_identity",
            "appearance_id",
            "sid_candidate",
            "id_event",
            "id_reject_reason",
            "id_display",
            "overlap_permit",
        ):
            track[field] = None
        track["identity_state"] = state
        track["identity_kind"] = state


def create_identity_v2_service(
    *,
    pipeline_config: Mapping[str, Any],
    pipeline_yaml_path: str | os.PathLike[str],
    camera_labels: Mapping[int, str],
    repo_root: str | os.PathLike[str],
    run_id: str,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[IdentityV2Service]:
    """Create the sole process-owned identity-v2 service for a runtime run."""

    env = os.environ if environ is None else environ
    mode = _read_mode(env)
    if mode is IdentityV2Mode.DISABLED:
        return None
    root = Path(repo_root).expanduser().resolve()
    yaml_path = Path(pipeline_yaml_path).expanduser().resolve()
    models = pipeline_config.get("models")
    reid = models.get("reid") if isinstance(models, Mapping) else None
    if not isinstance(reid, Mapping) or not bool(reid.get("enable", True)):
        raise IdentityV2ConfigurationError(
            "identity-v2 shadow/authoritative mode requires the active ReID model"
        )
    engine_path = _resolve_required_file(
        reid.get("engine"),
        field="models.reid.engine",
        repo_root=root,
        pipeline_yaml_path=yaml_path,
    )
    model_layer = _require_text(reid.get("layer"), "models.reid.layer")
    if "embedding_dim" not in reid:
        raise IdentityV2ConfigurationError("models.reid.embedding_dim is required")
    embedding_dim = int(reid["embedding_dim"])
    if embedding_dim <= 0:
        raise IdentityV2ConfigurationError("models.reid.embedding_dim must be positive")
    model_fingerprint = _sha256_file(engine_path)
    reid_transform_component = _loaded_reid_transform_component()
    model_semantic_profile_sha256 = identity_model_semantic_profile_sha256(
        reid_config=reid,
        model_fingerprint=model_fingerprint,
        model_layer=model_layer,
        embedding_dim=embedding_dim,
        repo_root=root,
        pipeline_yaml_path=yaml_path,
        reid_transform_component=reid_transform_component,
    )
    if (
        mode is IdentityV2Mode.AUTHORITATIVE
        and reid_transform_component.get("status") != "loaded"
    ):
        raise IdentityV2ConfigurationError(
            "authoritative identity-v2 requires the loaded noesis_reid_meta_ext "
            "embedding transform; no substitute extraction path is allowed"
        )
    topology_raw = str(
        env.get("NOESIS_IDENTITY_V2_CAMERA_TOPOLOGY", "config/camera_topology.yaml")
        or "config/camera_topology.yaml"
    ).strip()
    topology_path = Path(topology_raw).expanduser()
    if not topology_path.is_absolute():
        topology_path = root / topology_path
    coordinator_edges, proof_policies = _load_strict_topology(
        topology_path=topology_path.resolve(),
        camera_labels=camera_labels,
    )
    policy = _load_open_set_policy(
        environ=env,
        repo_root=root,
        model_fingerprint=model_fingerprint,
        model_layer=model_layer,
        embedding_dim=embedding_dim,
        model_semantic_profile_sha256=model_semantic_profile_sha256,
    )
    if (
        mode is IdentityV2Mode.AUTHORITATIVE
        and policy.calibration_status != "artifact_backed"
    ):
        raise IdentityV2ConfigurationError(
            "authoritative identity-v2 requires artifact-backed scoring calibration"
        )
    runtime_name = _identity_runtime_name(env, yaml_path)
    authority_runtime_profile_sha256 = identity_authority_runtime_profile_sha256(
        runtime=runtime_name
    )
    authority_cutover_artifact_id = None
    if mode is IdentityV2Mode.AUTHORITATIVE:
        authority_cutover_artifact_id = _load_authority_cutover_artifact(
            environ=env,
            repo_root=root,
            runtime=runtime_name,
            model_fingerprint=model_fingerprint,
            model_layer=model_layer,
            embedding_dim=embedding_dim,
            model_semantic_profile_sha256=model_semantic_profile_sha256,
            scoring_artifact_id=policy.calibration_artifact_id,
            authority_runtime_profile_sha256=authority_runtime_profile_sha256,
            topology_path=topology_path.resolve(),
            camera_ids=tuple(str(value) for value in camera_labels.values()),
        )
    evidence_recorder = None
    evidence_path_raw = str(
        env.get("NOESIS_IDENTITY_V2_EVIDENCE_PATH", "") or ""
    ).strip()
    if evidence_path_raw:
        if mode is not IdentityV2Mode.SHADOW:
            raise IdentityV2ConfigurationError(
                "identity-v2 evidence capture is allowed only in shadow mode"
            )
        evidence_path = Path(evidence_path_raw).expanduser()
        if not evidence_path.is_absolute():
            evidence_path = root / evidence_path
        session_id = _require_text(
            env.get("NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID"),
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID",
        )
        evidence_source = (
            str(env.get("NOESIS_IDENTITY_V2_EVIDENCE_SOURCE", "shadow") or "shadow")
            .strip()
            .lower()
        )
        evidence_runtime = (
            str(
                env.get("NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME", runtime_name)
                or runtime_name
            )
            .strip()
            .lower()
        )
        try:
            evidence_max_records = int(
                _bounded_env_number(
                    env,
                    "NOESIS_IDENTITY_V2_EVIDENCE_MAX_RECORDS",
                    DEFAULT_MAX_RECORDS,
                    minimum=100,
                    maximum=DEFAULT_MAX_RECORDS,
                    integer=True,
                )
            )
            evidence_max_bytes = int(
                _bounded_env_number(
                    env,
                    "NOESIS_IDENTITY_V2_EVIDENCE_MAX_BYTES",
                    DEFAULT_MAX_BYTES,
                    minimum=1024 * 1024,
                    maximum=DEFAULT_MAX_BYTES,
                    integer=True,
                )
            )
            evidence_max_age_s = float(
                _bounded_env_number(
                    env,
                    "NOESIS_IDENTITY_V2_EVIDENCE_MAX_AGE_S",
                    DEFAULT_MAX_AGE_S,
                    minimum=3600.0,
                    maximum=365.0 * 24.0 * 60.0 * 60.0,
                    integer=False,
                )
            )
            evidence_recorder = IdentityEvidenceRecorder(
                evidence_path,
                session_id=session_id,
                source=evidence_source,
                runtime=evidence_runtime,
                max_records=evidence_max_records,
                max_bytes=evidence_max_bytes,
                max_age_s=evidence_max_age_s,
                # The native DS9/replay media callback must never fsync.  The
                # test runtime intentionally remains synchronous so existing
                # unit fixtures can inspect the file before teardown.
                async_mode=evidence_runtime in {"ds9", "replay"},
            )
        except Exception as exc:
            raise IdentityV2ConfigurationError(
                f"identity-v2 evidence recorder is invalid: {exc}"
            ) from exc
    migration_review = None
    migration_review_raw = str(
        env.get("NOESIS_IDENTITY_V2_MIGRATION_REVIEW_REPORT", "") or ""
    ).strip()
    if migration_review_raw:
        migration_review_path = Path(migration_review_raw).expanduser()
        if not migration_review_path.is_absolute():
            migration_review_path = root / migration_review_path
        try:
            migration_review = load_migration_review(migration_review_path)
        except Exception as exc:
            raise IdentityV2ConfigurationError(
                f"identity-v2 migration review is invalid: {exc}"
            ) from exc
    store_raw = str(
        env.get("NOESIS_IDENTITY_V2_STORE", "~/.noesis/household/identity_v2.sqlite3")
        or "~/.noesis/household/identity_v2.sqlite3"
    ).strip()
    store_path = Path(store_raw).expanduser()
    if not store_path.is_absolute():
        store_path = root / store_path
    try:
        osd_cache_frames = int(
            str(env.get("NOESIS_IDENTITY_V2_OSD_CACHE_FRAMES", "8") or "8")
        )
    except (TypeError, ValueError) as exc:
        raise IdentityV2ConfigurationError(
            "NOESIS_IDENTITY_V2_OSD_CACHE_FRAMES must be an integer"
        ) from exc
    if not 2 <= osd_cache_frames <= 64:
        raise IdentityV2ConfigurationError(
            "NOESIS_IDENTITY_V2_OSD_CACHE_FRAMES must be within [2, 64]"
        )
    store = IdentityStore(store_path)
    try:
        runtime = IdentityV2Runtime(
            store,
            model_fingerprint=model_fingerprint,
            embedding_dim=embedding_dim,
            policy=policy,
        )
        coordinator = IdentityFrameCoordinator(
            runtime,
            camera_overlap_edges=coordinator_edges,
        )
        return IdentityV2Service(
            mode=mode,
            run_id=run_id,
            model_engine_path=engine_path,
            model_fingerprint=model_fingerprint,
            model_layer=model_layer,
            embedding_dim=embedding_dim,
            model_semantic_profile_sha256=model_semantic_profile_sha256,
            authority_runtime_profile_sha256=authority_runtime_profile_sha256,
            authority_cutover_artifact_id=authority_cutover_artifact_id,
            store=store,
            runtime=runtime,
            coordinator=coordinator,
            overlap_policies=proof_policies,
            evidence_recorder=evidence_recorder,
            migration_review=migration_review,
            osd_cache_frames=osd_cache_frames,
        )
    except Exception:
        store.close()
        raise


__all__ = [
    "IdentityFramePrimitive",
    "IdentityFrameServiceResult",
    "IdentityOsdDecision",
    "IdentityV2ConfigurationError",
    "IdentityV2Mode",
    "IdentityV2Service",
    "OverlapProofPolicy",
    "create_identity_v2_service",
    "identity_authority_runtime_profile_sha256",
    "identity_model_semantic_profile_sha256",
]
