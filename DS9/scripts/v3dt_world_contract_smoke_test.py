#!/usr/bin/env python3
"""Validate DS9 V3DT tracker geometry in the canonical backend world frame.

The gate is deliberately narrower than MV3DT acceptance.  It proves that one
occupied DS9 runtime session used the locked xzy camInfo profile, restored the
tracker's Z-up cuboid foot into Noesis' Y-up metric world, exposed the native
visibility/image-foot metadata, and advanced tracks on every configured camera.
It does not claim multi-view overlap association or time synchronization.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
for _path in (str(REPO_ROOT), str(DS9_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from noesis_core.private_paths import (  # noqa: E402
    PrivatePathError,
    atomic_create_private_file,
    read_private_file,
    require_fresh_private_file_bundle,
)
from noesis_core.strict_json import strict_json_loads  # noqa: E402
from noesis_core.v3dt_validation import (  # noqa: E402
    V3DTAxisMap,
    V3DTAxisMapError,
    v3dt_bbox3d_tracker_foot,
)
from scripts.internal_auth_client import (  # noqa: E402
    RequiredInternalAuth,
    add_auth_token_file_argument,
    connect_required_websocket,
    load_required_internal_auth,
)


CONTRACT = "noesis.ds9.v3dt_global_world_live_gate"
CONTRACT_VERSION = 2
CANONICAL_REPORT_FILENAME = "v3dt-world-contract.json"
CANONICAL_SOURCE_TRANSCRIPT_FILENAME = "v3dt-world-contract-source.json"
SOURCE_TRANSCRIPT_CONTRACT = "noesis.ds9.v3dt-world-source-transcript"
SOURCE_TRANSCRIPT_VERSION = 2
CONFIG_BINDING_CONTRACT = "noesis.ds9.v3dt-global-world-config-binding"
CONFIG_BINDING_VERSION = 1
SOURCE_PRIVACY_POLICY = {
    "access": "owner_private_0600",
    "payload_policy": "geometry_only_no_identity_or_media",
    "tracker_ids": "session_local_sha256_tokens",
    "stable_or_resident_identity": "absent",
    "raw_embedding_vectors": "absent",
    "image_frames": "absent",
    "camera_credentials": "absent",
    "secrets": "absent",
}

MAX_SOURCE_MESSAGES = 4096
MAX_SOURCE_TRACKS = 32768
MAX_SOURCE_TRANSCRIPT_BYTES = 32 * 1024 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,47}$")
RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRACK_TOKEN_RE = SHA256_RE

EXPECTED_WORLD_FRAME = "backend_world_m"
EXPECTED_WORLD_SOURCE = "bbox3d"
EXPECTED_AXIS_SPEC = "xzy"
EXPECTED_CAMERAS = ("living-room", "kitchen", "family-room")
EXPECTED_CAMERA_SET = frozenset(EXPECTED_CAMERAS)
EXPECTED_STREAM_SIZE = (1920, 1080)
EXPECTED_BINDING_PATHS = {
    "pipeline": "DS9/config/infer_v3dt.yaml",
    "effective_pipeline": "build/effective_pipeline_yolo26_seg.yaml",
    "cameras": "DS9/config/cameras_v3dt.yaml",
    "tracker": "build/config/v3dt/nvtracker_v3dt_runtime.yaml",
    "calibration": "config/camera_calibration.json",
    "alignment": "config/ply_alignment.json",
    "launch_plan": "launcher/launch-plan.json",
}
REPOSITORY_BINDING_KEYS = frozenset(
    {"pipeline", "cameras", "calibration", "alignment"}
)
SESSION_BUILD_BINDING_KEYS = frozenset({"effective_pipeline", "tracker"})
CONTAINER_REPO_ROOT = Path("/workspace")
CONTAINER_BUILD_ROOT = Path("/var/lib/noesis/build")
EXPECTED_CAMINFO_PATHS = {
    camera: f"DS9/config/v3dt/caminfo_baseline/camInfo_{camera}.yml"
    for camera in EXPECTED_CAMERAS
}
CONFIG_FILE_KEYS = frozenset(EXPECTED_BINDING_PATHS)
EXPECTED_BBOX3D_KEYS = frozenset(
    {
        "xCentre",
        "yCentre",
        "zCentre",
        "xLen",
        "yLen",
        "zLen",
        "xRot",
        "yRot",
        "zRot",
    }
)

# Static replay of 5,220 protected tracks showed exact xzy restoration.  The
# public world may still lag the raw measurement through the bounded ground
# state filter, so the public-world tolerance is intentionally looser.
DEFAULT_MIN_BBOX3D_COVERAGE = 0.95
MAX_PUBLIC_AXIS_ERROR_P95_M = 1.0
MAX_PUBLIC_AXIS_ERROR_MAX_M = 2.5
MAX_TRACKER_FLOOR_ERROR_P95_M = 0.30
MAX_PUBLIC_FLOOR_ERROR_P95_M = 0.05
MIN_AXIS_DISCRIMINATION_M = 0.50
MIN_AXIS_DISCRIMINATING_FRACTION = 0.90
MIN_OLD_RAW_REJECTION_MARGIN_M = 0.25
MIN_OLD_RAW_REJECTION_FRACTION = 0.95
# Protected three-camera replay measured native image-foot p95 values of
# 20.86 px, 15.33 px, and 25.36 px.  Forty pixels is a documented guard band.
MAX_NATIVE_IMAGE_FOOT_REPROJECTION_P95_PX = 40.0
MAX_DERIVED_IMAGE_BASE_REPROJECTION_P95_PX = 0.001


def _finite_number(
    value: object,
    *,
    absolute_bound: float | None = None,
) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    if absolute_bound is not None and abs(normalized) > float(absolute_bound):
        return None
    return normalized


def _canonical_float(value: object, *, absolute_bound: float) -> float | None:
    number = _finite_number(value, absolute_bound=absolute_bound)
    return round(number, 9) if number is not None else None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _visibility_value(value: object) -> float | None:
    normalized = _finite_number(value)
    if normalized is None or not 0.0 <= normalized <= 1.0:
        return None
    return round(normalized, 9)


def _image_point_values(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    normalized = tuple(
        _canonical_float(item, absolute_bound=65536.0) for item in value
    )
    if any(item is None for item in normalized):
        return None
    return float(normalized[0]), float(normalized[1])  # type: ignore[arg-type]


def _image_size_values(value: object) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    width = _nonnegative_int(value[0])
    height = _nonnegative_int(value[1])
    if width is None or height is None or not (8 < width <= 16384 and 8 < height <= 16384):
        return None
    return width, height


def _bbox3d_values(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or set(value) != EXPECTED_BBOX3D_KEYS:
        return None
    normalized: dict[str, float] = {}
    for key in EXPECTED_BBOX3D_KEYS:
        number = _finite_number(value.get(key), absolute_bound=1000.0)
        if number is None:
            return None
        normalized[key] = float(number)
    if any(normalized[key] <= 0.0 or normalized[key] > 10.0 for key in ("xLen", "yLen", "zLen")):
        return None
    return normalized


def _world_values(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    normalized = tuple(
        _finite_number(item, absolute_bound=1000.0) for item in value
    )
    if any(item is None for item in normalized):
        return None
    return tuple(float(item) for item in normalized)  # type: ignore[arg-type]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _metric_summary(values: Sequence[float]) -> dict[str, object]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "median": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1] if ordered else None,
    }


def _safe_repo_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} path is invalid")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} path must be a normalized repository-relative path")
    return path.as_posix()


def _file_binding(value: object, *, label: str, expected_path: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file binding schema drifted")
    path = _safe_repo_relative_path(value.get("path"), label=label)
    digest = value.get("sha256")
    if path != expected_path or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} file binding drifted")
    return {"path": path, "sha256": digest}


def _canonical_config_binding(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "contract",
        "contract_version",
        "session",
        "files",
        "caminfo",
        "semantics",
    }:
        raise ValueError("V3DT config binding schema drifted")
    if (
        type(value.get("schema_version")) is not int  # noqa: E721
        or value.get("schema_version") != 1
        or value.get("contract") != CONFIG_BINDING_CONTRACT
        or type(value.get("contract_version")) is not int  # noqa: E721
        or value.get("contract_version") != CONFIG_BINDING_VERSION
    ):
        raise ValueError("V3DT config binding envelope drifted")

    raw_session = value.get("session")
    if not isinstance(raw_session, Mapping) or set(raw_session) != {
        "session_id",
        "runtime_lane",
        "build_mount",
        "effective_pipeline_container_path",
        "tracker_container_path",
    }:
        raise ValueError("V3DT config session binding drifted")
    session_id = str(raw_session.get("session_id") or "").strip().lower()
    if (
        SESSION_RE.fullmatch(session_id) is None
        or raw_session.get("runtime_lane") != "v3dt"
        or raw_session.get("build_mount") != str(CONTAINER_BUILD_ROOT)
        or raw_session.get("effective_pipeline_container_path")
        != str(CONTAINER_BUILD_ROOT / "effective_pipeline_yolo26_seg.yaml")
        or raw_session.get("tracker_container_path")
        != str(CONTAINER_BUILD_ROOT / "config/v3dt/nvtracker_v3dt_runtime.yaml")
    ):
        raise ValueError("V3DT config session binding is not canonical")

    raw_files = value.get("files")
    if not isinstance(raw_files, Mapping) or set(raw_files) != CONFIG_FILE_KEYS:
        raise ValueError("V3DT config file inventory drifted")
    files = {
        key: _file_binding(
            raw_files.get(key),
            label=f"V3DT {key}",
            expected_path=EXPECTED_BINDING_PATHS[key],
        )
        for key in sorted(CONFIG_FILE_KEYS)
    }

    raw_caminfo = value.get("caminfo")
    if not isinstance(raw_caminfo, list) or len(raw_caminfo) != len(EXPECTED_CAMERAS):
        raise ValueError("V3DT camInfo binding inventory drifted")
    caminfo: list[dict[str, object]] = []
    for index, camera in enumerate(EXPECTED_CAMERAS):
        row = raw_caminfo[index]
        if not isinstance(row, Mapping) or set(row) != {
            "camera_id",
            "path",
            "sha256",
            "projection_key",
            "projection_matrix_3x4",
            "model_height_m",
            "model_radius_m",
        }:
            raise ValueError(f"V3DT camInfo {camera} schema drifted")
        path = _safe_repo_relative_path(row.get("path"), label=f"camInfo {camera}")
        digest = row.get("sha256")
        matrix = row.get("projection_matrix_3x4")
        if (
            row.get("camera_id") != camera
            or path != EXPECTED_CAMINFO_PATHS[camera]
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or row.get("projection_key") != "projectionMatrix_3x4_w2p"
            or not isinstance(matrix, list)
            or len(matrix) != 12
        ):
            raise ValueError(f"V3DT camInfo {camera} binding drifted")
        normalized_matrix = [
            _finite_number(item, absolute_bound=1_000_000.0) for item in matrix
        ]
        height = _finite_number(row.get("model_height_m"), absolute_bound=10.0)
        radius = _finite_number(row.get("model_radius_m"), absolute_bound=10.0)
        if (
            any(item is None for item in normalized_matrix)
            or height != 2.2
            or radius != 0.35
        ):
            raise ValueError(f"V3DT camInfo {camera} geometry drifted")
        caminfo.append(
            {
                "camera_id": camera,
                "path": path,
                "sha256": digest,
                "projection_key": "projectionMatrix_3x4_w2p",
                "projection_matrix_3x4": [float(item) for item in normalized_matrix],
                "model_height_m": float(height),
                "model_radius_m": float(radius),
            }
        )

    semantics = value.get("semantics")
    expected_semantic_keys = {
        "profile",
        "tracking_mode",
        "world_frame",
        "caminfo_world_axes",
        "camera_order",
        "stream_size",
        "enable_padding",
        "state_estimator_type",
        "output_foot_location",
        "output_visibility",
        "floor_y_m",
        "world_unit_scale_m",
        "calibration_pose_frame",
        "minimum_camera_center_separation_m",
    }
    if not isinstance(semantics, Mapping) or set(semantics) != expected_semantic_keys:
        raise ValueError("V3DT config semantic binding drifted")
    min_separation = _finite_number(
        semantics.get("minimum_camera_center_separation_m"), absolute_bound=1000.0
    )
    floor_y = _finite_number(semantics.get("floor_y_m"), absolute_bound=1000.0)
    world_unit_scale = _finite_number(
        semantics.get("world_unit_scale_m"), absolute_bound=1000.0
    )
    if (
        semantics.get("profile") != "sv3dt"
        or semantics.get("tracking_mode") != "v3dt"
        or semantics.get("world_frame") != EXPECTED_WORLD_FRAME
        or semantics.get("caminfo_world_axes") != EXPECTED_AXIS_SPEC
        or semantics.get("camera_order") != list(EXPECTED_CAMERAS)
        or semantics.get("stream_size") != list(EXPECTED_STREAM_SIZE)
        or type(semantics.get("enable_padding")) is not int  # noqa: E721
        or semantics.get("enable_padding") != 0
        or type(semantics.get("state_estimator_type")) is not int  # noqa: E721
        or semantics.get("state_estimator_type") != 3
        or type(semantics.get("output_foot_location")) is not int  # noqa: E721
        or semantics.get("output_foot_location") != 1
        or type(semantics.get("output_visibility")) is not int  # noqa: E721
        or semantics.get("output_visibility") != 1
        or floor_y != 0.0
        or world_unit_scale != 1.0
        or semantics.get("calibration_pose_frame") != EXPECTED_WORLD_FRAME
        or min_separation is None
        or min_separation <= 1.0
    ):
        raise ValueError("V3DT config semantics are not the locked global-world profile")
    canonical_semantics = dict(semantics)
    canonical_semantics["minimum_camera_center_separation_m"] = float(min_separation)
    return {
        "schema_version": 1,
        "contract": CONFIG_BINDING_CONTRACT,
        "contract_version": CONFIG_BINDING_VERSION,
        "session": dict(raw_session),
        "files": files,
        "caminfo": caminfo,
        "semantics": canonical_semantics,
    }


def _binding_sha256(binding: Mapping[str, object]) -> str:
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_config(
    path: Path,
    *,
    label: str,
    allowed_root: Path,
    logical_path: str,
    private_root: bool = False,
) -> tuple[bytes, object, str]:
    absolute = path.expanduser().absolute()
    root = allowed_root.expanduser().absolute()
    try:
        root_info = root.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} bound root is missing") from exc
    if root.is_symlink() or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(f"{label} bound root must be a real directory")
    if private_root and (stat.S_IMODE(root_info.st_mode) & 0o077):
        raise ValueError(f"{label} bound root must be owner-private")
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay below its bound root") from exc
    cursor = absolute
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"{label} path contains a symlink")
        cursor = cursor.parent
    info = absolute.lstat()
    if (
        absolute.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size <= 0
        or info.st_size > MAX_CONFIG_BYTES
    ):
        raise ValueError(
            f"{label} must be a single-link bounded non-empty regular file"
        )
    raw = absolute.read_bytes()
    try:
        payload = (
            strict_json_loads(raw, label=label)
            if absolute.suffix.lower() == ".json"
            else yaml.safe_load(raw.decode("utf-8"))
        )
    except Exception as exc:
        raise ValueError(f"{label} cannot be parsed: {exc}") from exc
    return raw, payload, logical_path


def _path_for_exact_binding(path: Path, *, key: str) -> Path:
    if key not in REPOSITORY_BINDING_KEYS:
        raise ValueError(f"V3DT {key} is not a repository-owned binding")
    absolute = path.expanduser().absolute()
    expected = (REPO_ROOT / EXPECTED_BINDING_PATHS[key]).absolute()
    if absolute != expected:
        raise ValueError(
            f"V3DT {key} config must be the canonical path {EXPECTED_BINDING_PATHS[key]}"
        )
    return absolute


def _load_launch_plan(
    *,
    launcher_dir: Path,
    session_id: str,
    runtime_lane: str,
    expected_build_root: Path,
) -> tuple[bytes, Mapping[str, object], Path]:
    launcher = launcher_dir.expanduser().absolute()
    launcher_info = launcher.lstat()
    if (
        stat.S_ISLNK(launcher_info.st_mode)
        or not stat.S_ISDIR(launcher_info.st_mode)
        or launcher_info.st_uid != os.geteuid()
        or stat.S_IMODE(launcher_info.st_mode) != 0o700
    ):
        raise ValueError("V3DT launch-plan root must be an owner-private directory")
    plan_path = launcher / "launch-plan.json"
    try:
        raw = read_private_file(
            plan_path,
            label="V3DT launch plan",
            max_bytes=MAX_CONFIG_BYTES,
        )
    except PrivatePathError as exc:
        raise ValueError(str(exc)) from exc
    payload = strict_json_loads(raw, label="V3DT launch plan")
    if not isinstance(payload, Mapping):
        raise ValueError("V3DT launch plan must be a mapping")
    session_paths = payload.get("session_paths")
    canonical_runtime = payload.get("canonical_runtime")
    if not isinstance(session_paths, Mapping) or not isinstance(canonical_runtime, Mapping):
        raise ValueError("V3DT launch plan lacks session/runtime bindings")
    runtime_root = _runtime_root_for_launcher(launcher, session_id=session_id)
    expected_session_paths = {
        "build": str(expected_build_root.expanduser().absolute()),
        "state": str(runtime_root / "state" / session_id),
        "depth": str(runtime_root / "depth" / session_id),
        "runtime_evidence": str(runtime_root / "evidence" / session_id / "runtime"),
        "launcher_evidence": str(launcher),
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != "noesis.ds9.canonical_runtime_container"
        or payload.get("mode") != "plan"
        or payload.get("ready_for_explicit_run") is not True
        or payload.get("session_id") != session_id
        or payload.get("runtime_lane") != runtime_lane
        or dict(session_paths) != expected_session_paths
        or canonical_runtime.get("lane") != "v3dt"
        or canonical_runtime.get("pipeline") != EXPECTED_BINDING_PATHS["pipeline"]
        or canonical_runtime.get("cameras") != EXPECTED_BINDING_PATHS["cameras"]
        or canonical_runtime.get("pgie_profile") != "yolo26_seg"
        or canonical_runtime.get("model_size") != "s"
        or canonical_runtime.get("tracking_mode") != "v3dt"
        or canonical_runtime.get("source_ids") != ["0", "1", "2"]
    ):
        raise ValueError("V3DT launch plan does not bind the canonical session")
    declared_build_root = Path(str(session_paths.get("build") or "")).expanduser()
    build_root = expected_build_root.expanduser()
    if not declared_build_root.is_absolute() or not build_root.is_absolute():
        raise ValueError("V3DT launch plan build root must be absolute")
    declared_build_root = declared_build_root.absolute()
    build_root = build_root.absolute()
    if declared_build_root != build_root:
        raise ValueError("V3DT launch plan build root differs from the expected session root")
    if build_root == REPO_ROOT or REPO_ROOT in build_root.parents:
        raise ValueError("V3DT session build root must be external to the checkout")
    return raw, payload, build_root


def _build_root_for_launcher(launcher_dir: Path, *, session_id: str) -> Path:
    return _runtime_root_for_launcher(
        launcher_dir,
        session_id=session_id,
    ) / "build" / session_id


def _runtime_root_for_launcher(launcher_dir: Path, *, session_id: str) -> Path:
    launcher = launcher_dir.expanduser().absolute()
    try:
        launcher_info = launcher.lstat()
    except OSError as exc:
        raise ValueError("V3DT launch-plan root must be an owner-private directory") from exc
    if (
        stat.S_ISLNK(launcher_info.st_mode)
        or not stat.S_ISDIR(launcher_info.st_mode)
        or launcher_info.st_uid != os.geteuid()
        or stat.S_IMODE(launcher_info.st_mode) != 0o700
    ):
        raise ValueError("V3DT launch-plan root must be an owner-private directory")
    session = launcher.parent
    evidence = session.parent
    if (
        launcher.name != "launcher"
        or session.name != session_id
        or evidence.name != "evidence"
    ):
        raise ValueError("V3DT launcher path is outside the canonical runtime session layout")
    return evidence.parent


def _container_path_to_host(
    value: object,
    *,
    build_root: Path,
    label: str,
) -> Path:
    raw = Path(str(value or ""))
    if not raw.is_absolute():
        raise ValueError(f"{label} must be an absolute container path")
    if raw == CONTAINER_REPO_ROOT or CONTAINER_REPO_ROOT in raw.parents:
        return REPO_ROOT / raw.relative_to(CONTAINER_REPO_ROOT)
    if raw == CONTAINER_BUILD_ROOT or CONTAINER_BUILD_ROOT in raw.parents:
        return build_root / raw.relative_to(CONTAINER_BUILD_ROOT)
    raise ValueError(f"{label} escapes the reviewed container mounts")


def _matrix4_from_column_major(value: object, *, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 16:
        raise ValueError(f"{label} must contain 16 column-major values")
    numbers = [_finite_number(item, absolute_bound=1_000_000.0) for item in value]
    if any(item is None for item in numbers):
        raise ValueError(f"{label} contains invalid values")
    return [
        [float(numbers[column * 4 + row]) for column in range(4)]
        for row in range(4)
    ]


def _camera_center_from_world_to_camera(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    rotation = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    translation = [float(matrix[row][3]) for row in range(3)]
    return tuple(
        -sum(rotation[row][column] * translation[row] for row in range(3))
        for column in range(3)
    )


def build_config_binding(
    *,
    pipeline_config: Path,
    cameras_config: Path,
    calibration_config: Path,
    alignment_config: Path,
    launcher_dir: Path,
    session_id: str,
    runtime_lane: str,
    expected_build_root: Path | None = None,
) -> dict[str, object]:
    session_id = str(session_id).strip().lower()
    runtime_lane = str(runtime_lane).strip().lower()
    bound_build_root = (
        expected_build_root.expanduser().absolute()
        if expected_build_root is not None
        else _build_root_for_launcher(launcher_dir, session_id=session_id)
    )
    launch_plan_raw, _launch_plan, build_root = _load_launch_plan(
        launcher_dir=launcher_dir,
        session_id=session_id,
        runtime_lane=runtime_lane,
        expected_build_root=bound_build_root,
    )
    paths = {
        "pipeline": _path_for_exact_binding(pipeline_config, key="pipeline"),
        "cameras": _path_for_exact_binding(cameras_config, key="cameras"),
        "calibration": _path_for_exact_binding(calibration_config, key="calibration"),
        "alignment": _path_for_exact_binding(alignment_config, key="alignment"),
        "effective_pipeline": build_root / "effective_pipeline_yolo26_seg.yaml",
        "tracker": build_root / "config/v3dt/nvtracker_v3dt_runtime.yaml",
    }
    loaded: dict[str, tuple[bytes, object, str]] = {}
    for key, path in paths.items():
        allowed_root = REPO_ROOT if key in REPOSITORY_BINDING_KEYS else build_root
        loaded[key] = _read_config(
            path,
            label=f"V3DT {key}",
            allowed_root=allowed_root,
            logical_path=EXPECTED_BINDING_PATHS[key],
            private_root=key in SESSION_BUILD_BINDING_KEYS,
        )
    loaded["launch_plan"] = (
        launch_plan_raw,
        _launch_plan,
        EXPECTED_BINDING_PATHS["launch_plan"],
    )
    payloads = {key: loaded[key][1] for key in loaded}
    if any(not isinstance(payloads[key], Mapping) for key in loaded):
        raise ValueError("V3DT bound configs must be mappings")

    pipeline = payloads["pipeline"]
    effective = payloads["effective_pipeline"]
    cameras = payloads["cameras"]
    tracker = payloads["tracker"]
    calibration = payloads["calibration"]
    alignment = payloads["alignment"]
    assert isinstance(pipeline, Mapping)
    assert isinstance(effective, Mapping)
    assert isinstance(cameras, Mapping)
    assert isinstance(tracker, Mapping)
    assert isinstance(calibration, Mapping)
    assert isinstance(alignment, Mapping)

    for label, document in (("pipeline", pipeline), ("effective pipeline", effective)):
        v3dt = document.get("v3dt")
        streammux = document.get("streammux")
        if not isinstance(v3dt, Mapping) or not isinstance(streammux, Mapping):
            raise ValueError(f"V3DT {label} lacks v3dt/streammux mappings")
        if (
            v3dt.get("profile") != "sv3dt"
            or v3dt.get("world_frame") != EXPECTED_WORLD_FRAME
            or v3dt.get("caminfo_world_axes") != EXPECTED_AXIS_SPEC
            or v3dt.get("camera_order") != list(EXPECTED_CAMERAS)
            or (streammux.get("width"), streammux.get("height")) != EXPECTED_STREAM_SIZE
            or streammux.get("enable-padding") != 0
        ):
            raise ValueError(f"V3DT {label} semantic profile drifted")
    if effective.get("tracking_mode") != "v3dt":
        raise ValueError("V3DT effective pipeline does not bind tracking_mode=v3dt")
    effective_tracker = effective.get("tracker")
    if not isinstance(effective_tracker, Mapping):
        raise ValueError("V3DT effective pipeline lacks tracker mapping")
    effective_tracker_path = _container_path_to_host(
        effective_tracker.get("config-file"),
        build_root=build_root,
        label="V3DT effective tracker path",
    )
    if effective_tracker_path.absolute() != paths["tracker"].absolute():
        raise ValueError("V3DT effective pipeline did not load the bound tracker config")

    state = tracker.get("StateEstimator")
    projection = tracker.get("ObjectModelProjection")
    if not isinstance(state, Mapping) or not isinstance(projection, Mapping):
        raise ValueError("V3DT tracker lacks StateEstimator/ObjectModelProjection")
    if (
        state.get("stateEstimatorType") != 3
        or projection.get("outputFootLocation") != 1
        or projection.get("outputVisibility") != 1
    ):
        raise ValueError("V3DT tracker projection contract drifted")

    camera_rows = cameras.get("cameras")
    if not isinstance(camera_rows, Mapping):
        raise ValueError("V3DT cameras config lacks cameras mapping")
    camera_order: list[str] = []
    for index in range(len(EXPECTED_CAMERAS)):
        row = camera_rows.get(index, camera_rows.get(str(index)))
        if not isinstance(row, Mapping):
            raise ValueError(f"V3DT cameras config lacks camera {index}")
        camera_order.append(str(row.get("name") or ""))
    if camera_order != list(EXPECTED_CAMERAS):
        raise ValueError("V3DT cameras config order drifted")

    calibration_rows = calibration.get("cameras")
    if not isinstance(calibration_rows, Mapping) or set(calibration_rows) != EXPECTED_CAMERA_SET:
        raise ValueError("V3DT calibration camera inventory drifted")
    centers: list[tuple[float, float, float]] = []
    for camera in EXPECTED_CAMERAS:
        row = calibration_rows.get(camera)
        if not isinstance(row, Mapping):
            raise ValueError(f"V3DT calibration lacks {camera}")
        pose = row.get("pose")
        if not isinstance(pose, Mapping) or pose.get("frame") != EXPECTED_WORLD_FRAME:
            raise ValueError(f"V3DT calibration pose frame drifted for {camera}")
        centers.append(
            _camera_center_from_world_to_camera(
                _matrix4_from_column_major(row.get("E"), label=f"{camera}.E")
            )
        )
    separations = [
        _distance(centers[left], centers[right])
        for left in range(len(centers))
        for right in range(left + 1, len(centers))
    ]
    minimum_separation = min(separations)
    if minimum_separation <= 1.0:
        raise ValueError("V3DT calibration camera centers are not meaningfully separated")

    units = alignment.get("units")
    floor_y = _finite_number(alignment.get("floor_y"), absolute_bound=1000.0)
    unit_scale = (
        _finite_number(units.get("s_obj_to_m"), absolute_bound=1000.0)
        if isinstance(units, Mapping)
        else None
    )
    if floor_y != 0.0 or unit_scale != 1.0:
        raise ValueError("V3DT alignment must bind floor_y=0 and meter units")

    raw_caminfo_paths = projection.get("cameraModelFilepath")
    if not isinstance(raw_caminfo_paths, list) or len(raw_caminfo_paths) != len(EXPECTED_CAMERAS):
        raise ValueError("V3DT tracker camInfo inventory drifted")
    caminfo: list[dict[str, object]] = []
    for index, camera in enumerate(EXPECTED_CAMERAS):
        absolute = _container_path_to_host(
            raw_caminfo_paths[index],
            build_root=build_root,
            label=f"V3DT camInfo path for {camera}",
        )
        expected = REPO_ROOT / EXPECTED_CAMINFO_PATHS[camera]
        if absolute.absolute() != expected.absolute():
            raise ValueError(f"V3DT tracker camInfo path drifted for {camera}")
        raw, document, relative = _read_config(
            absolute,
            label=f"V3DT camInfo {camera}",
            allowed_root=REPO_ROOT,
            logical_path=EXPECTED_CAMINFO_PATHS[camera],
        )
        if not isinstance(document, Mapping):
            raise ValueError(f"V3DT camInfo {camera} is not a mapping")
        matrix = document.get("projectionMatrix_3x4_w2p")
        model = document.get("modelInfo")
        if not isinstance(matrix, list) or len(matrix) != 12 or not isinstance(model, Mapping):
            raise ValueError(f"V3DT camInfo {camera} geometry drifted")
        caminfo.append(
            {
                "camera_id": camera,
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "projection_key": "projectionMatrix_3x4_w2p",
                "projection_matrix_3x4": [float(item) for item in matrix],
                "model_height_m": float(model.get("height")),
                "model_radius_m": float(model.get("radius")),
            }
        )

    binding = {
        "schema_version": 1,
        "contract": CONFIG_BINDING_CONTRACT,
        "contract_version": CONFIG_BINDING_VERSION,
        "session": {
            "session_id": session_id,
            "runtime_lane": runtime_lane,
            "build_mount": str(CONTAINER_BUILD_ROOT),
            "effective_pipeline_container_path": str(
                CONTAINER_BUILD_ROOT / "effective_pipeline_yolo26_seg.yaml"
            ),
            "tracker_container_path": str(
                CONTAINER_BUILD_ROOT / "config/v3dt/nvtracker_v3dt_runtime.yaml"
            ),
        },
        "files": {
            key: {
                "path": loaded[key][2],
                "sha256": hashlib.sha256(loaded[key][0]).hexdigest(),
            }
            for key in sorted(loaded)
        },
        "caminfo": caminfo,
        "semantics": {
            "profile": "sv3dt",
            "tracking_mode": "v3dt",
            "world_frame": EXPECTED_WORLD_FRAME,
            "caminfo_world_axes": EXPECTED_AXIS_SPEC,
            "camera_order": list(EXPECTED_CAMERAS),
            "stream_size": list(EXPECTED_STREAM_SIZE),
            "enable_padding": 0,
            "state_estimator_type": 3,
            "output_foot_location": 1,
            "output_visibility": 1,
            "floor_y_m": float(floor_y),
            "world_unit_scale_m": float(unit_scale),
            "calibration_pose_frame": EXPECTED_WORLD_FRAME,
            "minimum_camera_center_separation_m": float(minimum_separation),
        },
    }
    return _canonical_config_binding(binding)


def _continuity_token(
    *, session_id: str, camera_id: str, tracker_id: int | None
) -> str | None:
    if tracker_id is None:
        return None
    payload = (
        "noesis-v3dt-continuity-v2\x00"
        + session_id
        + "\x00"
        + camera_id
        + "\x00"
        + str(tracker_id)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SOURCE_TRACK_KEYS = frozenset(
    {
        "track_token",
        "frame_id",
        "camera_id",
        "image_size",
        "bbox3d",
        "world",
        "world_valid",
        "world_frame",
        "world_source",
        "visibility",
        "image_foot",
        "image_base",
    }
)


def _source_track(track: Mapping[str, Any], *, session_id: str) -> dict[str, object]:
    raw_camera = track.get("camera_id")
    camera = raw_camera.strip() if isinstance(raw_camera, str) else ""
    if camera not in EXPECTED_CAMERA_SET:
        camera = "unexpected"
    tracker_id = _nonnegative_int(track.get("tracker_id", track.get("track_id")))
    raw_bbox3d = track.get("bbox3d")
    bbox3d = {
        key: (
            _canonical_float(raw_bbox3d.get(key), absolute_bound=1000.0)
            if isinstance(raw_bbox3d, Mapping)
            else None
        )
        for key in sorted(EXPECTED_BBOX3D_KEYS)
    }
    raw_world = track.get("world")
    world = [
        (
            _canonical_float(raw_world[index], absolute_bound=1000.0)
            if isinstance(raw_world, (list, tuple)) and len(raw_world) > index
            else None
        )
        for index in range(3)
    ]
    image_size = _image_size_values(track.get("image_size"))
    image_foot = _image_point_values(track.get("image_foot"))
    image_base = _image_point_values(track.get("image_base"))
    return {
        "track_token": _continuity_token(
            session_id=session_id,
            camera_id=camera,
            tracker_id=tracker_id,
        ),
        "frame_id": _nonnegative_int(track.get("frame_id")),
        "camera_id": camera,
        "image_size": list(image_size) if image_size is not None else [None, None],
        "bbox3d": bbox3d,
        "world": world,
        "world_valid": (
            track.get("world_valid") if isinstance(track.get("world_valid"), bool) else None
        ),
        "world_frame": (
            EXPECTED_WORLD_FRAME
            if track.get("world_frame") == EXPECTED_WORLD_FRAME
            else "unexpected"
        ),
        "world_source": (
            EXPECTED_WORLD_SOURCE
            if track.get("world_source") == EXPECTED_WORLD_SOURCE
            else "unexpected"
        ),
        "visibility": _visibility_value(track.get("visibility")),
        "image_foot": list(image_foot) if image_foot is not None else [None, None],
        "image_base": list(image_base) if image_base is not None else [None, None],
    }


def _policy(min_bbox3d_coverage: float) -> dict[str, object]:
    return {
        "min_bbox3d_coverage": float(min_bbox3d_coverage),
        "max_public_axis_error_p95_m": MAX_PUBLIC_AXIS_ERROR_P95_M,
        "max_public_axis_error_max_m": MAX_PUBLIC_AXIS_ERROR_MAX_M,
        "max_tracker_floor_error_p95_m": MAX_TRACKER_FLOOR_ERROR_P95_M,
        "max_public_floor_error_p95_m": MAX_PUBLIC_FLOOR_ERROR_P95_M,
        "min_axis_discrimination_m": MIN_AXIS_DISCRIMINATION_M,
        "min_axis_discriminating_fraction": MIN_AXIS_DISCRIMINATING_FRACTION,
        "min_old_raw_rejection_margin_m": MIN_OLD_RAW_REJECTION_MARGIN_M,
        "min_old_raw_rejection_fraction": MIN_OLD_RAW_REJECTION_FRACTION,
        "max_native_image_foot_reprojection_p95_px": MAX_NATIVE_IMAGE_FOOT_REPROJECTION_P95_PX,
        "max_derived_image_base_reprojection_p95_px": MAX_DERIVED_IMAGE_BASE_REPROJECTION_P95_PX,
    }


def _replay_source_messages(
    messages: Sequence[object], *, session_id: str
) -> tuple[list[Mapping[str, Any]], int]:
    if len(messages) > MAX_SOURCE_MESSAGES:
        raise ValueError("V3DT source transcript message bound exceeded")
    tracks: list[Mapping[str, Any]] = []
    previous_observed_at_us = 0
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or set(message) != {
            "type",
            "observed_at_us",
            "tracks",
        }:
            raise ValueError(f"V3DT source message {index} schema drifted")
        if message.get("type") != "tracking":
            raise ValueError(f"V3DT source message {index} type drifted")
        observed_at_us = message.get("observed_at_us")
        if (
            isinstance(observed_at_us, bool)
            or not isinstance(observed_at_us, int)
            or observed_at_us <= 0
            or observed_at_us < previous_observed_at_us
        ):
            raise ValueError(f"V3DT source message {index} timestamp is invalid")
        previous_observed_at_us = observed_at_us
        raw_tracks = message.get("tracks")
        if not isinstance(raw_tracks, list):
            raise ValueError(f"V3DT source message {index} tracks must be a list")
        for track_index, track in enumerate(raw_tracks):
            if not isinstance(track, Mapping) or set(track) != SOURCE_TRACK_KEYS:
                raise ValueError(f"V3DT source track {index}:{track_index} schema drifted")
            token = track.get("track_token")
            canonical = dict(track)
            if token is not None and (
                not isinstance(token, str) or TRACK_TOKEN_RE.fullmatch(token) is None
            ):
                raise ValueError(f"V3DT source track {index}:{track_index} token drifted")
            # Canonicalization is idempotent without retaining the raw tracker ID.
            if canonical.get("camera_id") not in (*EXPECTED_CAMERAS, "unexpected"):
                raise ValueError(f"V3DT source track {index}:{track_index} camera drifted")
            bbox = canonical.get("bbox3d")
            if not isinstance(bbox, Mapping) or set(bbox) != EXPECTED_BBOX3D_KEYS:
                raise ValueError(f"V3DT source track {index}:{track_index} bbox schema drifted")
            for key, item in bbox.items():
                if item is not None and _canonical_float(item, absolute_bound=1000.0) != item:
                    raise ValueError(f"V3DT source track {index}:{track_index} bbox value drifted")
            for field, normalizer in (
                ("world", _world_values),
                ("image_size", _image_size_values),
                ("image_foot", _image_point_values),
                ("image_base", _image_point_values),
            ):
                raw_value = canonical.get(field)
                normalized = normalizer(raw_value)
                if normalized is not None and list(normalized) != raw_value:
                    raise ValueError(f"V3DT source track {index}:{track_index} {field} drifted")
                invalid_sentinel = (
                    [None, None, None] if field == "world" else [None, None]
                )
                if normalized is None and raw_value != invalid_sentinel:
                    raise ValueError(f"V3DT source track {index}:{track_index} {field} invalid")
            frame_id = canonical.get("frame_id")
            if frame_id is not None and _nonnegative_int(frame_id) != frame_id:
                raise ValueError(f"V3DT source track {index}:{track_index} frame ID drifted")
            world_valid = canonical.get("world_valid")
            if world_valid is not None and type(world_valid) is not bool:  # noqa: E721
                raise ValueError(f"V3DT source track {index}:{track_index} world validity drifted")
            if canonical.get("world_frame") not in (EXPECTED_WORLD_FRAME, "unexpected"):
                raise ValueError(f"V3DT source track {index}:{track_index} world frame drifted")
            if canonical.get("world_source") not in (EXPECTED_WORLD_SOURCE, "unexpected"):
                raise ValueError(f"V3DT source track {index}:{track_index} world source drifted")
            visibility = canonical.get("visibility")
            if visibility is not None and _visibility_value(visibility) != visibility:
                raise ValueError(f"V3DT source track {index}:{track_index} visibility drifted")
            tracks.append(canonical)
    if len(tracks) > MAX_SOURCE_TRACKS:
        raise ValueError("V3DT source transcript track bound exceeded")
    return tracks, len(messages)


def _source_transcript_document(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    config_binding: Mapping[str, object],
    min_bbox3d_coverage: float,
    messages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    canonical_binding = _canonical_config_binding(config_binding)
    binding_session = canonical_binding["session"]
    assert isinstance(binding_session, Mapping)
    if (
        binding_session.get("session_id") != str(session_id).strip().lower()
        or binding_session.get("runtime_lane") != str(runtime_lane).strip().lower()
    ):
        raise ValueError("V3DT source/session config binding drifted")
    replayed_tracks, replayed_message_count = _replay_source_messages(
        messages, session_id=session_id
    )
    return {
        "schema_version": 2,
        "contract": SOURCE_TRANSCRIPT_CONTRACT,
        "contract_version": SOURCE_TRANSCRIPT_VERSION,
        "session_id": str(session_id).strip().lower(),
        "runtime_lane": str(runtime_lane).strip().lower(),
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "privacy": dict(SOURCE_PRIVACY_POLICY),
        "config_binding": canonical_binding,
        "policy": _policy(min_bbox3d_coverage),
        "message_count": replayed_message_count,
        "track_count": len(replayed_tracks),
        "messages": [dict(message) for message in messages],
    }


def _caminfo_by_camera(config_binding: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = config_binding.get("caminfo")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("camera_id")): row
        for row in rows
        if isinstance(row, Mapping)
    }


def _project_tracker_point(
    point: Sequence[float],
    *,
    caminfo: Mapping[str, object],
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    matrix = caminfo.get("projection_matrix_3x4")
    if not isinstance(matrix, list) or len(matrix) != 12:
        return None
    try:
        x, y, z = (float(point[index]) for index in range(3))
        rows = [matrix[0:4], matrix[4:8], matrix[8:12]]
        denominator = sum(float(rows[2][index]) * (x, y, z, 1.0)[index] for index in range(4))
        if abs(denominator) < 1e-9:
            return None
        u = sum(float(rows[0][index]) * (x, y, z, 1.0)[index] for index in range(4)) / denominator
        v = sum(float(rows[1][index]) * (x, y, z, 1.0)[index] for index in range(4)) / denominator
        if caminfo.get("projection_key") == "projectionMatrix_3x4":
            u += image_size[0] * 0.5
            v += image_size[1] * 0.5
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return float(u), float(v)
    except Exception:
        return None


def _pixel_error(observed: Sequence[float], expected: Sequence[float]) -> float:
    return math.hypot(float(observed[0]) - float(expected[0]), float(observed[1]) - float(expected[1]))


def analyze_tracks(
    tracks: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    tracking_messages: int,
    config_binding: Mapping[str, object],
    min_bbox3d_coverage: float,
    source_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    session_id = str(session_id).strip().lower()
    runtime_lane = str(runtime_lane).strip().lower()
    if SESSION_RE.fullmatch(session_id) is None:
        raise ValueError("session_id must match [a-z0-9][a-z0-9-]{5,47}")
    if runtime_lane != "v3dt":
        raise ValueError("V3DT world evidence requires runtime lane 'v3dt'")
    if (
        RUNTIME_ID_RE.fullmatch(str(runtime_instance_id)) is None
        or RUNTIME_ID_RE.fullmatch(str(runtime_run_id)) is None
    ):
        raise ValueError("runtime instance/run identity is invalid")
    canonical_binding = _canonical_config_binding(config_binding)
    binding_session = canonical_binding["session"]
    assert isinstance(binding_session, Mapping)
    if (
        binding_session.get("session_id") != session_id
        or binding_session.get("runtime_lane") != runtime_lane
    ):
        raise ValueError("V3DT report/session config binding drifted")
    semantics = canonical_binding["semantics"]
    assert isinstance(semantics, Mapping)
    axis_map = V3DTAxisMap.parse(semantics.get("caminfo_world_axes"))
    floor_y = float(semantics["floor_y_m"])
    caminfo = _caminfo_by_camera(canonical_binding)

    total = len(tracks)
    bbox3d_valid = 0
    contract_valid = 0
    native_bridge_valid = 0
    invalid_bbox3d = 0
    invalid_visibility = 0
    invalid_image_foot = 0
    invalid_image_base = 0
    invalid_image_size = 0
    invalid_world = 0
    wrong_world_frame = 0
    wrong_world_source = 0
    wrong_camera = 0
    invalid_track_token = 0
    invalid_frame_id = 0
    frame_regressions = 0
    cameras: Counter[str] = Counter()
    last_frame_by_tracker: dict[tuple[str, str], int] = {}
    advancing_trackers_by_camera: dict[str, set[str]] = {
        camera: set() for camera in EXPECTED_CAMERAS
    }

    axis_errors: list[float] = []
    tracker_floor_errors: list[float] = []
    public_floor_errors: list[float] = []
    native_image_foot_errors: list[float] = []
    image_base_errors: list[float] = []
    discriminating_count = 0
    old_raw_rejected_count = 0

    for track in tracks:
        camera = str(track.get("camera_id") or "").strip()
        cameras[camera] += 1
        if camera not in EXPECTED_CAMERA_SET:
            wrong_camera += 1
        token = track.get("track_token")
        if not isinstance(token, str) or TRACK_TOKEN_RE.fullmatch(token) is None:
            invalid_track_token += 1
            token = None
        frame_id = _nonnegative_int(track.get("frame_id"))
        if frame_id is None:
            invalid_frame_id += 1
        if token is not None and frame_id is not None and camera in EXPECTED_CAMERA_SET:
            key = (camera, token)
            previous = last_frame_by_tracker.get(key)
            if previous is not None:
                if frame_id > previous:
                    advancing_trackers_by_camera[camera].add(token)
                elif frame_id < previous:
                    frame_regressions += 1
            if previous is None or frame_id > previous:
                last_frame_by_tracker[key] = frame_id

        bbox3d = _bbox3d_values(track.get("bbox3d"))
        if bbox3d is None:
            invalid_bbox3d += 1
        else:
            bbox3d_valid += 1
        visibility = _visibility_value(track.get("visibility"))
        image_foot = _image_point_values(track.get("image_foot"))
        image_base = _image_point_values(track.get("image_base"))
        image_size = _image_size_values(track.get("image_size"))
        if visibility is None:
            invalid_visibility += 1
        if image_foot is None:
            invalid_image_foot += 1
        if image_base is None:
            invalid_image_base += 1
        if image_size is None:
            invalid_image_size += 1
        if bbox3d is not None and visibility is not None and image_foot is not None:
            native_bridge_valid += 1

        world = _world_values(track.get("world"))
        if track.get("world_valid") is not True or world is None:
            invalid_world += 1
        if track.get("world_frame") != EXPECTED_WORLD_FRAME:
            wrong_world_frame += 1
        if track.get("world_source") != EXPECTED_WORLD_SOURCE:
            wrong_world_source += 1

        typed = bool(
            bbox3d is not None
            and visibility is not None
            and image_foot is not None
            and image_base is not None
            and image_size is not None
            and world is not None
            and track.get("world_valid") is True
            and track.get("world_frame") == EXPECTED_WORLD_FRAME
            and track.get("world_source") == EXPECTED_WORLD_SOURCE
            and camera in EXPECTED_CAMERA_SET
            and token is not None
            and frame_id is not None
        )
        if not typed:
            continue
        assert bbox3d is not None
        assert world is not None
        assert image_foot is not None
        assert image_base is not None
        assert image_size is not None
        contract_valid += 1
        try:
            tracker_foot = v3dt_bbox3d_tracker_foot(bbox3d)
            canonical_foot = axis_map.tracker_to_world(tracker_foot)
        except V3DTAxisMapError:
            invalid_bbox3d += 1
            continue
        axis_error = _distance(world, canonical_foot)
        old_raw_error = _distance(world, tracker_foot)
        axis_separation = _distance(canonical_foot, tracker_foot)
        axis_errors.append(axis_error)
        tracker_floor_errors.append(abs(float(canonical_foot[1]) - floor_y))
        public_floor_errors.append(abs(float(world[1]) - floor_y))
        if axis_separation >= MIN_AXIS_DISCRIMINATION_M:
            discriminating_count += 1
            if axis_error + MIN_OLD_RAW_REJECTION_MARGIN_M <= old_raw_error:
                old_raw_rejected_count += 1

        camera_projection = caminfo.get(camera)
        if camera_projection is not None:
            projected_foot = _project_tracker_point(
                tracker_foot, caminfo=camera_projection, image_size=image_size
            )
            opposite_endpoint = (
                bbox3d["xCentre"],
                bbox3d["yCentre"],
                bbox3d["zCentre"] + 0.5 * bbox3d["zLen"],
            )
            projected_base = _project_tracker_point(
                opposite_endpoint, caminfo=camera_projection, image_size=image_size
            )
            if projected_foot is not None:
                native_image_foot_errors.append(_pixel_error(image_foot, projected_foot))
            if projected_base is not None:
                image_base_errors.append(_pixel_error(image_base, projected_base))

    coverage = float(bbox3d_valid / total) if total else 0.0
    native_bridge_coverage = float(native_bridge_valid / bbox3d_valid) if bbox3d_valid else 0.0
    contract_coverage = float(contract_valid / total) if total else 0.0
    discriminating_fraction = (
        float(discriminating_count / contract_valid) if contract_valid else 0.0
    )
    rejection_fraction = (
        float(old_raw_rejected_count / discriminating_count)
        if discriminating_count
        else 0.0
    )
    continuity = {
        camera: len(advancing_trackers_by_camera[camera]) for camera in EXPECTED_CAMERAS
    }
    axis_summary = _metric_summary(axis_errors)
    tracker_floor_summary = _metric_summary(tracker_floor_errors)
    public_floor_summary = _metric_summary(public_floor_errors)
    native_foot_summary = _metric_summary(native_image_foot_errors)
    image_base_summary = _metric_summary(image_base_errors)

    failures: list[str] = []
    if tracking_messages < 2:
        failures.append("insufficient_tracking_messages")
    if total <= 0:
        failures.append("no_tracks")
    if set(cameras) != EXPECTED_CAMERA_SET:
        failures.append("camera_coverage")
    if coverage < float(min_bbox3d_coverage):
        failures.append("bbox3d_coverage")
    if native_bridge_coverage < float(min_bbox3d_coverage):
        failures.append("native_bridge_metadata_coverage")
    if contract_coverage < float(min_bbox3d_coverage):
        failures.append("typed_global_world_coverage")
    if contract_valid <= 0:
        failures.append("no_valid_global_bbox3d_world")
    if invalid_world:
        failures.append("invalid_world")
    if wrong_world_frame:
        failures.append("wrong_world_frame")
    if wrong_world_source:
        failures.append("wrong_world_source")
    if wrong_camera:
        failures.append("unexpected_camera")
    if invalid_track_token:
        failures.append("invalid_track_token")
    if invalid_frame_id:
        failures.append("invalid_frame_id")
    if frame_regressions:
        failures.append("frame_regression")
    if any(value <= 0 for value in continuity.values()):
        failures.append("missing_per_camera_tracker_continuity")
    if (
        axis_summary["p95"] is None
        or float(axis_summary["p95"]) > MAX_PUBLIC_AXIS_ERROR_P95_M
        or axis_summary["max"] is None
        or float(axis_summary["max"]) > MAX_PUBLIC_AXIS_ERROR_MAX_M
    ):
        failures.append("public_world_axis_error")
    if (
        tracker_floor_summary["p95"] is None
        or float(tracker_floor_summary["p95"]) > MAX_TRACKER_FLOOR_ERROR_P95_M
    ):
        failures.append("tracker_floor_error")
    if (
        public_floor_summary["p95"] is None
        or float(public_floor_summary["p95"]) > MAX_PUBLIC_FLOOR_ERROR_P95_M
    ):
        failures.append("public_floor_error")
    if discriminating_fraction < MIN_AXIS_DISCRIMINATING_FRACTION:
        failures.append("insufficient_axis_discrimination")
    if rejection_fraction < MIN_OLD_RAW_REJECTION_FRACTION:
        failures.append("old_raw_axis_tuple_not_rejected")
    if (
        native_foot_summary["count"] != contract_valid
        or native_foot_summary["p95"] is None
        or float(native_foot_summary["p95"])
        > MAX_NATIVE_IMAGE_FOOT_REPROJECTION_P95_PX
    ):
        failures.append("native_image_foot_reprojection")
    if (
        image_base_summary["count"] != contract_valid
        or image_base_summary["p95"] is None
        or float(image_base_summary["p95"])
        > MAX_DERIVED_IMAGE_BASE_REPROJECTION_P95_PX
    ):
        failures.append("derived_image_base_reprojection")

    config_summary = {
        "contract": CONFIG_BINDING_CONTRACT,
        "contract_version": CONFIG_BINDING_VERSION,
        "sha256": _binding_sha256(canonical_binding),
        "file_sha256": {
            key: canonical_binding["files"][key]["sha256"]  # type: ignore[index]
            for key in sorted(CONFIG_FILE_KEYS)
        },
        "caminfo_sha256": {
            str(row["camera_id"]): str(row["sha256"])
            for row in canonical_binding["caminfo"]  # type: ignore[union-attr]
        },
    }
    return {
        "schema_version": 2,
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "runtime_lane": runtime_lane,
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "ok": not failures,
        "status": "pass" if not failures else ("blocked" if total <= 0 else "failed"),
        "occupied_scene_observed": total > 0,
        "locked_world_contract": {
            "world_frame": EXPECTED_WORLD_FRAME,
            "world_source": EXPECTED_WORLD_SOURCE,
            "caminfo_world_axes": EXPECTED_AXIS_SPEC,
            "mv3dt_overlap_promotion": "not_claimed",
            "time_sync_promotion": "not_claimed",
        },
        "config_binding": config_summary,
        "policy": _policy(min_bbox3d_coverage),
        "tracking_messages": int(tracking_messages),
        "tracks_seen": total,
        "bbox3d_valid": bbox3d_valid,
        "bbox3d_coverage": coverage,
        "native_bridge_valid": native_bridge_valid,
        "native_bridge_coverage": native_bridge_coverage,
        "contract_valid_tracks": contract_valid,
        "contract_coverage": contract_coverage,
        "invalid_bbox3d": invalid_bbox3d,
        "invalid_visibility": invalid_visibility,
        "invalid_image_foot": invalid_image_foot,
        "invalid_image_base": invalid_image_base,
        "invalid_image_size": invalid_image_size,
        "invalid_world": invalid_world,
        "wrong_world_frame": wrong_world_frame,
        "wrong_world_source": wrong_world_source,
        "unexpected_camera": wrong_camera,
        "invalid_track_token": invalid_track_token,
        "invalid_frame_id": invalid_frame_id,
        "frame_regressions": frame_regressions,
        "cameras": {camera: int(cameras.get(camera, 0)) for camera in EXPECTED_CAMERAS},
        "advancing_trackers_by_camera": continuity,
        "public_world_axis_error_m": axis_summary,
        "tracker_floor_error_m": tracker_floor_summary,
        "public_floor_error_m": public_floor_summary,
        "old_raw_tuple_rejection": {
            "discriminating_count": discriminating_count,
            "discriminating_fraction": discriminating_fraction,
            "rejected_count": old_raw_rejected_count,
            "rejection_fraction": rejection_fraction,
            "policy": "canonical_xzy_must_beat_raw_tracker_tuple_by_locked_margin",
        },
        "native_image_foot_reprojection_px": native_foot_summary,
        "derived_image_base_reprojection_px": image_base_summary,
        "source_evidence": dict(source_evidence or {}),
        "failures": failures,
    }


def _encoded_private_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _source_evidence_metadata(
    *, encoded: bytes, document: Mapping[str, object]
) -> dict[str, object]:
    raw_messages = document.get("messages")
    messages = raw_messages if isinstance(raw_messages, list) else []
    timestamps = [
        int(message["observed_at_us"])
        for message in messages
        if isinstance(message, Mapping)
        and isinstance(message.get("observed_at_us"), int)
        and not isinstance(message.get("observed_at_us"), bool)
    ]
    return {
        "filename": CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "message_count": document.get("message_count"),
        "track_count": document.get("track_count"),
        "first_observed_at_us": min(timestamps) if timestamps else None,
        "last_observed_at_us": max(timestamps) if timestamps else None,
    }


def _read_private_json_object(
    path: Path,
    *,
    expected_filename: str,
    max_bytes: int,
    label: str,
) -> tuple[bytes, Mapping[str, object]]:
    candidate = Path(path).expanduser().absolute()
    if candidate.name != expected_filename:
        raise ValueError(f"{label} filename is not canonical")
    raw = read_private_file(candidate, label=label, max_bytes=max_bytes)
    if not raw:
        raise ValueError(f"{label} is empty")
    document = strict_json_loads(raw, label=label)
    if not isinstance(document, Mapping):
        raise ValueError(f"{label} must be an object")
    return raw, document


def validate_sealed_v3dt_world_report(
    report_path: Path,
    source_path: Path,
    *,
    pipeline_config: Path,
    cameras_config: Path,
    calibration_config: Path,
    alignment_config: Path,
    launcher_dir: Path,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    min_bbox3d_coverage: float = DEFAULT_MIN_BBOX3D_COVERAGE,
    expected_build_root: Path | None = None,
) -> Mapping[str, object]:
    """Strictly replay V3DT world evidence and its live config binding."""

    if Path(report_path).expanduser().absolute().parent != Path(
        source_path
    ).expanduser().absolute().parent:
        raise ValueError("V3DT world report and source must share one evidence directory")
    report_raw, report = _read_private_json_object(
        report_path,
        expected_filename=CANONICAL_REPORT_FILENAME,
        max_bytes=256 * 1024,
        label="V3DT world report",
    )
    source_raw, source = _read_private_json_object(
        source_path,
        expected_filename=CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
        label="V3DT world source transcript",
    )
    expected_source_fields = {
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
    }
    if set(source) != expected_source_fields:
        raise ValueError("V3DT world source transcript schema drifted")
    normalized_session = str(session_id).strip().lower()
    normalized_lane = str(runtime_lane).strip().lower()
    if (
        type(source.get("schema_version")) is not int  # noqa: E721
        or source.get("schema_version") != 2
        or source.get("contract") != SOURCE_TRANSCRIPT_CONTRACT
        or type(source.get("contract_version")) is not int  # noqa: E721
        or source.get("contract_version") != SOURCE_TRANSCRIPT_VERSION
        or source.get("session_id") != normalized_session
        or source.get("runtime_lane") != normalized_lane
        or source.get("runtime_instance_id") != str(runtime_instance_id)
        or source.get("runtime_run_id") != str(runtime_run_id)
        or source.get("privacy") != SOURCE_PRIVACY_POLICY
        or source.get("policy") != _policy(min_bbox3d_coverage)
    ):
        raise ValueError("V3DT world source identity/policy binding drifted")
    messages = source.get("messages")
    if not isinstance(messages, list):
        raise ValueError("V3DT world source messages must be a list")
    live_binding = build_config_binding(
        pipeline_config=Path(pipeline_config),
        cameras_config=Path(cameras_config),
        calibration_config=Path(calibration_config),
        alignment_config=Path(alignment_config),
        launcher_dir=Path(launcher_dir),
        session_id=normalized_session,
        runtime_lane=normalized_lane,
        expected_build_root=expected_build_root,
    )
    source_binding = source.get("config_binding")
    if not isinstance(source_binding, Mapping):
        raise ValueError("V3DT world source config binding is missing")
    if _canonical_config_binding(source_binding) != live_binding:
        raise ValueError("V3DT world source differs from live config/launcher binding")
    canonical_source = _source_transcript_document(
        session_id=normalized_session,
        runtime_lane=normalized_lane,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        config_binding=live_binding,
        min_bbox3d_coverage=min_bbox3d_coverage,
        messages=messages,
    )
    if source != canonical_source or source_raw != _encoded_private_json(canonical_source):
        raise ValueError("V3DT world source transcript is not canonical")
    tracks, tracking_messages = _replay_source_messages(
        messages,
        session_id=normalized_session,
    )
    source_evidence = _source_evidence_metadata(
        encoded=source_raw,
        document=source,
    )
    recomputed = analyze_tracks(
        tracks,
        session_id=normalized_session,
        runtime_lane=normalized_lane,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        tracking_messages=tracking_messages,
        config_binding=live_binding,
        min_bbox3d_coverage=min_bbox3d_coverage,
        source_evidence=source_evidence,
    )
    if report != recomputed or report_raw != _encoded_private_json(recomputed):
        raise ValueError("V3DT world report does not exactly replay from sealed source")
    if report.get("ok") is not True or report.get("status") != "pass":
        raise ValueError("V3DT world report is not a passing exact replay")
    return report


def _write_private_json(
    path: Path, payload: Mapping[str, object], *, max_bytes: int = 256 * 1024
) -> None:
    destination = path.expanduser().absolute()
    if destination.name not in {
        CANONICAL_REPORT_FILENAME,
        CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
    }:
        raise ValueError("V3DT world evidence output filename is not canonical")
    encoded = _encoded_private_json(payload)
    try:
        atomic_create_private_file(
            destination,
            encoded,
            label=f"immutable V3DT world evidence {destination.name}",
            max_bytes=max_bytes,
        )
    except PrivatePathError as exc:
        raise ValueError(str(exc)) from exc


async def collect_tracks(
    uri: str,
    auth: RequiredInternalAuth,
    *,
    duration_s: float,
    session_id: str,
) -> tuple[list[Mapping[str, Any]], int, list[dict[str, object]]]:
    tracks: list[Mapping[str, Any]] = []
    tracking_messages = 0
    source_messages: list[dict[str, object]] = []
    async with connect_required_websocket(uri, auth, max_size=None) as websocket:
        deadline = time.monotonic() + max(1.0, float(duration_s))
        while time.monotonic() < deadline:
            timeout_s = min(2.0, max(0.05, deadline - time.monotonic()))
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, str):
                continue
            try:
                payload = strict_json_loads(raw, label="V3DT world live message")
            except ValueError as exc:
                raise ValueError(
                    "V3DT world WebSocket emitted invalid strict JSON"
                ) from exc
            if not isinstance(payload, Mapping) or payload.get("type") != "tracking":
                continue
            if len(source_messages) >= MAX_SOURCE_MESSAGES:
                raise ValueError("V3DT source transcript message bound exceeded")
            tracking_messages += 1
            rows = payload.get("tracks")
            accepted = (
                [row for row in rows if isinstance(row, Mapping)]
                if isinstance(rows, list)
                else []
            )
            if len(tracks) + len(accepted) > MAX_SOURCE_TRACKS:
                raise ValueError("V3DT source transcript track bound exceeded")
            tracks.extend(accepted)
            source_messages.append(
                {
                    "type": "tracking",
                    "observed_at_us": time.time_ns() // 1_000,
                    "tracks": [
                        _source_track(row, session_id=session_id) for row in accepted
                    ],
                }
            )
    return tracks, tracking_messages, source_messages


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--runtime-lane", choices=("v3dt",), required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--runtime-run-id", required=True)
    parser.add_argument(
        "--min-bbox3d-coverage", type=float, default=DEFAULT_MIN_BBOX3D_COVERAGE
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=REPO_ROOT / EXPECTED_BINDING_PATHS["pipeline"],
    )
    parser.add_argument(
        "--cameras-config",
        type=Path,
        default=REPO_ROOT / EXPECTED_BINDING_PATHS["cameras"],
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=REPO_ROOT / EXPECTED_BINDING_PATHS["calibration"],
    )
    parser.add_argument(
        "--alignment-config",
        type=Path,
        default=REPO_ROOT / EXPECTED_BINDING_PATHS["alignment"],
    )
    parser.add_argument(
        "--launcher-evidence-dir",
        dest="launcher_evidence_dir",
        type=Path,
        required=True,
        help="Exact supervisor launcher evidence directory containing launch-plan.json.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--source-out", type=Path, required=True)
    add_auth_token_file_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 0.0 < float(args.min_bbox3d_coverage) <= 1.0:
        print(json.dumps({"ok": False, "error": "min bbox3d coverage must be in (0,1]"}))
        return 2
    try:
        config_binding = build_config_binding(
            pipeline_config=args.pipeline_config,
            cameras_config=args.cameras_config,
            calibration_config=args.calibration_config,
            alignment_config=args.alignment_config,
            launcher_dir=args.launcher_evidence_dir,
            session_id=str(args.session_id).strip().lower(),
            runtime_lane=str(args.runtime_lane).strip().lower(),
        )
        launcher = args.launcher_evidence_dir.expanduser().absolute()
        if args.source_out.expanduser().absolute().parent != launcher:
            raise ValueError("V3DT source output must remain in the bound launcher directory")
        if args.out is not None and args.out.expanduser().absolute().parent != launcher:
            raise ValueError("V3DT report output must remain in the bound launcher directory")
        output_paths = (args.source_out,) if args.out is None else (args.source_out, args.out)
        require_fresh_private_file_bundle(
            output_paths,
            label="V3DT behavior evidence bundle",
        )
        auth = load_required_internal_auth(args.auth_token_file)
        _tracks, _messages, source_messages = asyncio.run(
            collect_tracks(
                args.ws,
                auth,
                duration_s=float(args.duration),
                session_id=str(args.session_id).strip().lower(),
            )
        )
        source_document = _source_transcript_document(
            session_id=args.session_id,
            runtime_lane=args.runtime_lane,
            runtime_instance_id=args.runtime_instance_id,
            runtime_run_id=args.runtime_run_id,
            config_binding=config_binding,
            min_bbox3d_coverage=float(args.min_bbox3d_coverage),
            messages=source_messages,
        )
        source_encoded = _encoded_private_json(source_document)
        if len(source_encoded) > MAX_SOURCE_TRANSCRIPT_BYTES:
            raise ValueError("V3DT source transcript byte bound exceeded")
        replay_tracks, replay_message_count = _replay_source_messages(
            source_document["messages"],  # type: ignore[arg-type]
            session_id=str(args.session_id).strip().lower(),
        )
        _write_private_json(
            args.source_out,
            source_document,
            max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 1
    result = analyze_tracks(
        replay_tracks,
        session_id=args.session_id,
        runtime_lane=args.runtime_lane,
        runtime_instance_id=args.runtime_instance_id,
        runtime_run_id=args.runtime_run_id,
        tracking_messages=replay_message_count,
        config_binding=config_binding,
        min_bbox3d_coverage=float(args.min_bbox3d_coverage),
        source_evidence=_source_evidence_metadata(
            encoded=source_encoded, document=source_document
        ),
    )
    if args.out is not None:
        _write_private_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "blocked":
        return 2
    return 0 if bool(result["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
