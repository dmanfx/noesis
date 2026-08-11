from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


TRACK_CONTAINERS = (
    ("menonTrackDebug", "canonicalPaths"),
    ("menonTrackDebug", "canonicalInfos"),
    ("latestTrackInfos",),
    ("latestTrackingPaths",),
)

TRANSFORM_KEYS = {
    "world_to_menon_col_major",
    "world_to_scene_col_major",
    "worldToMenonColMajor",
    "worldToSceneColMajor",
}

BACKEND_POINT_KEYS = (
    "backendWorldPosition",
    "backend_world_position",
    "backendWorldRaw",
    "backend_world_raw",
    "backendWorldM",
    "backend_world_m",
    "backendWorld",
    "backend_world",
    "noesisWorld",
    "noesis_world",
    "world_point",
)

MENON_POINT_KEYS = (
    "menon_point",
    "menonPoint",
    "menonScene",
    "menon_scene",
    "scenePoint",
    "scene_point",
    "scene",
    "position",
)

BACKEND_FRAMES = {"backend_world_m", "noesis_backend_world_m"}
MENON_FRAMES = {"menon_scene", "virtual_twin_menon_scene", "scene"}

CAMERA_REPROJECTION_PATHS = (
    ("camera_reprojections",),
    ("cameraReprojections",),
    ("menonCameraReprojections",),
    ("menonCameraReprojectionDebug",),
    ("reprojectionDebugInfo", "camera_reprojections"),
    ("reprojectionDebugInfo", "cameraReprojections"),
    ("reprojectionCompareInfo", "camera_reprojections"),
    ("reprojectionCompareInfo", "cameraReprojections"),
    ("windowReprojectionCameraDebug", "camera_reprojections"),
    ("windowReprojectionCameraDebug", "cameraReprojections"),
)


def _is_seq(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _seq(value: Any) -> Sequence[Any]:
    return value if _is_seq(value) else []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _point3(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        for keys in (("x", "y", "z"), ("X", "Y", "Z")):
            point = [_finite_float(value.get(key)) for key in keys]
            if all(item is not None for item in point):
                return [float(item) for item in point if item is not None]
        for key in ("point", "position", "world", "scene"):
            nested = _point3(value.get(key))
            if nested is not None:
                return nested
        return None
    if not _is_seq(value) or len(value) < 3:
        return None
    point = [_finite_float(value[idx]) for idx in range(3)]
    if not all(item is not None for item in point):
        return None
    return [float(item) for item in point if item is not None]


def _col_major16(value: Any) -> list[float] | None:
    if not _is_seq(value) or len(value) != 16:
        return None
    out = [_finite_float(item) for item in value]
    if not all(item is not None for item in out):
        return None
    return [float(item) for item in out if item is not None]


def _get_path(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _iter_mappings(value: Any, *, max_depth: int = 6) -> Sequence[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []

    def visit(item: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(item, Mapping):
            found.append(item)
            for nested in item.values():
                visit(nested, depth + 1)
        elif _is_seq(item):
            for nested in item[:200]:
                visit(nested, depth + 1)

    visit(value, 0)
    return found


def _find_world_to_menon(payload: Mapping[str, Any]) -> list[float] | None:
    for path in (
        ("world_to_menon_col_major",),
        ("world_to_scene_col_major",),
        ("align", "scene_similarity", "world_to_scene_col_major"),
        ("calibrationData", "align", "scene_similarity", "world_to_scene_col_major"),
        ("calibrationData", "scene_similarity", "world_to_scene_col_major"),
        ("calibrationBundle", "align", "scene_similarity", "world_to_scene_col_major"),
        ("calibration_bundle", "align", "scene_similarity", "world_to_scene_col_major"),
    ):
        candidate = _col_major16(_get_path(payload, path))
        if candidate is not None:
            return candidate
    for item in _iter_mappings(payload):
        for key, value in item.items():
            if key in TRANSFORM_KEYS:
                candidate = _col_major16(value)
                if candidate is not None:
                    return candidate
    return None


def _find_floor_y(payload: Mapping[str, Any]) -> float:
    for path in (
        ("floor_y_scene",),
        ("menon_floor_y",),
        ("floorY",),
        ("menonTrackDebug", "floorY"),
        ("menonTrackDebug", "productionGeometry", "environmentFloorY"),
        ("calibrationData", "floor_y"),
        ("calibrationData", "align", "floor_y"),
    ):
        value = _finite_float(_get_path(payload, path))
        if value is not None:
            return value
    return 0.0


def _identity_value(item: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is None or value == "":
            continue
        return str(value)
    return None


def _identity_keys(item: Mapping[str, Any]) -> list[str]:
    entity_id = _identity_value(item, ("entity_id", "entityId"))
    camera_id = _identity_value(item, ("camera_id", "cameraId", "camera"))
    stable_id = _identity_value(item, ("stable_id", "stableId"))
    tracker_id = _identity_value(item, ("tracker_id", "trackerId", "track_id", "trackId", "id"))
    path_key = _identity_value(item, ("pathKey", "trackKey", "path_key", "track_key"))
    keys: list[str] = []
    if entity_id:
        keys.append(f"entity:{entity_id}")
    if camera_id and path_key:
        keys.append(f"camera_path:{camera_id}:{path_key}")
    if camera_id and stable_id:
        keys.append(f"camera_stable:{camera_id}:{stable_id}")
    if camera_id and tracker_id:
        keys.append(f"camera_tracker:{camera_id}:{tracker_id}")
    if path_key:
        keys.append(f"path:{path_key}")
    if stable_id:
        keys.append(f"stable:{stable_id}")
    if tracker_id:
        keys.append(f"tracker:{tracker_id}")
    return keys


def _entity_id(item: Mapping[str, Any]) -> str:
    entity_id = _identity_value(item, ("entity_id", "entityId"))
    if entity_id is not None:
        return entity_id
    stable_id = _identity_value(item, ("stable_id", "stableId"))
    if stable_id is not None:
        return f"stable-{stable_id}"
    tracker_id = _identity_value(item, ("tracker_id", "trackerId", "track_id", "trackId", "id"))
    if tracker_id is not None:
        return f"tracker-{tracker_id}"
    path_key = _identity_value(item, ("pathKey", "trackKey", "path_key", "track_key"))
    if path_key is not None:
        return f"path-{path_key}"
    return "entity"


def _frame_text(item: Mapping[str, Any]) -> str:
    values = [
        item.get("world_frame"),
        item.get("worldFrame"),
        item.get("coord_space"),
        item.get("coordSpace"),
        item.get("backendWorldCoordSpace"),
        item.get("coordinate_frame"),
        item.get("coordinateFrame"),
    ]
    return " ".join(str(value).strip().lower() for value in values if value is not None)


def _extract_backend_world(item: Mapping[str, Any]) -> tuple[list[float] | None, str | None]:
    for key in BACKEND_POINT_KEYS:
        point = _point3(item.get(key))
        if point is not None:
            return point, key
    for path in (
        ("transform", "backendWorldRaw"),
        ("virtualTwinTransform", "backendWorldRaw"),
        ("debug", "backendWorldRaw"),
    ):
        point = _point3(_get_path(item, path))
        if point is not None:
            return point, ".".join(path)
    frame = _frame_text(item)
    if any(frame_name in frame for frame_name in BACKEND_FRAMES):
        point = _point3(item.get("world"))
        if point is not None:
            return point, "world"
    return None, None


def _extract_menon_point(item: Mapping[str, Any], *, backend_source: str | None) -> tuple[list[float] | None, str | None]:
    for key in MENON_POINT_KEYS:
        point = _point3(item.get(key))
        if point is not None:
            return point, key
    frame = _frame_text(item)
    world_point = _point3(item.get("world"))
    if world_point is not None:
        if any(frame_name in frame for frame_name in MENON_FRAMES):
            return world_point, "world"
        if backend_source and backend_source != "world":
            return world_point, "world"
    return None, None


def _timestamp_s(
    item: Mapping[str, Any],
    *,
    seconds_keys: Sequence[str],
    millis_keys: Sequence[str],
    micros_keys: Sequence[str] = (),
) -> float | None:
    for key in seconds_keys:
        value = _finite_float(item.get(key))
        if value is not None:
            return value
    for key in millis_keys:
        value = _finite_float(item.get(key))
        if value is not None:
            return value / 1000.0
    for key in micros_keys:
        value = _finite_float(item.get(key))
        if value is not None:
            return value / 1_000_000.0
    return None


def _flatten_camera_records(container: Mapping[str, Any], source_path: str, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for camera_id, camera_payload in container.items():
        if isinstance(camera_payload, Mapping) and _is_seq(camera_payload.get("tracks")):
            parent = camera_payload
            tracks = camera_payload.get("tracks")
        elif _is_seq(camera_payload):
            parent = {}
            tracks = camera_payload
        else:
            continue
        for index, track in enumerate(_seq(tracks)):
            if not isinstance(track, Mapping):
                continue
            row = dict(track)
            row.setdefault("cameraId", camera_id)
            row.setdefault("updatedAtMs", parent.get("updatedAtMs"))
            row.setdefault("source", parent.get("source") or row.get("source"))
            row.setdefault("capturedAtMs", snapshot.get("capturedAtMs"))
            row["_browser_source_path"] = f"{source_path}.{camera_id}.tracks[{index}]"
            records.append(row)
    return records


def _flatten_records(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in _seq(snapshot.get("placements")):
        if isinstance(item, Mapping):
            row = dict(item)
            row.setdefault("capturedAtMs", snapshot.get("capturedAtMs"))
            row["_browser_source_path"] = "placements"
            records.append(row)
    for path in TRACK_CONTAINERS:
        container = _get_path(snapshot, path)
        if isinstance(container, Mapping):
            records.extend(_flatten_camera_records(container, ".".join(path), snapshot))
            continue
        if _is_seq(container):
            for index, item in enumerate(container):
                if not isinstance(item, Mapping):
                    continue
                row = dict(item)
                row.setdefault("capturedAtMs", snapshot.get("capturedAtMs"))
                row["_browser_source_path"] = f"{'.'.join(path)}[{index}]"
                records.append(row)
    return records


def _is_canonical_render_record(item: Mapping[str, Any]) -> bool:
    entity_id = _identity_value(item, ("entity_id", "entityId"))
    path_key = _identity_value(item, ("pathKey", "trackKey", "path_key", "track_key"))
    scene_transform_count = _finite_float(item.get("sceneTransformCount"))
    return bool(
        entity_id
        and path_key == f"entity:{entity_id}"
        and item.get("worldSource") == "canonical_world_snapshot"
        and item.get("sourceFrame") == "backend_world_m"
        and item.get("coordinateFrame") == "menon_scene"
        and scene_transform_count == 1.0
        and item.get("presenceActive") is True
        and item.get("lifecycle") != "lost"
        and _point3(item.get("backendWorldPosition")) is not None
        and _point3(item.get("position")) is not None
    )


CANONICAL_STATE_KEYS = {
    "contract",
    "contractVersion",
    "snapshotId",
    "producer",
    "runId",
    "sequence",
    "observedStartUs",
    "observedEndUs",
    "publishedAtUs",
    "frame",
    "units",
    "entities",
    "reason",
}
CANONICAL_PRODUCER_KEYS = {"runtime", "instance_id", "run_id", "software_revision"}
CANONICAL_ENTITY_KEYS = {
    "entity_id",
    "subject",
    "lifecycle",
    "position",
    "covariance",
    "velocity_mps",
    "room_id",
    "observed_at_us",
    "stale_after_us",
    "sources",
    "conflict",
    "conflict_reason",
}
CANONICAL_SUBJECT_KEYS = {
    "subject_id",
    "kind",
    "generation",
    "resident_uuid",
    "display_name",
    "stable_id",
}
CANONICAL_PRESENTATION_KEYS = {
    "contract",
    "contractVersion",
    "snapshotId",
    "runId",
    "sequence",
    "entityCount",
    "sourceFrame",
    "renderFrame",
    "sceneTransformCount",
    "reason",
    "lastError",
    "atMs",
}
CANONICAL_RENDER_KEYS = {
    "pathKey",
    "trackKey",
    "trackId",
    "entityId",
    "colorKey",
    "position",
    "backendWorldPosition",
    "sourceFrame",
    "coordinateFrame",
    "sceneTransformCount",
    "presenceActive",
    "trailOnly",
    "lifecycle",
    "observedAtUs",
    "staleAfterUs",
    "roomId",
    "conflict",
    "conflictReason",
    "subjectId",
    "identityKind",
    "identityLabel",
    "displayName",
    "residentUuid",
    "visitorGeneration",
    "stableId",
    "stable_id",
    "worldSource",
    "worldQuality",
}
SCENE_COHORT_KEYS = {
    "loading",
    "activeReleaseId",
    "activeParticipantIds",
    "lastCommittedAtMs",
    "lastAttemptAtMs",
    "lastError",
    "generation",
}
EXPECTED_SCENE_PARTICIPANTS = {
    "authored-environment",
    "virtual-twin-surface",
    "noesis-room-reconstruction",
}
AUTH_PROOF_KEYS = {"authenticated", "role", "sessionId", "expiresAt", "checkedAtMs", "origin"}


def _exact_keys(value: Any, expected: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def _strict_safe_int(value: Any, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _strict_point3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        return None
    return [float(item) for item in value]


def _strict_vector3(value: Any) -> list[float] | None:
    if not isinstance(value, Mapping) or set(value) != {"x", "y", "z"}:
        return None
    point = [value.get(key) for key in ("x", "y", "z")]
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in point):
        return None
    return [float(item) for item in point]


def _points_equal(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    left_point = _strict_point3(left)
    right_point = _point3(right)
    if left_point is None or right_point is None:
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(left_point, right_point, strict=True))


def _normalized_origin(value: Any) -> str | None:
    try:
        parsed = urlsplit(str(value))
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        port = parsed.port or (443 if scheme == "https" else 80)
    except Exception:
        return None
    if scheme not in {"http", "https"} or not hostname or parsed.username or parsed.password:
        return None
    host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{scheme}://{host}:{port}"


def _transform_col_major(point: Any, matrix: Sequence[float]) -> list[float] | None:
    source = _point3(point)
    if source is None or len(matrix) != 16:
        return None
    x, y, z = source
    w = (matrix[3] * x) + (matrix[7] * y) + (matrix[11] * z) + matrix[15]
    if not math.isfinite(w) or abs(w) <= 1e-12:
        return None
    transformed = [
        ((matrix[0] * x) + (matrix[4] * y) + (matrix[8] * z) + matrix[12]) / w,
        ((matrix[1] * x) + (matrix[5] * y) + (matrix[9] * z) + matrix[13]) / w,
        ((matrix[2] * x) + (matrix[6] * y) + (matrix[10] * z) + matrix[14]) / w,
    ]
    return transformed if all(math.isfinite(item) for item in transformed) else None


def _identity_label(subject: Mapping[str, Any]) -> str:
    kind = subject.get("kind")
    display_name = subject.get("display_name")
    stable_id = subject.get("stable_id")
    if kind == "resident":
        return display_name or "Resident"
    if kind == "visitor":
        return display_name or (f"Visitor {stable_id}" if stable_id else "Visitor")
    return "Provisional" if kind == "provisional" else "Unknown"


def _canonical_browser_context(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    auth = snapshot.get("authState")
    if not isinstance(auth, Mapping) or auth.get("authenticated") is not True or auth.get("role") not in {"owner", "operator"}:
        return None, "authenticated_browser_state_required"
    auth_proof = snapshot.get("authSessionProof")
    proof_checked_at_ms = _strict_safe_int(
        auth_proof.get("checkedAtMs"), 1
    ) if _exact_keys(auth_proof, AUTH_PROOF_KEYS) else None
    if (
        proof_checked_at_ms is None
        or auth_proof.get("authenticated") is not True
        or auth_proof.get("role") != auth.get("role")
        or not isinstance(auth_proof.get("sessionId"), str)
        or not auth_proof.get("sessionId")
        or not isinstance(auth_proof.get("expiresAt"), str)
        or not auth_proof.get("expiresAt")
        or not isinstance(auth_proof.get("origin"), str)
        or not auth_proof.get("origin")
        or auth_proof.get("origin") != _normalized_origin(snapshot.get("href"))
    ):
        return None, "fresh_auth_session_proof_required"

    state = snapshot.get("canonicalWorldState")
    if not _exact_keys(state, CANONICAL_STATE_KEYS):
        return None, "canonical_world_state_shape_invalid"
    producer = state.get("producer")
    if not _exact_keys(producer, CANONICAL_PRODUCER_KEYS):
        return None, "canonical_world_producer_shape_invalid"
    runtime = producer.get("runtime")
    run_id = producer.get("run_id")
    sequence = _strict_safe_int(state.get("sequence"))
    observed_start_us = _strict_safe_int(state.get("observedStartUs"), 1)
    observed_end_us = _strict_safe_int(state.get("observedEndUs"), 1)
    published_at_us = _strict_safe_int(state.get("publishedAtUs"), 1)
    if (
        state.get("contract") != "noesis.world.snapshot"
        or state.get("contractVersion") != 1
        or runtime not in {"ds8", "ds9"}
        or not all(isinstance(producer.get(key), str) and producer.get(key).strip() == producer.get(key) and producer.get(key)
                   for key in ("instance_id", "run_id", "software_revision"))
        or state.get("runId") != run_id
        or not isinstance(state.get("snapshotId"), str)
        or not state.get("snapshotId")
        or state.get("frame") != "backend_world_m"
        or state.get("units") != "meters"
        or sequence is None
        or observed_start_us is None
        or observed_end_us is None
        or published_at_us is None
        or not (observed_start_us <= observed_end_us <= published_at_us)
        or not isinstance(state.get("reason"), str)
    ):
        return None, "canonical_world_state_contract_invalid"

    captured_at_ms = _strict_safe_int(snapshot.get("capturedAtMs"), 1)
    if (
        captured_at_ms is None
        or proof_checked_at_ms < captured_at_ms
        or proof_checked_at_ms - captured_at_ms > 60_000
        or published_at_us > (captured_at_ms * 1000) + 5_000_000
    ):
        return None, "canonical_world_capture_clock_invalid"

    presentation = snapshot.get("canonicalWorldPresentation")
    debug = snapshot.get("menonTrackDebug")
    debug_presentation = debug.get("canonicalWorld") if isinstance(debug, Mapping) else None
    if not _exact_keys(presentation, CANONICAL_PRESENTATION_KEYS) or presentation != debug_presentation:
        return None, "canonical_presentation_coherence_invalid"
    if (
        presentation.get("contract") != state.get("contract")
        or presentation.get("contractVersion") != state.get("contractVersion")
        or presentation.get("snapshotId") != state.get("snapshotId")
        or presentation.get("runId") != state.get("runId")
        or presentation.get("sequence") != state.get("sequence")
        or presentation.get("sourceFrame") != "backend_world_m"
        or presentation.get("renderFrame") != "menon_scene"
        or presentation.get("lastError") is not None
        or _strict_safe_int(presentation.get("atMs"), 1) is None
        or presentation.get("atMs") > captured_at_ms
    ):
        return None, "canonical_presentation_contract_invalid"

    if not isinstance(debug, Mapping):
        return None, "canonical_debug_snapshot_missing"
    top_projection_mode = snapshot.get("trackingProjectionMode")
    if top_projection_mode not in (None, "canonical_world_snapshot"):
        return None, "canonical_projection_mode_invalid"
    if (
        debug.get("projectionMode") != "canonical_world_snapshot"
        or debug.get("pipeline") != "noesis_world_snapshot_v1_exact_scene_transform"
    ):
        return None, "canonical_debug_pipeline_invalid"
    paths = debug.get("canonicalPaths")
    infos = debug.get("canonicalInfos")
    if not isinstance(paths, list) or not isinstance(infos, list):
        return None, "canonical_debug_paths_invalid"
    if (
        _strict_safe_int(debug.get("canonicalPathCount")) != len(paths)
        or _strict_safe_int(debug.get("canonicalInfoCount")) != len(infos)
        or infos != paths
        or snapshot.get("latestTrackingPaths") != paths
        or snapshot.get("latestTrackInfos") != infos
    ):
        return None, "canonical_debug_paths_incoherent"

    entities = state.get("entities")
    if not isinstance(entities, list) or len(entities) != len(paths):
        return None, "canonical_entity_count_incoherent"
    if (
        presentation.get("entityCount") != len(paths)
        or presentation.get("sceneTransformCount") != (1 if paths else 0)
    ):
        return None, "canonical_presentation_count_incoherent"

    cohort = snapshot.get("promotedSceneCohort")
    if not _exact_keys(cohort, SCENE_COHORT_KEYS):
        return None, "scene_cohort_shape_invalid"
    participants = cohort.get("activeParticipantIds")
    if (
        cohort.get("loading") is not False
        or not isinstance(cohort.get("activeReleaseId"), str)
        or not cohort.get("activeReleaseId")
        or not isinstance(participants, list)
        or set(participants) != EXPECTED_SCENE_PARTICIPANTS
        or len(participants) != len(EXPECTED_SCENE_PARTICIPANTS)
        or _strict_safe_int(cohort.get("lastCommittedAtMs"), 1) is None
        or _strict_safe_int(cohort.get("lastAttemptAtMs"), 1) is None
        or cohort.get("lastError") is not None
        or _strict_safe_int(cohort.get("generation"), 1) is None
        or cohort.get("lastCommittedAtMs") > captured_at_ms
        or cohort.get("lastAttemptAtMs") > captured_at_ms
    ):
        return None, "scene_cohort_not_current"

    transform = _col_major16(
        _get_path(snapshot, ("calibrationData", "align", "scene_similarity", "world_to_scene_col_major"))
    )
    if transform is None:
        return None, "canonical_scene_transform_missing"

    entities_by_id: dict[str, Mapping[str, Any]] = {}
    for entity in entities:
        if not _exact_keys(entity, CANONICAL_ENTITY_KEYS) or not _exact_keys(entity.get("subject"), CANONICAL_SUBJECT_KEYS):
            return None, "canonical_entity_shape_invalid"
        entity_id = entity.get("entity_id")
        lifecycle = entity.get("lifecycle")
        subject = entity.get("subject")
        subject_kind = subject.get("kind")
        subject_id = subject.get("subject_id")
        resident_uuid = subject.get("resident_uuid")
        display_name = subject.get("display_name")
        generation = _strict_safe_int(subject.get("generation"))
        stable_id = subject.get("stable_id")
        observed_at_us = _strict_safe_int(entity.get("observed_at_us"), 1)
        stale_after_us = _strict_safe_int(entity.get("stale_after_us"), 1)
        if (
            not isinstance(entity_id, str)
            or not entity_id
            or entity_id in entities_by_id
            or lifecycle not in {"present", "held"}
            or _strict_vector3(entity.get("position")) is None
            or subject_kind not in {"resident", "visitor", "unknown", "provisional"}
            or not isinstance(subject_id, str)
            or not subject_id
            or generation is None
            or (stable_id is not None and _strict_safe_int(stable_id, 1) is None)
            or (display_name is not None and (not isinstance(display_name, str) or not display_name))
            or (subject_kind == "resident" and (not isinstance(resident_uuid, str) or not resident_uuid))
            or (subject_kind != "resident" and resident_uuid is not None)
            or (subject_kind in {"unknown", "provisional"} and display_name is not None)
            or observed_at_us is None
            or stale_after_us is None
            or observed_at_us > published_at_us
            or stale_after_us < observed_at_us
            or stale_after_us < captured_at_ms * 1000
        ):
            return None, "canonical_entity_contract_invalid"
        entities_by_id[entity_id] = entity

    accepted_paths: list[dict[str, Any]] = []
    for path in paths:
        if not _exact_keys(path, CANONICAL_RENDER_KEYS):
            return None, "canonical_render_shape_invalid"
        entity_id = path.get("entityId")
        entity = entities_by_id.get(entity_id)
        subject = entity.get("subject") if entity else None
        expected_position = _transform_col_major(entity.get("position") if entity else None, transform)
        if (
            entity is None
            or path.get("pathKey") != f"entity:{entity_id}"
            or path.get("trackKey") != f"entity:{entity_id}"
            or path.get("trackId") != entity_id
            or path.get("colorKey") != subject.get("subject_id")
            or path.get("sourceFrame") != "backend_world_m"
            or path.get("coordinateFrame") != "menon_scene"
            or _strict_safe_int(path.get("sceneTransformCount")) != 1
            or path.get("presenceActive") is not True
            or path.get("trailOnly") is not False
            or path.get("lifecycle") != entity.get("lifecycle")
            or path.get("observedAtUs") != entity.get("observed_at_us")
            or path.get("staleAfterUs") != entity.get("stale_after_us")
            or path.get("roomId") != entity.get("room_id")
            or path.get("conflict") != entity.get("conflict")
            or path.get("conflictReason") != entity.get("conflict_reason")
            or path.get("subjectId") != subject.get("subject_id")
            or path.get("identityKind") != subject.get("kind")
            or path.get("displayName") != subject.get("display_name")
            or path.get("residentUuid") != subject.get("resident_uuid")
            or path.get("visitorGeneration") != subject.get("generation")
            or path.get("stableId") != subject.get("stable_id")
            or path.get("stable_id") != subject.get("stable_id")
            or path.get("worldSource") != "canonical_world_snapshot"
            or path.get("worldQuality") != ("held" if entity.get("lifecycle") == "held" else "observed")
            or path.get("identityLabel") != _identity_label(subject)
            or not _points_equal(path.get("backendWorldPosition"), entity.get("position"))
            or not _points_equal(path.get("position"), expected_position, tolerance=1e-6)
        ):
            return None, "canonical_render_entity_mismatch"
        accepted_paths.append(dict(path))

    return {
        "state": state,
        "presentation": presentation,
        "paths": accepted_paths,
        "entity_ids": set(entities_by_id),
        "world_to_menon": transform,
        "runtime": runtime,
    }, "accepted"


def _first_peer_value(index: Mapping[str, Sequence[dict[str, Any]]], keys: Sequence[str], field: str) -> Any:
    for key in keys:
        for peer in index.get(key, ()):
            value = peer.get(field)
            if value is not None:
                return value
    return None


def _trace_trails_from_payload(
    snapshot: Mapping[str, Any],
    placements: Sequence[Mapping[str, Any]],
    *,
    allowed_entity_ids: set[str],
) -> list[dict[str, Any]]:
    explicit: list[dict[str, Any]] = []
    for item in _seq(snapshot.get("trails")):
        if isinstance(item, Mapping) and str(item.get("entity_id") or "") in allowed_entity_ids:
            explicit.append(dict(item))
    if explicit:
        return explicit

    by_entity: dict[tuple[str, str | None], list[Mapping[str, Any]]] = defaultdict(list)
    for item in placements:
        by_entity[(str(item.get("entity_id") or "entity"), item.get("camera_id"))].append(item)
    trails: list[dict[str, Any]] = []
    for (entity_id, camera_id), rows in by_entity.items():
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda row: float(row.get("noesis_ts_s") or row.get("menon_ts_s") or 0.0))
        trails.append(
            {
                "entity_id": entity_id,
                "camera_id": camera_id,
                "room": ordered[-1].get("room"),
                "noesis_world_points": [row["world_point"] for row in ordered if _point3(row.get("world_point")) is not None],
                "menon_points": [row["menon_point"] for row in ordered if _point3(row.get("menon_point")) is not None],
                "evidence": sorted({evidence for row in ordered for evidence in _seq(row.get("evidence"))}),
            }
        )
    return trails


def _first_value(primary: Mapping[str, Any], secondary: Mapping[str, Any], *keys: str) -> Any:
    for source in (primary, secondary):
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def _iter_reprojection_records(value: Any, source_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    direct_keys = {
        "camera_id",
        "cameraId",
        "camera",
        "source_frame",
        "sourceFrame",
        "menon_render",
        "menonRender",
        "render_path",
        "renderPath",
        "overlay_path",
        "overlayPath",
        "layers",
        "metrics",
        "anchor_mean_error_px",
        "anchorMeanErrorPx",
        "bbox_iou",
        "bboxIoU",
        "avatar_iou",
        "avatarIoU",
    }
    if isinstance(value, Mapping):
        if any(key in value for key in direct_keys):
            row = dict(value)
            row.setdefault("_browser_source_path", source_path)
            records.append(row)
            return records
        for key, nested in value.items():
            nested_path = f"{source_path}.{key}"
            if isinstance(nested, Mapping):
                row = dict(nested)
                row.setdefault("camera_id", key)
                row.setdefault("_browser_source_path", nested_path)
                records.append(row)
            elif _is_seq(nested):
                for index, item in enumerate(nested):
                    if not isinstance(item, Mapping):
                        continue
                    row = dict(item)
                    row.setdefault("camera_id", key)
                    row.setdefault("_browser_source_path", f"{nested_path}[{index}]")
                    records.append(row)
    elif _is_seq(value):
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            row.setdefault("_browser_source_path", f"{source_path}[{index}]")
            records.append(row)
    return records


def _trace_camera_reprojections_from_payload(
    snapshot: Mapping[str, Any],
    *,
    raw_snapshot_path: str,
    allowed_entity_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in CAMERA_REPROJECTION_PATHS:
        value = _get_path(snapshot, path)
        if value is None:
            continue
        for item in _iter_reprojection_records(value, ".".join(path)):
            metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}
            layers = item.get("layers") or item.get("renderedLayers") or item.get("requiredLayers") or metrics.get("layers")
            row: dict[str, Any] = {
                "camera_id": str(item.get("camera_id") or item.get("cameraId") or item.get("camera") or "camera"),
                "room": item.get("room") or item.get("zone"),
                "entity_id": item.get("entity_id") or item.get("stable_id") or item.get("stableId") or item.get("id"),
                "source_frame": _first_value(item, metrics, "source_frame", "sourceFrame", "source_frame_path", "sourceFramePath"),
                "menon_render": _first_value(item, metrics, "menon_render", "menonRender", "render_path", "renderPath"),
                "overlay_path": _first_value(item, metrics, "overlay_path", "overlayPath"),
                "layers": [str(layer) for layer in _seq(layers)],
                "evidence": [raw_snapshot_path, str(item.get("_browser_source_path") or ".".join(path))],
            }
            if str(row.get("entity_id") or "") not in allowed_entity_ids:
                continue
            for key, aliases in {
                "mean_error_px": ("mean_error_px", "meanErrorPx"),
                "max_error_px": ("max_error_px", "maxErrorPx"),
                "anchor_mean_error_px": ("anchor_mean_error_px", "anchorMeanErrorPx"),
                "anchor_max_error_px": ("anchor_max_error_px", "anchorMaxErrorPx"),
                "floor_grid_mean_error_px": ("floor_grid_mean_error_px", "floorGridMeanErrorPx"),
                "room_edge_mean_error_px": ("room_edge_mean_error_px", "roomEdgeMeanErrorPx"),
                "bbox_iou": ("bbox_iou", "bboxIoU"),
                "avatar_iou": ("avatar_iou", "avatarIoU"),
                "mask_iou": ("mask_iou", "maskIoU"),
            }.items():
                value = _finite_float(_first_value(item, metrics, *aliases))
                if value is not None:
                    row[key] = value
            has_signal = any(
                row.get(key)
                for key in (
                    "source_frame",
                    "menon_render",
                    "overlay_path",
                    "layers",
                    "mean_error_px",
                    "anchor_mean_error_px",
                    "bbox_iou",
                    "avatar_iou",
                    "mask_iou",
                )
            )
            if not has_signal:
                continue
            dedupe_key = (
                str(row.get("camera_id") or ""),
                str(row.get("source_frame") or ""),
                str(row.get("menon_render") or ""),
                str(row.get("overlay_path") or ""),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(row)
    return rows


def browser_snapshot_to_menon_trace(
    snapshot: Mapping[str, Any],
    *,
    run_id: str = "",
    page_url: str = "",
    screenshot_path: str | Path | None = None,
    raw_snapshot_path: str | Path = "menon/browser_snapshot.json",
) -> dict[str, Any]:
    """Convert Menon browser debug globals into the shared Menon trace contract."""

    canonical_context, admission_reason = _canonical_browser_context(snapshot)
    world_to_menon = canonical_context.get("world_to_menon") if canonical_context else None
    records = canonical_context.get("paths", []) if canonical_context else []
    candidates: list[dict[str, Any]] = []
    for record in records:
        if not _is_canonical_render_record(record):
            continue
        backend, backend_source = _extract_backend_world(record)
        menon, menon_source = _extract_menon_point(record, backend_source=backend_source)
        row = dict(record)
        row.setdefault("capturedAtMs", snapshot.get("capturedAtMs"))
        row["_backend_world"] = backend
        row["_backend_source"] = backend_source
        row["_menon_point"] = menon
        row["_menon_source"] = menon_source
        row["_identity_keys"] = _identity_keys(row)
        candidates.append(row)

    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        for key in row["_identity_keys"]:
            by_identity[key].append(row)

    evidence_path = str(raw_snapshot_path)
    placements: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in candidates:
        identity_keys = row["_identity_keys"]
        backend = row["_backend_world"] or _first_peer_value(by_identity, identity_keys, "_backend_world")
        menon = row["_menon_point"] or _first_peer_value(by_identity, identity_keys, "_menon_point")
        if backend is None or menon is None:
            continue
        camera_id = _identity_value(row, ("camera_id", "cameraId", "camera"))
        room = _identity_value(row, ("room", "room_id", "roomId", "zone", "expectedRoomZone", "roomZone"))
        entity_id = _entity_id(row)
        stable_id = _identity_value(row, ("stable_id", "stableId"))
        tracker_id = _identity_value(row, ("tracker_id", "trackerId", "track_id", "trackId", "id"))
        noesis_ts_s = _timestamp_s(
            row,
            seconds_keys=("noesis_ts_s", "noesisTsS", "frame_ts_s", "frameTsS", "timestamp_s", "timestampS", "ts_s", "tsS"),
            millis_keys=("noesis_ts_ms", "noesisTsMs", "frame_ts_ms", "frameTsMs"),
            micros_keys=("observed_at_us", "observedAtUs"),
        )
        menon_ts_s = _timestamp_s(
            row,
            seconds_keys=("menon_ts_s", "menonTsS", "render_ts_s", "renderTsS"),
            millis_keys=("menon_ts_ms", "menonTsMs", "render_ts_ms", "renderTsMs", "updatedAtMs", "capturedAtMs"),
        )
        dedupe_key = (
            entity_id,
            camera_id,
            tuple(round(value, 4) for value in backend),
            tuple(round(value, 4) for value in menon),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        placement = {
            "entity_id": entity_id,
            "camera_id": camera_id,
            "room": room,
            "world_point": backend,
            "menon_point": menon,
            "evidence": [evidence_path, str(row.get("_browser_source_path") or "browser_snapshot")],
            "transform_audit": [
                {"stage": "source_payload", "frame": "backend_world_m", "point": backend},
                {
                    "stage": "browser_trace_alignment",
                    "from_frame": "backend_world_m",
                    "to_frame": "menon_scene",
                    "matrix_col_major": world_to_menon,
                },
                {"stage": "render_anchor", "frame": "menon_scene", "point": menon},
            ],
        }
        if stable_id is not None:
            placement["stable_id"] = stable_id
        if tracker_id is not None:
            placement["tracker_id"] = tracker_id
        if noesis_ts_s is not None:
            placement["noesis_ts_s"] = noesis_ts_s
        if menon_ts_s is not None:
            placement["menon_ts_s"] = menon_ts_s
        placements.append(placement)

    cameras = sorted({str(item.get("camera_id")) for item in placements if item.get("camera_id")})
    rooms = sorted({str(item.get("room")) for item in placements if item.get("room")})
    producer_runtime = str(canonical_context.get("runtime") if canonical_context else "").strip().lower()
    pipeline_config = {
        "ds8": "config/infer.yaml",
        "ds9": "DS9/config/infer.yaml",
    }.get(producer_runtime)
    trace: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(run_id or snapshot.get("run_id") or "menon_browser_trace"),
        "source": {
            "repo": "Noesis_Devel",
            "runtime": producer_runtime or None,
            "pipeline_config": pipeline_config,
            "cameras_config": "config/cameras.yaml" if producer_runtime == "ds8" else None,
            "menon_available": True,
        },
        "scope": {"tiers": ["menon_trace", "menon_browser_capture"], "rooms": rooms, "cameras": cameras},
        "floor_y_scene": _find_floor_y(snapshot),
        "placements": placements,
        "trails": _trace_trails_from_payload(
            snapshot,
            placements,
            allowed_entity_ids=set(canonical_context.get("entity_ids", set())) if canonical_context else set(),
        ),
        "browser": {
            "page_url": str(page_url or snapshot.get("href") or snapshot.get("url") or ""),
            "captured_at_ms": snapshot.get("capturedAtMs"),
            "projection_mode": snapshot.get("trackingProjectionMode")
            or _get_path(snapshot, ("menonTrackDebug", "projectionMode")),
            "authenticated_role": _get_path(snapshot, ("authState", "role")),
            "canonical_admission": admission_reason,
            "snapshot_id": _get_path(snapshot, ("canonicalWorldState", "snapshotId")),
        },
        "artifacts": {"browser_snapshot_json": evidence_path},
    }
    production_geometry = _get_path(snapshot, ("menonTrackDebug", "productionGeometry"))
    if isinstance(production_geometry, Mapping):
        trace["production_geometry"] = dict(production_geometry)
    if world_to_menon is not None:
        trace["world_to_menon_col_major"] = world_to_menon
    camera_reprojections = _trace_camera_reprojections_from_payload(
        snapshot,
        raw_snapshot_path=evidence_path,
        allowed_entity_ids=set(canonical_context.get("entity_ids", set())) if canonical_context else set(),
    )
    if camera_reprojections:
        trace["camera_reprojections"] = camera_reprojections
    if screenshot_path is not None:
        trace["screenshots"] = [
            {
                "id": "menon_browser_capture",
                "kind": "menon_browser_screenshot",
                "path": str(screenshot_path),
            }
        ]
    return trace
