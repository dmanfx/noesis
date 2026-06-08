from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


TRACK_CONTAINERS = (
    ("menonTrackDebug", "rawByCamera"),
    ("menonTrackDebug", "rawProjectedInfos"),
    ("menonTrackDebug", "fusedInfos"),
    ("rawActiveTracksByCamera",),
    ("__menonRawActiveTracksByCamera",),
    ("latestRawTrackInfos",),
    ("latestTrackInfos",),
    ("latestRawTrackingPaths",),
    ("latestTrackingPaths",),
)

TRANSFORM_KEYS = {
    "world_to_menon_col_major",
    "world_to_scene_col_major",
    "worldToMenonColMajor",
    "worldToSceneColMajor",
}

BACKEND_POINT_KEYS = (
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
    camera_id = _identity_value(item, ("camera_id", "cameraId", "camera"))
    stable_id = _identity_value(item, ("stable_id", "stableId"))
    tracker_id = _identity_value(item, ("tracker_id", "trackerId", "track_id", "trackId", "id"))
    path_key = _identity_value(item, ("pathKey", "trackKey", "path_key", "track_key"))
    keys: list[str] = []
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


def _timestamp_s(item: Mapping[str, Any], *, seconds_keys: Sequence[str], millis_keys: Sequence[str]) -> float | None:
    for key in seconds_keys:
        value = _finite_float(item.get(key))
        if value is not None:
            return value
    for key in millis_keys:
        value = _finite_float(item.get(key))
        if value is not None:
            return value / 1000.0
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


def _first_peer_value(index: Mapping[str, Sequence[dict[str, Any]]], keys: Sequence[str], field: str) -> Any:
    for key in keys:
        for peer in index.get(key, ()):
            value = peer.get(field)
            if value is not None:
                return value
    return None


def _trace_trails_from_payload(snapshot: Mapping[str, Any], placements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    explicit: list[dict[str, Any]] = []
    for item in _seq(snapshot.get("trails")):
        if isinstance(item, Mapping):
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

    world_to_menon = _find_world_to_menon(snapshot)
    records = _flatten_records(snapshot)
    candidates: list[dict[str, Any]] = []
    for record in records:
        backend, backend_source = _extract_backend_world(record)
        menon, menon_source = _extract_menon_point(record, backend_source=backend_source)
        row = dict(record)
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
        room = _identity_value(row, ("room", "zone", "expectedRoomZone", "roomZone"))
        entity_id = _entity_id(row)
        stable_id = _identity_value(row, ("stable_id", "stableId"))
        tracker_id = _identity_value(row, ("tracker_id", "trackerId", "track_id", "trackId", "id"))
        noesis_ts_s = _timestamp_s(
            row,
            seconds_keys=("noesis_ts_s", "noesisTsS", "frame_ts_s", "frameTsS", "timestamp_s", "timestampS", "ts_s", "tsS"),
            millis_keys=("noesis_ts_ms", "noesisTsMs", "frame_ts_ms", "frameTsMs"),
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
    trace: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(run_id or snapshot.get("run_id") or "menon_browser_trace"),
        "source": {
            "repo": "Noesis_Devel",
            "pipeline_config": "config/infer.yaml",
            "cameras_config": "config/cameras.yaml",
            "menon_available": True,
        },
        "scope": {"tiers": ["menon_trace", "menon_browser_capture"], "rooms": rooms, "cameras": cameras},
        "floor_y_scene": _find_floor_y(snapshot),
        "placements": placements,
        "trails": _trace_trails_from_payload(snapshot, placements),
        "browser": {
            "page_url": str(page_url or snapshot.get("href") or snapshot.get("url") or ""),
            "captured_at_ms": snapshot.get("capturedAtMs"),
            "projection_mode": snapshot.get("trackingProjectionMode")
            or _get_path(snapshot, ("menonTrackDebug", "projectionMode")),
        },
        "artifacts": {"browser_snapshot_json": evidence_path},
    }
    if world_to_menon is not None:
        trace["world_to_menon_col_major"] = world_to_menon
    camera_reprojections = _trace_camera_reprojections_from_payload(snapshot, raw_snapshot_path=evidence_path)
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
