from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import SourceMetadata, ValidationCheck, ValidationReport
from .menon import (
    MenonAvatarSample,
    MenonBevTrailSample,
    MenonCameraReprojectionSample,
    MenonLatencySample,
    MenonObjectSample,
    MenonPlacementSample,
    MenonTrailSample,
    MenonTransformAuditSample,
    WorldBevSample,
    validate_bev_menon_trail_agreement,
    validate_menon_avatar_collision,
    validate_menon_avatar_orientation,
    validate_menon_avatar_scale,
    validate_menon_camera_reprojection,
    validate_menon_floor_contact,
    validate_menon_latency_alignment,
    validate_menon_object_placement,
    validate_menon_trail_agreement,
    validate_timestamp_alignment,
    validate_transform_audit,
    validate_world_bev_round_trip,
    validate_world_menon_placement_agreement,
    validate_world_menon_round_trip,
)


def load_menon_trace(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Menon trace must be a JSON object: {path}")
    return payload


def _seq(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else []


def _first_present(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item.get(key) is not None:
            return item.get(key)
    return None


def _string_seq(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in _seq(value)]


def _first_metric(item: Mapping[str, Any], metrics: Mapping[str, Any], *keys: str) -> Any:
    value = _first_present(item, *keys)
    if value is not None:
        return value
    return _first_present(metrics, *keys)


def _source_from_trace(payload: Mapping[str, Any], *, menon_root: str | None = None) -> SourceMetadata:
    source = payload.get("source") if isinstance(payload.get("source"), Mapping) else {}
    return SourceMetadata(
        repo="Noesis_Devel",
        git_revision=source.get("git_revision"),
        pipeline_config=source.get("pipeline_config"),
        cameras_config=source.get("cameras_config"),
        menon_available=source.get("menon_available") if source.get("menon_available") is not None else (bool(menon_root) if menon_root is not None else None),
        menon_revision=source.get("menon_revision"),
    )


def _world_to_menon(payload: Mapping[str, Any]) -> Sequence[float] | None:
    direct = payload.get("world_to_menon_col_major") or payload.get("world_to_scene_col_major")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes, bytearray)) and len(direct) == 16:
        return direct
    align = payload.get("align") if isinstance(payload.get("align"), Mapping) else {}
    similarity = align.get("scene_similarity") if isinstance(align.get("scene_similarity"), Mapping) else {}
    candidate = similarity.get("world_to_scene_col_major")
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)) and len(candidate) == 16:
        return candidate
    return None


def _world_to_bev(payload: Mapping[str, Any]) -> Sequence[float] | None:
    direct = payload.get("world_to_bev_col_major")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes, bytearray)) and len(direct) == 16:
        return direct
    return None


def parse_menon_placements(payload: Mapping[str, Any]) -> list[MenonPlacementSample]:
    rows: list[MenonPlacementSample] = []
    for item in _seq(payload.get("placements")):
        if not isinstance(item, Mapping):
            continue
        rows.append(
            MenonPlacementSample(
                entity_id=str(item.get("entity_id") or item.get("id") or item.get("stable_id") or item.get("stableId") or "entity"),
                world_point=item.get("world_point") or item.get("noesis_world") or item.get("world") or [],
                menon_point=item.get("menon_point") or item.get("menon_scene") or item.get("scene") or [],
                camera_id=item.get("camera_id") or item.get("cameraId") or item.get("camera"),
                room=item.get("room") or item.get("zone"),
                evidence=_seq(item.get("evidence")),
            )
        )
    return rows


def parse_world_bev_samples(payload: Mapping[str, Any]) -> list[WorldBevSample]:
    rows: list[WorldBevSample] = []
    for item in [*_seq(payload.get("world_bev_points")), *_seq(payload.get("world_bev_samples"))]:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            WorldBevSample(
                entity_id=str(item.get("entity_id") or item.get("id") or item.get("stable_id") or item.get("stableId") or "entity"),
                world_point=item.get("world_point") or item.get("noesis_world") or item.get("world") or [],
                bev_point_xz=item.get("bev_point_xz") or item.get("bev_xz") or item.get("bev") or [],
                bev_y=float(item.get("bev_y") or 0.0),
                camera_id=item.get("camera_id") or item.get("cameraId") or item.get("camera"),
                room=item.get("room") or item.get("zone"),
                evidence=_seq(item.get("evidence")),
            )
        )
    return rows


def parse_menon_trails(payload: Mapping[str, Any]) -> list[MenonTrailSample]:
    rows: list[MenonTrailSample] = []
    for item in _seq(payload.get("trails")):
        if not isinstance(item, Mapping):
            continue
        rows.append(
            MenonTrailSample(
                entity_id=str(item.get("entity_id") or item.get("id") or item.get("stable_id") or item.get("stableId") or "entity"),
                noesis_world_points=item.get("noesis_world_points") or item.get("world_points") or [],
                menon_points=item.get("menon_points") or item.get("menon_scene_points") or item.get("scene_points") or [],
                camera_id=item.get("camera_id") or item.get("cameraId") or item.get("camera"),
                room=item.get("room") or item.get("zone"),
                evidence=_seq(item.get("evidence")),
            )
        )
    return rows


def parse_bev_menon_trails(payload: Mapping[str, Any]) -> list[MenonBevTrailSample]:
    rows: list[MenonBevTrailSample] = []
    for item in [*_seq(payload.get("bev_trails")), *_seq(payload.get("bev_menon_trails"))]:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            MenonBevTrailSample(
                entity_id=str(item.get("entity_id") or item.get("id") or item.get("stable_id") or item.get("stableId") or "entity"),
                bev_points_xz=item.get("bev_points_xz") or item.get("bev_points") or item.get("bev_path_xz") or [],
                menon_points=item.get("menon_points") or item.get("menon_scene_points") or item.get("scene_points") or [],
                floor_y_world=float(item.get("floor_y_world") or item.get("world_floor_y") or 0.0),
                camera_id=item.get("camera_id") or item.get("cameraId") or item.get("camera"),
                room=item.get("room") or item.get("zone"),
                evidence=_seq(item.get("evidence")),
            )
        )
    return rows


def parse_menon_avatars(payload: Mapping[str, Any]) -> list[MenonAvatarSample]:
    rows: list[MenonAvatarSample] = []
    for item in [*_seq(payload.get("avatars")), *_seq(payload.get("avatar_samples"))]:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            MenonAvatarSample(
                entity_id=str(item.get("entity_id") or item.get("id") or item.get("stable_id") or item.get("stableId") or "entity"),
                menon_point=item.get("menon_point") or item.get("menon_scene") or item.get("scene") or item.get("position") or [],
                radius_scene=_first_present(item, "radius_scene", "radius", "collision_radius_scene"),
                expected_height_m=_first_present(item, "expected_height_m", "person_height_m"),
                observed_height_scene=_first_present(item, "observed_height_scene", "avatar_height_scene", "height_scene"),
                scene_units_per_m=float(item.get("scene_units_per_m") or 1.0),
                heading_deg=_first_present(item, "heading_deg", "yaw_deg", "orientation_deg"),
                movement_vector_xz=_first_present(item, "movement_vector_xz", "velocity_xz", "trail_delta_xz"),
                allowed_collision=bool(item.get("allowed_collision", False)),
                camera_id=item.get("camera_id") or item.get("cameraId") or item.get("camera"),
                room=item.get("room") or item.get("zone"),
                evidence=_seq(item.get("evidence")),
            )
        )
    return rows


def parse_menon_objects(payload: Mapping[str, Any]) -> list[MenonObjectSample]:
    rows: list[MenonObjectSample] = []
    for item in [*_seq(payload.get("objects")), *_seq(payload.get("object_samples"))]:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            MenonObjectSample(
                object_id=str(item.get("object_id") or item.get("id") or "object"),
                category=str(item.get("category") or item.get("class") or "object"),
                world_point=_first_present(item, "world_point", "world", "noesis_world"),
                menon_point=item.get("menon_point") or item.get("menon_scene") or item.get("scene") or item.get("position") or [],
                bbox_min_scene=_first_present(item, "bbox_min_scene", "bboxMinScene", "bbox_min", "bboxMin"),
                bbox_max_scene=_first_present(item, "bbox_max_scene", "bboxMaxScene", "bbox_max", "bboxMax"),
                expected_dimensions_m=_first_present(item, "expected_dimensions_m", "expectedDimensionsM", "expected_dims_m", "expectedDimsM"),
                observed_dimensions_scene=_first_present(item, "observed_dimensions_scene", "observedDimensionsScene", "dims_scene", "dimsScene"),
                scene_units_per_m=float(item.get("scene_units_per_m") or 1.0),
                support_y_scene=float(item.get("support_y_scene") or item.get("supportYScene") or 0.0),
                allowed_rooms=_string_seq(_first_present(item, "allowed_rooms", "allowedRooms", "compatible_rooms", "compatibleRooms")),
                allowed_collision=bool(item.get("allowed_collision", False)),
                room=item.get("room") or item.get("zone"),
                camera_id=item.get("camera_id") or item.get("cameraId") or item.get("camera"),
                evidence=_seq(item.get("evidence")),
            )
        )
    return rows


def parse_latency_samples(payload: Mapping[str, Any]) -> list[MenonLatencySample]:
    rows: list[MenonLatencySample] = []
    for item in [*_seq(payload.get("latency_samples")), *_seq(payload.get("latency"))]:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            MenonLatencySample(
                entity_id=str(item.get("entity_id") or item.get("id") or item.get("stable_id") or item.get("stableId") or "entity"),
                noesis_ts_s=float(_first_present(item, "noesis_ts_s", "frame_ts_s", "source_ts_s") or 0.0),
                telemetry_ts_s=_first_present(item, "telemetry_ts_s", "telemetryTsS"),
                menon_update_ts_s=_first_present(item, "menon_update_ts_s", "update_ts_s", "menonUpdateTsS"),
                menon_render_ts_s=_first_present(item, "menon_render_ts_s", "render_ts_s", "menonRenderTsS"),
                menon_display_ts_s=_first_present(item, "menon_display_ts_s", "display_ts_s", "menonDisplayTsS"),
                camera_id=item.get("camera_id") or item.get("cameraId") or item.get("camera"),
                room=item.get("room") or item.get("zone"),
                evidence=_seq(item.get("evidence")),
            )
        )
    return rows


def _wall_segments(raw_segments: Any) -> list[tuple[Sequence[float], Sequence[float]]]:
    segments: list[tuple[Sequence[float], Sequence[float]]] = []
    for item in _seq(raw_segments):
        if isinstance(item, Mapping):
            a = item.get("a") or item.get("start")
            b = item.get("b") or item.get("end")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) == 2:
            a, b = item[0], item[1]
        else:
            continue
        if isinstance(a, Sequence) and isinstance(b, Sequence) and len(a) >= 2 and len(b) >= 2:
            segments.append((a, b))
    return segments


def parse_transform_audits(payload: Mapping[str, Any]) -> list[MenonTransformAuditSample]:
    rows: list[MenonTransformAuditSample] = []
    for item in _seq(payload.get("placements")):
        if not isinstance(item, Mapping):
            continue
        stages = item.get("transform_audit") or item.get("transformAudit") or item.get("audit")
        if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes, bytearray)):
            continue
        rows.append(
            MenonTransformAuditSample(
                entity_id=str(item.get("entity_id") or item.get("id") or item.get("stable_id") or item.get("stableId") or "entity"),
                stages=[stage for stage in stages if isinstance(stage, Mapping)],
                camera_id=item.get("camera_id") or item.get("cameraId") or item.get("camera"),
                room=item.get("room") or item.get("zone"),
            )
        )
    for item in _seq(payload.get("transform_audits")):
        if not isinstance(item, Mapping):
            continue
        stages = item.get("stages") or item.get("transform_audit")
        rows.append(
            MenonTransformAuditSample(
                entity_id=str(item.get("entity_id") or item.get("id") or "entity"),
                stages=[stage for stage in _seq(stages) if isinstance(stage, Mapping)],
                camera_id=item.get("camera_id") or item.get("cameraId") or item.get("camera"),
                room=item.get("room") or item.get("zone"),
            )
        )
    return rows


def parse_camera_reprojections(payload: Mapping[str, Any]) -> list[MenonCameraReprojectionSample]:
    rows: list[MenonCameraReprojectionSample] = []
    for item in [*_seq(payload.get("camera_reprojections")), *_seq(payload.get("camera_reprojection_samples"))]:
        if not isinstance(item, Mapping):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}
        rows.append(
            MenonCameraReprojectionSample(
                camera_id=str(item.get("camera_id") or item.get("cameraId") or item.get("camera") or "camera"),
                source_frame=item.get("source_frame") or item.get("sourceFrame") or item.get("source_frame_path"),
                menon_render=item.get("menon_render") or item.get("menonRender") or item.get("render_path"),
                overlay_path=item.get("overlay_path") or item.get("overlayPath"),
                layers=_seq(item.get("layers")),
                mean_error_px=_first_metric(item, metrics, "mean_error_px", "meanErrorPx"),
                max_error_px=_first_metric(item, metrics, "max_error_px", "maxErrorPx"),
                anchor_mean_error_px=_first_metric(item, metrics, "anchor_mean_error_px", "anchorMeanErrorPx"),
                anchor_max_error_px=_first_metric(item, metrics, "anchor_max_error_px", "anchorMaxErrorPx"),
                floor_grid_mean_error_px=_first_metric(item, metrics, "floor_grid_mean_error_px", "floorGridMeanErrorPx"),
                room_edge_mean_error_px=_first_metric(item, metrics, "room_edge_mean_error_px", "roomEdgeMeanErrorPx"),
                bbox_iou=_first_metric(item, metrics, "bbox_iou", "bboxIoU"),
                avatar_iou=_first_metric(item, metrics, "avatar_iou", "avatarIoU"),
                mask_iou=_first_metric(item, metrics, "mask_iou", "maskIoU"),
                entity_id=item.get("entity_id") or item.get("id") or item.get("stable_id") or item.get("stableId"),
                room=item.get("room") or item.get("zone"),
                evidence=_seq(item.get("evidence")),
            )
        )
    return rows


def _timestamps(payload: Mapping[str, Any], key: str) -> list[float]:
    out: list[float] = []
    for value in _seq(payload.get(key)):
        try:
            out.append(float(value))
        except Exception:
            continue
    if out:
        return out
    for item in _seq(payload.get("placements")):
        if not isinstance(item, Mapping):
            continue
        raw = item.get(key)
        if raw is None:
            continue
        try:
            out.append(float(raw))
        except Exception:
            continue
    return out


def build_menon_trace_report(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    menon_root: str | None = None,
    require_menon_root: bool = False,
) -> ValidationReport:
    world_to_menon = _world_to_menon(payload)
    world_to_bev = _world_to_bev(payload)
    placements = parse_menon_placements(payload)
    world_bev_samples = parse_world_bev_samples(payload)
    trails = parse_menon_trails(payload)
    bev_trails = parse_bev_menon_trails(payload)
    avatars = parse_menon_avatars(payload)
    objects = parse_menon_objects(payload)
    latency_samples = parse_latency_samples(payload)
    audits = parse_transform_audits(payload)
    camera_reprojections = parse_camera_reprojections(payload)
    report = ValidationReport(
        run_id=run_id,
        source=_source_from_trace(payload, menon_root=menon_root),
        scope=payload.get("scope") if isinstance(payload.get("scope"), Mapping) else {"tiers": ["menon_trace"]},
    )
    if require_menon_root and not menon_root:
        report.add_check(
            ValidationCheck(
                id="MENON.checkout_available",
                domain="menon",
                name="menon_checkout_available",
                status="blocked",
                failure_type="infrastructure_failure",
                detail="Menon root was required but not provided.",
                suggested_next_diagnostic="Pass --menon-root or set MENON_ROOT.",
            )
        )
    if world_to_menon is None:
        report.add_check(
            ValidationCheck(
                id="MENON.world_transform_present",
                domain="menon",
                name="world_to_menon_transform_present",
                status="fail",
                failure_type="transform_failure",
                detail="Trace does not include world_to_menon_col_major or align.scene_similarity.world_to_scene_col_major.",
                suggested_next_diagnostic="Export scene_similarity.world_to_scene_col_major from the calibration bundle or Menon debug state.",
            )
        )
        return report
    report.add_check(validate_world_menon_round_trip(world_to_menon, placements))
    report.add_check(validate_world_menon_placement_agreement(world_to_menon, placements))
    if world_to_bev is not None or world_bev_samples:
        report.add_check(validate_world_bev_round_trip(world_to_bev or [], world_bev_samples))
    report.add_check(
        validate_menon_floor_contact(
            placements,
            floor_y_scene=float(payload.get("floor_y_scene") or payload.get("menon_floor_y") or 0.0),
        )
    )
    report.add_check(validate_menon_trail_agreement(world_to_menon, trails))
    if bev_trails or "bev_trails" in payload or "bev_menon_trails" in payload:
        report.add_check(validate_bev_menon_trail_agreement(world_to_menon, bev_trails))
    if avatars or "avatars" in payload or "avatar_samples" in payload:
        collision = payload.get("collision") if isinstance(payload.get("collision"), Mapping) else {}
        wall_segments = _wall_segments(collision.get("wall_segments_xz") or payload.get("wall_segments_xz"))
        obstacle_boxes = _seq(collision.get("obstacle_boxes_xz") or payload.get("obstacle_boxes_xz"))
        report.add_check(validate_menon_avatar_scale(avatars))
        report.add_check(validate_menon_avatar_collision(avatars, wall_segments, obstacle_boxes))
        report.add_check(validate_menon_avatar_orientation(avatars))
    if objects or "objects" in payload or "object_samples" in payload:
        collision = payload.get("collision") if isinstance(payload.get("collision"), Mapping) else {}
        wall_segments = _wall_segments(collision.get("wall_segments_xz") or payload.get("wall_segments_xz"))
        obstacle_boxes = _seq(collision.get("object_obstacle_boxes_xz") or collision.get("obstacle_boxes_xz") or payload.get("obstacle_boxes_xz"))
        report.add_check(validate_menon_object_placement(world_to_menon, objects, wall_segments, obstacle_boxes))
    if camera_reprojections or "camera_reprojections" in payload or "camera_reprojection_samples" in payload:
        report.add_check(validate_menon_camera_reprojection(camera_reprojections))
    if latency_samples or "latency_samples" in payload or "latency" in payload:
        report.add_check(validate_menon_latency_alignment(latency_samples))
    report.add_check(validate_transform_audit(audits))
    report.add_check(validate_timestamp_alignment(_timestamps(payload, "noesis_ts_s"), _timestamps(payload, "menon_ts_s")))
    return report
