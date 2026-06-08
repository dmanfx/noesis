from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .camera import CameraCalibration, ReprojectionAnchor, project_world_to_pixel


def _blank_image(width: int, height: int) -> np.ndarray:
    image = np.full((int(height), int(width), 3), 245, dtype=np.uint8)
    step = max(32, min(width, height) // 12)
    for x in range(0, width, step):
        image[:, x : x + 1] = (225, 225, 225)
    for y in range(0, height, step):
        image[y : y + 1, :] = (225, 225, 225)
    return image


def _load_image(path: str | Path | None, width: int, height: int) -> np.ndarray:
    if path:
        try:
            import cv2

            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                if image.shape[1] != width or image.shape[0] != height:
                    image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
                return image
        except Exception:
            pass
    return _blank_image(width, height)


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    try:
        import cv2

        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    except Exception:
        return


def _draw_circle(image: np.ndarray, point: tuple[float, float], color: tuple[int, int, int], radius: int = 5) -> None:
    try:
        import cv2

        cv2.circle(image, (int(round(point[0])), int(round(point[1]))), int(radius), color, -1, lineType=cv2.LINE_AA)
    except Exception:
        return


def _draw_line(image: np.ndarray, a: tuple[float, float], b: tuple[float, float], color: tuple[int, int, int], thickness: int = 1) -> None:
    try:
        import cv2

        cv2.line(
            image,
            (int(round(a[0])), int(round(a[1]))),
            (int(round(b[0])), int(round(b[1]))),
            color,
            int(thickness),
            lineType=cv2.LINE_AA,
        )
    except Exception:
        return


def _draw_rect(image: np.ndarray, bbox: Sequence[float], color: tuple[int, int, int], thickness: int = 2) -> None:
    try:
        import cv2

        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, int(thickness), lineType=cv2.LINE_AA)
    except Exception:
        _draw_line(image, (float(bbox[0]), float(bbox[1])), (float(bbox[2]), float(bbox[1])), color, thickness=thickness)
        _draw_line(image, (float(bbox[2]), float(bbox[1])), (float(bbox[2]), float(bbox[3])), color, thickness=thickness)
        _draw_line(image, (float(bbox[2]), float(bbox[3])), (float(bbox[0]), float(bbox[3])), color, thickness=thickness)
        _draw_line(image, (float(bbox[0]), float(bbox[3])), (float(bbox[0]), float(bbox[1])), color, thickness=thickness)


def _draw_polyline(image: np.ndarray, points: Sequence[Sequence[float]], color: tuple[int, int, int], thickness: int = 2) -> None:
    clean = [_point2(point) for point in points]
    clean = [point for point in clean if point is not None]
    if len(clean) < 2:
        return
    for left, right in zip(clean, clean[1:] + clean[:1]):
        _draw_line(image, (left[0], left[1]), (right[0], right[1]), color, thickness=thickness)


def _draw_ellipse(
    image: np.ndarray,
    center: tuple[float, float],
    axes: tuple[float, float],
    angle_deg: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    try:
        import cv2

        cv2.ellipse(
            image,
            (int(round(center[0])), int(round(center[1]))),
            (max(1, int(round(axes[0]))), max(1, int(round(axes[1])))),
            float(angle_deg),
            0.0,
            360.0,
            color,
            int(thickness),
            lineType=cv2.LINE_AA,
        )
    except Exception:
        # Polygon fallback keeps artifacts useful when OpenCV drawing is not available.
        steps = 48
        points = []
        theta = np.deg2rad(float(angle_deg))
        rot = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
        for t in np.linspace(0.0, 2.0 * np.pi, steps):
            local = np.asarray([np.cos(t) * axes[0], np.sin(t) * axes[1]], dtype=np.float64)
            px = rot @ local + np.asarray(center, dtype=np.float64)
            points.append((float(px[0]), float(px[1])))
        for left, right in zip(points, points[1:] + points[:1]):
            _draw_line(image, left, right, color, thickness=thickness)


def _point3(value: Any) -> list[float] | None:
    try:
        if isinstance(value, Mapping):
            value = value.get("point") or value.get("world") or value.get("world_point")
        point = [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def _point2(value: Any) -> list[float] | None:
    try:
        if isinstance(value, Mapping):
            value = value.get("point") or value.get("xz") or value.get("world_xz")
        point = [float(value[0]), float(value[1])]
    except Exception:
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def _bbox4(value: Any) -> list[float] | None:
    try:
        if isinstance(value, Mapping):
            value = value.get("bbox") or value.get("bbox_xyxy") or value.get("xyxy") or value.get("rect")
        bbox = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except Exception:
        return None
    if not np.all(np.isfinite(bbox)) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _label(value: Any, default: str = "") -> str:
    if isinstance(value, Mapping):
        for key in ("label", "entity_id", "stable_id", "id"):
            if value.get(key) is not None:
                return str(value.get(key))
    return default


def _mask_points(value: Any) -> list[list[float]]:
    if isinstance(value, Mapping):
        value = value.get("points") or value.get("polygon") or value.get("mask")
    return [point for point in (_point2(item) for item in _point_sequence(value)) if point is not None]


def _point_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _project(calibration: CameraCalibration, point: Sequence[float]) -> tuple[float, float] | None:
    try:
        u, v, z = project_world_to_pixel(calibration, point)
        if z <= 0.0:
            return None
        width, height = calibration.image_size
        pad = max(width, height) * 0.10
        if u < -pad or u > width + pad or v < -pad or v > height + pad:
            return None
        return u, v
    except Exception:
        return None


def render_camera_reprojection_overlay(
    calibration: CameraCalibration,
    anchors: Sequence[ReprojectionAnchor],
    output_path: str | Path,
    *,
    background_path: str | Path | None = None,
    room_outline_world: Sequence[Sequence[float]] = (),
    detected_footpoints_world: Sequence[Any] = (),
    mesh_edges_world: Sequence[Any] = (),
    image_bboxes_xyxy: Sequence[Any] = (),
    projected_bboxes_xyxy: Sequence[Any] = (),
    avatar_bboxes_xyxy: Sequence[Any] = (),
    mask_polygons_xy: Sequence[Any] = (),
    floor_grid_x: tuple[float, float, float] = (-3.0, 3.0, 1.0),
    floor_grid_z: tuple[float, float, float] = (0.0, 8.0, 1.0),
) -> Path:
    width, height = calibration.image_size
    image = _load_image(background_path, int(width), int(height))

    x_min, x_max, x_step = floor_grid_x
    z_min, z_max, z_step = floor_grid_z
    x_values = np.arange(float(x_min), float(x_max) + 1e-6, max(1e-6, float(x_step)))
    z_values = np.arange(float(z_min), float(z_max) + 1e-6, max(1e-6, float(z_step)))
    floor_y = float(calibration.floor_y)
    for x in x_values:
        prev = None
        for z in z_values:
            current = _project(calibration, [float(x), floor_y, float(z)])
            if prev is not None and current is not None:
                _draw_line(image, prev, current, (180, 180, 180), thickness=1)
            prev = current
    for z in z_values:
        prev = None
        for x in x_values:
            current = _project(calibration, [float(x), floor_y, float(z)])
            if prev is not None and current is not None:
                _draw_line(image, prev, current, (180, 180, 180), thickness=1)
            prev = current

    outline = [_point3(point) for point in room_outline_world]
    outline = [point for point in outline if point is not None]
    for left, right in zip(outline, outline[1:]):
        a = _project(calibration, left)
        b = _project(calibration, right)
        if a is not None and b is not None:
            _draw_line(image, a, b, (220, 120, 0), thickness=2)

    for raw_edge in mesh_edges_world:
        if isinstance(raw_edge, Mapping):
            left = _point3(raw_edge.get("a") or raw_edge.get("start"))
            right = _point3(raw_edge.get("b") or raw_edge.get("end"))
        elif isinstance(raw_edge, Sequence) and not isinstance(raw_edge, (str, bytes, bytearray)) and len(raw_edge) >= 2:
            left = _point3(raw_edge[0])
            right = _point3(raw_edge[1])
        else:
            left = right = None
        if left is None or right is None:
            continue
        a = _project(calibration, left)
        b = _project(calibration, right)
        if a is not None and b is not None:
            _draw_line(image, a, b, (120, 120, 220), thickness=1)

    for idx, raw_point in enumerate(detected_footpoints_world):
        label = None
        if isinstance(raw_point, Mapping):
            label = raw_point.get("id") or raw_point.get("entity_id") or raw_point.get("stable_id")
        point = _point3(raw_point)
        if point is None:
            continue
        projected = _project(calibration, point)
        if projected is None:
            continue
        _draw_circle(image, projected, (255, 0, 255), radius=5)
        _draw_text(image, str(label or idx), (int(projected[0]) + 6, int(projected[1]) + 14), (150, 0, 150))

    for idx, raw_bbox in enumerate(image_bboxes_xyxy):
        bbox = _bbox4(raw_bbox)
        if bbox is None:
            continue
        _draw_rect(image, bbox, (0, 160, 0), thickness=2)
        _draw_text(image, f"det:{_label(raw_bbox, str(idx))}", (int(bbox[0]), max(12, int(bbox[1]) - 6)), (0, 120, 0))

    for idx, raw_bbox in enumerate(projected_bboxes_xyxy):
        bbox = _bbox4(raw_bbox)
        if bbox is None:
            continue
        _draw_rect(image, bbox, (0, 0, 220), thickness=2)
        _draw_text(image, f"proj:{_label(raw_bbox, str(idx))}", (int(bbox[0]), min(int(height) - 8, int(bbox[3]) + 16)), (0, 0, 160))

    for idx, raw_bbox in enumerate(avatar_bboxes_xyxy):
        bbox = _bbox4(raw_bbox)
        if bbox is None:
            continue
        _draw_rect(image, bbox, (220, 80, 0), thickness=1)
        _draw_text(image, f"avatar:{_label(raw_bbox, str(idx))}", (int(bbox[0]), min(int(height) - 8, int(bbox[3]) + 32)), (160, 60, 0))

    for raw_mask in mask_polygons_xy:
        points = _mask_points(raw_mask)
        if points:
            _draw_polyline(image, points, (180, 0, 180), thickness=2)

    for anchor in anchors:
        projected = _project(calibration, anchor.world_point)
        try:
            expected = (float(anchor.expected_pixel[0]), float(anchor.expected_pixel[1]))
        except Exception:
            expected = None
        if expected is not None:
            _draw_circle(image, expected, (0, 180, 0), radius=5)
        if projected is not None:
            _draw_circle(image, projected, (0, 0, 220), radius=4)
            _draw_text(image, anchor.anchor_id, (int(projected[0]) + 6, int(projected[1]) - 6), (0, 0, 120))
        if expected is not None and projected is not None:
            _draw_line(image, expected, projected, (0, 160, 255), thickness=2)

    _draw_text(image, f"camera={calibration.camera_id}", (12, 22), (40, 40, 40))
    _draw_text(image, "green=expected/detected red=projected magenta=foot/mask", (12, 44), (40, 40, 40))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        ok = cv2.imwrite(str(target), image)
        if not ok:
            raise RuntimeError("cv2.imwrite returned false")
    except Exception:
        from PIL import Image

        Image.fromarray(image[:, :, ::-1]).save(target)
    return target


def _bounds_from_points(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    if not points:
        return -1.0, 1.0, -1.0, 1.0
    arr = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    x_min = float(np.min(arr[:, 0]))
    x_max = float(np.max(arr[:, 0]))
    z_min = float(np.min(arr[:, 1]))
    z_max = float(np.max(arr[:, 1]))
    span = max(x_max - x_min, z_max - z_min, 1.0)
    margin = span * 0.12
    return x_min - margin, x_max + margin, z_min - margin, z_max + margin


def render_bev_diagnostic_overlay(
    output_path: str | Path,
    *,
    room_polygons_xz: Sequence[Sequence[Sequence[float]]] = (),
    wall_segments_xz: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
    doorway_segments_xz: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
    camera_frustums_xz: Sequence[Sequence[Sequence[float]]] = (),
    track_paths_xz: Sequence[Sequence[Sequence[float]]] = (),
    raw_footpoints_xz: Sequence[Any] = (),
    confidence_ellipses_xz: Sequence[Mapping[str, Any]] = (),
    track_annotations_xz: Sequence[Mapping[str, Any]] = (),
    width: int = 900,
    height: int = 700,
) -> Path:
    image = np.full((int(height), int(width), 3), 250, dtype=np.uint8)
    all_points: list[list[float]] = []
    for polygon in room_polygons_xz:
        all_points.extend(point for point in (_point2(p) for p in polygon) if point is not None)
    for a, b in wall_segments_xz:
        pa = _point2(a)
        pb = _point2(b)
        if pa is not None:
            all_points.append(pa)
        if pb is not None:
            all_points.append(pb)
    for polygon in camera_frustums_xz:
        all_points.extend(point for point in (_point2(p) for p in polygon) if point is not None)
    for path in track_paths_xz:
        all_points.extend(point for point in (_point2(p) for p in path) if point is not None)
    for raw in raw_footpoints_xz:
        point = _point2(raw)
        if point is not None:
            all_points.append(point)
    for raw in confidence_ellipses_xz:
        point = _point2(raw.get("center") or raw.get("point") or raw.get("xz"))
        if point is None:
            continue
        radius_x = float(raw.get("radius_x_m") or raw.get("radius_m") or 0.0)
        radius_z = float(raw.get("radius_z_m") or raw.get("radius_m") or 0.0)
        all_points.extend([[point[0] - radius_x, point[1] - radius_z], [point[0] + radius_x, point[1] + radius_z]])
    for raw in track_annotations_xz:
        point = _point2(raw.get("point") or raw.get("xz") or raw.get("center"))
        if point is not None:
            all_points.append(point)

    x_min, x_max, z_min, z_max = _bounds_from_points(all_points)
    pad = 42

    def to_px(point: Sequence[float]) -> tuple[float, float]:
        x = float(point[0])
        z = float(point[1])
        u = pad + (x - x_min) / max(1e-9, x_max - x_min) * (width - 2 * pad)
        v = height - pad - (z - z_min) / max(1e-9, z_max - z_min) * (height - 2 * pad)
        return float(u), float(v)

    for x in np.linspace(x_min, x_max, 9):
        _draw_line(image, to_px([x, z_min]), to_px([x, z_max]), (235, 235, 235), thickness=1)
    for z in np.linspace(z_min, z_max, 7):
        _draw_line(image, to_px([x_min, z]), to_px([x_max, z]), (235, 235, 235), thickness=1)

    for polygon in room_polygons_xz:
        points = [point for point in (_point2(p) for p in polygon) if point is not None]
        if len(points) < 2:
            continue
        closed = points + [points[0]]
        for left, right in zip(closed, closed[1:]):
            _draw_line(image, to_px(left), to_px(right), (70, 70, 70), thickness=2)

    for wall_a, wall_b in wall_segments_xz:
        a = _point2(wall_a)
        b = _point2(wall_b)
        if a is not None and b is not None:
            _draw_line(image, to_px(a), to_px(b), (20, 20, 20), thickness=3)

    for door_a, door_b in doorway_segments_xz:
        a = _point2(door_a)
        b = _point2(door_b)
        if a is not None and b is not None:
            _draw_line(image, to_px(a), to_px(b), (0, 170, 0), thickness=5)

    for frustum in camera_frustums_xz:
        points = [point for point in (_point2(p) for p in frustum) if point is not None]
        for left, right in zip(points, points[1:]):
            _draw_line(image, to_px(left), to_px(right), (230, 150, 0), thickness=2)
        if points:
            _draw_circle(image, to_px(points[0]), (230, 150, 0), radius=5)

    for path_idx, path in enumerate(track_paths_xz):
        points = [point for point in (_point2(p) for p in path) if point is not None]
        for left, right in zip(points, points[1:]):
            _draw_line(image, to_px(left), to_px(right), (0, 80, 220), thickness=2)
        for point in points:
            _draw_circle(image, to_px(point), (0, 80, 220), radius=4)
        if points:
            _draw_text(image, f"track {path_idx}", (int(to_px(points[-1])[0]) + 6, int(to_px(points[-1])[1]) - 6), (0, 60, 160))

    for idx, raw in enumerate(raw_footpoints_xz):
        label = None
        if isinstance(raw, Mapping):
            label = raw.get("id") or raw.get("entity_id") or raw.get("stable_id")
        point = _point2(raw)
        if point is None:
            continue
        px = to_px(point)
        _draw_circle(image, px, (255, 0, 255), radius=5)
        _draw_text(image, str(label or idx), (int(px[0]) + 6, int(px[1]) + 14), (150, 0, 150))

    for raw in confidence_ellipses_xz:
        center = _point2(raw.get("center") or raw.get("point") or raw.get("xz"))
        if center is None:
            continue
        radius_x_m = float(raw.get("radius_x_m") or raw.get("radius_m") or 0.15)
        radius_z_m = float(raw.get("radius_z_m") or raw.get("radius_m") or 0.15)
        center_px = to_px(center)
        x_axis_px = abs(to_px([center[0] + radius_x_m, center[1]])[0] - center_px[0])
        z_axis_px = abs(to_px([center[0], center[1] + radius_z_m])[1] - center_px[1])
        _draw_ellipse(image, center_px, (x_axis_px, z_axis_px), float(raw.get("heading_deg") or 0.0), (180, 80, 0), thickness=2)

    for idx, raw in enumerate(track_annotations_xz):
        point = _point2(raw.get("point") or raw.get("xz") or raw.get("center"))
        if point is None:
            continue
        stable_id = raw.get("stable_id") if raw.get("stable_id") is not None else raw.get("stableId")
        reid = raw.get("reid_confidence") if raw.get("reid_confidence") is not None else raw.get("reidConfidence")
        projection = raw.get("projection_confidence") if raw.get("projection_confidence") is not None else raw.get("projectionConfidence")
        parts = [f"SID {stable_id}" if stable_id is not None else f"track {idx}"]
        if projection is not None:
            parts.append(f"P {float(projection):.2f}")
        if reid is not None:
            parts.append(f"R {float(reid):.2f}")
        px = to_px(point)
        _draw_circle(image, px, (40, 40, 40), radius=3)
        _draw_text(image, " ".join(parts), (int(px[0]) + 8, int(px[1]) - 10), (20, 20, 20))

    _draw_text(image, "BEV diagnostic overlay", (12, 24), (40, 40, 40))
    _draw_text(image, "black=walls green=doorways orange=frustums blue=tracks magenta=raw brown=uncertainty", (12, 46), (40, 40, 40))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        ok = cv2.imwrite(str(target), image)
        if not ok:
            raise RuntimeError("cv2.imwrite returned false")
    except Exception:
        from PIL import Image

        Image.fromarray(image[:, :, ::-1]).save(target)
    return target
