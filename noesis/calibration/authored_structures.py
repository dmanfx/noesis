"""Extract room-local structural evidence from an authored OBJ scene.

The extractor is deliberately geometry-only.  The caller supplies the exact
authored floor group(s) selected by the existing semantic room map; this module
does not infer room labels, use camera/person positions, or apply a scene scale.
All returned coordinates and offsets remain in the OBJ's source scene units.
An optional initial scale is used only to turn metric tolerances into scene-unit
tolerances.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from noesis.validation.authored_scene import AuthoredSceneGeometry, ObjTriangle


Point2 = tuple[float, float]
Interval = tuple[float, float]


@dataclass(frozen=True)
class AuthoredFloorBoundary:
    """One directed boundary loop of a selected authored floor surface."""

    points_xz: tuple[Point2, ...]
    heights_y: tuple[float, ...]
    signed_area_scene2: float
    is_hole: bool


@dataclass(frozen=True)
class AuthoredFloorPolygon:
    """A connected, upward-facing floor patch from the selected OBJ groups."""

    floor_groups: tuple[str, ...]
    boundaries: tuple[AuthoredFloorBoundary, ...]
    floor_height: float
    height_range: Interval
    area_scene2: float
    source_triangles: tuple[int, ...]


@dataclass(frozen=True)
class AuthoredOpening:
    """A geometric void between supported vertical face intervals.

    ``interval`` is measured along the canonical horizontal tangent of the
    source plane. ``start_xz`` and ``end_xz`` are the corresponding source
    scene-plane endpoints. This describes missing mesh coverage, not a
    semantic doorway/window classification.
    """

    opening_id: str
    plane_id: str
    interval: Interval
    start_xz: Point2
    end_xz: Point2
    vertical_span: Interval
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class AuthoredWallSegment:
    """A contiguous supported interval on an authored near-vertical plane.

    These fields intentionally match the structural-registration input shape.
    They are source-scene coordinates; the initial scale is never applied.
    """

    segment_id: str
    start_xz: Point2
    end_xz: Point2
    normal_xz: Point2
    offset: float
    support_count: int
    vertical_span: Interval
    source_refs: tuple[str, ...]
    plane_id: str
    floor_groups: tuple[str, ...]
    floor_boundary_distance_scene: float


@dataclass(frozen=True)
class AuthoredStructureExtraction:
    source_sha256: str
    floor_groups: tuple[str, ...]
    floor_polygons: tuple[AuthoredFloorPolygon, ...]
    wall_segments: tuple[AuthoredWallSegment, ...]
    openings: tuple[AuthoredOpening, ...]
    tolerance_scene_units: float
    initial_scene_units_per_meter: float | None


@dataclass
class _FloorComponent:
    triangles: list[ObjTriangle]
    groups: set[str]


@dataclass
class _PlaneCluster:
    normal: Point2
    offset: float
    triangles: list[ObjTriangle]


def _finite_positive(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _point_key(point: Sequence[float], *, decimals: int = 7) -> tuple[float, ...]:
    return tuple(round(float(value), decimals) for value in point)


def _triangle_points(geometry: AuthoredSceneGeometry, triangle: ObjTriangle) -> tuple[tuple[float, float, float], ...]:
    return tuple(geometry.vertices[index] for index in triangle.vertex_indices)


def _floor_components(
    geometry: AuthoredSceneGeometry,
    triangles: Sequence[ObjTriangle],
    requested_groups: set[str],
) -> list[_FloorComponent]:
    edge_owners: dict[tuple[tuple[float, ...], tuple[float, ...]], list[int]] = defaultdict(list)
    for position, triangle in enumerate(triangles):
        points = _triangle_points(geometry, triangle)
        keys = [_point_key(point) for point in points]
        for index in range(3):
            edge = tuple(sorted((keys[index], keys[(index + 1) % 3])))
            edge_owners[edge].append(position)

    adjacency: list[set[int]] = [set() for _ in triangles]
    for owners in edge_owners.values():
        for first in owners:
            adjacency[first].update(other for other in owners if other != first)

    seen: set[int] = set()
    components: list[_FloorComponent] = []
    for start in range(len(triangles)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members: list[ObjTriangle] = []
        groups: set[str] = set()
        while stack:
            current = stack.pop()
            triangle = triangles[current]
            members.append(triangle)
            groups.update(group for group in triangle.groups if group in requested_groups)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(_FloorComponent(members, groups))
    return components


def _floor_boundaries(
    geometry: AuthoredSceneGeometry,
    triangles: Sequence[ObjTriangle],
) -> tuple[AuthoredFloorBoundary, ...]:
    # Up-facing OBJ triangles have a clockwise XZ projection. Cancel directed
    # shared edges so separate selected groups that tile one room form a union.
    directed: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = defaultdict(int)
    point_by_key: dict[tuple[float, ...], tuple[float, float, float]] = {}
    for triangle in triangles:
        points = _triangle_points(geometry, triangle)
        keys = [_point_key((point[0], point[2])) for point in points]
        for point, key in zip(points, keys):
            point_by_key[key] = point
        for index in range(3):
            edge = (keys[index], keys[(index + 1) % 3])
            reverse = (edge[1], edge[0])
            if directed[reverse] > 0:
                directed[reverse] -= 1
            else:
                directed[edge] += 1

    remaining = {edge for edge, count in directed.items() if count > 0}
    outgoing: dict[tuple[float, ...], list[tuple[float, ...]]] = defaultdict(list)
    for start, end in remaining:
        outgoing[start].append(end)
    for ends in outgoing.values():
        ends.sort()

    loops: list[AuthoredFloorBoundary] = []
    while remaining:
        first = min(remaining)
        start, current = first
        path = [start]
        remaining.remove(first)
        for _ in range(len(directed) + 1):
            if current == start:
                break
            path.append(current)
            choices = [end for end in outgoing.get(current, ()) if (current, end) in remaining]
            if not choices:
                path = []
                break
            # For manifold floor boundaries there is one choice. Sorting makes
            # malformed branches deterministic without inventing a closure.
            following = choices[0]
            remaining.remove((current, following))
            current = following
        if len(path) < 3 or current != start:
            continue
        points = tuple((float(point_by_key[key][0]), float(point_by_key[key][2])) for key in path)
        heights = tuple(float(point_by_key[key][1]) for key in path)
        signed_area = 0.5 * sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        # The outer loop of a +Y surface is clockwise in XZ (negative area);
        # holes run in the opposite direction when triangle winding is sound.
        loops.append(
            AuthoredFloorBoundary(
                points_xz=points,
                heights_y=heights,
                signed_area_scene2=float(signed_area),
                is_hole=signed_area > 0.0,
            )
        )
    return tuple(sorted(loops, key=lambda loop: (loop.is_hole, -abs(loop.signed_area_scene2))))


def _floor_polygon(
    geometry: AuthoredSceneGeometry,
    component: _FloorComponent,
) -> AuthoredFloorPolygon:
    weighted_height = 0.0
    area_xz = 0.0
    all_y: list[float] = []
    for triangle in component.triangles:
        points = _triangle_points(geometry, triangle)
        signed_twice_area = (
            (points[1][0] - points[0][0]) * (points[2][2] - points[0][2])
            - (points[1][2] - points[0][2]) * (points[2][0] - points[0][0])
        )
        weight = abs(signed_twice_area) * 0.5
        area_xz += weight
        weighted_height += weight * sum(point[1] for point in points) / 3.0
        all_y.extend(point[1] for point in points)
    height = weighted_height / area_xz if area_xz > 1e-12 else sum(all_y) / max(1, len(all_y))
    boundaries = _floor_boundaries(geometry, component.triangles)
    # Ring orientation carries outer/hole sign; sum absolute areas by sign.
    boundary_area = sum(
        -abs(loop.signed_area_scene2) if not loop.is_hole else abs(loop.signed_area_scene2)
        for loop in boundaries
    )
    area = max(0.0, -boundary_area) if boundaries else area_xz
    return AuthoredFloorPolygon(
        floor_groups=tuple(sorted(component.groups)),
        boundaries=boundaries,
        floor_height=float(height),
        height_range=(float(min(all_y)), float(max(all_y))),
        area_scene2=float(area),
        source_triangles=tuple(sorted(triangle.index for triangle in component.triangles)),
    )


def _canonical_plane(triangle: ObjTriangle, points: Sequence[Sequence[float]]) -> tuple[Point2, float] | None:
    nx, nz = float(triangle.normal[0]), float(triangle.normal[2])
    length = math.hypot(nx, nz)
    if length <= 1e-12:
        return None
    nx, nz = nx / length, nz / length
    if nx < -1e-12 or (abs(nx) <= 1e-12 and nz < 0.0):
        nx, nz = -nx, -nz
    offset = nx * float(points[0][0]) + nz * float(points[0][2])
    return (nx, nz), offset


def _wall_plane_clusters(
    geometry: AuthoredSceneGeometry,
    *,
    vertical_normal_tolerance_deg: float,
    plane_tolerance: float,
    plane_angle_tolerance_deg: float,
) -> list[_PlaneCluster]:
    max_abs_ny = math.sin(math.radians(vertical_normal_tolerance_deg))
    candidates: list[tuple[Point2, float, ObjTriangle]] = []
    for triangle in geometry.triangles:
        if triangle.area_scene2 <= 1e-12 or abs(float(triangle.normal[1])) > max_abs_ny:
            continue
        points = _triangle_points(geometry, triangle)
        plane = _canonical_plane(triangle, points)
        if plane is not None:
            candidates.append((plane[0], plane[1], triangle))
    candidates.sort(key=lambda item: (math.atan2(item[0][1], item[0][0]), item[1], item[2].index))

    cos_tolerance = math.cos(math.radians(plane_angle_tolerance_deg))
    clusters: list[_PlaneCluster] = []
    for normal, offset, triangle in candidates:
        match: _PlaneCluster | None = None
        match_score = math.inf
        for cluster in clusters:
            alignment = cluster.normal[0] * normal[0] + cluster.normal[1] * normal[1]
            offset_error = abs(cluster.offset - offset)
            if alignment >= cos_tolerance and offset_error <= plane_tolerance:
                score = offset_error + (1.0 - alignment) * max(1.0, plane_tolerance)
                if score < match_score:
                    match, match_score = cluster, score
        if match is None:
            clusters.append(_PlaneCluster(normal, offset, [triangle]))
            continue
        match.triangles.append(triangle)
        weights = [max(item.area_scene2, 1e-12) for item in match.triangles]
        nx = sum(
            _canonical_plane(item, _triangle_points(geometry, item))[0][0] * weight
            for item, weight in zip(match.triangles, weights)
        )
        nz = sum(
            _canonical_plane(item, _triangle_points(geometry, item))[0][1] * weight
            for item, weight in zip(match.triangles, weights)
        )
        norm = math.hypot(nx, nz)
        if norm > 1e-12:
            match.normal = (nx / norm, nz / norm)
        match.offset = sum(
            _canonical_plane(item, _triangle_points(geometry, item))[1] * weight
            for item, weight in zip(match.triangles, weights)
        ) / sum(weights)
    return sorted(clusters, key=lambda item: (item.normal, item.offset, min(t.index for t in item.triangles)))


def _point_in_floor(point: Point2, polygon: AuthoredFloorPolygon, geometry: AuthoredSceneGeometry) -> bool:
    # Triangle containment avoids assuming rectangular or convex room geometry.
    for triangle_id in polygon.source_triangles:
        triangle = geometry.triangles[triangle_id]
        points = _triangle_points(geometry, triangle)
        a, b, c = ((point3[0], point3[2]) for point3 in points)
        cross = lambda p1, p2, p3: (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        d1, d2, d3 = cross(a, b, point), cross(b, c, point), cross(c, a, point)
        if not ((d1 < -1e-8 or d2 < -1e-8 or d3 < -1e-8) and (d1 > 1e-8 or d2 > 1e-8 or d3 > 1e-8)):
            return True
    return False


def _point_segment_distance(point: Point2, start: Point2, end: Point2) -> float:
    dx, dz = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dz * dz
    if length2 <= 1e-18:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dz) / length2))
    return math.hypot(point[0] - (start[0] + fraction * dx), point[1] - (start[1] + fraction * dz))


def _boundary_matches(
    plane: _PlaneCluster,
    polygon: AuthoredFloorPolygon,
    geometry: AuthoredSceneGeometry,
    *,
    association_tolerance: float,
    angle_tolerance_deg: float,
) -> tuple[list[Interval], list[tuple[Point2, Point2]], float | None]:
    nx, nz = plane.normal
    tangent = (-nz, nx)
    cos_angle = math.cos(math.radians(angle_tolerance_deg))
    intervals: list[Interval] = []
    inward_candidates: list[tuple[Point2, Point2]] = []
    boundary_distances: list[float] = []
    for boundary in polygon.boundaries:
        ring = list(boundary.points_xz)
        for start, end in zip(ring, ring[1:] + ring[:1]):
            dx, dz = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dz)
            if length <= 1e-9:
                continue
            edge_tangent = (dx / length, dz / length)
            if abs(tangent[0] * edge_tangent[0] + tangent[1] * edge_tangent[1]) < cos_angle:
                continue
            distance = abs(nx * start[0] + nz * start[1] - plane.offset)
            if distance > association_tolerance:
                continue
            first_s = tangent[0] * start[0] + tangent[1] * start[1]
            second_s = tangent[0] * end[0] + tangent[1] * end[1]
            intervals.append((min(first_s, second_s), max(first_s, second_s)))
            inward_candidates.append((start, end))
            boundary_distances.append(distance)
    return (
        _merge_intervals(intervals, 1e-8),
        inward_candidates,
        min(boundary_distances) if boundary_distances else None,
    )


def _inside_direction(
    plane: _PlaneCluster,
    polygons: Sequence[AuthoredFloorPolygon],
    geometry: AuthoredSceneGeometry,
    boundary_edges: Sequence[tuple[Point2, Point2]],
    *,
    probe_distance: float,
) -> Point2:
    nx, nz = plane.normal
    tangent = (-nz, nx)
    if boundary_edges:
        start, end = boundary_edges[0]
        mid = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length > 1e-9:
            edge_tangent = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
            candidate = (-edge_tangent[1], edge_tangent[0])
            plus = (mid[0] + candidate[0] * probe_distance, mid[1] + candidate[1] * probe_distance)
            minus = (mid[0] - candidate[0] * probe_distance, mid[1] - candidate[1] * probe_distance)
            if any(_point_in_floor(plus, polygon, geometry) for polygon in polygons):
                return candidate
            if any(_point_in_floor(minus, polygon, geometry) for polygon in polygons):
                return (-candidate[0], -candidate[1])
    all_points = [point for triangle in plane.triangles for point in _triangle_points(geometry, triangle)]
    s = sum(tangent[0] * point[0] + tangent[1] * point[2] for point in all_points) / max(1, len(all_points))
    center = (nx * plane.offset + tangent[0] * s, nz * plane.offset + tangent[1] * s)
    plus = (center[0] + nx * probe_distance, center[1] + nz * probe_distance)
    minus = (center[0] - nx * probe_distance, center[1] - nz * probe_distance)
    plus_inside = any(_point_in_floor(plus, polygon, geometry) for polygon in polygons)
    minus_inside = any(_point_in_floor(minus, polygon, geometry) for polygon in polygons)
    if minus_inside and not plus_inside:
        return (-nx, -nz)
    return (nx, nz)


def _merge_intervals(intervals: Iterable[Interval], tolerance: float) -> list[Interval]:
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals if math.isfinite(a) and math.isfinite(b))
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + tolerance:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(item[0], item[1]) for item in merged]


def _clip_intervals(
    intervals: Sequence[Interval],
    domains: Sequence[Interval],
    *,
    extension: float,
    tolerance: float,
) -> list[Interval]:
    if not domains:
        return list(intervals)
    clipped: list[Interval] = []
    for start, end in intervals:
        for domain_start, domain_end in domains:
            low, high = domain_start - extension, domain_end + extension
            left, right = max(start, low), min(end, high)
            if right - left > tolerance:
                clipped.append((left, right))
    return _merge_intervals(clipped, tolerance)


def _triangle_slice_interval(
    points: Sequence[Sequence[float]],
    normal: Point2,
    tangent: Point2,
    y: float,
) -> Interval | None:
    projected = [(tangent[0] * float(point[0]) + tangent[1] * float(point[2]), float(point[1])) for point in points]
    intersections: list[float] = []
    for first, second in zip(projected, projected[1:] + projected[:1]):
        if (first[1] < y < second[1]) or (second[1] < y < first[1]):
            fraction = (y - first[1]) / (second[1] - first[1])
            intersections.append(first[0] + fraction * (second[0] - first[0]))
    if len(intersections) < 2:
        return None
    return (min(intersections), max(intersections))


def _source_refs(triangles: Iterable[ObjTriangle]) -> tuple[str, ...]:
    refs = {
        f"{'+'.join(triangle.groups)}#triangle:{triangle.index}"
        for triangle in triangles
    }
    return tuple(sorted(refs))


def _coverage_rows(
    geometry: AuthoredSceneGeometry,
    plane: _PlaneCluster,
    *,
    domain_intervals: Sequence[Interval],
    endpoint_extension: float,
    coordinate_tolerance: float,
) -> list[tuple[float, float, list[Interval], dict[tuple[float, float], set[int]]]]:
    levels = sorted(
        {
            round(float(point[1]) / max(coordinate_tolerance, 1e-9)) * max(coordinate_tolerance, 1e-9)
            for triangle in plane.triangles
            for point in _triangle_points(geometry, triangle)
        }
    )
    tangent = (-plane.normal[1], plane.normal[0])
    rows: list[tuple[float, float, list[Interval], dict[tuple[float, float], set[int]]]] = []
    for low, high in zip(levels, levels[1:]):
        if high - low <= coordinate_tolerance:
            continue
        y = (low + high) * 0.5
        raw: list[tuple[Interval, int]] = []
        for triangle in plane.triangles:
            points = _triangle_points(geometry, triangle)
            interval = _triangle_slice_interval(points, plane.normal, tangent, y)
            if interval is not None and interval[1] - interval[0] > coordinate_tolerance:
                raw.append((interval, triangle.index))
        intervals = _clip_intervals(
            _merge_intervals((interval for interval, _ in raw), coordinate_tolerance),
            domain_intervals,
            extension=endpoint_extension,
            tolerance=coordinate_tolerance,
        )
        sources: dict[tuple[float, float], set[int]] = {}
        for interval in intervals:
            contributors = {
                triangle_id
                for source_interval, triangle_id in raw
                if source_interval[0] <= interval[1] + coordinate_tolerance
                and source_interval[1] >= interval[0] - coordinate_tolerance
            }
            sources[interval] = contributors
        rows.append((low, high, intervals, sources))
    return rows


def _merge_vertical_records(
    records: list[tuple[Interval, Interval, set[int]]],
    *,
    tolerance: float,
) -> list[tuple[Interval, Interval, set[int]]]:
    result: list[tuple[Interval, Interval, set[int]]] = []
    for interval, y_span, sources in sorted(records, key=lambda item: (item[0], item[1])):
        match = None
        for index in range(len(result) - 1, -1, -1):
            prior_interval, prior_y, _ = result[index]
            if prior_y[1] < y_span[0] - tolerance:
                break
            if (
                abs(prior_interval[0] - interval[0]) <= tolerance
                and abs(prior_interval[1] - interval[1]) <= tolerance
                and abs(prior_y[1] - y_span[0]) <= tolerance
            ):
                match = index
                break
        if match is None:
            result.append((interval, y_span, set(sources)))
        else:
            old_interval, old_y, old_sources = result[match]
            old_sources.update(sources)
            result[match] = (old_interval, (old_y[0], y_span[1]), old_sources)
    return result


def _line_point(normal: Point2, tangent: Point2, offset: float, coordinate: float) -> Point2:
    return (normal[0] * offset + tangent[0] * coordinate, normal[1] * offset + tangent[1] * coordinate)


def extract_authored_structures(
    geometry: AuthoredSceneGeometry,
    floor_groups: Sequence[str],
    *,
    initial_scene_units_per_meter: float | None = None,
    tolerance_m: float = 0.01,
    wall_to_floor_association_m: float = 0.20,
    min_wall_length_m: float = 0.15,
    min_wall_height_m: float = 0.60,
    opening_min_width_m: float = 0.04,
    vertical_normal_tolerance_deg: float = 8.0,
    plane_angle_tolerance_deg: float = 2.0,
    boundary_angle_tolerance_deg: float = 8.0,
) -> AuthoredStructureExtraction:
    """Extract selected floor patches and spatially adjacent wall face evidence.

    ``floor_groups`` are exact OBJ group names from the existing semantic room
    map. No group-name prefix or camera label is used to discover walls: walls
    are selected by measured coplanarity and proximity to the selected floor
    boundary. Floor boundaries without vertical face support are never emitted
    as walls, and unsupported intervals remain gaps.

    ``initial_scene_units_per_meter`` affects tolerances only. It does not scale
    any vertex, plane, endpoint, height, or offset in the returned data.
    """

    if not floor_groups:
        raise ValueError("at least one exact authored floor group is required")
    requested = tuple(sorted({str(group).strip() for group in floor_groups if str(group).strip()}))
    if not requested:
        raise ValueError("floor_groups must contain non-empty exact group names")
    if len(requested) != len(floor_groups):
        raise ValueError("floor_groups contains an empty or duplicate name")
    scale = _finite_positive(initial_scene_units_per_meter, name="initial_scene_units_per_meter")
    metric_values = (tolerance_m, wall_to_floor_association_m, min_wall_length_m, min_wall_height_m, opening_min_width_m)
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in metric_values):
        raise ValueError("structural tolerances must be finite and non-negative")
    for value, name in (
        (vertical_normal_tolerance_deg, "vertical_normal_tolerance_deg"),
        (plane_angle_tolerance_deg, "plane_angle_tolerance_deg"),
        (boundary_angle_tolerance_deg, "boundary_angle_tolerance_deg"),
    ):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) < 45.0:
            raise ValueError(f"{name} must be finite and in [0, 45)")

    selected_floor_triangles = [
        triangle
        for triangle in geometry.triangles
        if any(group in requested for group in triangle.groups)
        and float(triangle.normal[1]) >= math.cos(math.radians(geometry.horizontal_tolerance_deg))
        and triangle.area_scene2 > 1e-12
    ]
    if not selected_floor_triangles:
        raise ValueError("the selected authored floor groups have no upward-facing floor triangles")

    components = _floor_components(geometry, selected_floor_triangles, set(requested))
    floors = tuple(
        _floor_polygon(geometry, component)
        for component in components
        if component.triangles
    )
    if not floors:
        raise ValueError("the selected authored floor groups produced no floor polygons")

    all_floor_points = [
        point
        for triangle in selected_floor_triangles
        for point in _triangle_points(geometry, triangle)
    ]
    extent = max(
        math.hypot(max(p[0] for p in all_floor_points) - min(p[0] for p in all_floor_points),
                   max(p[2] for p in all_floor_points) - min(p[2] for p in all_floor_points)),
        max(p[1] for p in all_floor_points) - min(p[1] for p in all_floor_points),
        1.0,
    )
    if scale is None:
        # A dimensionless fallback is only for numerical robustness. Callers
        # doing metric registration should provide their provisional scale.
        coordinate_tolerance = max(extent * 1e-7, 1e-8)
        association_tolerance = max(extent * 0.02, coordinate_tolerance * 4)
        min_wall_length = 0.0
        min_wall_height = 0.0
        opening_min_width = coordinate_tolerance * 4
    else:
        coordinate_tolerance = max(float(tolerance_m) * scale, 1e-7)
        association_tolerance = max(float(wall_to_floor_association_m) * scale, coordinate_tolerance)
        min_wall_length = float(min_wall_length_m) * scale
        min_wall_height = float(min_wall_height_m) * scale
        opening_min_width = max(float(opening_min_width_m) * scale, coordinate_tolerance)

    plane_tolerance = max(coordinate_tolerance, extent * 1e-8)
    planes = _wall_plane_clusters(
        geometry,
        vertical_normal_tolerance_deg=float(vertical_normal_tolerance_deg),
        plane_tolerance=plane_tolerance,
        plane_angle_tolerance_deg=float(plane_angle_tolerance_deg),
    )
    all_segments: list[AuthoredWallSegment] = []
    all_openings: list[AuthoredOpening] = []
    for plane_index, plane in enumerate(planes):
        canonical_normal = plane.normal
        tangent = (-canonical_normal[1], canonical_normal[0])
        all_vertices = [point for triangle in plane.triangles for point in _triangle_points(geometry, triangle)]
        plane_s = [tangent[0] * point[0] + tangent[1] * point[2] for point in all_vertices]
        plane_y = [float(point[1]) for point in all_vertices]
        if max(plane_s) - min(plane_s) < min_wall_length or max(plane_y) - min(plane_y) < min_wall_height:
            continue

        matching_floors: list[AuthoredFloorPolygon] = []
        matched_domains: list[Interval] = []
        matched_edges: list[tuple[Point2, Point2]] = []
        matched_boundary_distances: list[float] = []
        for floor in floors:
            # Structural wall faces must overlap the authored floor's vertical
            # neighborhood; this rejects unrelated tall geometry at a distance.
            if min(plane_y) > floor.height_range[1] + association_tolerance:
                continue
            if max(plane_y) < floor.height_range[0] - association_tolerance:
                continue
            domains, edges, boundary_distance = _boundary_matches(
                plane,
                floor,
                geometry,
                association_tolerance=association_tolerance,
                angle_tolerance_deg=float(boundary_angle_tolerance_deg),
            )
            if domains:
                matching_floors.append(floor)
                matched_domains.extend(domains)
                matched_edges.extend(edges)
                if boundary_distance is not None:
                    matched_boundary_distances.append(boundary_distance)
        if not matching_floors:
            continue

        matched_domains = _merge_intervals(matched_domains, coordinate_tolerance)
        plane_id = f"wall-plane-{plane_index:04d}"
        inward = _inside_direction(
            plane,
            matching_floors,
            geometry,
            matched_edges,
            probe_distance=max(plane_tolerance * 2.0, min(association_tolerance * 0.1, extent * 1e-3)),
        )
        signed_offset = inward[0] * canonical_normal[0] * plane.offset + inward[1] * canonical_normal[1] * plane.offset
        # Distance from the floor edge is allowed for wall thickness, but the
        # longitudinal extent remains clipped to the actual selected boundary.
        endpoint_extension = min(association_tolerance, max(plane_tolerance * 2.0, (0.05 * scale if scale else extent * 1e-4)))
        domain = matched_domains
        rows = _coverage_rows(
            geometry,
            plane,
            domain_intervals=domain,
            endpoint_extension=endpoint_extension,
            coordinate_tolerance=coordinate_tolerance,
        )
        segment_records: list[tuple[Interval, Interval, set[int]]] = []
        opening_records: list[tuple[Interval, Interval, set[int]]] = []
        for low, high, intervals, sources in rows:
            for interval in intervals:
                if interval[1] - interval[0] >= min_wall_length:
                    segment_records.append((interval, (low, high), set(sources.get(interval, set()))))
            for left, right in zip(intervals, intervals[1:]):
                gap = (left[1], right[0])
                if gap[1] - gap[0] >= opening_min_width:
                    refs = set(sources.get(left, set())) | set(sources.get(right, set()))
                    opening_records.append((gap, (low, high), refs))

        floor_groups_for_plane = tuple(sorted({group for floor in matching_floors for group in floor.floor_groups}))
        for record_index, (interval, y_span, triangle_ids) in enumerate(
            _merge_vertical_records(segment_records, tolerance=coordinate_tolerance)
        ):
            start = _line_point(canonical_normal, tangent, plane.offset, interval[0])
            end = _line_point(canonical_normal, tangent, plane.offset, interval[1])
            refs = _source_refs(t for t in plane.triangles if t.index in triangle_ids)
            all_segments.append(
                AuthoredWallSegment(
                    segment_id=f"{plane_id}-segment-{record_index:04d}",
                    start_xz=start,
                    end_xz=end,
                    normal_xz=inward,
                    offset=float(signed_offset),
                    support_count=len(triangle_ids),
                    vertical_span=y_span,
                    source_refs=refs,
                    plane_id=plane_id,
                    floor_groups=floor_groups_for_plane,
                    floor_boundary_distance_scene=min(matched_boundary_distances),
                )
            )
        for record_index, (interval, y_span, triangle_ids) in enumerate(
            _merge_vertical_records(opening_records, tolerance=coordinate_tolerance)
        ):
            all_openings.append(
                AuthoredOpening(
                    opening_id=f"{plane_id}-opening-{record_index:04d}",
                    plane_id=plane_id,
                    interval=interval,
                    start_xz=_line_point(canonical_normal, tangent, plane.offset, interval[0]),
                    end_xz=_line_point(canonical_normal, tangent, plane.offset, interval[1]),
                    vertical_span=y_span,
                    source_refs=_source_refs(t for t in plane.triangles if t.index in triangle_ids),
                )
            )

    all_segments.sort(key=lambda item: (item.plane_id, item.vertical_span, item.start_xz, item.segment_id))
    all_openings.sort(key=lambda item: (item.plane_id, item.vertical_span, item.interval, item.opening_id))
    return AuthoredStructureExtraction(
        source_sha256=geometry.source_sha256,
        floor_groups=requested,
        floor_polygons=floors,
        wall_segments=tuple(all_segments),
        openings=tuple(all_openings),
        tolerance_scene_units=coordinate_tolerance,
        initial_scene_units_per_meter=scale,
    )


__all__ = [
    "AuthoredFloorBoundary",
    "AuthoredFloorPolygon",
    "AuthoredOpening",
    "AuthoredStructureExtraction",
    "AuthoredWallSegment",
    "extract_authored_structures",
]
