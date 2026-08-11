"""Offline physical validation of canonical tracks against an authored OBJ scene.

The promoted Sweet Home 3D OBJ is already expressed in ``menon_scene``.  This
module therefore applies the declared backend-world-to-scene similarity exactly
once to canonical world points, then compares those points with upward-facing,
near-horizontal surfaces from authoritative ``ground_*`` and ``room_*`` OBJ
groups.  It is validation-only code and has no runtime or rendering side
effects.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import numpy as np

from .core import (
    CheckStatus,
    ConfidenceScores,
    FailureType,
    SourceMetadata,
    ValidationCheck,
    ValidationReport,
)
from .transforms import matrix_from_col_major, transform_point


ORACLE_CONTRACT = "noesis.validation.authored_scene_track_geometry"
ORACLE_CONTRACT_VERSION = 1
DEFAULT_MAX_OBJ_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_VERTICES = 5_000_000
DEFAULT_MAX_FACES = 5_000_000
DEFAULT_MAX_FACE_VERTICES = 16_384
_AUTHORITY_PREFIXES = ("ground_", "room_")
_STATUS_ORDER = {
    CheckStatus.PASS.value: 0,
    CheckStatus.WARNING.value: 1,
    CheckStatus.BLOCKED.value: 2,
    CheckStatus.FAIL.value: 3,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_point3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"), value.get("z"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) < 3:
        return None
    try:
        point = tuple(float(item) for item in value[:3])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in point):
        return None
    return point  # type: ignore[return-value]


def _authority_group(groups: Sequence[str]) -> str | None:
    for group in groups:
        if str(group).startswith(_AUTHORITY_PREFIXES):
            return str(group)
    return None


def _cross_2d(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(a[0]))


def _point_in_triangle_2d(
    point: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    *,
    epsilon: float = 1e-9,
) -> bool:
    d1 = _cross_2d(a, b, point)
    d2 = _cross_2d(b, c, point)
    d3 = _cross_2d(c, a, point)
    has_negative = d1 < -epsilon or d2 < -epsilon or d3 < -epsilon
    has_positive = d1 > epsilon or d2 > epsilon or d3 > epsilon
    return not (has_negative and has_positive)


def _segments_intersect_2d(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
    *,
    epsilon: float = 1e-10,
) -> bool:
    ab_c = _cross_2d(a, b, c)
    ab_d = _cross_2d(a, b, d)
    cd_a = _cross_2d(c, d, a)
    cd_b = _cross_2d(c, d, b)
    return (ab_c * ab_d < -epsilon) and (cd_a * cd_b < -epsilon)


def _newell_normal(indices: Sequence[int], vertices: Sequence[tuple[float, float, float]]) -> np.ndarray:
    normal = np.zeros(3, dtype=np.float64)
    for offset, index in enumerate(indices):
        current = vertices[index]
        following = vertices[indices[(offset + 1) % len(indices)]]
        normal[0] += (current[1] - following[1]) * (current[2] + following[2])
        normal[1] += (current[2] - following[2]) * (current[0] + following[0])
        normal[2] += (current[0] - following[0]) * (current[1] + following[1])
    return normal


def _project_polygon(
    indices: Sequence[int], vertices: Sequence[tuple[float, float, float]]
) -> tuple[list[tuple[float, float]], int]:
    normal = _newell_normal(indices, vertices)
    if float(np.linalg.norm(normal)) <= 1e-12:
        raise ValueError("OBJ face is degenerate")
    drop_axis = int(np.argmax(np.abs(normal)))
    axes = [axis for axis in range(3) if axis != drop_axis]
    return [
        (float(vertices[index][axes[0]]), float(vertices[index][axes[1]]))
        for index in indices
    ], drop_axis


def _remove_collinear_polygon_vertices(
    indices: list[int], points: list[tuple[float, float]], *, epsilon: float = 1e-12
) -> tuple[list[int], list[tuple[float, float]]]:
    changed = True
    while changed and len(indices) > 3:
        changed = False
        for offset in range(len(indices)):
            previous = points[(offset - 1) % len(points)]
            current = points[offset]
            following = points[(offset + 1) % len(points)]
            if abs(_cross_2d(previous, current, following)) <= epsilon:
                del indices[offset]
                del points[offset]
                changed = True
                break
    return indices, points


def _triangulate_polygon(
    raw_indices: Sequence[int],
    vertices: Sequence[tuple[float, float, float]],
) -> list[tuple[int, int, int]]:
    indices: list[int] = []
    for index in raw_indices:
        if not indices or index != indices[-1]:
            indices.append(index)
    if len(indices) > 1 and indices[0] == indices[-1]:
        indices.pop()
    if len(indices) < 3:
        raise ValueError("OBJ face has fewer than three distinct vertices")
    if len(indices) == 3:
        return [(indices[0], indices[1], indices[2])]

    points, _ = _project_polygon(indices, vertices)
    indices, points = _remove_collinear_polygon_vertices(indices, points)
    if len(indices) == 3:
        return [(indices[0], indices[1], indices[2])]

    # Fail closed on a self-intersecting polygon instead of inventing geometry.
    for first in range(len(points)):
        first_next = (first + 1) % len(points)
        for second in range(first + 1, len(points)):
            second_next = (second + 1) % len(points)
            if first in (second, second_next) or first_next in (second, second_next):
                continue
            if _segments_intersect_2d(
                points[first], points[first_next], points[second], points[second_next]
            ):
                raise ValueError("OBJ face polygon is self-intersecting")

    signed_area = 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if abs(signed_area) <= 1e-12:
        raise ValueError("OBJ face has zero projected area")
    orientation = 1.0 if signed_area > 0.0 else -1.0

    remaining = list(range(len(indices)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ear_found = False
        for cursor, current in enumerate(remaining):
            previous = remaining[(cursor - 1) % len(remaining)]
            following = remaining[(cursor + 1) % len(remaining)]
            if orientation * _cross_2d(points[previous], points[current], points[following]) <= 1e-12:
                continue
            if any(
                other not in (previous, current, following)
                and _point_in_triangle_2d(
                    points[other], points[previous], points[current], points[following]
                )
                for other in remaining
            ):
                continue
            triangles.append((indices[previous], indices[current], indices[following]))
            del remaining[cursor]
            ear_found = True
            break
        if not ear_found:
            raise ValueError("OBJ face could not be triangulated")
    triangles.append(tuple(indices[index] for index in remaining))  # type: ignore[arg-type]
    return triangles


@dataclass(frozen=True)
class ObjTriangle:
    index: int
    groups: tuple[str, ...]
    vertex_indices: tuple[int, int, int]
    normal: tuple[float, float, float]
    area_scene2: float


@dataclass(frozen=True)
class WalkableTriangle:
    triangle: ObjTriangle
    authority_group: str


@dataclass(frozen=True)
class BoundaryEdge:
    authority_group: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]


@dataclass(frozen=True)
class SimilarityTransform:
    matrix: np.ndarray
    col_major: tuple[float, ...]
    scene_units_per_meter: float
    determinant: float
    relative_scale_spread: float

    @classmethod
    def from_col_major(
        cls,
        values: Sequence[float],
        *,
        affine_tolerance: float = 1e-9,
        similarity_tolerance: float = 1e-6,
    ) -> "SimilarityTransform":
        matrix = matrix_from_col_major(values, name="world_to_scene_col_major")
        if not np.allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=affine_tolerance, rtol=0.0):
            raise ValueError("world_to_scene_col_major is not an affine transform")
        linear = matrix[:3, :3]
        singular_values = np.linalg.svd(linear, compute_uv=False)
        scale = float(np.mean(singular_values))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("world_to_scene_col_major has an invalid scale")
        spread = float((np.max(singular_values) - np.min(singular_values)) / scale)
        if spread > float(similarity_tolerance):
            raise ValueError(
                "world_to_scene_col_major is not a uniform similarity "
                f"(relative scale spread {spread:.6g})"
            )
        determinant = float(np.linalg.det(linear))
        if determinant <= 0.0:
            raise ValueError("world_to_scene_col_major is mirrored or singular")
        return cls(
            matrix=matrix,
            col_major=tuple(float(item) for item in values),
            scene_units_per_meter=scale,
            determinant=determinant,
            relative_scale_spread=spread,
        )

    def apply(self, point_m: Sequence[float]) -> tuple[float, float, float]:
        point = _finite_point3(point_m)
        if point is None:
            raise ValueError("backend world point must contain three finite values")
        transformed = transform_point(point, self.matrix)
        return tuple(float(item) for item in transformed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_to_scene_col_major": list(self.col_major),
            "scene_units_per_meter": self.scene_units_per_meter,
            "determinant": self.determinant,
            "relative_scale_spread": self.relative_scale_spread,
        }


@dataclass(frozen=True)
class GeometryThresholds:
    support_good_m: float = 0.12
    support_fail_m: float = 0.25
    outside_warning_m: float = 0.15

    def __post_init__(self) -> None:
        values = (self.support_good_m, self.support_fail_m, self.outside_warning_m)
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("geometry thresholds must be finite and non-negative")
        if float(self.support_good_m) > float(self.support_fail_m):
            raise ValueError("support_good_m cannot exceed support_fail_m")

    def to_dict(self) -> dict[str, float]:
        return {
            "support_good_m": float(self.support_good_m),
            "support_fail_m": float(self.support_fail_m),
            "outside_warning_m": float(self.outside_warning_m),
        }


@dataclass(frozen=True)
class AuthoredSceneGeometry:
    source_path: Path
    source_sha256: str
    vertices: tuple[tuple[float, float, float], ...]
    triangles: tuple[ObjTriangle, ...]
    walkable_triangles: tuple[WalkableTriangle, ...]
    boundary_edges: tuple[BoundaryEdge, ...]
    horizontal_tolerance_deg: float

    @classmethod
    def from_obj(
        cls,
        path: str | Path,
        *,
        horizontal_tolerance_deg: float = 15.0,
        max_bytes: int = DEFAULT_MAX_OBJ_BYTES,
        max_vertices: int = DEFAULT_MAX_VERTICES,
        max_faces: int = DEFAULT_MAX_FACES,
        max_face_vertices: int = DEFAULT_MAX_FACE_VERTICES,
    ) -> "AuthoredSceneGeometry":
        source_path = Path(path).expanduser().resolve()
        info = source_path.stat()
        if not source_path.is_file():
            raise ValueError(f"authored OBJ is not a regular file: {source_path}")
        if info.st_size <= 0 or info.st_size > int(max_bytes):
            raise ValueError(
                f"authored OBJ size must be in 1..{int(max_bytes)} bytes; found {info.st_size}"
            )
        tolerance = float(horizontal_tolerance_deg)
        if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 90.0:
            raise ValueError("horizontal_tolerance_deg must be in [0, 90)")

        vertices: list[tuple[float, float, float]] = []
        triangles: list[ObjTriangle] = []
        active_groups: tuple[str, ...] = ("default",)
        object_name = "default"
        face_count = 0
        with source_path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                keyword = fields[0]
                if keyword == "v":
                    if len(fields) < 4:
                        raise ValueError(f"OBJ vertex is incomplete at line {line_number}")
                    try:
                        vertex = tuple(float(item) for item in fields[1:4])
                    except ValueError as exc:
                        raise ValueError(f"OBJ vertex is invalid at line {line_number}") from exc
                    if not all(math.isfinite(item) for item in vertex):
                        raise ValueError(f"OBJ vertex is non-finite at line {line_number}")
                    vertices.append(vertex)  # type: ignore[arg-type]
                    if len(vertices) > int(max_vertices):
                        raise ValueError("authored OBJ exceeds the vertex limit")
                    continue
                if keyword == "o":
                    object_name = " ".join(fields[1:]).strip() or "default"
                    continue
                if keyword == "g":
                    active_groups = tuple(fields[1:]) or (f"object:{object_name}",)
                    continue
                if keyword != "f":
                    continue

                face_count += 1
                if face_count > int(max_faces):
                    raise ValueError("authored OBJ exceeds the face limit")
                if len(fields) - 1 > int(max_face_vertices):
                    raise ValueError(f"OBJ face exceeds the vertex limit at line {line_number}")
                raw_indices: list[int] = []
                for token in fields[1:]:
                    vertex_token = token.split("/", 1)[0]
                    try:
                        obj_index = int(vertex_token)
                    except ValueError as exc:
                        raise ValueError(f"OBJ face index is invalid at line {line_number}") from exc
                    if obj_index == 0:
                        raise ValueError(f"OBJ face index cannot be zero at line {line_number}")
                    index = obj_index - 1 if obj_index > 0 else len(vertices) + obj_index
                    if index < 0 or index >= len(vertices):
                        raise ValueError(f"OBJ face index is out of range at line {line_number}")
                    raw_indices.append(index)
                try:
                    face_triangles = _triangulate_polygon(raw_indices, vertices)
                except ValueError as exc:
                    raise ValueError(f"{exc} at OBJ line {line_number}") from exc
                for indices in face_triangles:
                    points = np.asarray([vertices[index] for index in indices], dtype=np.float64)
                    cross = np.cross(points[1] - points[0], points[2] - points[0])
                    cross_norm = float(np.linalg.norm(cross))
                    normal = cross / cross_norm if cross_norm > 1e-12 else np.zeros(3, dtype=np.float64)
                    triangles.append(
                        ObjTriangle(
                            index=len(triangles),
                            groups=active_groups,
                            vertex_indices=tuple(indices),
                            normal=tuple(float(item) for item in normal),
                            area_scene2=0.5 * cross_norm,
                        )
                    )

        if not vertices:
            raise ValueError("authored OBJ contains no vertices")
        if not triangles:
            raise ValueError("authored OBJ contains no triangulated faces")

        minimum_up_component = math.cos(math.radians(tolerance))
        walkable: list[WalkableTriangle] = []
        for triangle in triangles:
            group = _authority_group(triangle.groups)
            if group is None or triangle.area_scene2 <= 1e-12:
                continue
            # Only top faces are walkable.  This rejects the matching underside
            # of Sweet Home 3D room slabs and all vertical wall faces.
            if float(triangle.normal[1]) < minimum_up_component:
                continue
            walkable.append(WalkableTriangle(triangle=triangle, authority_group=group))

        boundaries = _derive_boundary_edges(vertices, walkable)
        return cls(
            source_path=source_path,
            source_sha256=_sha256_file(source_path),
            vertices=tuple(vertices),
            triangles=tuple(triangles),
            walkable_triangles=tuple(walkable),
            boundary_edges=tuple(boundaries),
            horizontal_tolerance_deg=tolerance,
        )

    def inventory(self) -> dict[str, Any]:
        groups = Counter(
            surface.authority_group for surface in self.walkable_triangles
        )
        vertices = np.asarray(self.vertices, dtype=np.float64)
        return {
            "path": str(self.source_path),
            "sha256": self.source_sha256,
            "vertex_count": len(self.vertices),
            "triangle_count": len(self.triangles),
            "walkable_triangle_count": len(self.walkable_triangles),
            "walkable_group_count": len(groups),
            "walkable_groups": dict(sorted(groups.items())),
            "boundary_edge_count": len(self.boundary_edges),
            "horizontal_tolerance_deg": self.horizontal_tolerance_deg,
            "bounds_scene": {
                "min": [float(item) for item in np.min(vertices, axis=0)],
                "max": [float(item) for item in np.max(vertices, axis=0)],
            },
        }


def _position_key(point: Sequence[float], *, decimals: int = 7) -> tuple[float, float, float]:
    return tuple(round(float(item), decimals) for item in point)  # type: ignore[return-value]


def _derive_boundary_edges(
    vertices: Sequence[tuple[float, float, float]],
    walkable: Sequence[WalkableTriangle],
) -> list[BoundaryEdge]:
    counts: Counter[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = Counter()
    examples: dict[
        tuple[str, tuple[float, float, float], tuple[float, float, float]],
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ] = {}
    for surface in walkable:
        indices = surface.triangle.vertex_indices
        for offset in range(3):
            start = vertices[indices[offset]]
            end = vertices[indices[(offset + 1) % 3]]
            start_key = _position_key(start)
            end_key = _position_key(end)
            low, high = sorted((start_key, end_key))
            key = (surface.authority_group, low, high)
            counts[key] += 1
            examples[key] = (start, end)
    return [
        BoundaryEdge(authority_group=key[0], start=examples[key][0], end=examples[key][1])
        for key, count in counts.items()
        if count == 1
    ]


def _point_in_triangle_xz(point: Sequence[float], triangle: np.ndarray) -> bool:
    return _point_in_triangle_2d(
        (float(point[0]), float(point[2])),
        (float(triangle[0, 0]), float(triangle[0, 2])),
        (float(triangle[1, 0]), float(triangle[1, 2])),
        (float(triangle[2, 0]), float(triangle[2, 2])),
        epsilon=1e-8,
    )


def _support_y(point: Sequence[float], triangle: np.ndarray, normal: Sequence[float]) -> float:
    normal_np = np.asarray(normal, dtype=np.float64)
    if abs(float(normal_np[1])) <= 1e-12:
        raise ValueError("walkable triangle has a vertical plane")
    origin = triangle[0]
    return float(
        origin[1]
        - (
            normal_np[0] * (float(point[0]) - origin[0])
            + normal_np[2] * (float(point[2]) - origin[2])
        )
        / normal_np[1]
    )


def _point_segment_distance_xz(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    point_xz = np.asarray([point[0], point[2]], dtype=np.float64)
    start_xz = np.asarray([start[0], start[2]], dtype=np.float64)
    end_xz = np.asarray([end[0], end[2]], dtype=np.float64)
    delta = end_xz - start_xz
    norm_squared = float(np.dot(delta, delta))
    if norm_squared <= 1e-18:
        return float(np.linalg.norm(point_xz - start_xz))
    fraction = float(np.dot(point_xz - start_xz, delta) / norm_squared)
    fraction = min(1.0, max(0.0, fraction))
    return float(np.linalg.norm(point_xz - (start_xz + fraction * delta)))


@dataclass(frozen=True)
class JournalObservation:
    journal_sequence: int
    sample_id: str
    camera_id: str
    observed_at_us: int | None
    world_point_m: tuple[float, float, float] | None
    world_source: str | None
    world_quality: str | None
    zone_label: str | None
    zone_source: str | None
    zone_authoritative: bool
    expected_room: str | None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class OracleSampleResult:
    sample_id: str
    camera_id: str
    journal_sequence: int | None
    observed_at_us: int | None
    status: str
    world_point_m: tuple[float, float, float] | None
    scene_point: tuple[float, float, float] | None
    world_source: str | None
    zone_label: str | None
    zone_source: str | None
    zone_authoritative: bool
    expected_room: str | None
    contained: bool | None
    support_group: str | None
    support_y_scene: float | None
    vertical_support_scene: float | None
    vertical_support_m: float | None
    signed_boundary_distance_scene: float | None
    signed_boundary_distance_m: float | None
    expected_room_groups: tuple[str, ...] | None
    expected_room_contained: bool | None
    expected_room_boundary_distance_m: float | None
    first_divergence: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "camera_id": self.camera_id,
            "journal_sequence": self.journal_sequence,
            "observed_at_us": self.observed_at_us,
            "status": self.status,
            "world_point_m": list(self.world_point_m) if self.world_point_m is not None else None,
            "scene_point": list(self.scene_point) if self.scene_point is not None else None,
            "world_source": self.world_source,
            "zone_label": self.zone_label,
            "zone_source": self.zone_source,
            "zone_authoritative": self.zone_authoritative,
            "expected_room": self.expected_room,
            "contained": self.contained,
            "support_group": self.support_group,
            "support_y_scene": self.support_y_scene,
            "vertical_support_scene": self.vertical_support_scene,
            "vertical_support_m": self.vertical_support_m,
            "signed_boundary_distance_scene": self.signed_boundary_distance_scene,
            "signed_boundary_distance_m": self.signed_boundary_distance_m,
            "expected_room_groups": (
                list(self.expected_room_groups)
                if self.expected_room_groups is not None
                else None
            ),
            "expected_room_contained": self.expected_room_contained,
            "expected_room_boundary_distance_m": self.expected_room_boundary_distance_m,
            "first_divergence": dict(self.first_divergence) if self.first_divergence else None,
        }
        return {key: value for key, value in payload.items() if value is not None}


def _nearest_boundary_distance(
    geometry: AuthoredSceneGeometry,
    point: Sequence[float],
    *,
    authority_group: str | None = None,
) -> float:
    edges = [
        edge
        for edge in geometry.boundary_edges
        if authority_group is None or edge.authority_group == authority_group
    ]
    if not edges and authority_group is not None:
        edges = list(geometry.boundary_edges)
    if not edges:
        return math.inf
    return min(_point_segment_distance_xz(point, edge.start, edge.end) for edge in edges)


def score_world_point(
    *,
    geometry: AuthoredSceneGeometry,
    similarity: SimilarityTransform,
    world_point_m: Sequence[float],
    thresholds: GeometryThresholds = GeometryThresholds(),
    sample_id: str = "sample",
    camera_id: str = "unknown",
    journal_sequence: int | None = None,
    observed_at_us: int | None = None,
    world_source: str | None = None,
    zone_label: str | None = None,
    zone_source: str | None = None,
    zone_authoritative: bool = False,
    expected_room: str | None = None,
    expected_room_groups: Sequence[str] | None = None,
) -> OracleSampleResult:
    world_point = _finite_point3(world_point_m)
    if world_point is None:
        return OracleSampleResult(
            sample_id=sample_id,
            camera_id=camera_id,
            journal_sequence=journal_sequence,
            observed_at_us=observed_at_us,
            status=CheckStatus.FAIL.value,
            world_point_m=None,
            scene_point=None,
            world_source=world_source,
            zone_label=zone_label,
            zone_source=zone_source,
            zone_authoritative=zone_authoritative,
            expected_room=expected_room,
            contained=None,
            support_group=None,
            support_y_scene=None,
            vertical_support_scene=None,
            vertical_support_m=None,
            signed_boundary_distance_scene=None,
            signed_boundary_distance_m=None,
            expected_room_groups=(
                tuple(str(group) for group in expected_room_groups)
                if expected_room_groups is not None
                else None
            ),
            expected_room_contained=None,
            expected_room_boundary_distance_m=None,
            first_divergence={
                "stage": "backend_world_input",
                "reason": "world_position_invalid",
                "failure_type": FailureType.PROJECTION.value,
            },
        )

    scene_point = similarity.apply(world_point)
    candidates: list[tuple[float, int, WalkableTriangle, float]] = []
    for surface in geometry.walkable_triangles:
        triangle = np.asarray(
            [geometry.vertices[index] for index in surface.triangle.vertex_indices],
            dtype=np.float64,
        )
        if not _point_in_triangle_xz(scene_point, triangle):
            continue
        support_y = _support_y(scene_point, triangle, surface.triangle.normal)
        vertical_error = abs(float(scene_point[1]) - support_y)
        room_preference = 0 if surface.authority_group.startswith("room_") else 1
        candidates.append((vertical_error, room_preference, surface, support_y))

    if not candidates:
        boundary_scene = _nearest_boundary_distance(geometry, scene_point)
        boundary_m = boundary_scene / similarity.scene_units_per_meter
        status = (
            CheckStatus.WARNING.value
            if math.isfinite(boundary_m) and boundary_m <= float(thresholds.outside_warning_m)
            else CheckStatus.FAIL.value
        )
        return OracleSampleResult(
            sample_id=sample_id,
            camera_id=camera_id,
            journal_sequence=journal_sequence,
            observed_at_us=observed_at_us,
            status=status,
            world_point_m=world_point,
            scene_point=scene_point,
            world_source=world_source,
            zone_label=zone_label,
            zone_source=zone_source,
            zone_authoritative=zone_authoritative,
            expected_room=expected_room,
            contained=False,
            support_group=None,
            support_y_scene=None,
            vertical_support_scene=None,
            vertical_support_m=None,
            signed_boundary_distance_scene=-boundary_scene if math.isfinite(boundary_scene) else None,
            signed_boundary_distance_m=-boundary_m if math.isfinite(boundary_m) else None,
            expected_room_groups=(
                tuple(str(group) for group in expected_room_groups)
                if expected_room_groups is not None
                else None
            ),
            expected_room_contained=False if expected_room_groups else None,
            expected_room_boundary_distance_m=None,
            first_divergence={
                "stage": "authored_floor_containment",
                "reason": "outside_walkable_surface",
                "failure_type": FailureType.SEMANTIC.value,
                "outside_distance_m": boundary_m if math.isfinite(boundary_m) else None,
                "warning_band_m": float(thresholds.outside_warning_m),
            },
        )

    vertical_scene, _, surface, support_y = min(candidates, key=lambda row: (row[0], row[1]))
    vertical_m = vertical_scene / similarity.scene_units_per_meter
    boundary_scene = _nearest_boundary_distance(
        geometry, scene_point, authority_group=surface.authority_group
    )
    boundary_m = boundary_scene / similarity.scene_units_per_meter
    if vertical_m <= float(thresholds.support_good_m):
        status = CheckStatus.PASS.value
        divergence = None
    elif vertical_m <= float(thresholds.support_fail_m):
        status = CheckStatus.WARNING.value
        divergence = {
            "stage": "authored_floor_vertical_support",
            "reason": "vertical_support_outside_good_band",
            "failure_type": FailureType.PROJECTION.value,
            "vertical_support_m": vertical_m,
            "good_max_m": float(thresholds.support_good_m),
            "fail_max_m": float(thresholds.support_fail_m),
        }
    else:
        status = CheckStatus.FAIL.value
        divergence = {
            "stage": "authored_floor_vertical_support",
            "reason": "vertical_support_exceeds_limit",
            "failure_type": FailureType.PROJECTION.value,
            "vertical_support_m": vertical_m,
            "fail_max_m": float(thresholds.support_fail_m),
        }

    semantic_groups = (
        tuple(dict.fromkeys(str(group) for group in expected_room_groups if str(group)))
        if expected_room_groups is not None
        else None
    )
    expected_room_contained: bool | None = None
    expected_room_boundary_m: float | None = None
    if semantic_groups:
        semantic_surfaces = [
            row for row in geometry.walkable_triangles if row.authority_group in semantic_groups
        ]
        expected_room_contained = any(
            _point_in_triangle_xz(
                scene_point,
                np.asarray(
                    [geometry.vertices[index] for index in row.triangle.vertex_indices],
                    dtype=np.float64,
                ),
            )
            for row in semantic_surfaces
        )
        semantic_boundary_scene = min(
            (
                _nearest_boundary_distance(
                    geometry, scene_point, authority_group=authority_group
                )
                for authority_group in semantic_groups
            ),
            default=math.inf,
        )
        if math.isfinite(semantic_boundary_scene):
            expected_room_boundary_m = semantic_boundary_scene / similarity.scene_units_per_meter
        if divergence is None and not expected_room_contained:
            status = (
                CheckStatus.WARNING.value
                if expected_room_boundary_m is not None
                and expected_room_boundary_m <= float(thresholds.outside_warning_m)
                else CheckStatus.FAIL.value
            )
            divergence = {
                "stage": "expected_room_semantic_containment",
                "reason": "outside_expected_room_surface",
                "failure_type": FailureType.SEMANTIC.value,
                "expected_room": expected_room,
                "expected_room_groups": list(semantic_groups),
                "outside_distance_m": expected_room_boundary_m,
                "warning_band_m": float(thresholds.outside_warning_m),
            }
    return OracleSampleResult(
        sample_id=sample_id,
        camera_id=camera_id,
        journal_sequence=journal_sequence,
        observed_at_us=observed_at_us,
        status=status,
        world_point_m=world_point,
        scene_point=scene_point,
        world_source=world_source,
        zone_label=zone_label,
        zone_source=zone_source,
        zone_authoritative=zone_authoritative,
        expected_room=expected_room,
        contained=True,
        support_group=surface.authority_group,
        support_y_scene=support_y,
        vertical_support_scene=vertical_scene,
        vertical_support_m=vertical_m,
        signed_boundary_distance_scene=boundary_scene if math.isfinite(boundary_scene) else None,
        signed_boundary_distance_m=boundary_m if math.isfinite(boundary_m) else None,
        expected_room_groups=semantic_groups,
        expected_room_contained=expected_room_contained,
        expected_room_boundary_distance_m=expected_room_boundary_m,
        first_divergence=divergence,
    )


def _unavailable_result(observation: JournalObservation) -> OracleSampleResult:
    invalid = observation.unavailable_reason == "world_position_invalid"
    return OracleSampleResult(
        sample_id=observation.sample_id,
        camera_id=observation.camera_id,
        journal_sequence=observation.journal_sequence,
        observed_at_us=observation.observed_at_us,
        status=CheckStatus.FAIL.value if invalid else CheckStatus.BLOCKED.value,
        world_point_m=None,
        scene_point=None,
        world_source=observation.world_source,
        zone_label=observation.zone_label,
        zone_source=observation.zone_source,
        zone_authoritative=observation.zone_authoritative,
        expected_room=observation.expected_room,
        contained=None,
        support_group=None,
        support_y_scene=None,
        vertical_support_scene=None,
        vertical_support_m=None,
        signed_boundary_distance_scene=None,
        signed_boundary_distance_m=None,
        expected_room_groups=None,
        expected_room_contained=None,
        expected_room_boundary_distance_m=None,
        first_divergence={
            "stage": "producer_world_position",
            "reason": observation.unavailable_reason or "world_position_unavailable",
            "failure_type": (
                FailureType.PROJECTION.value if invalid else FailureType.DATA_QUALITY.value
            ),
        },
    )


def read_person_observations(
    journal_path: str | Path,
    *,
    cameras: Iterable[str] = (),
    limit: int = 0,
) -> list[JournalObservation]:
    path = Path(journal_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"world journal is not a regular file: {path}")
    camera_filter = {str(camera).strip() for camera in cameras if str(camera).strip()}
    effective_limit = max(0, int(limit))
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(records)").fetchall()
        }
        required = {"sequence", "contract", "payload_json"}
        if not required.issubset(columns):
            raise ValueError("world journal records table does not match the required contract")
        # A SQL LIMIT is valid only when no camera filter follows it.  When a
        # camera subset is requested, stream all rows through a bounded deque
        # so ``limit`` means the newest matching observations rather than the
        # newest observations from unrelated cameras.
        if effective_limit > 0 and not camera_filter:
            rows = connection.execute(
                "SELECT sequence, payload_json FROM ("
                "SELECT sequence, payload_json FROM records WHERE contract = ? "
                "ORDER BY sequence DESC LIMIT ?"
                ") ORDER BY sequence ASC",
                ("noesis.observation.person", effective_limit),
            )
        else:
            rows = connection.execute(
                "SELECT sequence, payload_json FROM records WHERE contract = ? ORDER BY sequence ASC",
                ("noesis.observation.person",),
            )
        observations: list[JournalObservation] | deque[JournalObservation]
        observations = (
            deque(maxlen=effective_limit)
            if camera_filter and effective_limit > 0
            else []
        )
        for raw_sequence, payload_json in rows:
            try:
                payload = json.loads(payload_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid observation JSON at journal sequence {raw_sequence}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"observation is not an object at journal sequence {raw_sequence}")
            body = payload.get("payload") if isinstance(payload.get("payload"), Mapping) else {}
            tracklet = body.get("tracklet") if isinstance(body.get("tracklet"), Mapping) else {}
            camera_id = str(tracklet.get("camera_id") or body.get("camera_id") or "unknown")
            if camera_filter and camera_id not in camera_filter:
                continue
            world = body.get("world")
            point: tuple[float, float, float] | None = None
            unavailable_reason: str | None = None
            world_source: str | None = None
            world_quality: str | None = None
            zone_label = str(body.get("zone")) if body.get("zone") else None
            zone_source = (
                str(body.get("zone_source")) if body.get("zone_source") else None
            )
            zone_authoritative = body.get("zone_authoritative") is True
            expected_room = (
                zone_label
                if zone_authoritative and zone_source == "nvdsanalytics_roi"
                else None
            )
            if world is None:
                unavailable_reason = "world_position_unavailable"
            elif isinstance(world, Mapping):
                point = _finite_point3(world.get("position"))
                world_source = str(world.get("source")) if world.get("source") else None
                world_quality = str(world.get("quality")) if world.get("quality") else None
                if point is None:
                    unavailable_reason = "world_position_invalid"
            else:
                unavailable_reason = "world_position_invalid"
            observed_at_raw = payload.get("observed_at_us")
            try:
                observed_at_us = int(observed_at_raw) if observed_at_raw is not None else None
            except (TypeError, ValueError):
                observed_at_us = None
            sequence = int(raw_sequence)
            observations.append(
                JournalObservation(
                    journal_sequence=sequence,
                    sample_id=f"journal:{sequence}",
                    camera_id=camera_id,
                    observed_at_us=observed_at_us,
                    world_point_m=point,
                    world_source=world_source,
                    world_quality=world_quality,
                    zone_label=zone_label,
                    zone_source=zone_source,
                    zone_authoritative=zone_authoritative,
                    expected_room=expected_room,
                    unavailable_reason=unavailable_reason,
                )
            )
        return list(observations)
    finally:
        connection.close()


_SIMILARITY_PATHS: tuple[tuple[str, ...], ...] = (
    ("world_to_scene_col_major",),
    ("world_to_menon_scene_col_major",),
    ("scene_similarity", "world_to_scene_col_major"),
    ("align", "scene_similarity", "world_to_scene_col_major"),
    ("pose_correction", "world_to_menon_scene_col_major"),
)


def _mapping_path(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def load_similarity(path: str | Path) -> SimilarityTransform:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("similarity JSON must contain an object")
    for key_path in _SIMILARITY_PATHS:
        candidate = _mapping_path(payload, key_path)
        if (
            isinstance(candidate, Sequence)
            and not isinstance(candidate, (str, bytes, bytearray))
            and len(candidate) == 16
        ):
            return SimilarityTransform.from_col_major(candidate)
    raise ValueError(
        "similarity JSON does not contain a supported world-to-scene column-major matrix"
    )


def _worst_sample_status(results: Sequence[OracleSampleResult]) -> str:
    if not results:
        return CheckStatus.BLOCKED.value
    return max((result.status for result in results), key=lambda status: _STATUS_ORDER[status])


def _sample_status_counts(results: Sequence[OracleSampleResult]) -> dict[str, int]:
    counts = Counter(result.status for result in results)
    return {
        status.value: int(counts.get(status.value, 0))
        for status in CheckStatus
        if status != CheckStatus.SKIPPED
    }


def _physical_status(
    result: OracleSampleResult, thresholds: GeometryThresholds
) -> str:
    if result.world_point_m is None:
        return result.status
    if result.contained is False:
        outside_m = (
            abs(float(result.signed_boundary_distance_m))
            if result.signed_boundary_distance_m is not None
            else math.inf
        )
        return (
            CheckStatus.WARNING.value
            if outside_m <= float(thresholds.outside_warning_m)
            else CheckStatus.FAIL.value
        )
    support_m = (
        float(result.vertical_support_m)
        if result.vertical_support_m is not None
        else math.inf
    )
    if support_m <= float(thresholds.support_good_m):
        return CheckStatus.PASS.value
    if support_m <= float(thresholds.support_fail_m):
        return CheckStatus.WARNING.value
    return CheckStatus.FAIL.value


def _camera_summary(
    results: Sequence[OracleSampleResult], thresholds: GeometryThresholds
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[OracleSampleResult]] = defaultdict(list)
    for result in results:
        grouped[result.camera_id].append(result)
    summary: dict[str, dict[str, Any]] = {}
    for camera_id, rows in sorted(grouped.items()):
        world_rows = [row for row in rows if row.world_point_m is not None]
        contained = sum(row.contained is True for row in world_rows)
        vertical_values = sorted(
            float(row.vertical_support_m)
            for row in world_rows
            if row.vertical_support_m is not None and math.isfinite(float(row.vertical_support_m))
        )
        physical_counts = Counter(_physical_status(row, thresholds) for row in world_rows)
        semantic_rows = [
            row for row in world_rows if row.expected_room_contained is not None
        ]
        summary[camera_id] = {
            "status": _worst_sample_status(rows),
            "observation_count": len(rows),
            "world_point_count": len(world_rows),
            "world_coverage": len(world_rows) / len(rows) if rows else 0.0,
            "contained_count": contained,
            "containment_rate": contained / len(world_rows) if world_rows else None,
            "status_counts": _sample_status_counts(rows),
            "authored_floor_status_counts": {
                status.value: int(physical_counts.get(status.value, 0))
                for status in (CheckStatus.PASS, CheckStatus.WARNING, CheckStatus.FAIL)
            },
            "declared_expected_rooms": sorted(
                {str(row.expected_room) for row in rows if row.expected_room}
            ),
            "expected_room_evaluated_count": len(semantic_rows),
            "expected_room_contained_count": sum(
                row.expected_room_contained is True for row in semantic_rows
            ),
            "vertical_support_max_m": max(vertical_values) if vertical_values else None,
            "vertical_support_p95_m": (
                float(np.percentile(vertical_values, 95)) if vertical_values else None
            ),
        }
    return summary


def _placement_check(
    camera_id: str,
    rows: Sequence[OracleSampleResult],
    thresholds: GeometryThresholds,
) -> ValidationCheck:
    world_rows = [row for row in rows if row.world_point_m is not None]
    if not world_rows:
        return ValidationCheck(
            id=f"MENON.authored_floor_placement.{camera_id}",
            domain="menon",
            name="authored_floor_placement",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.DATA_QUALITY,
            camera=camera_id,
            metric={"world_point_count": 0},
            detail="No canonical world positions were available for authored-scene placement validation.",
            suggested_next_diagnostic="Inspect producer floor/depth rejection evidence for this camera.",
        )
    physical_statuses = [_physical_status(row, thresholds) for row in world_rows]
    worst = max(physical_statuses, key=lambda status: _STATUS_ORDER[status])
    failures = sum(status == CheckStatus.FAIL.value for status in physical_statuses)
    warnings = sum(status == CheckStatus.WARNING.value for status in physical_statuses)
    contained = sum(row.contained is True for row in world_rows)
    return ValidationCheck(
        id=f"MENON.authored_floor_placement.{camera_id}",
        domain="menon",
        name="authored_floor_placement",
        status=worst,
        failure_type=(
            None if worst == CheckStatus.PASS.value else FailureType.SEMANTIC
        ),
        camera=camera_id,
        metric={
            "world_point_count": len(world_rows),
            "contained_count": contained,
            "containment_rate": contained / len(world_rows),
            "failure_count": failures,
            "warning_count": warnings,
        },
        detail=(
            "Canonical world positions agree with authored walkable surfaces."
            if worst == CheckStatus.PASS.value
            else "One or more canonical world positions diverge from authored walkable surfaces."
        ),
        suggested_next_diagnostic=(
            None
            if worst == CheckStatus.PASS.value
            else "Inspect the first-divergence row, camera pose, and scene similarity before changing presentation smoothing."
        ),
    )


def _normalize_room_label(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _normalized_room_group_map(
    room_group_map: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    if room_group_map is None:
        return normalized
    for label, raw_groups in room_group_map.items():
        if isinstance(raw_groups, (str, bytes, bytearray)) or not isinstance(raw_groups, Sequence):
            raise ValueError(f"room group mapping for {label!r} must be a sequence")
        key = _normalize_room_label(str(label))
        groups = tuple(dict.fromkeys(str(group).strip() for group in raw_groups if str(group).strip()))
        if not key or not groups:
            raise ValueError(f"room group mapping for {label!r} is empty")
        normalized[key] = groups
    return normalized


def _expected_room_check(
    camera_id: str,
    rows: Sequence[OracleSampleResult],
    *,
    room_mapping_provided: bool,
) -> ValidationCheck:
    declared = [row for row in rows if row.expected_room]
    if not declared:
        return ValidationCheck(
            id=f"SEMANTIC.expected_room_containment.{camera_id}",
            domain="semantic",
            name="expected_room_containment",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.SEMANTIC,
            camera=camera_id,
            metric={"declared_room_sample_count": 0},
            detail="The journal does not declare an expected room for this camera's observations.",
            suggested_next_diagnostic="Preserve a producer-owned room label or supply an explicit room authority map.",
        )
    evaluated = [row for row in declared if row.expected_room_contained is not None]
    if not room_mapping_provided or len(evaluated) != len(declared):
        missing_rooms = sorted(
            {
                str(row.expected_room)
                for row in declared
                if row.expected_room_contained is None
            }
        )
        return ValidationCheck(
            id=f"SEMANTIC.expected_room_containment.{camera_id}",
            domain="semantic",
            name="expected_room_containment",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.SEMANTIC,
            camera=camera_id,
            metric={
                "declared_room_sample_count": len(declared),
                "evaluated_sample_count": len(evaluated),
                "unmapped_room_labels": missing_rooms,
            },
            detail=(
                "Generic authored-floor support is available, but exact room labels cannot be "
                "derived from opaque OBJ room group IDs without an explicit authority map."
            ),
            suggested_next_diagnostic=(
                "Provide a reviewed room-label-to-OBJ-group JSON mapping; do not infer it from camera IDs."
            ),
        )
    outside = [row for row in evaluated if row.expected_room_contained is False]
    warnings = sum(row.status == CheckStatus.WARNING.value for row in outside)
    failures = sum(row.status == CheckStatus.FAIL.value for row in outside)
    status = (
        CheckStatus.FAIL
        if failures
        else CheckStatus.WARNING
        if warnings or outside
        else CheckStatus.PASS
    )
    return ValidationCheck(
        id=f"SEMANTIC.expected_room_containment.{camera_id}",
        domain="semantic",
        name="expected_room_containment",
        status=status,
        failure_type=None if status == CheckStatus.PASS else FailureType.SEMANTIC,
        camera=camera_id,
        metric={
            "evaluated_sample_count": len(evaluated),
            "contained_count": len(evaluated) - len(outside),
            "outside_count": len(outside),
        },
        detail=(
            "Track positions remain within their explicitly mapped authored room surfaces."
            if status == CheckStatus.PASS
            else "One or more tracks are outside their explicitly mapped authored room surfaces."
        ),
        suggested_next_diagnostic=(
            None
            if status == CheckStatus.PASS
            else "Inspect source-camera rays, room transitions, and the reviewed room authority map."
        ),
    )


def build_journal_oracle_report(
    *,
    journal_path: str | Path,
    obj_path: str | Path,
    similarity_path: str | Path,
    run_id: str,
    cameras: Iterable[str] = (),
    journal_limit: int = 0,
    max_divergences: int = 100,
    thresholds: GeometryThresholds = GeometryThresholds(),
    horizontal_tolerance_deg: float = 15.0,
    room_group_map: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    geometry = AuthoredSceneGeometry.from_obj(
        obj_path, horizontal_tolerance_deg=horizontal_tolerance_deg
    )
    similarity = load_similarity(similarity_path)
    observations = read_person_observations(
        journal_path, cameras=cameras, limit=journal_limit
    )
    normalized_room_map = _normalized_room_group_map(room_group_map)
    authored_groups = {row.authority_group for row in geometry.walkable_triangles}
    unknown_mapped_groups = sorted(
        {
            group
            for groups in normalized_room_map.values()
            for group in groups
            if group not in authored_groups
        }
    )
    results: list[OracleSampleResult] = []
    for observation in observations:
        if observation.world_point_m is None:
            results.append(_unavailable_result(observation))
            continue
        results.append(
            score_world_point(
                geometry=geometry,
                similarity=similarity,
                world_point_m=observation.world_point_m,
                thresholds=thresholds,
                sample_id=observation.sample_id,
                camera_id=observation.camera_id,
                journal_sequence=observation.journal_sequence,
                observed_at_us=observation.observed_at_us,
                world_source=observation.world_source,
                zone_label=observation.zone_label,
                zone_source=observation.zone_source,
                zone_authoritative=observation.zone_authoritative,
                expected_room=observation.expected_room,
                expected_room_groups=(
                    normalized_room_map.get(_normalize_room_label(observation.expected_room))
                    if observation.expected_room
                    else None
                ),
            )
        )

    checks: list[ValidationCheck] = [
        ValidationCheck(
            id="SCENE.authored_walkable_inventory",
            domain="scene",
            name="authored_walkable_inventory",
            status=(
                CheckStatus.PASS if geometry.walkable_triangles else CheckStatus.FAIL
            ),
            failure_type=(None if geometry.walkable_triangles else FailureType.SCENE),
            metric={
                "walkable_triangle_count": len(geometry.walkable_triangles),
                "walkable_group_count": len(
                    {row.authority_group for row in geometry.walkable_triangles}
                ),
            },
            detail=(
                "Authored OBJ contains upward-facing ground/room floor surfaces."
                if geometry.walkable_triangles
                else "Authored OBJ contains no authoritative walkable floor surfaces."
            ),
            suggested_next_diagnostic=(
                None
                if geometry.walkable_triangles
                else "Inspect promoted OBJ group names, winding, and floor normals."
            ),
        ),
        ValidationCheck(
            id="TRANSFORM.world_to_scene_similarity",
            domain="transform",
            name="world_to_scene_similarity",
            status=CheckStatus.PASS,
            metric={
                "scene_units_per_meter": similarity.scene_units_per_meter,
                "relative_scale_spread": similarity.relative_scale_spread,
                "determinant": similarity.determinant,
            },
            detail="The exact column-major transform is an affine, right-handed uniform similarity.",
        ),
        ValidationCheck(
            id="SCENE.expected_room_authority",
            domain="scene",
            name="expected_room_authority",
            status=(
                CheckStatus.FAIL
                if unknown_mapped_groups
                else CheckStatus.PASS
                if normalized_room_map
                else CheckStatus.BLOCKED
            ),
            failure_type=(
                FailureType.SCENE
                if unknown_mapped_groups
                else FailureType.SEMANTIC
                if not normalized_room_map
                else None
            ),
            metric={
                "mapping_provided": bool(normalized_room_map),
                "mapped_room_count": len(normalized_room_map),
                "unknown_obj_groups": unknown_mapped_groups,
            },
            detail=(
                "The explicit semantic room map references authored walkable OBJ groups."
                if normalized_room_map and not unknown_mapped_groups
                else "The semantic room map references unknown authored OBJ groups."
                if unknown_mapped_groups
                else "OBJ room group IDs are opaque; generic floor support does not prove expected-room containment."
            ),
            suggested_next_diagnostic=(
                None
                if normalized_room_map and not unknown_mapped_groups
                else "Create or correct a reviewed room-label-to-OBJ-group authority artifact."
            ),
        ),
        ValidationCheck(
            id="MENON.rendered_track_placement",
            domain="menon",
            name="rendered_track_placement",
            status=CheckStatus.BLOCKED,
            failure_type=FailureType.SYNC,
            metric={"rendered_sample_count": 0},
            detail=(
                "A Noesis world journal contains no Menon committed/rendered positions; "
                "authored-floor support cannot prove browser placement."
            ),
            suggested_next_diagnostic=(
                "Join this report with a canonical Menon browser trace carrying entity, world, committed-scene, and rendered positions."
            ),
        ),
    ]
    if not results:
        checks.append(
            ValidationCheck(
                id="TRACK.journal_observations",
                domain="tracking",
                name="journal_observations",
                status=CheckStatus.BLOCKED,
                failure_type=FailureType.DATA_QUALITY,
                metric={"observation_count": 0},
                detail="The selected journal window contains no person observations.",
                suggested_next_diagnostic="Capture an occupied journal interval or expand the selected window.",
            )
        )
    grouped: dict[str, list[OracleSampleResult]] = defaultdict(list)
    for result in results:
        grouped[result.camera_id].append(result)
    for camera_id, rows in sorted(grouped.items()):
        world_count = sum(row.world_point_m is not None for row in rows)
        coverage = world_count / len(rows) if rows else 0.0
        coverage_status = (
            CheckStatus.BLOCKED
            if world_count == 0
            else CheckStatus.PASS
            if coverage >= 0.90
            else CheckStatus.WARNING
        )
        checks.append(
            ValidationCheck(
                id=f"TRACK.world_geometry_coverage.{camera_id}",
                domain="tracking",
                name="world_geometry_coverage",
                status=coverage_status,
                failure_type=(
                    None if coverage_status == CheckStatus.PASS else FailureType.DATA_QUALITY
                ),
                camera=camera_id,
                metric={
                    "observation_count": len(rows),
                    "world_point_count": world_count,
                    "world_coverage": coverage,
                },
                threshold={"good_minimum": 0.90},
                detail=(
                    "Canonical world-position coverage meets the validation band."
                    if coverage_status == CheckStatus.PASS
                    else "Canonical world-position coverage is incomplete for this camera."
                ),
                suggested_next_diagnostic=(
                    None
                    if coverage_status == CheckStatus.PASS
                    else "Inspect producer rejection reasons before evaluating Menon presentation behavior."
                ),
            )
        )
        checks.append(_placement_check(camera_id, rows, thresholds))
        checks.append(
            _expected_room_check(
                camera_id,
                rows,
                room_mapping_provided=bool(normalized_room_map),
            )
        )

    pass_count = sum(result.status == CheckStatus.PASS.value for result in results)
    world_count = sum(result.world_point_m is not None for result in results)
    geometry_confidence = pass_count / world_count if world_count else 0.0
    end_to_end_confidence = pass_count / len(results) if results else 0.0
    validation_report = ValidationReport(
        run_id=str(run_id),
        created_at=_utc_now_iso(),
        source=SourceMetadata(repo="Noesis_Devel"),
        scope={
            "tiers": ["offline", "persisted_journal", "authored_scene"],
            "cameras": sorted(grouped),
            "observation_count": len(results),
            "world_point_count": world_count,
        },
        confidence=ConfidenceScores(
            projection=geometry_confidence,
            geometry=geometry_confidence,
            semantic=geometry_confidence,
            end_to_end=end_to_end_confidence,
        ),
        checks=checks,
    )
    report = validation_report.to_dict()

    divergences = [result for result in results if result.first_divergence is not None]
    physical_divergences = [
        result
        for result in divergences
        if str(result.first_divergence.get("stage", "")).startswith("authored_floor")
    ]
    physical_statuses = [
        _physical_status(result, thresholds)
        for result in results
        if result.world_point_m is not None
    ]
    effective_transform_status = (
        max(physical_statuses, key=lambda status: _STATUS_ORDER[status])
        if physical_statuses
        else CheckStatus.BLOCKED.value
    )
    report["oracle"] = {
        "contract": ORACLE_CONTRACT,
        "contract_version": ORACLE_CONTRACT_VERSION,
        "thresholds": thresholds.to_dict(),
        "scene": geometry.inventory(),
        "similarity": {
            **similarity.to_dict(),
            "path": str(Path(similarity_path).expanduser().resolve()),
            "sha256": _sha256_file(Path(similarity_path).expanduser().resolve()),
        },
        "journal": {
            "path": str(Path(journal_path).expanduser().resolve()),
            "sha256": _sha256_file(Path(journal_path).expanduser().resolve()),
            "limit": int(journal_limit),
        },
        "sample_status_counts": _sample_status_counts(results),
        "camera_summary": _camera_summary(results, thresholds),
        "room_semantics": {
            "mapping_provided": bool(normalized_room_map),
            "mapping": {
                key: list(groups) for key, groups in sorted(normalized_room_map.items())
            },
            "unknown_obj_groups": unknown_mapped_groups,
            "limitation": (
                None
                if normalized_room_map
                else "Opaque room_* OBJ group IDs are not equivalent to household room labels; expected-room containment is blocked."
            ),
        },
        "first_divergence": divergences[0].to_dict() if divergences else None,
        "first_physical_divergence": (
            physical_divergences[0].to_dict() if physical_divergences else None
        ),
        "divergence_count": len(divergences),
        "divergences_truncated": len(divergences) > max(0, int(max_divergences)),
        "divergences": [
            result.to_dict() for result in divergences[: max(0, int(max_divergences))]
        ],
        "stage_first_divergence": {
            "producer_world": next(
                (
                    result.to_dict()
                    for result in divergences
                    if result.first_divergence
                    and result.first_divergence.get("stage") == "producer_world_position"
                ),
                None,
            ),
            "world_to_scene_transform": {
                "status": effective_transform_status,
                "structural_status": CheckStatus.PASS.value,
                "first_divergence": (
                    physical_divergences[0].to_dict() if physical_divergences else None
                ),
                "evaluated_world_point_count": world_count,
                "detail": (
                    "The declared matrix is a valid right-handed uniform similarity and each "
                    "finite backend_world_m point was transformed exactly once. Status also "
                    "reflects whether the transformed output lands on authored walkable geometry."
                ),
                "causal_limitation": (
                    "A geometric divergence proves the effective producer-plus-transform output "
                    "is invalid; independent surveyed anchors or source rays are required to "
                    "attribute it uniquely to producer world geometry or calibration."
                    if physical_divergences
                    else None
                ),
            },
            "authored_floor_support": (
                physical_divergences[0].to_dict() if physical_divergences else None
            ),
            "expected_room_semantic_containment": next(
                (
                    result.to_dict()
                    for result in divergences
                    if result.first_divergence
                    and result.first_divergence.get("stage")
                    == "expected_room_semantic_containment"
                ),
                {
                    "status": CheckStatus.BLOCKED.value,
                    "reason": "room_label_to_obj_group_authority_unavailable",
                    "detail": "Generic authored-floor support does not establish expected-room containment.",
                }
                if not normalized_room_map
                else None,
            ),
            "menon_rendered_placement": {
                "status": CheckStatus.BLOCKED.value,
                "first_divergence": None,
                "reason": "menon_render_evidence_unavailable_in_world_journal",
                "detail": "A canonical Menon browser trace is required for this stage.",
            },
        },
    }
    return report


__all__ = [
    "AuthoredSceneGeometry",
    "BoundaryEdge",
    "GeometryThresholds",
    "JournalObservation",
    "ObjTriangle",
    "OracleSampleResult",
    "ORACLE_CONTRACT",
    "ORACLE_CONTRACT_VERSION",
    "SimilarityTransform",
    "WalkableTriangle",
    "build_journal_oracle_report",
    "load_similarity",
    "read_person_observations",
    "score_world_point",
]
