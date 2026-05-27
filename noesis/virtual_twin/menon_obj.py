from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StructuralSurface:
    surface_id: str
    label: str
    normal: np.ndarray
    centroid: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    area: float
    vertex_count: int
    material: str | None = None
    object_name: str | None = None
    triangles: np.ndarray | None = None


def _classify_obj_surface(normal: np.ndarray, name: str | None, material: str | None) -> str:
    text = f"{name or ''} {material or ''}".lower()
    if "floor" in text:
        return "floor"
    if "wall" in text:
        return "wall"
    if "ceil" in text:
        return "ceiling"
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-9:
        return "unknown"
    n = n / norm
    axis = int(np.argmax(np.abs(n)))
    # SweetHome/Three scenes in Menon commonly use Y as vertical, matching backend_world_m.
    if axis == 1:
        return "floor_or_ceiling"
    return "wall"


def _triangulate(face: list[int]) -> list[tuple[int, int, int]]:
    if len(face) < 3:
        return []
    return [(face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)]


def _surface_from_faces(
    *,
    surface_id: str,
    vertices: np.ndarray,
    faces: list[list[int]],
    object_name: str | None,
    material: str | None,
) -> StructuralSurface | None:
    weighted_normal = np.zeros(3, dtype=np.float64)
    weighted_centroid = np.zeros(3, dtype=np.float64)
    area_total = 0.0
    used_vertices: set[int] = set()
    triangles: list[np.ndarray] = []
    for face in faces:
        for tri in _triangulate(face):
            p0, p1, p2 = vertices[list(tri)]
            normal = np.cross(p1 - p0, p2 - p0)
            area = float(np.linalg.norm(normal) * 0.5)
            if area <= 1e-12:
                continue
            tri_normal = normal / (2.0 * area)
            centroid = (p0 + p1 + p2) / 3.0
            weighted_normal += tri_normal * area
            weighted_centroid += centroid * area
            area_total += area
            used_vertices.update(tri)
            triangles.append(np.stack([p0, p1, p2], axis=0))
    if area_total <= 1e-12:
        return None
    used = vertices[sorted(used_vertices)]
    normal = weighted_normal / max(float(np.linalg.norm(weighted_normal)), 1e-12)
    centroid = weighted_centroid / area_total
    label = _classify_obj_surface(normal, object_name, material)
    return StructuralSurface(
        surface_id=surface_id,
        label=label,
        normal=normal.astype(np.float32),
        centroid=centroid.astype(np.float32),
        bounds_min=np.min(used, axis=0).astype(np.float32),
        bounds_max=np.max(used, axis=0).astype(np.float32),
        area=area_total,
        vertex_count=len(used_vertices),
        material=material,
        object_name=object_name,
        triangles=np.asarray(triangles, dtype=np.float32) if triangles else None,
    )


def parse_menon_obj_surfaces(path: Path, *, min_area: float = 0.05) -> list[StructuralSurface]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Menon structural OBJ missing: {source}")
    vertices: list[list[float]] = []
    groups: list[tuple[str | None, str | None, list[list[int]]]] = []
    object_name: str | None = None
    material: str | None = None
    faces: list[list[int]] = []

    def flush() -> None:
        nonlocal faces
        if faces:
            groups.append((object_name, material, faces))
            faces = []

    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            head, _, rest = line.partition(" ")
            if head == "v":
                parts = rest.split()
                if len(parts) >= 3:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif head in {"o", "g"}:
                flush()
                object_name = rest.strip() or None
            elif head == "usemtl":
                flush()
                material = rest.strip() or None
            elif head == "f":
                idxs: list[int] = []
                for token in rest.split():
                    raw_idx = token.split("/")[0]
                    if not raw_idx:
                        continue
                    idx = int(raw_idx)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1
                    idxs.append(idx)
                if len(idxs) >= 3:
                    faces.append(idxs)
    flush()

    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
        raise ValueError(f"Menon structural OBJ has no valid vertices: {source}")
    surfaces: list[StructuralSurface] = []
    for idx, (obj_name, mat, group_faces) in enumerate(groups):
        surface = _surface_from_faces(
            surface_id=f"surface_{idx:04d}",
            vertices=verts,
            faces=group_faces,
            object_name=obj_name,
            material=mat,
        )
        if surface is None or float(surface.area) < float(min_area):
            continue
        surfaces.append(surface)
    return surfaces


def surface_to_json(surface: StructuralSurface) -> dict[str, object]:
    return {
        "surface_id": surface.surface_id,
        "label": surface.label,
        "normal": [float(x) for x in surface.normal],
        "centroid": [float(x) for x in surface.centroid],
        "bounds_min": [float(x) for x in surface.bounds_min],
        "bounds_max": [float(x) for x in surface.bounds_max],
        "area": float(surface.area),
        "vertex_count": int(surface.vertex_count),
        "triangle_count": int(surface.triangles.shape[0]) if surface.triangles is not None else 0,
        "material": surface.material,
        "object_name": surface.object_name,
    }


__all__ = ["StructuralSurface", "parse_menon_obj_surfaces", "surface_to_json"]
