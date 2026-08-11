from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from noesis.virtual_twin.menon_obj import StructuralSurface, parse_menon_obj_surfaces
from noesis.virtual_twin.registration import SourcePlaneSurface, register_planes_to_menon_surfaces


def _rotation_y(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _surface(surface_id: str, label: str, centroid: np.ndarray, normal: np.ndarray, area: float) -> StructuralSurface:
    centroid = np.asarray(centroid, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    return StructuralSurface(
        surface_id=surface_id,
        label=label,
        centroid=centroid.astype(np.float32),
        normal=(normal / np.linalg.norm(normal)).astype(np.float32),
        bounds_min=(centroid - 0.5).astype(np.float32),
        bounds_max=(centroid + 0.5).astype(np.float32),
        area=float(area),
        vertex_count=4,
        material=label,
        object_name=surface_id,
    )


def test_menon_obj_parser_extracts_floor_and_wall_surfaces(tmp_path: Path) -> None:
    obj = tmp_path / "room.obj"
    obj.write_text(
        "\n".join(
            [
                "o floor_main",
                "usemtl floor",
                "v 0 0 0",
                "v 4 0 0",
                "v 4 0 3",
                "v 0 0 3",
                "f 1 2 3 4",
                "o wall_north",
                "usemtl wall",
                "v 0 0 3",
                "v 4 0 3",
                "v 4 2.5 3",
                "v 0 2.5 3",
                "f 5 6 7 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    surfaces = parse_menon_obj_surfaces(obj)

    labels = {surface.label for surface in surfaces}
    assert "floor" in labels
    assert "wall" in labels
    assert all(surface.bounds_min.shape == (3,) and surface.bounds_max.shape == (3,) for surface in surfaces)


def test_registration_recovers_known_similarity_transform() -> None:
    scale = 97.5
    rotation = _rotation_y(17.0)
    translation = np.asarray([125.0, -6.0, 430.0], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation

    target_centroids = [
        np.asarray([125.0, -6.0, 430.0], dtype=np.float64),
        np.asarray([420.0, 0.0, 430.0], dtype=np.float64),
        np.asarray([125.0, 0.0, 735.0], dtype=np.float64),
        np.asarray([500.0, 0.0, 810.0], dtype=np.float64),
    ]
    target_normals = [
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    ]
    labels = ["floor", "wall", "wall", "wall"]
    areas = [400.0, 300.0, 200.0, 100.0]
    targets = [
        _surface(f"target_{idx}", label, centroid, normal, area)
        for idx, (label, centroid, normal, area) in enumerate(zip(labels, target_centroids, target_normals, areas))
    ]

    inv_linear = np.linalg.inv(matrix[:3, :3])
    sources = []
    for idx, (label, centroid, normal, area) in enumerate(zip(labels, target_centroids, target_normals, areas)):
        source_centroid = inv_linear @ (centroid - translation)
        source_normal = rotation.T @ normal
        sources.append(
            SourcePlaneSurface(
                plane_id=f"source_{idx}",
                label=label,
                centroid=source_centroid,
                normal=source_normal,
                confidence=float(area),
                support_pixels=int(area),
            )
        )

    result = register_planes_to_menon_surfaces(sources, targets, min_correspondences=4)
    recovered = np.asarray(result["world_to_menon_scene_col_major"], dtype=np.float64).reshape((4, 4), order="F")

    assert result["correspondence_count"] == 4
    assert result["position_rmse_scene_units"] < 1e-3
    assert np.allclose(recovered, matrix, atol=1e-3)
    assert math.isclose(float(result["scene_per_m"]), scale, abs_tol=1e-9)
