from __future__ import annotations

from pathlib import Path

import pytest

from noesis.calibration.authored_structures import extract_authored_structures
from noesis.validation.authored_scene import AuthoredSceneGeometry


def _write_obj(path: Path, faces: list[tuple[str, list[tuple[float, float, float]]]]) -> Path:
    lines: list[str] = []
    vertex_index = 1
    for group, points in faces:
        lines.append(f"g {group}")
        lines.extend(f"v {x} {y} {z}" for x, y, z in points)
        indices = list(range(vertex_index, vertex_index + len(points)))
        lines.append("f " + " ".join(map(str, indices)))
        vertex_index += len(points)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _quad(
    group: str,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    d: tuple[float, float, float],
) -> tuple[str, list[tuple[float, float, float]]]:
    return group, [a, b, c, d]


def _concave_room_faces() -> list[tuple[str, list[tuple[float, float, float]]]]:
    # A clockwise XZ L-shaped floor; the OBJ loader ear-clips this polygon.
    floor = (
        "room_semantic_group_17",
        [(0, 0, 0), (0, 0, 8), (3, 0, 8), (3, 0, 3), (8, 0, 3), (8, 0, 0)],
    )
    walls = [
        # The west wall has a doorway-sized geometric gap, with a header above.
        _quad("arbitrary_wall_material_a", (0, 0, 0), (0, 3, 0), (0, 3, 3), (0, 0, 3)),
        _quad("arbitrary_wall_material_a", (0, 0, 5), (0, 3, 5), (0, 3, 8), (0, 0, 8)),
        _quad("arbitrary_wall_material_a", (0, 2, 3), (0, 3, 3), (0, 3, 5), (0, 2, 5)),
        # A parallel authored face demonstrates that wall thickness stays
        # geometric; the extractor must not snap it to the other wall plane.
        _quad("other_uninformative_group", (0.15, 0, 0), (0.15, 3, 0), (0.15, 3, 3), (0.15, 0, 3)),
        _quad("other_uninformative_group", (0.15, 0, 5), (0.15, 3, 5), (0.15, 3, 8), (0.15, 0, 8)),
        _quad("other_uninformative_group", (0.15, 2, 3), (0.15, 3, 3), (0.15, 3, 5), (0.15, 2, 5)),
        # This tall plane is not associated with the selected floor footprint.
        _quad("unrelated_vertical_object", (20, 0, 0), (20, 3, 0), (20, 3, 8), (20, 0, 8)),
    ]
    return [floor, *walls]


def test_extracts_concave_floor_and_only_spatially_supported_wall_faces(tmp_path: Path) -> None:
    geometry = AuthoredSceneGeometry.from_obj(_write_obj(tmp_path / "concave.obj", _concave_room_faces()))

    result = extract_authored_structures(
        geometry,
        ["room_semantic_group_17"],
        initial_scene_units_per_meter=10.0,
        wall_to_floor_association_m=0.2,
        min_wall_length_m=0.1,
        min_wall_height_m=0.2,
    )

    assert len(result.floor_polygons) == 1
    floor = result.floor_polygons[0]
    assert floor.floor_groups == ("room_semantic_group_17",)
    assert floor.floor_height == pytest.approx(0.0)
    assert floor.height_range == pytest.approx((0.0, 0.0))
    assert floor.area_scene2 == pytest.approx(39.0)
    assert len(floor.boundaries) == 1
    assert len(floor.boundaries[0].points_xz) == 6
    assert result.tolerance_scene_units == pytest.approx(0.1)
    # The hint only sets tolerances: coordinates and floor area are unchanged.
    assert result.floor_polygons[0].boundaries[0].points_xz[0] in {
        (0.0, 0.0), (0.0, 8.0), (3.0, 8.0), (3.0, 3.0), (8.0, 3.0), (8.0, 0.0)
    }

    west_walls = [
        wall
        for wall in result.wall_segments
        if abs(wall.normal_xz[0]) > 0.99 and abs(wall.normal_xz[1]) < 0.01
        and abs(wall.start_xz[0] - wall.end_xz[0]) < 1e-7
    ]
    assert west_walls
    assert sorted({wall.offset for wall in west_walls}) == pytest.approx([0.0, 0.15])
    assert all(wall.normal_xz == pytest.approx((1.0, 0.0)) for wall in west_walls)
    for offset in (0.0, 0.15):
        distances = [wall.floor_boundary_distance_scene for wall in west_walls if abs(wall.offset - offset) < 1e-7]
        assert distances
        assert distances == pytest.approx([offset] * len(distances))
    assert all("unrelated_vertical_object" not in " ".join(wall.source_refs) for wall in west_walls)

    openings = [opening for opening in result.openings if abs(opening.start_xz[0]) < 1e-7]
    assert len(openings) == 1
    assert openings[0].interval == pytest.approx((3.0, 5.0))
    assert openings[0].vertical_span == pytest.approx((0.0, 2.0))

    low_segments = [wall for wall in west_walls if wall.vertical_span == pytest.approx((0.0, 2.0))]
    assert sorted((min(wall.start_xz[1], wall.end_xz[1]), max(wall.start_xz[1], wall.end_xz[1])) for wall in low_segments) == [
        (0.0, 3.0), (0.0, 3.0), (5.0, 8.0), (5.0, 8.0)
    ]


def test_does_not_turn_floor_edges_without_vertical_mesh_into_walls(tmp_path: Path) -> None:
    geometry = AuthoredSceneGeometry.from_obj(
        _write_obj(
            tmp_path / "floor_only.obj",
            [("room_exact", [(0, 0, 0), (0, 0, 4), (4, 0, 4), (4, 0, 0)])],
        )
    )

    result = extract_authored_structures(geometry, ["room_exact"], initial_scene_units_per_meter=1.0)

    assert len(result.floor_polygons) == 1
    assert len(result.floor_polygons[0].boundaries) == 1
    assert result.wall_segments == ()
    assert result.openings == ()


def test_selected_floor_group_union_removes_internal_tiling_seam(tmp_path: Path) -> None:
    geometry = AuthoredSceneGeometry.from_obj(
        _write_obj(
            tmp_path / "tiled_floor.obj",
            [
                ("room_left", [(0, 0, 0), (0, 0, 4), (2, 0, 4), (2, 0, 0)]),
                ("room_right", [(2, 0, 0), (2, 0, 4), (4, 0, 4), (4, 0, 0)]),
            ],
        )
    )

    result = extract_authored_structures(geometry, ["room_left", "room_right"])

    assert len(result.floor_polygons) == 1
    assert result.floor_polygons[0].floor_groups == ("room_left", "room_right")
    assert result.floor_polygons[0].area_scene2 == pytest.approx(16.0)
    assert len(result.floor_polygons[0].boundaries) == 1
    boundary = result.floor_polygons[0].boundaries[0].points_xz
    boundary_edges = set(zip(boundary, boundary[1:] + boundary[:1]))
    assert len(boundary) == 6  # Collinear outer vertices are retained from each source tile.
    assert ((2.0, 0.0), (2.0, 4.0)) not in boundary_edges
    assert ((2.0, 4.0), (2.0, 0.0)) not in boundary_edges
    assert result.source_sha256 == geometry.source_sha256


def test_rejects_nonexistent_exact_floor_group_instead_of_using_room_labels(tmp_path: Path) -> None:
    geometry = AuthoredSceneGeometry.from_obj(_write_obj(tmp_path / "one_room.obj", _concave_room_faces()))

    with pytest.raises(ValueError, match="no upward-facing floor triangles"):
        extract_authored_structures(geometry, ["living-room"])
