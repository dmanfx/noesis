from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from noesis.validation.authored_scene import (
    AuthoredSceneGeometry,
    GeometryThresholds,
    SimilarityTransform,
    build_journal_oracle_report,
    load_similarity,
    read_person_observations,
    score_world_point,
)
from scripts.noesis_validation_authored_scene import _load_room_group_map


FIXTURES = Path(__file__).parent / "fixtures" / "authored_scene_oracle"
OBJ = FIXTURES / "simple_home.obj"
SIMILARITY = FIXTURES / "identity_similarity.json"
ROOM_MAP = FIXTURES / "room_group_map.json"


def _observation(
    *,
    camera: str,
    sequence: int,
    world: list[float] | None,
    room: str,
    zone_source: str = "nvdsanalytics_roi",
    zone_authoritative: bool = True,
) -> dict[str, object]:
    world_payload = (
        {
            "position": {"x": world[0], "y": world[1], "z": world[2]},
            "frame": "backend_world_m",
            "units": "meters",
            "source": "pose_floor_only",
            "quality": "good",
        }
        if world is not None
        else None
    )
    return {
        "contract": "noesis.observation.person",
        "contract_version": 1,
        "observation_id": f"private-id-{sequence}",
        "observed_at_us": 1_000_000 + sequence,
        "payload": {
            "tracklet": {
                "camera_id": camera,
                "source_id": 0,
                "frame_id": sequence,
                "tracker_id": sequence,
            },
            "zone": room,
            "zone_source": zone_source,
            "zone_authoritative": zone_authoritative,
            "world": world_payload,
        },
    }


def _journal(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE records ("
            "sequence INTEGER PRIMARY KEY, recorded_at_us INTEGER NOT NULL, "
            "contract TEXT NOT NULL, payload_json TEXT NOT NULL)"
        )
        rows = [
            _observation(camera="living-room", sequence=1, world=[2.0, 0.05, 2.0], room="Living Room"),
            # Physically supported by the house floor, but semantically in Kitchen.
            _observation(camera="living-room", sequence=2, world=[7.0, 0.05, 2.0], room="Living Room"),
            _observation(camera="living-room", sequence=3, world=None, room="Living Room"),
            _observation(camera="kitchen", sequence=4, world=[7.0, 0.20, 2.0], room="Kitchen"),
            _observation(camera="kitchen", sequence=5, world=[10.10, 0.0, 5.0], room="Kitchen"),
            _observation(camera="kitchen", sequence=6, world=[12.0, 0.0, 5.0], room="Kitchen"),
        ]
        connection.executemany(
            "INSERT INTO records(sequence, recorded_at_us, contract, payload_json) "
            "VALUES (?, ?, ?, ?)",
            [
                (
                    index,
                    1_000_000 + index,
                    "noesis.observation.person",
                    json.dumps(payload, separators=(",", ":")),
                )
                for index, payload in enumerate(rows, start=1)
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def test_obj_loader_triangulates_and_derives_only_upward_authority_surfaces() -> None:
    geometry = AuthoredSceneGeometry.from_obj(OBJ)
    inventory = geometry.inventory()

    assert inventory["triangle_count"] == 12
    assert inventory["walkable_triangle_count"] == 6
    assert inventory["walkable_groups"] == {
        "ground_main": 2,
        "room_kitchen": 2,
        "room_living": 2,
    }
    assert "room_underfloor" not in inventory["walkable_groups"]
    assert inventory["boundary_edge_count"] == 12


def test_obj_loader_ear_clips_concave_negative_index_face(tmp_path: Path) -> None:
    obj = tmp_path / "concave.obj"
    obj.write_text(
        "\n".join(
            [
                "g room_concave",
                "v 0 0 0",
                "v 0 0 2",
                "v 1 0 2",
                "v 1 0 1",
                "v 2 0 1",
                "v 2 0 0",
                "f -6 -5 -4 -3 -2 -1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    geometry = AuthoredSceneGeometry.from_obj(obj)
    similarity = SimilarityTransform.from_col_major(
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    )

    assert len(geometry.walkable_triangles) == 4
    supported = score_world_point(
        geometry=geometry,
        similarity=similarity,
        world_point_m=[0.5, 0.0, 1.5],
    )
    notch = score_world_point(
        geometry=geometry,
        similarity=similarity,
        world_point_m=[1.5, 0.0, 1.5],
    )
    assert supported.status == "pass"
    assert notch.contained is False
    assert notch.first_divergence["stage"] == "authored_floor_containment"


def test_similarity_applies_exact_column_major_scale_rotation_and_translation() -> None:
    similarity = SimilarityTransform.from_col_major(
        [
            0.0,
            0.0,
            -10.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            0.0,
            100.0,
            5.0,
            200.0,
            1.0,
        ]
    )
    assert similarity.scene_units_per_meter == pytest.approx(10.0)
    assert similarity.apply([1.0, 2.0, 3.0]) == pytest.approx((130.0, 25.0, 190.0))

    with pytest.raises(ValueError, match="uniform similarity"):
        SimilarityTransform.from_col_major(
            [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]
        )


def test_floor_support_boundary_and_expected_room_are_separate_metrics() -> None:
    geometry = AuthoredSceneGeometry.from_obj(OBJ)
    similarity = load_similarity(SIMILARITY)
    thresholds = GeometryThresholds()

    valid = score_world_point(
        geometry=geometry,
        similarity=similarity,
        world_point_m=[2.0, 0.05, 2.0],
        thresholds=thresholds,
        expected_room="Living Room",
        expected_room_groups=["room_living"],
    )
    wrong_room = score_world_point(
        geometry=geometry,
        similarity=similarity,
        world_point_m=[7.0, 0.05, 2.0],
        thresholds=thresholds,
        expected_room="Living Room",
        expected_room_groups=["room_living"],
    )
    vertical_warning = score_world_point(
        geometry=geometry,
        similarity=similarity,
        world_point_m=[2.0, 0.20, 2.0],
        thresholds=thresholds,
    )
    near_boundary = score_world_point(
        geometry=geometry,
        similarity=similarity,
        world_point_m=[10.10, 0.0, 5.0],
        thresholds=thresholds,
    )
    far_outside = score_world_point(
        geometry=geometry,
        similarity=similarity,
        world_point_m=[12.0, 0.0, 5.0],
        thresholds=thresholds,
    )

    assert valid.status == "pass"
    assert valid.contained is True
    assert valid.expected_room_contained is True
    assert valid.signed_boundary_distance_m == pytest.approx(2.0)

    assert wrong_room.contained is True
    assert wrong_room.vertical_support_m == pytest.approx(0.05)
    assert wrong_room.expected_room_contained is False
    assert wrong_room.status == "fail"
    assert wrong_room.first_divergence["stage"] == "expected_room_semantic_containment"

    assert vertical_warning.status == "warning"
    assert vertical_warning.first_divergence["stage"] == "authored_floor_vertical_support"
    assert near_boundary.status == "warning"
    assert near_boundary.signed_boundary_distance_m == pytest.approx(-0.10)
    assert far_outside.status == "fail"


def test_journal_reader_is_read_only_bounded_and_does_not_persist_private_ids(tmp_path: Path) -> None:
    journal = _journal(tmp_path / "world.db")
    observations = read_person_observations(journal, cameras=["living-room"], limit=2)

    assert [row.journal_sequence for row in observations] == [2, 3]
    assert observations[0].sample_id == "journal:2"
    assert observations[0].zone_label == "Living Room"
    assert observations[0].zone_source == "nvdsanalytics_roi"
    assert observations[0].zone_authoritative is True
    assert observations[0].expected_room == "Living Room"
    assert observations[1].unavailable_reason == "world_position_unavailable"
    assert all("private-id" not in row.sample_id for row in observations)


def test_camera_default_zone_is_not_promoted_to_expected_room(tmp_path: Path) -> None:
    journal = tmp_path / "camera-default.db"
    payload = _observation(
        camera="kitchen",
        sequence=1,
        world=[7.0, 0.0, 2.0],
        room="Kitchen",
        zone_source="camera_default",
        zone_authoritative=False,
    )
    connection = sqlite3.connect(journal)
    try:
        connection.execute(
            "CREATE TABLE records ("
            "sequence INTEGER PRIMARY KEY, recorded_at_us INTEGER NOT NULL, "
            "contract TEXT NOT NULL, payload_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO records(sequence, recorded_at_us, contract, payload_json) "
            "VALUES (?, ?, ?, ?)",
            (1, 1_000_001, "noesis.observation.person", json.dumps(payload)),
        )
        connection.commit()
    finally:
        connection.close()

    observation = read_person_observations(journal)[0]
    assert observation.zone_label == "Kitchen"
    assert observation.zone_source == "camera_default"
    assert observation.zone_authoritative is False
    assert observation.expected_room is None


def test_report_emits_stage_specific_first_divergence_and_room_limitation(tmp_path: Path) -> None:
    journal = _journal(tmp_path / "world.db")
    room_map = json.loads(ROOM_MAP.read_text(encoding="utf-8"))["rooms"]
    report = build_journal_oracle_report(
        journal_path=journal,
        obj_path=OBJ,
        similarity_path=SIMILARITY,
        run_id="synthetic",
        room_group_map=room_map,
    )

    assert report["summary"]["status"] == "fail"
    oracle = report["oracle"]
    assert oracle["contract"] == "noesis.validation.authored_scene_track_geometry"
    assert oracle["first_divergence"]["journal_sequence"] == 2
    stages = oracle["stage_first_divergence"]
    assert stages["producer_world"]["journal_sequence"] == 3
    assert stages["world_to_scene_transform"]["status"] == "fail"
    assert stages["world_to_scene_transform"]["structural_status"] == "pass"
    assert stages["world_to_scene_transform"]["first_divergence"]["journal_sequence"] == 4
    assert stages["authored_floor_support"]["journal_sequence"] == 4
    assert stages["expected_room_semantic_containment"]["journal_sequence"] == 2
    assert stages["menon_rendered_placement"]["status"] == "blocked"

    living = oracle["camera_summary"]["living-room"]
    assert living["world_coverage"] == pytest.approx(2 / 3)
    assert living["authored_floor_status_counts"] == {
        "pass": 2,
        "warning": 0,
        "fail": 0,
    }
    assert living["expected_room_evaluated_count"] == 2
    assert living["expected_room_contained_count"] == 1

    report_without_map = build_journal_oracle_report(
        journal_path=journal,
        obj_path=OBJ,
        similarity_path=SIMILARITY,
        run_id="synthetic-no-map",
    )
    limitation = report_without_map["oracle"]["room_semantics"]
    assert limitation["mapping_provided"] is False
    assert "expected-room containment is blocked" in limitation["limitation"]
    assert (
        report_without_map["oracle"]["stage_first_divergence"][
            "expected_room_semantic_containment"
        ]["status"]
        == "blocked"
    )


def test_room_group_map_is_bound_to_the_exact_authored_scene(tmp_path: Path) -> None:
    bound_map = tmp_path / "bound-room-map.json"
    bound_map.write_text(
        json.dumps(
            {
                "authored_scene_sha256": (
                    "9a3ff9b2b1183ee4db3f92e286691bf9631391051c1898efb7cb8e3997dcff58"
                ),
                "rooms": {"Living Room": ["room_living"]},
            }
        ),
        encoding="utf-8",
    )

    assert _load_room_group_map(bound_map, authored_scene_path=OBJ) == {
        "Living Room": ["room_living"]
    }

    other_obj = tmp_path / "other.obj"
    other_obj.write_text(OBJ.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the supplied OBJ"):
        _load_room_group_map(bound_map, authored_scene_path=other_obj)
