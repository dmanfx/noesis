from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

from scripts.noesis_validation_track_geometry_acceptance import (
    main,
    run_track_geometry_acceptance,
)


FIXTURES = Path(__file__).parent / "fixtures" / "authored_scene_oracle"
OBJ = FIXTURES / "simple_home.obj"
SIMILARITY = FIXTURES / "identity_similarity.json"
IDENTITY = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def _journal(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE records (sequence INTEGER PRIMARY KEY, "
            "recorded_at_us INTEGER NOT NULL, contract TEXT NOT NULL, "
            "payload_json TEXT NOT NULL)"
        )
        observations = (
            (1, "living-room", "Living Room", [2.0, 0.05, 2.0]),
            (2, "kitchen", "Kitchen", [7.0, 0.05, 2.0]),
        )
        for sequence, camera, room, point in observations:
            payload = {
                "contract": "noesis.observation.person",
                "contract_version": 1,
                "observed_at_us": 1_000_000 + sequence,
                "payload": {
                    "tracklet": {"camera_id": camera, "tracker_id": sequence},
                    "zone": room,
                    "zone_source": "nvdsanalytics_roi",
                    "zone_authoritative": True,
                    "world": {
                        "position": {"x": point[0], "y": point[1], "z": point[2]},
                        "frame": "backend_world_m",
                        "units": "meters",
                        "source": "pose_floor_only",
                    },
                },
            }
            connection.execute(
                "INSERT INTO records(sequence, recorded_at_us, contract, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    sequence,
                    1_000_000 + sequence,
                    "noesis.observation.person",
                    json.dumps(payload),
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return path


def _production_geometry() -> dict[str, object]:
    endpoint = [2.0, 0.05, 2.0]
    return {
        "contract": "menon.production-track-geometry-debug",
        "contractVersion": 1,
        "bounded": True,
        "validationTruncated": False,
        "worldToSceneMatrix": IDENTITY,
        "validations": [
            {
                "contract": "menon.track.scene.validation",
                "contractVersion": 1,
                "status": "pass",
                "structuralStatus": "pass",
                "physicalStatus": "pass",
                "identity": {"entityId": "entity-1", "pathKey": "entity:entity-1"},
                "checks": [
                    {"code": "canonical_frame_contract", "status": "pass"},
                    {"code": "production_visual_present", "status": "pass"},
                    {
                        "code": "world_to_scene_exact",
                        "status": "pass",
                        "expectedScenePoint": endpoint,
                        "actualScenePoint": endpoint,
                        "toleranceScene": 1e-6,
                    },
                    {
                        "code": "visual_root_identity",
                        "status": "pass",
                        "maxElementError": 0.0,
                        "toleranceScene": 1e-6,
                    },
                    *[
                        {
                            "code": code,
                            "status": "pass",
                            "expectedEndpoint": endpoint,
                            "actualEndpoint": endpoint,
                            "toleranceScene": 1e-6,
                        }
                        for code in (
                            "rendered_trail_endpoint_exact",
                            "rendered_trail_world_endpoint_exact",
                            "decorator_input_endpoint_exact",
                        )
                    ],
                    {
                        "code": "decorator_world_anchor_alignment",
                        "status": "pass",
                        "trailEndpointWorld": endpoint,
                        "decoratorAnchorWorld": endpoint,
                        "toleranceScene": 1e-6,
                    },
                ],
            }
        ],
        "stageVisibility": {
            "bounded": True,
            "truncated": False,
            "entries": [
                {
                    "key": "entity:entity-1",
                    "canonicalWorld": {"entityPresent": True},
                    "presentation": {"entityPresent": True, "rejectionReason": None},
                    "renderer": {
                        "visualPresent": True,
                        "markerObjectPresent": True,
                        "markerEndpointPresent": True,
                        "markerVisible": True,
                        "rejectionReason": None,
                    },
                    "trail": {
                        "geometryPresent": True,
                        "movementSegmentPresent": False,
                        "currentPointPresent": True,
                        "rejectionReason": "trail_has_no_movement_segment_yet",
                    },
                }
            ],
        },
    }


def _inputs(tmp_path: Path) -> dict[str, Path]:
    room_map = tmp_path / "room-map.json"
    room_map.write_text(
        json.dumps(
            {
                "authored_scene_sha256": (
                    "9a3ff9b2b1183ee4db3f92e286691bf9631391051c1898efb7cb8e3997dcff58"
                ),
                "rooms": {
                    "Living Room": ["room_living"],
                    "Kitchen": ["room_kitchen"],
                },
            }
        ),
        encoding="utf-8",
    )
    trace = tmp_path / "menon-trace.json"
    trace.write_text(
        json.dumps(
            {
                "run_id": "trace",
                "world_to_scene_col_major": IDENTITY,
                "production_geometry": _production_geometry(),
            }
        ),
        encoding="utf-8",
    )
    return {
        "journal": _journal(tmp_path / "world.db"),
        "room_map": room_map,
        "trace": trace,
    }


def _run(tmp_path: Path, *, run_id: str = "acceptance") -> tuple[Path, dict]:
    inputs = _inputs(tmp_path)
    return run_track_geometry_acceptance(
        journal_path=inputs["journal"],
        obj_path=OBJ,
        similarity_path=SIMILARITY,
        room_group_map_path=inputs["room_map"],
        menon_trace_path=inputs["trace"],
        output_dir=tmp_path / "private-output",
        run_id=run_id,
    )


def test_unified_acceptance_separates_three_passing_gates_and_private_evidence(
    tmp_path: Path,
) -> None:
    result_path, report = _run(tmp_path)

    assert report["summary"]["status"] == "pass"
    assert {name: gate["status"] for name, gate in report["gates"].items()} == {
        "producer_coverage": "pass",
        "physical_placement": "pass",
        "menon_renderer": "pass",
    }
    assert report["bindings"]["world_to_scene_max_matrix_error"] == 0.0
    assert report["bindings"]["authored_scene_sha256"] == (
        "9a3ff9b2b1183ee4db3f92e286691bf9631391051c1898efb7cb8e3997dcff58"
    )
    assert stat.S_IMODE(result_path.parent.stat().st_mode) == 0o700
    for artifact in result_path.parent.rglob("*"):
        expected = 0o700 if artifact.is_dir() else 0o600
        assert stat.S_IMODE(artifact.stat().st_mode) == expected


def test_transform_mismatch_fails_physical_and_renderer_without_hiding_coverage(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = json.loads(inputs["trace"].read_text(encoding="utf-8"))
    payload["world_to_scene_col_major"][12] = 1.0
    inputs["trace"].write_text(json.dumps(payload), encoding="utf-8")

    _, report = run_track_geometry_acceptance(
        journal_path=inputs["journal"],
        obj_path=OBJ,
        similarity_path=SIMILARITY,
        room_group_map_path=inputs["room_map"],
        menon_trace_path=inputs["trace"],
        output_dir=tmp_path / "private-output",
        run_id="mismatch",
    )

    assert report["summary"]["status"] == "fail"
    assert report["gates"]["producer_coverage"]["status"] == "pass"
    assert report["gates"]["physical_placement"]["status"] == "fail"
    assert report["gates"]["menon_renderer"]["status"] == "fail"


def test_missing_production_geometry_blocks_renderer_and_cli_returns_nonzero(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = json.loads(inputs["trace"].read_text(encoding="utf-8"))
    payload.pop("production_geometry")
    inputs["trace"].write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "--journal", str(inputs["journal"]),
            "--obj", str(OBJ),
            "--similarity", str(SIMILARITY),
            "--room-group-map", str(inputs["room_map"]),
            "--menon-trace", str(inputs["trace"]),
            "--output-dir", str(tmp_path / "private-output"),
            "--run-id", "missing-production",
        ]
    )
    report = json.loads(
        (tmp_path / "private-output/missing-production/track_geometry_acceptance.json").read_text(
            encoding="utf-8"
        )
    )

    assert exit_code == 1
    assert report["summary"]["status"] == "blocked"
    assert report["gates"]["menon_renderer"]["status"] == "blocked"


def test_unbound_room_map_fails_closed_before_any_gate_can_pass(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    room_map = json.loads(inputs["room_map"].read_text(encoding="utf-8"))
    room_map["authored_scene_sha256"] = "0" * 64
    inputs["room_map"].write_text(json.dumps(room_map), encoding="utf-8")

    _, report = run_track_geometry_acceptance(
        journal_path=inputs["journal"],
        obj_path=OBJ,
        similarity_path=SIMILARITY,
        room_group_map_path=inputs["room_map"],
        menon_trace_path=inputs["trace"],
        output_dir=tmp_path / "private-output",
        run_id="bad-room-map",
    )

    assert report["summary"]["status"] == "fail"
    assert report["checks"][0]["id"] == "ACCEPTANCE.input"
    assert {gate["status"] for gate in report["gates"].values()} == {"blocked"}
