from __future__ import annotations

import fcntl
import json
import os
import stat
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from noesis.validation.alignment_walk import (
    AlignmentWalkCapture,
    EphemeralIds,
    _calibration_for_session,
    _evaluate_camera_candidates,
    _fit_fixed_center_rotation,
    append_waypoint_marker,
    assign_waypoints,
    build_waypoint_calibration_evidence,
    create_run_directory,
    payload_sha256,
    read_ndjson_private,
    sanitize_tracking_message,
    validate_waypoint_manifest,
    verify_run,
    write_alignment_report,
    write_waypoint_calibration_report,
)


def _calibration_bundle() -> dict[str, object]:
    return {
        "align": {
            "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
            "floor_y": 0.0,
            "units": {"s_obj_to_m": 1.0},
            "scene_similarity": {
                "s_obj_to_m": 1.0,
                "world_to_scene_col_major": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ]
            },
        },
        "cameras": {
            "K": {"family-room": [1000.0, 1000.0, 640.0, 360.0]},
            "E": {
                "family-room": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ]
            },
            "pose": {},
        },
        "meta": {"world_frame": "backend_world_m", "units": "meters"},
        "metric_scale": 1.0,
    }


def _scene_binding(*, world_digest: str | None = None) -> dict[str, str]:
    identity = [float(value) for value in np.eye(4).flatten(order="F")]
    digest = payload_sha256(
        {
            "world_to_scene_col_major": identity,
            "s_obj_to_m": 1.0,
        }
    )
    return {
        "release_id": "menon-scene-test-v1",
        "authored_scene_sha256": "e" * 64,
        "world_to_scene_sha256": world_digest or digest,
    }


def _waypoints() -> dict[str, object]:
    return {
        "contract": "noesis.alignment.walk_waypoints",
        "contract_version": 1,
        "waypoints": [
            {
                "id": "family-center",
                "camera_id": "family-room",
                "label": "Family room center",
                "rooms": ["Family Room"],
                "expected_scene_xz": [0.0, 0.0],
                "pause_s": 3.0,
            }
        ],
    }


def _world_snapshot(sequence: int, first: list[float], second: list[float]) -> dict[str, object]:
    entities = []
    for entity_id, point, camera, tracker_id in (
        ("resident:real-name", first, "family-room", 111111),
        ("visitor:sensitive", second, "family-room", 222222),
    ):
        entities.append(
            {
                "entity_id": entity_id,
                "subject": {"kind": "resident", "subject_id": entity_id},
                "lifecycle": "present",
                "position": {"x": point[0], "y": point[1], "z": point[2]},
                "covariance": {"values": [1.0] * 9},
                "velocity_mps": {"x": 0.0, "y": 0.0, "z": 0.0},
                "room_id": "Family Room",
                "observed_at_us": 1_000_000 + sequence,
                "stale_after_us": 2_000_000 + sequence,
                "sources": [
                    {
                        "observation_id": (
                            f"sensitive-observation-{sequence}-{tracker_id}"
                        ),
                        "camera_id": camera,
                        "zone": "Family Room",
                        "zone_source": "nvdsanalytics_roi",
                        "zone_authoritative": True,
                        "observed_at_us": 1_000_000 + sequence,
                        "position": {"x": point[0], "y": point[1], "z": point[2]},
                        "covariance": {"values": [1.0] * 9},
                        "accepted": True,
                        "rejection_reason": None,
                    }
                ],
                "conflict": False,
                "conflict_reason": None,
            }
        )
    return {
        "contract": "noesis.world.snapshot",
        "contract_version": 1,
        "snapshot_id": f"snapshot-sensitive-{sequence}",
        "producer": {"runtime": "ds9", "instance_id": "instance", "run_id": "runtime-run"},
        "sequence": sequence,
        "observed_start_us": 1_000_000 + sequence,
        "observed_end_us": 1_000_000 + sequence,
        "published_at_us": 1_000_100 + sequence,
        "frame": "backend_world_m",
        "units": "meters",
        "entities": entities,
    }


def _tracking_message(
    frame: int,
    walker: list[float],
    other: list[float],
    *,
    walker_depth: float = 2.0,
) -> dict[str, object]:
    tracks = []
    observations = []
    for tracker_id, point, depth in (
        (111111, walker, walker_depth),
        (222222, other, 5.0),
    ):
        image_foot = (
            [
                1000.0 * float(point[0]) / float(point[2]) + 640.0,
                1000.0 * float(point[1]) / float(point[2]) + 360.0,
            ]
            if float(point[2]) > 1e-9
            else [25.0, 60.0]
        )
        tracks.append(
            {
                "tracker_id": tracker_id,
                "tracker_lifecycle_generation": 1,
                "stable_id": 987654,
                "identity_v2": {"resident": "sensitive"},
                "reid_identity": "private-name",
                "embedding": [0.1, 0.2],
                "embedding_present": True,
                "embedding_sequence": 7,
                "embedding_model_sha256": "e" * 64,
                "embedding_dimension": 2,
                "camera_id": "family-room",
                "frame_id": frame,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "image_size": [1280, 720],
                "image_foot": image_foot,
                "image_base": image_foot,
                "confidence": 0.9,
                "tracker_confidence": 0.8,
                "zone": "Family Room",
                "zone_source": "nvdsanalytics_roi",
                "zone_authoritative": True,
                "depth_status": "ok",
                "depth_anchor_source": "lower_body_band",
                "depth_anchor_m": depth,
                "depth_registered_m": depth * 0.9,
                "depth_used_m": depth * 0.9,
                "depth_registration_status": "applied",
                "depth_registration_id": "reg-v1",
                "depth_sample_count": 64,
                "depth_valid_fraction": 0.9,
                "floor_world_raw_m": point,
                "depth_world_raw_m": point,
                "world_measurement_raw_m": point,
                "world_prediction_m": point,
                "world_innovation_m": 0.1,
                "world_measurement_accepted": True,
                "world_floor_range_m": 7.5,
                "world_floor_range_limit_m": 22.0,
                "world_floor_incidence_sin": 0.25,
                "world_floor_admitted": True,
                "world": point,
                "world_valid": True,
                "world_frame": "backend_world_m",
                "world_source": "pose_depth_fused",
                "world_quality": "good",
                "motion_mode": "walk",
            }
        )
        observations.append(
            {
                "observation_id": (
                    f"sensitive-observation-{frame}-{tracker_id}"
                ),
                "calibration": {"sha256": "a" * 64},
                "model": {"sha256": "b" * 64},
                "config": {"sha256": "c" * 64},
                "payload": {
                    "tracklet": {
                        "camera_id": "family-room",
                        "tracker_id": tracker_id,
                        "frame_id": frame,
                    },
                    "embedding_sequence": 7,
                },
            }
        )
    return {
        "type": "tracking",
        "source_id": 2,
        "camera_id": "family-room",
        "frame_id": frame,
        "media_pts_ns": frame * 1_000_000,
        "captured_at_us": 900_000 + frame,
        "observed_at_us": 1_000_000 + frame,
        "tracking_publication_sequence": frame,
        "tracks": tracks,
        "observations": observations,
        "world_snapshot": _world_snapshot(frame, walker, other),
    }


def _room_zones(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "name": "Family Room",
                    "bounds": {"minX": -1.0, "maxX": 1.0, "minZ": -1.0, "maxZ": 1.0},
                    "floorY": 0.0,
                },
                {
                    "name": "Kitchen",
                    "bounds": {"minX": 2.0, "maxX": 3.0, "minZ": 2.0, "maxZ": 3.0},
                    "floorY": 0.0,
                },
                {
                    "name": "Kitchen2",
                    "bounds": {"minX": 3.0, "maxX": 4.0, "minZ": 2.0, "maxZ": 3.0},
                    "floorY": 0.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_capture_sanitizes_identity_and_persists_private_durable_evidence(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    run_dir = create_run_directory(root, "walk-1")
    capture = AlignmentWalkCapture(run_dir=run_dir, run_id="walk-1", ws_uri="ws://127.0.0.1:6008?secret=no")
    capture.copy_waypoints(_waypoints())
    capture.process_message({"type": "calibration-bundle", "data": _calibration_bundle()})

    base_ns = 10_000_000_000
    positions = (-0.8, -0.3, -0.05, 0.0, 0.0, 0.0)
    for index, x in enumerate(positions):
        capture.process_message(
            _tracking_message(index, [x, 0.0, 0.0], [5.0, 0.0, 5.0]),
            received_at_us=1_000_000 + index * 500_000,
            received_monotonic_ns=base_ns + index * 500_000_000,
        )
        if index == 3:
            append_waypoint_marker(
                run_dir,
                waypoint_id="family-center",
                actor="walker-a",
                recorded_at_us=1_000_000 + index * 500_000,
                monotonic_ns=base_ns + index * 500_000_000,
            )

    # A duplicate standalone snapshot is intentionally ignored by this capture path.
    capture.process_message({"type": "world_snapshot", "payload": _world_snapshot(99, [0, 0, 0], [5, 0, 5])})
    capture.finish()

    verification = verify_run(run_dir)
    assert verification["ok"] is True, verification
    assert verification["counts"] == {
        "markers.ndjson": 1,
        "samples.ndjson": 12,
        "world.ndjson": 6,
    }
    for path in run_dir.rglob("*"):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600)

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (run_dir / "samples.ndjson", run_dir / "world.ndjson")
    )
    for forbidden in (
        "private-name",
        "resident:real-name",
        "visitor:sensitive",
        "sensitive-observation",
        '"stable_id"',
        '"embedding"',
        '"subject"',
        '"entity_id"',
    ):
        assert forbidden not in serialized
    samples = read_ndjson_private(run_dir / "samples.ndjson")
    assert {row["tracklet_key"] for row in samples} == {"tracklet-1", "tracklet-2"}
    assert samples[0]["calibration_sha256"] == "a" * 64
    assert samples[0]["world_measurement_raw_m"] == [-0.8, 0.0, 0.0]
    assert samples[0]["world_floor_range_m"] == 7.5
    assert samples[0]["world_floor_range_limit_m"] == 22.0
    assert samples[0]["world_floor_incidence_sin"] == 0.25
    assert samples[0]["world_floor_admitted"] is True
    assert samples[0]["zone"] == "Family Room"
    assert samples[0]["zone_source"] == "nvdsanalytics_roi"
    assert samples[0]["zone_authoritative"] is True
    world = read_ndjson_private(run_dir / "world.ndjson")
    assert world[0]["entities"][0]["position"] == [-0.8, 0.0, 0.0]
    assert world[0]["entities"][0]["entity_key"] == "entity-1"
    assert samples[0]["observation_key"] == (
        world[0]["entities"][0]["sources"][0]["observation_key"]
    )
    assert world[0]["entities"][0]["sources"][0]["zone"] == "Family Room"
    assert world[0]["entities"][0]["sources"][0]["zone_source"] == (
        "nvdsanalytics_roi"
    )
    assert world[0]["entities"][0]["sources"][0]["zone_authoritative"] is True
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    assert session["ws_uri"] == "ws://127.0.0.1:6008"
    assert session["privacy"]["embeddings_persisted"] is False


def test_private_ndjson_reader_waits_for_an_in_progress_append_lock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "samples.ndjson"
    path.write_text('{"sequence":0}\n', encoding="utf-8")
    path.chmod(0o600)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    completed = threading.Event()
    result: list[dict[str, object]] = []

    def read_rows() -> None:
        result.extend(read_ndjson_private(path))
        completed.set()

    thread = threading.Thread(target=read_rows)
    thread.start()
    time.sleep(0.05)
    assert completed.is_set() is False
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)
    thread.join(timeout=1.0)
    assert completed.is_set() is True
    assert result == [{"sequence": 0}]


def test_report_assigns_arriving_walker_among_multiple_people(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    run_dir = create_run_directory(root, "walk-report")
    capture = AlignmentWalkCapture(run_dir=run_dir, run_id="walk-report", ws_uri="ws://127.0.0.1:6008")
    capture.copy_waypoints(_waypoints())
    capture.process_message({"type": "calibration-bundle", "data": _calibration_bundle()})
    base_ns = 20_000_000_000
    for index, x in enumerate((-0.8, -0.3, -0.05, 0.0, 0.0, 0.0)):
        capture.process_message(
            _tracking_message(index, [x, 0.0, 0.0], [5.0, 0.0, 5.0]),
            received_at_us=2_000_000 + index * 500_000,
            received_monotonic_ns=base_ns + index * 500_000_000,
        )
        if index == 3:
            append_waypoint_marker(
                run_dir,
                waypoint_id="family-center",
                actor="walker-a",
                tracklet_key="tracklet-1",
                monotonic_ns=base_ns + index * 500_000_000,
            )
    capture.finish()

    result = write_alignment_report(run_dir, room_zones_path=_room_zones(tmp_path / "room-zones.json"))
    assert result["assignment_count"] == 1
    assignments = json.loads((run_dir / "waypoint_assignments.json").read_text(encoding="utf-8"))
    assert assignments[0]["status"] == "assigned"
    assert assignments[0]["selected"]["tracklet_key"] == "tracklet-1"
    assert assignments[0]["selected"]["error_m"] == 0.0
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    waypoint_check = next(check for check in report["checks"] if check["id"] == "ALIGN.waypoint.family-center")
    assert waypoint_check["status"] == "pass"
    room_check = next(check for check in report["checks"] if check["id"] == "ALIGN.room.family-room")
    # All-person containment is intentionally separate from selected-walker fitting.
    assert room_check["metric"]["inside_ratio"] == 0.5


def test_multi_person_equal_candidates_are_reported_ambiguous() -> None:
    identity = np.eye(4, dtype=np.float64)
    base_ns = 30_000_000_000
    samples = []
    for key in ("tracklet-1", "tracklet-2"):
        for offset, x in ((-1_000_000_000, -0.4), (0, 0.0), (1_000_000_000, 0.0)):
            samples.append(
                {
                    "camera_id": "family-room",
                    "tracklet_key": key,
                    "received_monotonic_ns": base_ns + offset,
                    "world_valid": True,
                    "world": [x, 0.0, 0.0],
                }
            )
    markers = [
        {
            "waypoint_id": "family-center",
            "phase": "arrived",
            "monotonic_ns": base_ns,
            "actor": "walker-a",
        }
    ]
    assignments = assign_waypoints(
        samples,
        markers,
        _waypoints(),
        world_to_scene=identity,
        scene_to_m=1.0,
    )
    assert assignments[0]["status"] == "ambiguous"
    assert assignments[0]["score_margin"] == 0.0


def test_automatic_assignment_ignores_poisoned_current_world_motion() -> None:
    marker_ns = 31_000_000_000
    samples = []
    for index, (offset, u, world_x) in enumerate(
        (
            (-1_000_000_000, 400.0, 1000.0),
            (0, 520.0, -1000.0),
            (1_000_000_000, 520.0, 5000.0),
        )
    ):
        samples.append(
            {
                "sequence": index,
                "camera_id": "family-room",
                "tracklet_key": "tracklet-1",
                "received_monotonic_ns": marker_ns + offset,
                "image_foot": [u, 400.0],
                "image_size": [1280.0, 720.0],
                "world_valid": True,
                "world": [world_x, 0.0, 0.0],
            }
        )
    for index, (offset, world_x) in enumerate(
        (
            (-1_000_000_000, -10000.0),
            (0, 10000.0),
            (1_000_000_000, -20000.0),
        ),
        start=3,
    ):
        samples.append(
            {
                "sequence": index,
                "camera_id": "family-room",
                "tracklet_key": "tracklet-2",
                "received_monotonic_ns": marker_ns + offset,
                "image_foot": [700.0, 400.0],
                "image_size": [1280.0, 720.0],
                "world_valid": True,
                "world": [world_x, 0.0, 0.0],
            }
        )
    assignments = assign_waypoints(
        samples,
        [
            {
                "waypoint_id": "family-center",
                "phase": "arrived",
                "monotonic_ns": marker_ns,
            }
        ],
        _waypoints(),
        world_to_scene=np.eye(4, dtype=np.float64),
        scene_to_m=1.0,
    )
    assert assignments[0]["status"] == "assigned"
    assert assignments[0]["selected"]["tracklet_key"] == "tracklet-1"
    assert (
        assignments[0]["selected"]["pre_marker_motion_m"]
        < assignments[0]["candidates"][1]["pre_marker_motion_m"]
    )


def test_guided_waypoint_manifest_requires_complete_fit_holdout_declaration() -> None:
    payload = _waypoints()
    waypoint = payload["waypoints"][0]  # type: ignore[index]
    waypoint["expected_scene_xyz"] = [0.0, 0.0, 2.0]  # type: ignore[index]
    try:
        validate_waypoint_manifest(payload)
    except ValueError as exc:
        assert "both expected_scene_xyz and split" in str(exc)
    else:
        raise AssertionError("partial guided waypoint declaration was accepted")

    waypoint["split"] = "training"  # type: ignore[index]
    try:
        validate_waypoint_manifest(payload)
    except ValueError as exc:
        assert "fit or holdout" in str(exc)
    else:
        raise AssertionError("unsupported guided waypoint split was accepted")

    waypoint["split"] = "holdout"  # type: ignore[index]
    normalized = validate_waypoint_manifest(payload)
    assert normalized["waypoints"][0]["expected_scene_xyz"] == [0.0, 0.0, 2.0]
    assert normalized["waypoints"][0]["split"] == "holdout"


def test_exact_waypoint_binding_is_nearest_marker_sample_without_ground_truth_selection() -> None:
    identity = np.eye(4, dtype=np.float64)
    marker_ns = 40_000_000_000
    samples = [
        {
            "sequence": index,
            "camera_id": "family-room",
            "tracklet_key": "tracklet-1",
            "frame_id": 100 + index,
            "media_pts_ns": 1_000_000 + index,
            "received_at_us": 2_000_000 + index,
            "received_monotonic_ns": marker_ns + offset,
            "image_foot": [640.0, 360.0],
            "image_size": [1280.0, 720.0],
            "world_valid": True,
            "world": [x, 0.0, 2.0],
        }
        for index, (offset, x) in enumerate(
            ((-1_000_000_000, -0.5), (100_000_000, 0.8), (800_000_000, 0.0))
        )
    ]
    samples[1]["world_valid"] = False
    manifest = {
        "waypoints": [
            {
                "id": "family-center",
                "camera_id": "family-room",
                "expected_scene_xz": [0.0, 2.0],
            }
        ]
    }
    assignments = assign_waypoints(
        samples,
        [
            {
                "waypoint_id": "family-center",
                "phase": "arrived",
                "monotonic_ns": marker_ns,
            }
        ],
        manifest,
        world_to_scene=identity,
        scene_to_m=1.0,
    )
    exact = assignments[0]["selected"]["exact_sample"]
    # Frame 101 is closest in time even though it is world-invalid and frame
    # 102 is closest to truth. Exact evidence must expose, not skip, that gap.
    assert exact["sample_sequence"] == 1
    assert exact["frame_id"] == 101
    assert exact["media_pts_ns"] == 1_000_001
    assert exact["marker_delta_ms"] == 100.0


def test_explicit_marker_tracklet_binds_without_valid_current_world() -> None:
    marker_ns = 41_000_000_000
    samples = [
        {
            "sequence": index,
            "camera_id": "family-room",
            "tracklet_key": tracklet_key,
            "frame_id": 200 + index,
            "media_pts_ns": 2_000_000 + index,
            "captured_at_us": 3_000_000 + index,
            "observed_at_us": 3_100_000 + index,
            "received_at_us": 3_200_000 + index,
            "received_monotonic_ns": marker_ns + index,
            "image_foot": [500.0 + index, 400.0],
            "image_size": [1280.0, 720.0],
            "world_valid": False,
            "world": None,
        }
        for index, tracklet_key in enumerate(("tracklet-1", "tracklet-2"))
    ]
    assignments = assign_waypoints(
        samples,
        [
            {
                "waypoint_id": "family-center",
                "phase": "arrived",
                "monotonic_ns": marker_ns,
                "tracklet_key": "tracklet-2",
            }
        ],
        _waypoints(),
        world_to_scene=np.eye(4, dtype=np.float64),
        scene_to_m=1.0,
    )
    assignment = assignments[0]
    assert assignment["status"] == "assigned"
    assert assignment["selection_source"] == "explicit_tracklet_marker"
    assert assignment["selected"]["tracklet_key"] == "tracklet-2"
    assert assignment["selected"]["exact_sample"]["frame_id"] == 201
    assert assignment["selected"]["error_m"] is None


def test_scene_similarity_scale_is_authoritative_over_legacy_align_units(
    tmp_path: Path,
) -> None:
    bundle = _calibration_bundle()
    similarity = bundle["align"]["scene_similarity"]  # type: ignore[index]
    similarity["world_to_scene_col_major"] = [  # type: ignore[index]
        10.0, 0.0, 0.0, 0.0,
        0.0, 10.0, 0.0, 0.0,
        0.0, 0.0, 10.0, 0.0,
        3.0, 4.0, 5.0, 1.0,
    ]
    similarity["s_obj_to_m"] = 0.1  # type: ignore[index]
    bundle["align"]["units"]["s_obj_to_m"] = 1.0  # type: ignore[index]
    run_dir = create_run_directory(tmp_path / "evidence", "scale-authority")
    capture = AlignmentWalkCapture(
        run_dir=run_dir,
        run_id="scale-authority",
        ws_uri="ws://127.0.0.1:6008",
    )
    capture.process_message({"type": "calibration-bundle", "data": bundle})
    capture.process_message(
        _tracking_message(1, [0.0, 0.0, 2.0], [5.0, 0.0, 5.0]),
        received_at_us=1_000_000,
        received_monotonic_ns=1_000_000_000,
    )
    capture.finish()
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    _calibration, matrix, scene_to_m = _calibration_for_session(run_dir, session)
    assert scene_to_m == 0.1
    assert np.allclose(np.diag(matrix)[:3], [10.0, 10.0, 10.0])


def test_scene_binding_is_persisted_and_rejects_active_similarity_mismatch(
    tmp_path: Path,
) -> None:
    matching_dir = create_run_directory(tmp_path / "evidence", "scene-match")
    matching = AlignmentWalkCapture(
        run_dir=matching_dir,
        run_id="scene-match",
        ws_uri="ws://127.0.0.1:6008",
        scene_binding=_scene_binding(),
    )
    matching.process_message(
        {"type": "calibration-bundle", "data": _calibration_bundle()}
    )
    matching.process_message(
        _tracking_message(1, [0.0, 0.0, 2.0], [5.0, 0.0, 5.0]),
        received_at_us=1_000_000,
        received_monotonic_ns=1_000_000_000,
    )
    matching.finish()
    session = json.loads(
        (matching_dir / "session.json").read_text(encoding="utf-8")
    )
    assert session["scene_binding"] == _scene_binding()
    assert session["scene_binding_verification"]["status"] == "partial"
    assert session["scene_binding_verification"]["verified_fields"] == [
        "world_to_scene_sha256"
    ]

    mismatch_dir = create_run_directory(tmp_path / "evidence", "scene-mismatch")
    mismatch = AlignmentWalkCapture(
        run_dir=mismatch_dir,
        run_id="scene-mismatch",
        ws_uri="ws://127.0.0.1:6008",
        scene_binding=_scene_binding(world_digest="0" * 64),
    )
    with pytest.raises(RuntimeError, match="world_to_scene_sha256"):
        mismatch.process_message(
            {"type": "calibration-bundle", "data": _calibration_bundle()}
        )
    mismatch.finish(status="failed", error="scene binding mismatch")
    failed_session = json.loads(
        (mismatch_dir / "session.json").read_text(encoding="utf-8")
    )
    assert failed_session["status"] == "failed"
    assert failed_session["scene_binding_verification"]["status"] == "mismatch"


def test_waypoint_evidence_supports_advisory_fit_holdout_and_physical_depth_fit() -> None:
    identity_col_major = [float(value) for value in np.eye(4).flatten(order="F")]
    calibration = {
        "cameras": {
            "K": {"family-room": [100.0, 100.0, 50.0, 40.0]},
            "E": {"family-room": identity_col_major},
        }
    }
    points = [
        ("fit-a", "fit", [0.0, 0.0, 2.0]),
        ("fit-b", "fit", [1.0, 0.2, 3.0]),
        ("fit-c", "fit", [-1.0, -0.2, 4.0]),
        ("fit-d", "fit", [0.5, 0.4, 5.0]),
        ("holdout-a", "holdout", [0.25, 0.1, 3.5]),
    ]
    samples = []
    assignments = []
    waypoints = []
    bundle_sha = "d" * 64
    for sequence, (waypoint_id, split, point) in enumerate(points):
        u = 100.0 * point[0] / point[2] + 50.0
        v = 100.0 * point[1] / point[2] + 40.0
        sample = {
            "sequence": sequence,
            "camera_id": "family-room",
            "tracklet_key": f"tracklet-{sequence + 1}",
            "frame_id": 100 + sequence,
            "media_pts_ns": 10_000 + sequence,
            "captured_at_us": 15_000 + sequence,
            "observed_at_us": 16_000 + sequence,
            "received_at_us": 20_000 + sequence,
            "received_monotonic_ns": 30_000 + sequence,
            "image_foot": [u, v],
            "image_size": [1280.0, 720.0],
            "calibration_bundle_sha256": bundle_sha,
            "calibration_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "depth_anchor_m": point[2] + 0.2,
            "depth_registered_m": point[2],
            "depth_used_m": point[2],
            "depth_registration_status": "ok",
            "depth_registration_id": "reg-v1",
            "floor_world_raw_m": [point[0] + 0.1, point[1], point[2]],
            "depth_world_raw_m": point,
            "world_measurement_raw_m": point,
            "world_prediction_m": point,
            "world": point,
            "world_valid": True,
        }
        samples.append(sample)
        assignments.append(
            {
                "waypoint_id": waypoint_id,
                "camera_id": "family-room",
                "marker_monotonic_ns": 30_000 + sequence,
                "status": "assigned",
                "score_margin": 1.0,
                "selected": {
                    "tracklet_key": sample["tracklet_key"],
                    "exact_sample": {
                        "sample_sequence": sequence,
                        "camera_id": "family-room",
                        "tracklet_key": sample["tracklet_key"],
                        "frame_id": sample["frame_id"],
                        "media_pts_ns": sample["media_pts_ns"],
                        "captured_at_us": sample["captured_at_us"],
                        "observed_at_us": sample["observed_at_us"],
                        "received_at_us": sample["received_at_us"],
                        "received_monotonic_ns": sample["received_monotonic_ns"],
                        "marker_delta_ms": 0.0,
                        "image_foot": sample["image_foot"],
                        "image_size": sample["image_size"],
                    },
                },
            }
        )
        waypoints.append(
            {
                "id": waypoint_id,
                "camera_id": "family-room",
                "expected_scene_xyz": point,
                "split": split,
            }
        )
    evidence, metrics, candidate = build_waypoint_calibration_evidence(
        samples,
        assignments,
        {"waypoints": waypoints},
        calibration=calibration,
        calibration_bundle_sha256=bundle_sha,
        source_capture_artifact_index_sha256="f" * 64,
        world_to_scene=np.eye(4, dtype=np.float64),
        scene_to_m=1.0,
    )
    assert candidate["status"] == "complete"
    assert candidate["advisory_only"] is True
    assert candidate["active_config_modified"] is False
    assert candidate["fit_waypoint_count"] == 4
    assert {
        row["anchor_id"] for row in candidate["solver"]["correspondences"]
    } == {"fit-a", "fit-b", "fit-c", "fit-d"}
    assert len(candidate["similarity"]["sha256"]) == 64
    first = evidence[0]
    assert first["status"] == "complete"
    assert first["binding"]["frame_id"] == 100
    assert first["calibration"]["K"] == [100.0, 100.0, 50.0, 40.0]
    assert first["calibration"]["image_size"] == [1280.0, 720.0]
    assert first["calibration"]["E_world_to_camera_col_major"] == identity_col_major
    assert len(first["calibration"]["similarity"]["sha256"]) == 64
    assert first["depth"]["physical_registration_pair"] == {
        "raw_depth_m": 2.2,
        "registered_depth_m": 2.0,
        "target_optical_depth_m": 2.0,
    }
    candidate_pair = first["depth"]["candidate_physical_registration_pair"]
    assert candidate_pair["raw_depth_m"] == 2.2
    assert candidate_pair["registered_depth_m"] == 2.0
    assert np.isclose(candidate_pair["target_optical_depth_m"], 2.0)
    assert np.isclose(first["camera_metrics"]["raw_depth_error_m"], 0.2)
    assert first["camera_metrics"]["registered_depth_error_m"] == 0.0
    assert np.isclose(
        first["stages"]["active"]["floor_candidate"]["error_m"], 0.1
    )
    assert first["stages"]["active"]["final_world"]["error_m"] == 0.0
    assert first["observed_camera_ray_geometry"]["camera_center_backend_world_m"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert metrics["splits"]["fit"]["binding_coverage"] == 1.0
    assert metrics["splits"]["holdout"]["binding_coverage"] == 1.0
    assert (
        metrics["splits"]["holdout"]["candidate_stages"]["final_world"][
            "error_m"
        ]["max"]
        < 1e-12
    )
    baseline_camera = candidate["camera_calibration_candidates"]["cameras"][
        "family-room"
    ]
    poisoned_samples = json.loads(json.dumps(samples))
    for index, sample in enumerate(poisoned_samples):
        sample["world"] = [1000.0 + index, 500.0, -1000.0 + index]
        sample["world_valid"] = False
    poisoned_evidence, _poisoned_metrics, poisoned_candidate = (
        build_waypoint_calibration_evidence(
            poisoned_samples,
            assignments,
            {"waypoints": waypoints},
            calibration=calibration,
            calibration_bundle_sha256=bundle_sha,
            source_capture_artifact_index_sha256="f" * 64,
            world_to_scene=np.eye(4, dtype=np.float64),
            scene_to_m=1.0,
        )
    )
    assert poisoned_candidate["status"] == "blocked"
    assert all(
        "producer_final_world_unavailable" in row["diagnostic_flags"]
        for row in poisoned_evidence
    )
    poisoned_camera = poisoned_candidate["camera_calibration_candidates"][
        "cameras"
    ]["family-room"]
    assert poisoned_camera["rotation"]["admission_status"] == "admissible"
    assert (
        poisoned_camera["depth_registration"]["admission_status"]
        == baseline_camera["depth_registration"]["admission_status"]
    )
    assert (
        poisoned_camera["rotation"]["candidate_sha256"]
        == baseline_camera["rotation"]["candidate_sha256"]
    )
    assert (
        poisoned_camera["depth_registration"].get("candidate_sha256")
        == baseline_camera["depth_registration"].get("candidate_sha256")
    )
    assert (
        poisoned_camera["depth_registration"].get("reason")
        == baseline_camera["depth_registration"].get("reason")
    )


def test_waypoint_calibration_report_seals_private_fit_holdout_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = create_run_directory(tmp_path / "evidence", "guided-fit-holdout")
    points = [
        ("fit-a", "fit", [0.0, 0.0, 2.0]),
        ("fit-b", "fit", [1.0, 0.2, 2.0]),
        ("fit-c", "fit", [0.0, -0.2, 3.0]),
        ("fit-d", "fit", [1.0, 0.4, 3.0]),
        ("holdout-a", "holdout", [0.5, 0.1, 2.5]),
    ]
    manifest = {
        "contract": "noesis.alignment.walk_waypoints",
        "contract_version": 1,
        "waypoints": [
            {
                "id": waypoint_id,
                "camera_id": "family-room",
                "expected_scene_xz": [point[0], point[2]],
                "expected_scene_xyz": point,
                "split": split,
                "pause_s": 2.0,
            }
            for waypoint_id, split, point in points
        ],
    }
    capture = AlignmentWalkCapture(
        run_dir=run_dir,
        run_id="guided-fit-holdout",
        ws_uri="ws://127.0.0.1:6008",
        scene_binding=_scene_binding(),
    )
    capture.copy_waypoints(manifest)
    capture.process_message({"type": "calibration-bundle", "data": _calibration_bundle()})
    base_ns = 60_000_000_000
    frame = 0
    for waypoint_index, (waypoint_id, _split, point) in enumerate(points):
        marker_ns = base_ns + waypoint_index * 10_000_000_000
        for offset_ns in (-500_000_000, *range(0, 1_000_000_000, 100_000_000)):
            message = _tracking_message(
                frame,
                point,
                [5.0, 0.0, 5.0],
                walker_depth=(float(point[2]) - 0.25) / 0.9,
            )
            for track in message["tracks"]:  # type: ignore[index]
                track["world"] = [  # type: ignore[index]
                    1000.0 + frame,
                    500.0,
                    -1000.0 + frame,
                ]
                track["world_valid"] = False  # type: ignore[index]
            capture.process_message(
                message,
                received_at_us=3_000_000 + frame * 100_000,
                received_monotonic_ns=marker_ns + offset_ns,
            )
            if offset_ns == 0:
                append_waypoint_marker(
                    run_dir,
                    waypoint_id=waypoint_id,
                    actor="walker-a",
                    tracklet_key="tracklet-1",
                    monotonic_ns=marker_ns,
                )
            frame += 1
    capture.finish()

    result = write_waypoint_calibration_report(run_dir)
    assert result["status"] == "warning"
    assert result["advisory_similarity_status"] == "blocked"
    assert result["camera_candidate_status"] == "complete"
    assert result["calibration_candidate_status"] == "admissible"
    assert result["candidate_summary"] == {
        "fit_solver_status": "admissible",
        "calibration_candidate_status": "admissible",
        "advisory_only": True,
        "active_config_modified": False,
        "camera_candidate_status": "complete",
        "advisory_similarity_status": "blocked",
    }
    assert result["fit_waypoint_count"] == 4
    assert result["holdout_waypoint_count"] == 1
    evidence = json.loads(
        (run_dir / "waypoint_calibration_evidence.json").read_text(encoding="utf-8")
    )
    assert len(evidence) == 5
    assert all(row["status"] == "complete" for row in evidence)
    assert all(row["scene_binding"] == _scene_binding() for row in evidence)
    assert evidence[-1]["split"] == "holdout"
    assert evidence[-1]["binding"]["frame_id"] == 45
    candidate_payload = json.loads(
        (run_dir / "waypoint_candidate_similarity.json").read_text(
            encoding="utf-8"
        )
    )
    assert candidate_payload["candidate_summary"] == result["candidate_summary"]
    camera_candidate = candidate_payload["camera_calibration_candidates"][
        "cameras"
    ]["family-room"]
    assert camera_candidate["rotation"]["admission_status"] == "admissible"
    assert camera_candidate["rotation"]["holdout_used_by_solver"] is False
    assert camera_candidate["rotation"]["fit_waypoint_ids"] == [
        "fit-a",
        "fit-b",
        "fit-c",
        "fit-d",
    ]
    assert camera_candidate["rotation"]["camera_center_preservation_error_m"] < 1e-12
    assert camera_candidate["depth_registration"]["admission_status"] == "admissible"
    assert camera_candidate["depth_registration"]["holdout_used_by_solver"] is False
    assert "holdout-a" not in camera_candidate["depth_registration"][
        "fit_waypoint_ids"
    ]
    holdout_metrics = camera_candidate["metrics"]["splits"]["holdout"]
    assert holdout_metrics["rotation"]["angular_ray_error_deg"]["coverage"] == 1.0
    assert holdout_metrics["depth"]["mapped_optical_depth_error_m"]["coverage"] == 1.0
    derived_index = json.loads(
        (run_dir / "waypoint_calibration_artifact_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(derived_index["source_capture_artifact_index_sha256"]) == 64
    assert set(derived_index["artifacts"]) == {
        "waypoint_calibration_evidence.json",
        "waypoint_calibration_metrics.json",
        "waypoint_candidate_similarity.json",
        "waypoint_camera_calibration_candidates.json",
        "waypoint_calibration_report.json",
        "waypoint_calibration_report.md",
    }
    for path in run_dir.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (
            0o700 if path.is_dir() else 0o600
        )


def _rotation_solver_row(
    *,
    waypoint_id: str,
    split: str,
    sample_sequence: int,
    center: np.ndarray,
    observed_camera_ray: np.ndarray,
    expected_world: np.ndarray,
    post_marker_raw_depth_m: float | None = None,
) -> dict[str, object]:
    observed_camera_ray = observed_camera_ray / np.linalg.norm(
        observed_camera_ray
    )
    source_e = np.eye(4, dtype=np.float64)
    source_e[:3, 3] = -center
    image_foot = [
        100.0 * float(observed_camera_ray[0]) / float(observed_camera_ray[2])
        + 50.0,
        100.0 * float(observed_camera_ray[1]) / float(observed_camera_ray[2])
        + 40.0,
    ]
    sample = {
        "sample_sequence": sample_sequence,
        "camera_id": "family-room",
        "tracklet_key": "tracklet-guided",
        "frame_id": 1000 + sample_sequence,
        "media_pts_ns": 10_000 + sample_sequence,
        "image_foot": image_foot,
        "raw_depth_m": post_marker_raw_depth_m,
        "registered_depth_m": post_marker_raw_depth_m,
        "calibration_bundle_sha256": "b" * 64,
        "floor_world_raw_m": expected_world.tolist(),
        "depth_world_raw_m": expected_world.tolist(),
        "final_world_m": expected_world.tolist(),
        "world_valid": True,
    }
    return {
        "waypoint_id": waypoint_id,
        "camera_id": "family-room",
        "split": split,
        "status": "complete",
        "binding": {
            "sample_sequence": sample_sequence,
            "tracklet_key": "tracklet-guided",
            "image_foot": image_foot,
        },
        "calibration": {
            "K": [100.0, 100.0, 50.0, 40.0],
            "E_world_to_camera_col_major": [
                float(value) for value in source_e.flatten(order="F")
            ],
            "bundle_sha256": "b" * 64,
            "similarity": {"sha256": "s" * 64},
            "floor_y_backend_m": 0.0,
        },
        "observed_camera_ray_geometry": {
            "camera_center_backend_world_m": center.tolist(),
            "observed_unit_ray_camera": observed_camera_ray.tolist(),
        },
        "expected_camera_geometry": {
            "backend_world_m": expected_world.tolist(),
        },
        "depth": {"registered_m": post_marker_raw_depth_m},
        "post_marker_samples": [sample],
    }


def test_fixed_center_rotation_uses_fit_only_and_preserves_nonzero_center() -> None:
    center = np.asarray([2.0, 1.5, -1.0], dtype=np.float64)
    angle = np.deg2rad(17.0)
    expected_rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    observed_rays = (
        np.asarray([-0.35, -0.20, 1.0]),
        np.asarray([0.40, -0.10, 1.0]),
        np.asarray([-0.20, 0.45, 1.0]),
        np.asarray([0.30, 0.35, 1.0]),
    )
    rows = [
        _rotation_solver_row(
            waypoint_id=f"fit-{index}",
            split="fit",
            sample_sequence=index,
            center=center,
            observed_camera_ray=ray,
            expected_world=(
                center
                + expected_rotation @ (ray / np.linalg.norm(ray)) * (3.0 + index)
            ),
        )
        for index, ray in enumerate(observed_rays)
    ]
    rows.append(
        _rotation_solver_row(
            waypoint_id="holdout-poison",
            split="holdout",
            sample_sequence=99,
            center=center,
            observed_camera_ray=np.asarray([0.0, 0.0, 1.0]),
            expected_world=center + np.asarray([20.0, 0.0, 1.0]),
        )
    )

    candidate = _fit_fixed_center_rotation(
        "family-room",
        rows,
        source_capture_artifact_index_sha256="c" * 64,
        physical_target_similarity_sha256="d" * 64,
    )

    assert candidate["status"] == "complete"
    assert candidate["admission_status"] == "admissible"
    assert candidate["holdout_used_by_solver"] is False
    assert candidate["fit_waypoint_ids"] == [
        "fit-0",
        "fit-1",
        "fit-2",
        "fit-3",
    ]
    assert candidate["fixed_camera_center_backend_world_m"] == center.tolist()
    assert candidate["candidate_camera_center_backend_world_m"] == pytest.approx(
        center.tolist(), abs=1e-12
    )
    assert candidate["camera_center_preservation_error_m"] < 1e-12
    solved_rotation = np.asarray(
        candidate["camera_to_world_rotation_row_major"], dtype=np.float64
    ).reshape(3, 3)
    assert solved_rotation == pytest.approx(expected_rotation, abs=1e-10)
    assert candidate["determinant"] == pytest.approx(1.0, abs=1e-12)


def test_fixed_center_rotation_corrects_reflection_and_rejects_bad_fit() -> None:
    center = np.zeros(3, dtype=np.float64)
    reflection = np.diag([-1.0, 1.0, 1.0])
    observed_rays = (
        np.asarray([-0.8, -0.4, 1.0]),
        np.asarray([0.9, -0.3, 1.0]),
        np.asarray([-0.4, 0.8, 1.0]),
        np.asarray([0.7, 0.9, 0.4]),
    )
    rows = [
        _rotation_solver_row(
            waypoint_id=f"fit-reflected-{index}",
            split="fit",
            sample_sequence=index,
            center=center,
            observed_camera_ray=ray,
            expected_world=reflection @ (ray / np.linalg.norm(ray)) * 4.0,
        )
        for index, ray in enumerate(observed_rays)
    ]

    candidate = _fit_fixed_center_rotation(
        "family-room",
        rows,
        source_capture_artifact_index_sha256="c" * 64,
        physical_target_similarity_sha256="d" * 64,
    )

    assert candidate["status"] == "complete"
    assert candidate["reflection_correction_applied"] is True
    assert candidate["determinant"] == pytest.approx(1.0, abs=1e-12)
    assert candidate["admission_status"] == "rejected"
    assert any(
        "reflective" in reason
        for reason in candidate["admission_rejection_reasons"]
    )
    assert candidate["fit_metrics"]["angular_error_deg"]["p95"] > 5.0


def test_fixed_center_rotation_blocks_degenerate_directions() -> None:
    center = np.zeros(3, dtype=np.float64)
    rows = [
        _rotation_solver_row(
            waypoint_id=f"fit-degenerate-{index}",
            split="fit",
            sample_sequence=index,
            center=center,
            observed_camera_ray=np.asarray([0.0, 0.0, 1.0]),
            expected_world=np.asarray([0.0, 0.0, 2.0 + index]),
        )
        for index in range(3)
    ]

    candidate = _fit_fixed_center_rotation(
        "family-room",
        rows,
        source_capture_artifact_index_sha256="c" * 64,
        physical_target_similarity_sha256="d" * 64,
    )

    assert candidate["status"] == "blocked"
    assert candidate["admission_status"] == "blocked"
    assert "degenerate" in candidate["reason"]


def test_camera_candidate_metrics_expose_bad_holdout_geometry_and_depth() -> None:
    center = np.zeros(3, dtype=np.float64)
    fit_rays = (
        np.asarray([-0.3, -0.2, 1.0]),
        np.asarray([0.4, -0.1, 1.0]),
        np.asarray([-0.2, 0.5, 1.0]),
    )
    rows = [
        _rotation_solver_row(
            waypoint_id=f"fit-{index}",
            split="fit",
            sample_sequence=index,
            center=center,
            observed_camera_ray=ray,
            expected_world=(ray / np.linalg.norm(ray)) * 3.0,
            post_marker_raw_depth_m=3.0,
        )
        for index, ray in enumerate(fit_rays)
    ]
    rows.append(
        _rotation_solver_row(
            waypoint_id="holdout-bad",
            split="holdout",
            sample_sequence=10,
            center=center,
            observed_camera_ray=np.asarray([0.0, 0.0, 1.0]),
            expected_world=np.asarray([3.0, 0.0, 3.0]),
            post_marker_raw_depth_m=8.0,
        )
    )
    rotation = _fit_fixed_center_rotation(
        "family-room",
        rows,
        source_capture_artifact_index_sha256="c" * 64,
        physical_target_similarity_sha256="d" * 64,
    )
    assert rotation["admission_status"] == "admissible"
    depth = {
        "status": "complete",
        "admission_status": "admissible",
        "knots_raw_m": [1.0, 10.0],
        "knots_physical_optical_depth_m": [1.0, 10.0],
    }

    metrics = _evaluate_camera_candidates(
        "family-room",
        rows,
        rotation_candidate=rotation,
        depth_candidate=depth,
    )

    holdout = metrics["splits"]["holdout"]
    assert holdout["rotation"]["angular_ray_error_deg"]["coverage"] == 1.0
    assert holdout["rotation"]["angular_ray_error_deg"]["distribution"][
        "p95"
    ] > 40.0
    assert holdout["rotation"]["image_reprojection_error_px"]["distribution"][
        "p95"
    ] == pytest.approx(100.0)
    assert holdout["depth"]["mapped_optical_depth_error_m"]["coverage"] == 1.0
    assert holdout["depth"]["mapped_optical_depth_error_m"]["distribution"][
        "p95"
    ] == pytest.approx(5.0)


def test_report_never_admits_fit_candidate_when_holdout_fails(
    tmp_path: Path,
) -> None:
    run_dir = create_run_directory(
        tmp_path / "evidence", "guided-bad-holdout"
    )
    points = [
        ("fit-a", "fit", [0.0, 0.0, 2.0]),
        ("fit-b", "fit", [1.0, 0.2, 2.0]),
        ("fit-c", "fit", [0.0, -0.2, 3.0]),
        ("fit-d", "fit", [1.0, 0.4, 3.0]),
        ("holdout-a", "holdout", [0.5, 0.1, 2.5]),
    ]
    manifest = {
        "contract": "noesis.alignment.walk_waypoints",
        "contract_version": 1,
        "waypoints": [
            {
                "id": waypoint_id,
                "camera_id": "family-room",
                "expected_scene_xz": [point[0], point[2]],
                "expected_scene_xyz": point,
                "split": split,
            }
            for waypoint_id, split, point in points
        ],
    }
    capture = AlignmentWalkCapture(
        run_dir=run_dir,
        run_id="guided-bad-holdout",
        ws_uri="ws://127.0.0.1:6008",
        scene_binding=_scene_binding(),
    )
    capture.copy_waypoints(manifest)
    capture.process_message(
        {"type": "calibration-bundle", "data": _calibration_bundle()}
    )
    frame = 0
    base_ns = 90_000_000_000
    for waypoint_index, (waypoint_id, split, expected_point) in enumerate(
        points
    ):
        marker_ns = base_ns + waypoint_index * 10_000_000_000
        observed_point = (
            [3.0, 0.0, 3.0] if split == "holdout" else expected_point
        )
        for offset_ns in range(0, 1_000_000_000, 100_000_000):
            capture.process_message(
                _tracking_message(
                    frame,
                    observed_point,
                    [5.0, 0.0, 5.0],
                    walker_depth=(
                        (float(observed_point[2]) - 0.25) / 0.9
                    ),
                ),
                received_at_us=8_000_000 + frame * 100_000,
                received_monotonic_ns=marker_ns + offset_ns,
            )
            if offset_ns == 0:
                append_waypoint_marker(
                    run_dir,
                    waypoint_id=waypoint_id,
                    tracklet_key="tracklet-1",
                    monotonic_ns=marker_ns,
                )
            frame += 1
    capture.finish()
    result = write_waypoint_calibration_report(run_dir)
    assert result["fit_solver_status"] == "admissible"
    assert result["calibration_candidate_status"] == "rejected"
    assert result["candidate_summary"]["calibration_candidate_status"] == (
        "rejected"
    )
    candidate_payload = json.loads(
        (run_dir / "waypoint_candidate_similarity.json").read_text(
            encoding="utf-8"
        )
    )
    assert candidate_payload["candidate_summary"] == result["candidate_summary"]
    assert result["status"] == "fail"


def test_sanitizer_uses_ephemeral_ids_and_omits_sensitive_fields() -> None:
    ids = EphemeralIds()
    message = _tracking_message(4, [0.0, 0.0, 0.0], [5.0, 0.0, 5.0])
    rows = sanitize_tracking_message(
        message,
        ids=ids,
        calibration_bundle_sha256="d" * 64,
        received_at_us=1,
        received_monotonic_ns=2,
        first_sequence=0,
    )
    assert [row["tracklet_key"] for row in rows] == ["tracklet-1", "tracklet-2"]
    assert all("stable_id" not in row and "embedding" not in row and "identity_v2" not in row for row in rows)
    assert rows[0]["calibration_bundle_sha256"] == "d" * 64


def test_verify_detects_artifact_tampering(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path / "evidence", "walk-tamper")
    capture = AlignmentWalkCapture(run_dir=run_dir, run_id="walk-tamper", ws_uri="ws://127.0.0.1:6008")
    capture.process_message({"type": "calibration-bundle", "data": _calibration_bundle()})
    capture.finish()
    with (run_dir / "samples.ndjson").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
        handle.flush()
        os.fsync(handle.fileno())
    verification = verify_run(run_dir)
    assert verification["ok"] is False
    assert any("digest mismatch" in error or "invalid contract" in error for error in verification["errors"])


def test_capture_fails_closed_if_calibration_changes_mid_walk(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path / "evidence", "walk-calibration-change")
    capture = AlignmentWalkCapture(
        run_dir=run_dir,
        run_id="walk-calibration-change",
        ws_uri="ws://user:secret@127.0.0.1:6008/path?token=sensitive",
    )
    first = _calibration_bundle()
    second = _calibration_bundle()
    second["align"]["floor_y"] = 0.25  # type: ignore[index]
    capture.process_message({"type": "calibration-bundle", "data": first})
    capture.process_message({"type": "calibration-bundle", "data": second})
    capture.finish()
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    assert session["status"] == "failed"
    assert session["error"] == "calibration bundle changed during capture"
    assert session["ws_uri"] == "ws://127.0.0.1:6008/path"
    assert "secret" not in (run_dir / "session.json").read_text(encoding="utf-8")
