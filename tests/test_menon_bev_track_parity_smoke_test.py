from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from scripts import menon_bev_track_parity_smoke_test as gate


REPO_ROOT = Path(__file__).resolve().parents[1]


def _calibration() -> gate._CameraCalibration:
    world_to_camera = np.array(
        [
            [1.0, 0.0, 0.0, -10.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -20.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return gate._CameraCalibration(
        source_id=0,
        camera_id="living-room",
        intrinsics=np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        world_to_camera=world_to_camera,
        floor_y=0.0,
        image_size=(100, 100),
    )


def _tracking(
    *,
    observed_at_us: int = 2_000_000,
    camera_id: str = "living-room",
    source_id: int = 0,
    frame_id: int = 7,
    tracks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    default_tracks: list[dict[str, object]] = [
        {
            "tracker_id": 44,
            "stable_id": 5,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "world": [11.0, 0.0, 22.0],
            "world_valid": True,
            "world_frame": "backend_world_m",
            "image_foot": [60.0, 90.0],
            "bbox": [50.0, 10.0, 20.0, 80.0],
            "depth_status": "ok",
            "depth_registration_status": "ok",
            "depth_registered_m": 4.0,
            "depth_used_m": 4.0,
        }
    ]
    return {
        "type": "tracking",
        "source_id": source_id,
        "frame_id": frame_id,
        "observed_at_us": observed_at_us,
        "camera_id": camera_id,
        "image_size": [100, 100],
        "tracks": default_tracks if tracks is None else tracks,
    }


def _bev(
    *,
    frame: str = "camera_local_ground_m",
    frame_mode: str = "camera_local",
    observed_at_us: int = 2_000_000,
    display_source: str = "world_to_camera_local",
    point: tuple[float, float] = (1.0, 2.0),
    camera_id: str = "living-room",
    source_id: int = 0,
    frame_id: int = 7,
    footpoints: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    default_footpoints: list[dict[str, object]] = [
        {
            "x": point[0],
            "y": point[1],
            "method": "image_foot",
            "stableId": 5,
            "trackerId": 44,
            "frameId": frame_id,
            "displaySource": display_source,
        }
    ]
    return {
        "type": "bev-frame",
        "cameraId": camera_id,
        "sourceId": source_id,
        "frameId": frame_id,
        "observedAtUs": observed_at_us,
        "frame": frame,
        "world_frame": frame,
        "frame_mode": frame_mode,
        "units": "meters",
        "xMin": -5.0,
        "xMax": 5.0,
        "zMin": 0.0,
        "zMax": 8.0,
        "boundsSource": "active_floorplan",
        "floorplanCoordinateSpace": "floorplan_normalized_v1",
        "floorplanBounds": {
            "min_x": -5.0,
            "max_x": 5.0,
            "min_z": 0.0,
            "max_z": 8.0,
        },
        "floorplanGridShape": [16, 20],
        "floorplanGridResM": 0.5,
        "floorplanSnapshotTsUs": 1_500_000,
        "floorplanTsUs": 1_600_000,
        "floorplanSnapshotId": f"snapshot-{camera_id}",
        "floorplanSnapshotContentSha256": "a" * 64,
        "floorplanCalibrationFingerprint": "b" * 64,
        "footpoints": default_footpoints if footpoints is None else footpoints,
        "trails": [],
    }


def _authority(camera_id: str = "living-room") -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "snapshot_ts_us": 1_500_000,
        "floorplan_ts_us": 1_600_000,
        "snapshot_id": f"snapshot-{camera_id}",
        "snapshot_content_sha256": "a" * 64,
        "calibration_fingerprint": "b" * 64,
        "grid_shape": [16, 20],
        "grid_res_m": 0.5,
        "bounds": {
            "min_x": -5.0,
            "max_x": 5.0,
            "min_z": 0.0,
            "max_z": 8.0,
        },
        "ray_to_floorplan_alignment": None,
    }


def _kitchen_calibration() -> gate._CameraCalibration:
    living_room = _calibration()
    return gate._CameraCalibration(
        source_id=1,
        camera_id="kitchen",
        intrinsics=living_room.intrinsics.copy(),
        world_to_camera=living_room.world_to_camera.copy(),
        floor_y=living_room.floor_y,
        image_size=living_room.image_size,
    )


def _evaluate(
    *messages: dict[str, object],
    calibrations: dict[str, gate._CameraCalibration] | None = None,
    authority: dict[str, dict[str, object]] | None = None,
    acquisition_started_at_us: int = 1_000_000,
    acquisition_finished_at_us: int = 3_000_000,
) -> dict[str, object]:
    reviewed = calibrations or {"living-room": _calibration()}
    return gate._evaluate_messages(
        messages,
        contract=gate._FrameContract(
            frame="camera_local_ground_m",
            frame_mode="camera_local",
            units="meters",
        ),
        calibrations=reviewed,
        floorplan_authority=(
            authority
            if authority is not None
            else {camera_id: _authority(camera_id) for camera_id in reviewed}
        ),
        acquisition_started_at_us=acquisition_started_at_us,
        acquisition_finished_at_us=acquisition_finished_at_us,
    )


def test_camera_local_labels_and_calibrated_world_transform_pass() -> None:
    summary = _evaluate(_tracking(), _bev())

    assert summary["local_frame_violations"] == 0
    assert summary["exact_frame_associations"] == 1
    assert summary["comparisons"] == 1
    assert summary["p95_err_m"] == 0.0


def test_relabeling_world_coordinates_as_camera_local_fails() -> None:
    summary = _evaluate(
        _tracking(),
        _bev(frame="backend_world_m", frame_mode="world"),
    )

    assert summary["local_frame_violations"] == 1


def test_incompatible_world_and_local_coordinates_are_never_subtracted_directly() -> None:
    tracking = _tracking()
    track = tracking["tracks"][0]
    expected = gate._expected_local_point(
        display_source="world_to_camera_local",
        point_method="image_foot",
        track=track,
        tracking=tracking,
        bev=_bev(),
        calibration=_calibration(),
        bounds=(-5.0, 5.0, 0.0, 8.0),
        authority=_authority(),
    )

    assert expected == (1.0, 2.0)
    assert expected != (11.0, 22.0)


def test_registered_depth_uses_independent_camera_unprojection() -> None:
    summary = _evaluate(
        _tracking(),
        _bev(display_source="registered_depth_anchor", point=(0.4, 4.0)),
    )

    assert summary["comparisons"] == 1
    assert summary["p95_err_m"] == 0.0


def test_floor_contact_uses_independent_ray_plane_intersection() -> None:
    downward_camera = gate._CameraCalibration(
        source_id=0,
        camera_id="living-room",
        intrinsics=np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        world_to_camera=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        floor_y=0.0,
        image_size=(100, 100),
    )

    point = gate._ray_floor_to_camera_local(
        u=50.0,
        v=100.0,
        calibration=downward_camera,
    )

    assert point is not None
    assert point[0] == 0.0
    assert point[1] == 4.0


def test_unsupported_display_source_fails_closed() -> None:
    summary = _evaluate(
        _tracking(),
        _bev(display_source="world", point=(11.0, 22.0)),
    )

    assert summary["display_source_violations"] == 1
    assert summary["comparisons"] == 0


def test_stale_frame_association_fails() -> None:
    summary = _evaluate(
        _tracking(observed_at_us=2_000_000),
        _bev(observed_at_us=2_000_001),
    )

    assert summary["stale_association_violations"] == 1
    assert summary["exact_frame_associations"] == 0
    assert summary["comparisons"] == 0


def test_wall_clock_stale_messages_cannot_satisfy_a_new_acquisition() -> None:
    summary = _evaluate(
        _tracking(observed_at_us=900_000),
        _bev(observed_at_us=900_000),
    )

    assert summary["freshness_violations"] == 2
    assert summary["paired_bev_frame_count"] == 0


def test_duplicate_bev_frame_is_rejected_instead_of_reusing_one_track_frame() -> None:
    bev = _bev()
    summary = _evaluate(_tracking(), bev, copy.deepcopy(bev))

    assert summary["paired_bev_frame_count"] == 1
    assert summary["duplicate_bev_frames"] == 1


def test_every_configured_camera_requires_an_exact_pair_even_when_empty() -> None:
    calibrations = {
        "living-room": _calibration(),
        "kitchen": _kitchen_calibration(),
    }
    authority = {camera: _authority(camera) for camera in calibrations}
    missing = _evaluate(
        _tracking(),
        _bev(),
        calibrations=calibrations,
        authority=authority,
    )
    assert missing["paired_camera_count"] == 1
    assert missing["missing_paired_cameras"] == ["kitchen"]

    complete = _evaluate(
        _tracking(),
        _bev(),
        _tracking(camera_id="kitchen", source_id=1, frame_id=8, tracks=[]),
        _bev(camera_id="kitchen", source_id=1, frame_id=8, footpoints=[]),
        calibrations=calibrations,
        authority=authority,
    )
    assert complete["paired_camera_count"] == 2
    assert complete["missing_paired_cameras"] == []
    assert complete["footpoint_total"] == 1


def test_floorplan_snapshot_identity_is_bound_to_each_bev_frame() -> None:
    bev = _bev()
    bev["floorplanSnapshotId"] = "substituted-snapshot"
    summary = _evaluate(_tracking(), bev)

    assert summary["floorplan_authority_violations"] == 1
    assert summary["paired_bev_frame_count"] == 0


def test_declared_geometry_without_independent_inputs_fails_closed() -> None:
    tracking = _tracking()
    track = tracking["tracks"][0]
    assert isinstance(track, dict)
    track.pop("depth_registered_m")
    summary = _evaluate(
        tracking,
        _bev(display_source="registered_depth_anchor", point=(0.4, 4.0)),
    )

    assert summary["geometry_unavailable"] == 1
    assert summary["comparisons"] == 0


def test_registered_depth_oracle_rejects_invalid_status_and_used_mismatch() -> None:
    for updates in (
        {"depth_status": "no_depth"},
        {"depth_used_m": 8.0},
    ):
        tracking = _tracking()
        track = tracking["tracks"][0]
        assert isinstance(track, dict)
        track.update(updates)
        summary = _evaluate(
            tracking,
            _bev(display_source="registered_depth_anchor", point=(0.4, 4.0)),
        )

        assert summary["geometry_unavailable"] == 1
        assert summary["comparisons"] == 0


def test_all_reviewed_runtime_configs_and_ownership_lock_camera_local_frame() -> None:
    for relative in (
        "config/infer.yaml",
        "config/infer_v3dt_baseline.yaml",
        "DS9/config/infer.yaml",
        "DS9/config/infer_v3dt.yaml",
    ):
        contract = gate._load_frame_contract(REPO_ROOT / relative)
        assert contract == gate._FrameContract(
            frame="camera_local_ground_m",
            frame_mode="camera_local",
            units="meters",
        )

    ownership = yaml.safe_load(
        (REPO_ROOT / "DS9/docs/runtime_ownership.yaml").read_text(encoding="utf-8")
    )
    capability = next(
        row
        for row in ownership["capabilities"]
        if row["id"] == "config.local_bev_frame"
    )
    bindings = capability["evidence"]["repository_source"]
    assert {
        row["yaml_equals"]["bev.frame"] for row in bindings.values()
    } == {"camera_local_ground_m"}
