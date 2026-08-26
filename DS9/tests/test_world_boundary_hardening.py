from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from noesis.calibration.world_fusion_policy import (
    CameraWorldFusionProfile,
    WorldFusionPolicy,
)
from noesis.pipelines import hooks
from noesis.telemetry.person_ground_state import PersonGroundState
from noesis_core.runtime_publication import RuntimePublicationGate


def _extrinsics_for_camera(camera_world: tuple[float, float, float]) -> list[float]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = -np.asarray(camera_world, dtype=np.float64)
    return list(matrix.flatten(order="F"))


def _calibration(
    camera_world: tuple[float, float, float],
    *,
    revision: str,
    transform: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        camera_id="cam0",
        intrinsics=np.array(
            [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        extrinsics_col_major=_extrinsics_for_camera(camera_world),
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=1.0,
        world_frame_id="backend_world_m",
        world_frame_revision=revision,
        frame_transform_sha256=transform,
    )


def _processor(raw: SimpleNamespace, active: SimpleNamespace) -> hooks._AnalyticsTelemetryProcessor:
    provider = SimpleNamespace(
        snapshot=lambda _source_id, _camera_id: raw,
        world_snapshot=lambda _source_id, _camera_id: active,
    )
    policy = WorldFusionPolicy(
        policy_id="world-boundary-test",
        evidence={},
        cameras={
            "cam0": CameraWorldFusionProfile(
                camera_id="cam0",
                calibration_fingerprint_sha256="a" * 64,
                registration_id="reg-test",
                floor_weight_scale=1.0,
                depth_weight_scale=1.0,
                floor_only_allowed=True,
                floor_ray_max_range_m=20.0,
            )
        },
    )
    return hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={"models": {"pose": {"kpt_threshold": 0.35}}}),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=provider,
        world_fusion_policy=policy,
    )


def _seeded_track(tracker_id: int, *, source: str, world: list[float]) -> dict[str, object]:
    return {
        "tracker_id": tracker_id,
        "bbox": [100.0, 100.0, 80.0, 180.0],
        "bbox3d": {"x": world[0], "y": world[1], "z": world[2]},
        "world": list(world),
        "world_valid": True,
        "world_source": source,
        "world_frame": "backend_world_m",
    }


def test_bbox3d_range_uses_revision_bound_world_snapshot_with_nonidentity_transform() -> None:
    # The active target-frame camera is translated 10 m from the raw source
    # frame. The candidate is inside the 20 m envelope only in that active
    # frame; a raw-snapshot check would reject it.
    raw = _calibration((0.0, 2.2, -6.0), revision="rev-1", transform="raw")
    active = _calibration((10.0, 2.2, -6.0), revision="rev-1", transform="active")
    processor = _processor(raw, active)
    track = _seeded_track(11, source="bbox3d", world=[20.0, 0.0, 10.0])

    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_source_label="bbox3d",
        world_now_ts=100.0,
    )

    assert track["world_valid"] is True
    assert track["world_frame_revision"] == "rev-1"
    assert track["world_observation_range_admitted"] is True
    assert track["world_observation_range_m"] < 20.0


def test_non_bbox3d_preseed_is_admitted_or_rejected_by_canonical_range_gate() -> None:
    raw = _calibration((0.0, 2.2, -6.0), revision="rev-1", transform="raw")
    active = _calibration((10.0, 2.2, -6.0), revision="rev-1", transform="active")
    processor = _processor(raw, active)
    track = _seeded_track(12, source="external_preseed", world=[40.0, 0.0, 10.0])
    track["world_frame_revision"] = "rev-1"
    track["world_transform_sha256"] = "active"
    track.pop("bbox3d")

    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_now_ts=100.0,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "world_observation_range_exceeded"
    assert track["world_observation_range_admitted"] is False


def test_out_of_range_seeded_hold_is_invalid_and_does_not_publish() -> None:
    raw = _calibration((0.0, 2.2, -6.0), revision="rev-1", transform="raw")
    active = _calibration((10.0, 2.2, -6.0), revision="rev-1", transform="active")
    processor = _processor(raw, active)
    processor._world_state_by_track[(0, 13)] = PersonGroundState(  # type: ignore[attr-defined]
        world_x=40.0,
        world_z=10.0,
        filtered_ts=99.0,
        last_good_world=(40.0, 0.0, 10.0),
        last_good_ts=99.0,
    )
    track = _seeded_track(13, source="bbox3d", world=[40.0, 0.0, 10.0])

    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_source_label="bbox3d",
        world_now_ts=100.0,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "world_observation_range_exceeded"
    assert track["trail_append_allowed"] is False


def test_refinement_exception_clears_arbitrary_world_valid() -> None:
    provider = SimpleNamespace(
        snapshot=lambda _source_id, _camera_id: (_ for _ in ()).throw(RuntimeError("calibration boom")),
        world_snapshot=lambda _source_id, _camera_id: (_ for _ in ()).throw(RuntimeError("calibration boom")),
    )
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=provider,
    )
    track = _seeded_track(14, source="external_preseed", world=[1.0, 0.0, 1.0])
    track.pop("bbox3d")

    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_now_ts=100.0,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "world_refinement_failed"


def test_v3dt_bbox3d_without_calibration_is_invalid() -> None:
    raw = _calibration((0.0, 2.2, -6.0), revision="rev-1", transform="raw")
    active = _calibration((10.0, 2.2, -6.0), revision="rev-1", transform="active")
    processor = _processor(raw, active)
    processor._tracking_mode = "v3dt"  # type: ignore[attr-defined]
    processor.bev_calibration = None
    track = _seeded_track(15, source="bbox3d", world=[1.0, 0.0, 1.0])

    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_now_ts=100.0,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "calibration_unavailable"
