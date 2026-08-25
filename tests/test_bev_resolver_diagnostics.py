from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from noesis.telemetry.bev import BevRenderer, CalibrationSnapshot, Footpoint, FloorplanSpace
from noesis_core.coordinate_frames import CameraGroundFrame


def _source(*, revision: str = "world-r7", frame_id: int = 42) -> dict:
    covariance = [
        0.04, 0.0, 0.01,
        0.0, 0.25, 0.0,
        0.01, 0.0, 0.09,
    ]
    return {
        "contract": "noesis.world_resolver_diagnostics",
        "version": 1,
        "camera_id": "family-room",
        "source_id": 2,
        "tracker_id": 17,
        "frame_id": frame_id,
        "observed_at_us": 123_456,
        "world_frame": "backend_world_m",
        "world_frame_revision": revision,
        "calibration_revision": "calibration-r9",
        "pcf_revision": "pcf-r3",
        "selected_id": "ankle-1",
        "selected_kind": "floor_ray",
        "decision": "selected_best_supported_candidate",
        "reason": "floor_ray_has_better_support",
        "resolved": {
            "position": {"x": 1.0, "y": 0.0, "z": 3.0},
            "covariance": {"values": covariance},
        },
        "candidates": [
            {
                "id": "ankle-1",
                "kind": "floor_ray",
                "position": {"x": 1.0, "y": 0.0, "z": 3.0},
                "covariance": {"values": covariance},
                "status": "measured",
                "selected": True,
                "pcf": {
                    "prior_id": "pcf-family",
                    "revision_id": "pcf-r3",
                    "status": "observed",
                    "inside_extent": True,
                    "inside_authored_space": True,
                    "extent_outside_distance_m": 0.0,
                    "evidence_observed": True,
                    "observed_confidence": 0.91,
                    "boundary_signed_distance_m": 0.8,
                    "reasons": ["inside_room"],
                },
            },
            {
                "id": "depth-1",
                "kind": "registered_depth",
                "position": {"x": 1.2, "y": 0.0, "z": 3.1},
                "covariance": {"values": covariance},
                "status": "alternate",
                "selected": False,
            },
        ],
        "legacy": {
            "id": "legacy",
            "kind": "legacy_policy",
            "position": {"x": 1.7, "y": 0.0, "z": 3.8},
            "covariance": {"values": covariance},
            "source": "legacy_policy_fixture",
        },
        "disagreement": {"distance_m": 0.71, "reason": "legacy_differs"},
    }


def _normalize(
    source: dict,
    *,
    floorplan_space: FloorplanSpace | None = None,
    expected_source_id: int = 2,
    debug_source_id: int | None = None,
):
    debug = {"world_resolver": source}
    if debug_source_id is not None:
        debug["source_id"] = debug_source_id
    fp = Footpoint(
        frame_id=42,
        tracker_id=17,
        world_x=1.0,
        world_z=3.0,
        debug=debug,
    )
    calib = CalibrationSnapshot(
        camera_id="family-room",
        intrinsics=np.eye(3),
        extrinsics_col_major=np.eye(4).reshape(-1).tolist(),
        floor_y=0.0,
        image_size=(1280, 720),
        world_frame_id="backend_world_m",
        world_frame_revision="world-r7",
        camera_calibration_sha256="calibration-r9",
    )
    return BevRenderer._normalize_resolver_diagnostics(
        fp=fp,
        source=source,
        expected_camera_id="family-room",
        expected_source_id=expected_source_id,
        expected_observed_at_us=123_456,
        expected_world_frame="backend_world_m",
        expected_world_revision="world-r7",
        frame_mode="world",
        calib=calib,
        R_wc=np.eye(3),
        C_world=np.zeros(3),
        camera_ground_frame=None,
        floorplan_alignment=None,
        floorplan_space=floorplan_space,
    )


def test_resolver_diagnostics_preserve_exact_cohort_and_bound_candidates() -> None:
    payload = _normalize(_source())
    assert payload is not None
    assert payload["contract"] == "noesis.world_resolver_diagnostics"
    assert payload["worldFrameRevision"] == "world-r7"
    assert payload["frameId"] == 42
    assert len(payload["candidates"]) == 2
    assert payload["resolved"]["covarianceXZ"] == [[0.04, 0.01], [0.01, 0.09]]
    assert payload["candidates"][0]["display"] == {"x": 1.0, "z": 3.0}
    assert payload["candidates"][0]["pcf"]["extentOutsideDistanceM"] == 0.0
    assert payload["legacy"]["source"] == "legacy_policy_fixture"
    assert payload["sourceId"] == 2
    assert payload["sensorId"] == 2
    assert payload["pcfRevision"] == "pcf-r3"


def test_resolver_diagnostics_keep_world_revision_separate_from_floorplan_snapshot_id() -> None:
    source = _source()
    floorplan = FloorplanSpace(
        x_range=(-2.0, 4.0),
        z_range=(0.0, 8.0),
        snapshot_id="pcf-family",
        snapshot_content_sha256="a" * 64,
        world_frame="backend_world_m",
        world_frame_revision="world-r7",
        calibration_fingerprint="b" * 64,
    )
    payload = _normalize(source, floorplan_space=floorplan)
    assert payload is not None
    assert payload["worldFrameRevision"] == "world-r7"
    assert payload["pcfRevision"] == "pcf-r3"
    assert payload["floorplanSnapshotId"] == "pcf-family"
    assert payload["floorplanWorldFrameRevision"] == "world-r7"

    mismatched_frame = FloorplanSpace(
        x_range=(-2.0, 4.0),
        z_range=(0.0, 8.0),
        snapshot_id="pcf-family",
        snapshot_content_sha256="a" * 64,
        world_frame="backend_world_m",
        world_frame_revision="world-old",
        calibration_fingerprint="b" * 64,
    )
    assert _normalize(source, floorplan_space=mismatched_frame) is None


def test_resolver_diagnostics_accept_raw_source_id_when_bev_uses_mapped_sensor_id() -> None:
    source = _source()
    source["source_id"] = 20
    source["sensor_id"] = 2
    payload = _normalize(
        source,
        expected_source_id=2,
        debug_source_id=20,
    )
    assert payload is not None
    assert payload["sourceId"] == 20
    assert payload["sensorId"] == 2

    source["sensor_id"] = 3
    assert _normalize(
        source,
        expected_source_id=2,
        debug_source_id=20,
    ) is None


def test_resolver_diagnostics_fail_closed_on_contract_revision_or_covariance() -> None:
    assert _normalize({**_source(), "contract": "wrong"}) is None
    assert _normalize(_source(revision="world-old")) is None
    assert _normalize({**_source(), "frame_id": "42"}) is None
    assert _normalize({**_source(), "tracker_id": 18}) is None
    assert _normalize({**_source(), "observed_at_us": 123_457}) is None
    assert _normalize({**_source(), "calibration_revision": "calibration-old"}) is None
    malformed = _source()
    malformed["resolved"]["covariance"]["values"][2] = 0.9
    malformed["resolved"]["covariance"]["values"][6] = 0.1
    assert _normalize(malformed) is None


def test_resolver_diagnostics_reject_more_than_four_candidates() -> None:
    source = _source()
    source["candidates"] = list(source["candidates"]) + [
        {
            "id": f"extra-{index}",
            "kind": "pose_scale",
            "position": {"x": 1.0, "y": 0.0, "z": 3.0},
            "covariance": {"values": [0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04]},
        }
        for index in range(3)
    ]
    assert _normalize(source) is None


def test_resolver_diagnostics_can_be_candidate_only_without_empty_resolved_stub() -> None:
    source = _source()
    source.pop("resolved")
    payload = _normalize(source)
    assert payload is not None
    assert "resolved" not in payload
    assert len(payload["candidates"]) == 2


def test_resolver_covariance_follows_world_to_local_and_floorplan_jacobian() -> None:
    source = _source()
    fp = Footpoint(
        frame_id=42,
        tracker_id=17,
        world_x=1.0,
        world_z=3.0,
        debug={"world_resolver": source},
    )
    calib = CalibrationSnapshot(
        camera_id="family-room",
        intrinsics=np.eye(3),
        extrinsics_col_major=np.eye(4).reshape(-1).tolist(),
        floor_y=0.0,
        image_size=(1280, 720),
        world_frame_id="backend_world_m",
        world_frame_revision="world-r7",
        camera_calibration_sha256="calibration-r9",
    )
    theta = np.deg2rad(31.0)
    camera_ground_frame = CameraGroundFrame(
        camera_world_m=np.zeros(3),
        camera_right_world=np.array([np.cos(theta), 0.0, -np.sin(theta)]),
        camera_forward_world=np.array([np.sin(theta), 0.0, np.cos(theta)]),
    )
    alignment = np.array([[1.0, 0.25, 0.3], [-0.15, 0.85, -0.2]])
    payload = BevRenderer._normalize_resolver_diagnostics(
        fp=fp,
        source=source,
        expected_camera_id="family-room",
        expected_source_id=2,
        expected_observed_at_us=123_456,
        expected_world_frame="backend_world_m",
        expected_world_revision="world-r7",
        frame_mode="camera_local",
        calib=calib,
        R_wc=np.eye(3),
        C_world=np.zeros(3),
        camera_ground_frame=camera_ground_frame,
        floorplan_alignment=alignment,
        floorplan_space=None,
    )
    assert payload is not None
    world_covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    horizontal_jacobian = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    expected_jacobian = alignment[:, :2] @ horizontal_jacobian
    expected = expected_jacobian @ world_covariance @ expected_jacobian.T
    actual = np.asarray(payload["resolved"]["covarianceXZ"])
    assert np.allclose(actual, expected, atol=1e-10)
    assert payload["resolved"]["world"] == {"x": 1.0, "z": 3.0}
    assert payload["resolved"]["display"] != payload["resolved"]["world"]


def test_resolver_diagnostics_require_exact_track_lifecycle_and_transform() -> None:
    source = _source()
    source.update(
        track_key="2:17:3",
        tracker_lifecycle_generation=3,
        world_transform_sha256="a" * 64,
    )
    fp = Footpoint(
        frame_id=42,
        tracker_id=17,
        track_key="2:17:3",
        tracker_lifecycle_generation=3,
        world_transform_sha256="a" * 64,
        world_x=1.0,
        world_z=3.0,
        debug={"world_resolver": source},
    )
    calib = CalibrationSnapshot(
        camera_id="family-room",
        intrinsics=np.eye(3),
        extrinsics_col_major=np.eye(4).reshape(-1).tolist(),
        floor_y=0.0,
        image_size=(1280, 720),
        world_frame_id="backend_world_m",
        world_frame_revision="world-r7",
        camera_calibration_sha256="calibration-r9",
        frame_transform_sha256="a" * 64,
    )
    kwargs = {
        "fp": fp,
        "source": source,
        "expected_camera_id": "family-room",
        "expected_source_id": 2,
        "expected_observed_at_us": 123_456,
        "expected_world_frame": "backend_world_m",
        "expected_world_revision": "world-r7",
        "frame_mode": "world",
        "calib": calib,
        "R_wc": np.eye(3),
        "C_world": np.zeros(3),
        "camera_ground_frame": None,
        "floorplan_alignment": None,
        "floorplan_space": None,
    }
    payload = BevRenderer._normalize_resolver_diagnostics(**kwargs)
    assert payload is not None
    assert payload["trackKey"] == "2:17:3"
    assert payload["trackerLifecycleGeneration"] == 3
    assert payload["worldTransformSha256"] == "a" * 64

    stale_lifecycle = dict(source)
    stale_lifecycle["tracker_lifecycle_generation"] = 4
    assert BevRenderer._normalize_resolver_diagnostics(
        **{**kwargs, "source": stale_lifecycle}
    ) is None
    stale_transform = dict(source)
    stale_transform["world_transform_sha256"] = "b" * 64
    assert BevRenderer._normalize_resolver_diagnostics(
        **{**kwargs, "source": stale_transform}
    ) is None


class _IdentityWs:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    @staticmethod
    def response_model_timing_since(_started_ns):
        return {"duration_ms": 0.0}

    def broadcast_bev_sync(self, payload, **_kwargs):
        self.messages.append(payload)
        return SimpleNamespace(submission_id=len(self.messages), message_count=1)


def _identity_calibration(transform: str) -> CalibrationSnapshot:
    return CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.6, 0.0, 1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
        world_frame_id="backend_world_m",
        world_frame_revision="active-revision",
        frame_transform_sha256=transform,
    )


def test_canonical_bev_admission_requires_frame_id_revision_and_transform_match() -> None:
    ws = _IdentityWs()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )
    renderer.config_per_cam["cam0"] = renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    calibration = _identity_calibration("a" * 64)
    point = Footpoint(
        frame_id=42,
        tracker_id=17,
        track_key="2:17:3",
        tracker_lifecycle_generation=3,
        world_x=0.4,
        world_z=3.59,
        canonical_world_required=True,
        world_frame="backend_world_m",
        world_frame_revision="active-revision",
        world_transform_sha256="a" * 64,
    )

    renderer.render_and_publish(
        "cam0", calibration, footpoints=[point], frame_id=42, timestamp_us=1_000_000
    )
    assert len(ws.messages[-1]["footpoints"]) == 1
    renderer.render_and_publish(
        "cam0", calibration, footpoints=[point], frame_id=43, timestamp_us=2_000_000
    )
    assert ws.messages[-1]["footpoints"] == []
    assert ws.messages[-1]["droppedFootpoints"][0]["reason"] == (
        "canonical_world_frame_id_mismatch"
    )
    stale = Footpoint(**{**point.__dict__, "frame_id": 43, "world_transform_sha256": "b" * 64})
    renderer.render_and_publish(
        "cam0", calibration, footpoints=[stale], frame_id=43, timestamp_us=3_000_000
    )
    assert ws.messages[-1]["footpoints"] == []
    assert ws.messages[-1]["droppedFootpoints"][0]["reason"] == (
        "canonical_world_transform_mismatch"
    )
