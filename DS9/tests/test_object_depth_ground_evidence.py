from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from geometry.homography import project_world_to_image
from noesis.calibration.world_fusion_policy import (
    CameraWorldFusionProfile,
    WorldFusionPolicy,
)
from noesis.metadata.object_depth import ObjectDepthResult
from noesis.pipelines import hooks
from noesis_core.runtime_publication import RuntimePublicationGate


def _pose_payload() -> dict[str, Any]:
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, point in {
        5: (25.0, 40.0),
        6: (75.0, 40.0),
        11: (35.0, 100.0),
        12: (65.0, 100.0),
        13: (35.0, 145.0),
        14: (65.0, 145.0),
        15: (30.0, 190.0),
        16: (70.0, 190.0),
    }.items():
        keypoints[index] = [point[0], point[1], 0.9]
    return {
        "bbox": [100.0, 100.0, 100.0, 200.0],
        "keypoints_roi": keypoints,
    }


def _anchor_calibration() -> SimpleNamespace:
    intrinsics = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_world = np.array([0.0, 2.2, -6.0], dtype=np.float64)
    forward = np.array([0.0, -0.25, 1.0], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    down /= np.linalg.norm(down)
    rotation_cw = np.column_stack([right, down, forward]).T
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = rotation_cw
    extrinsics[:3, 3] = -rotation_cw @ camera_world
    return SimpleNamespace(
        camera_id="cam0",
        intrinsics=intrinsics,
        extrinsics_col_major=list(extrinsics.flatten(order="F")),
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=1.0,
    )


def _depth_only_policy() -> WorldFusionPolicy:
    profile = CameraWorldFusionProfile(
        camera_id="cam0",
        calibration_fingerprint_sha256="a" * 64,
        registration_id="reg-test",
        floor_weight_scale=0.0,
        depth_weight_scale=1.0,
        floor_only_allowed=False,
        floor_ray_max_range_m=22.0,
    )
    return WorldFusionPolicy(
        policy_id="test-depth-only",
        evidence={},
        cameras={"cam0": profile},
    )


def _world_pose_and_bbox(
    calib: SimpleNamespace,
    *,
    foot_z: float,
) -> tuple[np.ndarray, list[float]]:
    def project(world_xyz: tuple[float, float, float]) -> tuple[float, float]:
        uv = project_world_to_image(
            world_xyz,
            calib.intrinsics,
            calib.extrinsics_col_major,
            calib.image_size,
            unit_scale=1.0,
        )
        assert uv is not None
        return float(uv[0]), float(uv[1])

    foot_u, foot_v = project((0.0, 0.0, float(foot_z)))
    _head_u, head_v = project((0.0, 1.8, float(foot_z)))
    bbox = [foot_u - 40.0, head_v, 80.0, foot_v - head_v]
    keypoints = np.zeros((17, 3), dtype=np.float64)
    world_points = {
        "left_shoulder": (-0.18, 1.47, foot_z),
        "right_shoulder": (0.18, 1.47, foot_z),
        "left_hip": (-0.10, 1.00, foot_z),
        "right_hip": (0.10, 1.00, foot_z),
        "left_knee": (-0.10, 0.50, foot_z),
        "right_knee": (0.10, 0.50, foot_z),
        "left_ankle": (-0.10, 0.00, foot_z),
        "right_ankle": (0.10, 0.00, foot_z),
    }
    for name, world_xyz in world_points.items():
        u, v = project(tuple(float(value) for value in world_xyz))
        keypoints[hooks._POSE_KPT_INDEX[name]] = [u, v, 0.98]
    return keypoints, [float(value) for value in bbox]


class _MaskedStatsDevice:
    def __init__(self, *, contaminate_contact: bool = False) -> None:
        self.contaminate_contact = bool(contaminate_contact)
        self.calls: list[tuple[int, int, int, int, int]] = []
        self.capsule_calls: list[tuple[list[list[float]], int]] = []

    def sample_pose_capsule_stats(
        self,
        capsules: list[list[float]],
        max_samples: int,
    ) -> dict[str, float | int | None]:
        self.capsule_calls.append((capsules, max_samples))
        active = 40
        contaminated = self.contaminate_contact
        p10 = 3.0 if contaminated else 4.0
        p90 = 5.0 if contaminated else 4.0
        return {
            "roi_area_px": active,
            "capsule_count": len(capsules),
            "sampled_area_px": active,
            "sampled_capsule_area_px": active,
            "sample_count": active,
            "valid_fraction": 1.0,
            "depth_center": None,
            "depth_median": 4.0,
            "depth_mean": 4.0,
            "depth_p10": p10,
            "depth_p90": p90,
            "depth_min": p10,
            "depth_max": p90,
        }

    def sample_roi_stats(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        max_samples: int,
    ) -> dict[str, float | int]:
        active = int(width) * int(height)
        self.calls.append((left, top, width, height, active))
        contaminated = self.contaminate_contact
        p10 = 3.0 if contaminated else 4.0
        p90 = 5.0 if contaminated else 4.0
        return {
            "roi_area_px": int(width) * int(height),
            "mask_area_px": active,
            "sample_count": active,
            "valid_fraction": 1.0 if active > 0 else 0.0,
            "depth_center": 4.0,
            "depth_median": 4.0 if active > 0 else None,
            "depth_mean": 4.0 if active > 0 else None,
            "depth_p10": p10 if active > 0 else None,
            "depth_p90": p90 if active > 0 else None,
            "depth_min": p10 if active > 0 else None,
            "depth_max": p90 if active > 0 else None,
        }


def _run_pose_depth_sample(
    monkeypatch: Any,
    *,
    contaminate_contact: bool = False,
    ankles_available: bool = True,
) -> tuple[dict[str, Any], _MaskedStatsDevice, hooks._ObjectDepthFusionProcessor]:
    pts_us = 2_000_000
    device = _MaskedStatsDevice(contaminate_contact=contaminate_contact)
    store = hooks._AlignedDepthFrameStore()
    store.put(
        hooks._AlignedDepthFrame(
            key=(0, 50, pts_us),
            source_id=0,
            frame_id=50,
            pts_us=pts_us,
            depth_map=None,
            valid_mask=None,
            frame_w=1920,
            frame_h=1080,
            depth_w=518,
            depth_h=294,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            depth_device_frame=device,
        )
    )
    processor = hooks._ObjectDepthFusionProcessor(
        depth_store=store,
        fallback_frame_size=(1920, 1080),
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        depth_every_n_frames=2,
    )
    attached: list[dict[str, Any]] = []
    pose_payload = _pose_payload()
    if not ankles_available:
        for index in (15, 16):
            pose_payload["keypoints_roi"][index][2] = 0.0
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        SimpleNamespace(extract_pose_features=lambda _obj: json.dumps(pose_payload)),
    )
    monkeypatch.setattr(
        hooks,
        "noesis_depth_meta_ext",
        SimpleNamespace(
            extract_object_mask=lambda _obj: None,
            attach_object_depth=lambda _batch, _obj, payload, _replace: (
                attached.append(json.loads(payload)) or True
            ),
        ),
    )
    obj_meta = SimpleNamespace(
        object_id=7,
        class_id=0,
        confidence=0.9,
        rect_params=SimpleNamespace(
            left=100.0,
            top=100.0,
            width=100.0,
            height=200.0,
        ),
    )
    frame_meta = SimpleNamespace(
        source_id=0,
        frame_number=50,
        frame_width=1920,
        frame_height=1080,
        buf_pts=pts_us * 1000,
        object_items=[obj_meta],
    )
    processor.handle_servicemaker_frame(SimpleNamespace(frame_items=[frame_meta]), frame_meta)
    assert len(attached) == 1
    return attached[0], device, processor


def test_pose_capsule_uses_observed_ankles_for_native_depth_support(monkeypatch) -> None:
    payload, device, _processor = _run_pose_depth_sample(monkeypatch)

    assert device.calls == []
    assert len(device.capsule_calls) == 1
    capsules, _max_samples = device.capsule_calls[0]
    assert len(capsules) == 2
    assert all(len(capsule) == 6 for capsule in capsules)
    assert all(capsule[4] < capsule[5] for capsule in capsules)
    assert payload["status"] == "ok"
    assert payload["sampling_mode"] == "pose_capsule_native"
    assert payload["anchor_source"] == "pose_ankle_support"
    assert payload["anchor_uv"] == [150.0, 290.0]
    assert payload["anchor_depth_m"] == 4.0
    assert payload["anchor_depth_spread_m"] == 0.0
    assert payload["evidence_quality"] == "good"
    assert payload["measurement_frame_id"] == 50
    assert payload["measurement_ts_us"] == 2_000_000
    assert payload["measurement_age_us"] == 0
    assert payload["measurement_cached"] is False


def test_pose_contact_mask_exactly_matches_compound_native_predicate() -> None:
    processor = object.__new__(hooks._ObjectDepthFusionProcessor)
    payload = _pose_payload()
    keypoints = np.asarray(payload["keypoints_roi"], dtype=np.float32)
    keypoints[:, 0] += 100.0
    keypoints[:, 1] += 100.0

    _body, contact, _uv, visible, contacts = processor._pose_capsule_masks(
        keypoints,
        crop_rect=(100, 100, 200, 300),
        bbox=(100.0, 100.0, 100.0, 200.0),
    )

    assert visible == 2
    assert len(contacts) == 2
    grid_y, grid_x = np.ogrid[100:300, 100:200]
    px = np.asarray(grid_x, dtype=np.float32) + 0.5
    py = np.asarray(grid_y, dtype=np.float32) + 0.5
    expected = np.zeros(contact.shape, dtype=bool)
    old_wide_capsule = np.zeros(contact.shape, dtype=bool)
    ankle_disks = np.zeros(contact.shape, dtype=bool)
    for ax, ay, bx, by, line_radius, ankle_radius in contacts:
        dx = bx - ax
        dy = by - ay
        length_sq = (dx * dx) + (dy * dy)
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / length_sq, 0.0, 1.0)
        segment_distance_sq = np.square(px - (ax + t * dx)) + np.square(
            py - (ay + t * dy)
        )
        ankle_distance_sq = np.square(px - bx) + np.square(py - by)
        expected |= (segment_distance_sq <= line_radius * line_radius) | (
            ankle_distance_sq <= ankle_radius * ankle_radius
        )
        old_wide_capsule |= segment_distance_sq <= ankle_radius * ankle_radius
        ankle_disks |= ankle_distance_sq <= ankle_radius * ankle_radius

    assert np.array_equal(contact, expected)
    # The prior native geometry admitted a furniture-prone strip beside the
    # lower leg.  The corrected primitive removes that strip while retaining
    # the full ankle support disk.
    removed_lower_leg_strip = old_wide_capsule & ~contact & ~ankle_disks
    assert int(np.count_nonzero(removed_lower_leg_strip)) > 0


def test_pose_capsule_rejects_furniture_prone_depth_spread(monkeypatch) -> None:
    payload, _device, _processor = _run_pose_depth_sample(
        monkeypatch,
        contaminate_contact=True,
    )

    assert payload["status"] == "ambiguous_depth"
    assert payload["evidence_quality"] == "rejected"
    assert payload["evidence_reason"] == "depth_spread_exceeded"
    assert payload["anchor_depth_spread_m"] == 2.0
    assert "anchor_source" not in payload
    assert "anchor_depth_m" not in payload
    assert "anchor_uv" not in payload


def test_pose_without_observed_ankles_does_not_sample_gpu_depth(monkeypatch) -> None:
    payload, device, _processor = _run_pose_depth_sample(
        monkeypatch,
        ankles_available=False,
    )

    assert device.calls == []
    assert payload["status"] == "no_ground_contact"
    assert payload["evidence_quality"] == "rejected"
    assert payload["evidence_reason"] == "pose_ankles_unavailable"
    assert "anchor_source" not in payload
    assert "anchor_depth_m" not in payload


def test_cached_depth_keeps_original_measurement_time_and_provenance(monkeypatch) -> None:
    payload, _device, processor = _run_pose_depth_sample(monkeypatch)
    obj_meta = SimpleNamespace(object_id=7, confidence=0.8)

    cached = processor._cached_payload(
        source_id=0,
        frame_id=51,
        pts_us=2_100_000,
        obj_meta=obj_meta,
        bbox=payload["bbox"],
    )

    assert cached is not None
    assert cached["frame_id"] == 51
    assert cached["ts_us"] == 2_100_000
    assert cached["measurement_frame_id"] == 50
    assert cached["measurement_ts_us"] == 2_000_000
    assert cached["measurement_age_us"] == 100_000
    assert cached["measurement_cached"] is True
    assert cached["sampling_mode"] == "pose_capsule_native"
    assert cached["anchor_source"] == "pose_ankle_support"


def test_cached_depth_is_diagnostic_only_for_current_world_observation() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor.depth_registration = None
    processor._project_pixel_to_world_observation = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("stale depth must not be projected as a current observation")
    )
    result = ObjectDepthResult(
        source_id=0,
        frame_id=51,
        object_id=7,
        class_id=0,
        bbox=(10.0, 20.0, 30.0, 40.0),
        score=0.9,
        sampling_mode="pose_capsule_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=84,
        valid_fraction=0.88,
        anchor_uv=(25.0, 59.0),
        anchor_source="pose_ankle_support",
        anchor_depth_m=5.5,
        anchor_sample_count=84,
        anchor_valid_fraction=0.88,
        measurement_frame_id=50,
        measurement_ts_us=2_000_000,
        measurement_age_us=100_000,
        measurement_cached=True,
        depth_tensor_frame_id=49,
        depth_tensor_ts_us=1_966_667,
        depth_tensor_age_frames=1,
        depth_tensor_age_us=33_333,
        ts_us=2_100_000,
    )

    observation = processor._depth_observation_from_anchor(
        calib=_anchor_calibration(),
        anchor=hooks._PoseAnchorCandidate(
            u=25.0,
            v=59.0,
            source="pose_ankle_floor",
            contact_basis="pose:ankle_pair",
        ),
        depth_result=result,
        flip_u=False,
        flip_v=False,
    )

    assert observation.world_point is None
    assert observation.weight == 0.0
    assert observation.reason == "depth_measurement_not_current"
    assert observation.raw_depth_m == 5.5
    assert observation.registered_depth_m is None


def test_fresh_cached_fresh_depth_publishes_bounded_monotonic_cv_prediction(
    monkeypatch,
) -> None:
    calibration = _anchor_calibration()
    provider = SimpleNamespace(
        snapshot=lambda _sensor_id, _camera_id: calibration,
        world_snapshot=lambda _sensor_id, _camera_id: calibration,
    )
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={"models": {"pose": {"kpt_threshold": 0.35}}}
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=provider,
        world_fusion_policy=_depth_only_policy(),
    )
    now = [100.0]
    monkeypatch.setattr(hooks.time, "time", lambda: now[0])
    camera_world = np.array([0.0, 2.2, -6.0], dtype=np.float64)

    def run_frame(
        frame_id: int,
        at_s: float,
        foot_z: float,
        measured_z: float,
        measurement_frame_id: int,
        measurement_ts_us: int,
        cached: bool,
    ) -> dict[str, Any]:
        now[0] = float(at_s)
        keypoints, bbox = _world_pose_and_bbox(calibration, foot_z=foot_z)
        current_ts_us = int(round(float(at_s) * 1_000_000.0))
        range_m = float(
            np.linalg.norm(
                np.array([0.0, 0.0, measured_z], dtype=np.float64)
                - camera_world
            )
        )
        depth_result = ObjectDepthResult(
            source_id=0,
            frame_id=frame_id,
            object_id=840,
            class_id=0,
            bbox=bbox,
            score=0.91,
            sampling_mode="pose_capsule_native",
            status="ok",
            unit="m",
            is_metric=True,
            sample_count=96,
            valid_fraction=0.96,
            anchor_uv=(
                float((keypoints[15, 0] + keypoints[16, 0]) * 0.5),
                float((keypoints[15, 1] + keypoints[16, 1]) * 0.5),
            ),
            anchor_source="pose_ankle_support",
            anchor_depth_m=range_m,
            anchor_sample_count=96,
            anchor_valid_fraction=0.96,
            measurement_frame_id=measurement_frame_id,
            measurement_ts_us=measurement_ts_us,
            measurement_age_us=max(0, current_ts_us - measurement_ts_us),
            measurement_cached=cached,
            ts_us=current_ts_us,
        )
        track: dict[str, Any] = {
            "tracker_id": 840,
            "frame_id": frame_id,
            "bbox": bbox,
            "image_size": list(calibration.image_size),
        }
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=keypoints,
            depth_result=depth_result,
        )
        processor._apply_public_depth_fields(track, depth_result)
        return track

    first = run_frame(1, 100.0, 6.0, 6.0, 1, 100_000_000, False)
    second = run_frame(2, 100.2, 6.8, 6.8, 2, 100_200_000, False)
    state = processor._world_state_by_track[(0, 840)]
    last_good_before_cached = tuple(state.last_good_world or ())
    last_good_ts_before_cached = float(state.last_good_ts)

    cached_one = run_frame(3, 100.267, 7.0, 6.8, 2, 100_200_000, True)
    cached_two = run_frame(4, 100.333, 7.2, 6.8, 2, 100_200_000, True)

    assert first["world_measurement_accepted"] is True
    assert second["world_measurement_accepted"] is True
    assert "_world_current_image_foot_calib" not in second
    assert second["motion_mode"] == "walk"
    for predicted in (cached_one, cached_two):
        assert predicted["world_source"] == "image_motion_prediction"
        assert "_world_current_image_foot_calib" not in predicted
        assert predicted["world_measurement_accepted"] is False
        assert predicted["world_rejection_reason"] == "depth_measurement_not_current"
        assert predicted["world_quality_reason"] == (
            "depth_measurement_not_current,predicted_from_accepted_image_motion"
        )
        assert predicted["world_prediction_provenance"]["non_authoritative"] is True
        assert predicted["world_prediction_provenance"]["origin"] == (
            "last_accepted_image_foot"
        )
        assert predicted["world_prediction_provenance"]["state_integrated"] is True
        assert predicted["world"] == pytest.approx(
            predicted["world_filter_prediction"]
        )
        assert predicted["depth_measurement_cached"] is True
        assert predicted["depth_used_m"] is None
        assert predicted["trail_break_required"] is False
        assert predicted["trail_append_allowed"] is True
        assert predicted["world_contact_basis"] == "pose:ankle_pair"
    assert second["world"][2] < cached_one["world"][2] < cached_two["world"][2]
    assert tuple(state.last_good_world or ()) == pytest.approx(
        last_good_before_cached
    )
    assert state.last_good_ts == pytest.approx(last_good_ts_before_cached)

    # A metric outlier beyond the calibrated camera envelope is rejected
    # before it can seed or perturb the accepted metric state.  The independent
    # accepted-image-foot continuation may still advance the bounded process
    # posterior, while retaining the rejection and non-authoritative status.
    rejected = run_frame(5, 100.4, 7.4, 20.0, 5, 100_400_000, False)
    assert rejected["world_source"] == "image_motion_prediction"
    assert rejected["world_measurement_accepted"] is False
    assert rejected["world_rejection_reason"] == "world_observation_range_exceeded"
    assert rejected["world_quality_reason"] == (
        "world_observation_range_exceeded,predicted_from_accepted_image_motion"
    )
    assert rejected["world_prediction_provenance"]["type"] == (
        "bbox_affine_floor_projection"
    )
    assert rejected["world_prediction_provenance"]["state_integrated"] is True
    assert rejected["trail_append_allowed"] is True
    assert tuple(state.last_good_world or ()) == pytest.approx(
        last_good_before_cached
    )
    assert state.last_good_ts == pytest.approx(last_good_ts_before_cached)

    # The existing 0.40 s display TTL is a hard fail-closed boundary.
    expired = run_frame(6, 100.601, 7.6, 20.0, 5, 100_400_000, True)
    assert expired["world_valid"] is False
    assert "world_source" not in expired
    assert expired["world_measurement_accepted"] is False

    fresh = run_frame(7, 100.8, 7.8, 7.8, 7, 100_800_000, False)
    assert fresh["world_measurement_accepted"] is True
    assert fresh["world_source"] == "pose_depth_only"
    assert fresh["world"][2] > cached_two["world"][2]
    assert fresh["trail_break_required"] is False
    assert state.last_good_ts == pytest.approx(100.8)


def test_world_anchor_rejects_legacy_bbox_rectangle_even_if_marked_ok() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    bbox_depth = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=7,
        class_id=0,
        bbox=(10.0, 20.0, 30.0, 40.0),
        score=0.9,
        sampling_mode="bbox_band_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=600,
        valid_fraction=1.0,
        anchor_uv=(25.0, 59.0),
        anchor_source="lower_body_band",
        anchor_depth_m=5.5,
        anchor_sample_count=600,
        anchor_valid_fraction=1.0,
    )

    assert processor._resolve_person_depth_anchor(bbox_depth) is None


def test_public_depth_fields_expose_and_clear_evidence_provenance() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    result = ObjectDepthResult(
        source_id=0,
        frame_id=51,
        object_id=7,
        class_id=0,
        bbox=(10.0, 20.0, 30.0, 40.0),
        score=0.9,
        sampling_mode="pose_capsule_native",
        status="ambiguous_depth",
        unit="m",
        is_metric=True,
        sample_count=84,
        valid_fraction=0.88,
        depth_spread_m=1.4,
        anchor_depth_spread_m=2.0,
        evidence_quality="rejected",
        evidence_reason="depth_spread_exceeded",
        measurement_frame_id=50,
        measurement_ts_us=2_000_000,
        measurement_age_us=100_000,
        measurement_cached=True,
        depth_tensor_frame_id=49,
        depth_tensor_ts_us=1_966_667,
        depth_tensor_age_frames=1,
        depth_tensor_age_us=33_333,
    )
    track: dict[str, Any] = {}

    processor._apply_public_depth_fields(track, result)

    assert track["depth_evidence_quality"] == "rejected"
    assert track["depth_evidence_reason"] == "depth_spread_exceeded"
    assert track["depth_spread_m"] == 1.4
    assert track["depth_anchor_spread_m"] == 2.0
    assert track["depth_measurement_frame_id"] == 50
    assert track["depth_measurement_ts_us"] == 2_000_000
    assert track["depth_measurement_age_us"] == 100_000
    assert track["depth_measurement_cached"] is True
    assert track["depth_tensor_frame_id"] == 49
    assert track["depth_tensor_ts_us"] == 1_966_667
    assert track["depth_tensor_age_frames"] == 1
    assert track["depth_tensor_age_us"] == 33_333

    processor._apply_public_depth_fields(track, None)

    assert track["depth_evidence_quality"] is None
    assert track["depth_evidence_reason"] is None
    assert track["depth_spread_m"] is None
    assert track["depth_anchor_spread_m"] is None
    assert track["depth_measurement_frame_id"] is None
    assert track["depth_measurement_ts_us"] is None
    assert track["depth_measurement_age_us"] is None
    assert track["depth_measurement_cached"] is None
    assert track["depth_tensor_frame_id"] is None
    assert track["depth_tensor_ts_us"] is None
    assert track["depth_tensor_age_frames"] is None
    assert track["depth_tensor_age_us"] is None


def test_seated_pose_without_ankles_does_not_reuse_upright_gravity_drop(
    monkeypatch,
) -> None:
    calibration = _anchor_calibration()
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={"models": {"pose": {"kpt_threshold": 0.35}}}
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=SimpleNamespace(
            snapshot=lambda _sensor_id, _camera_id: calibration
        ),
    )
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, point in {
        5: (600.0, 200.0),
        6: (680.0, 200.0),
        11: (610.0, 300.0),
        12: (670.0, 300.0),
        13: (650.0, 315.0),
        14: (700.0, 315.0),
    }.items():
        keypoints[index] = [point[0], point[1], 0.98]
    pose_payload = {
        "bbox": [560.0, 180.0, 180.0, 190.0],
        "keypoints_abs": keypoints,
    }
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        SimpleNamespace(
            extract_pose_features=lambda _obj: json.dumps(pose_payload)
        ),
    )
    state = hooks._WorldAnchorState(
        ts=float(hooks.time.time()),
        height_ref_scene=1.8,
    )
    processor._world_state_by_track[(0, 77)] = state
    track = {"tracker_id": 77, "bbox": pose_payload["bbox"]}

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
    )

    assert state.posture == "sitting"
    assert track["world_valid"] is False
    assert "world" not in track
    assert track.get("world_source") != "gravity_drop"


def test_no_ground_contact_drops_leg_extension_anchor_even_when_posture_unknown() -> None:
    calibration = _anchor_calibration()
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={"models": {"pose": {"kpt_threshold": 0.35}}}
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=SimpleNamespace(
            snapshot=lambda _sensor_id, _camera_id: calibration
        ),
    )
    # Hip/knee geometry alone would previously be extrapolated to a synthetic
    # ankle and floor-projected.  The native depth result explicitly says that
    # no ground contact was observed, so that estimate must fail closed.
    keypoints = np.zeros((17, 3), dtype=np.float64)
    for index, point in {
        5: (600.0, 200.0),
        6: (680.0, 200.0),
        11: (610.0, 300.0),
        12: (670.0, 300.0),
        13: (650.0, 355.0),
        14: (700.0, 355.0),
    }.items():
        keypoints[index] = [point[0], point[1], 0.98]
    bbox = [560.0, 180.0, 180.0, 300.0]
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=78,
        class_id=0,
        bbox=bbox,
        score=0.95,
        sampling_mode="pose_capsule_native",
        status="no_ground_contact",
        unit="m",
        is_metric=True,
        sample_count=600,
        valid_fraction=1.0,
        depth_median=8.0,
        depth_center=8.0,
    )
    state = hooks._WorldAnchorState(
        ts=float(hooks.time.time()),
        height_ref_scene=1.8,
    )
    processor._world_state_by_track[(0, 78)] = state
    track = {"tracker_id": 78, "frame_id": 1, "bbox": bbox}

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        pose_kpts_abs=keypoints,
        depth_result=depth_result,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert "world_source" not in track
    assert state.last_good_world is None


def test_universal_resolver_keeps_independent_upright_body_scale_when_depth_has_no_contact(
    monkeypatch,
) -> None:
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "world-r1"
    calibration.frame_transform_sha256 = "a" * 64
    calibration.camera_calibration_sha256 = "b" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "models": {"pose": {"kpt_threshold": 0.35}},
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    }
                },
            }
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=SimpleNamespace(
            snapshot=lambda _sensor_id, _camera_id: calibration
        ),
    )
    keypoints, bbox = _world_pose_and_bbox(calibration, foot_z=6.0)
    keypoints[hooks._POSE_KPT_INDEX["left_ankle"], 2] = 0.0
    keypoints[hooks._POSE_KPT_INDEX["right_ankle"], 2] = 0.0
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=79,
        class_id=0,
        bbox=bbox,
        score=0.95,
        sampling_mode="pose_capsule_native",
        status="no_ground_contact",
        unit="m",
        is_metric=True,
        sample_count=400,
        valid_fraction=1.0,
        depth_median=8.0,
        depth_center=8.0,
    )
    state = hooks._WorldAnchorState(
        height_ref_scene=1.8,
        motion_mode="walk",
        posture="standing",
    )
    processor._world_state_by_track[(0, 79)] = state
    monkeypatch.setattr(
        processor,
        "_gravity_drop_world",
        lambda *_args, **_kwargs: np.asarray([0.0, 0.0, 6.0]),
    )
    track = {
        "tracker_id": 79,
        "frame_id": 1,
        "source_id": 0,
        "observed_at_us": 1_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "image_size": [1280, 720],
    }

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        pose_kpts_abs=keypoints,
        depth_result=depth_result,
        world_now_ts=1.0,
    )

    assert track["world_valid"] is True
    assert track["world_source"] == "gravity_drop"
    assert track["world_quality"] == "estimated"
    assert track["world"] == pytest.approx([0.0, 0.0, 6.0])
    assert track["world_resolver_selected_id"] == "gravity_reconstruction"


def test_stationary_seated_hold_extends_only_for_trusted_lifecycle(monkeypatch) -> None:
    calibration = _anchor_calibration()
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={"models": {"pose": {"kpt_threshold": 0.35}}}
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=SimpleNamespace(
            snapshot=lambda _sensor_id, _camera_id: calibration
        ),
    )
    keypoints = np.zeros((17, 3), dtype=np.float64)
    for index, point in {
        5: (600.0, 200.0),
        6: (680.0, 200.0),
        11: (610.0, 300.0),
        12: (670.0, 300.0),
        13: (650.0, 315.0),
        14: (700.0, 315.0),
    }.items():
        keypoints[index] = [point[0], point[1], 0.98]
    bbox = [560.0, 180.0, 180.0, 190.0]
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=19,
        class_id=0,
        bbox=bbox,
        score=0.95,
        sampling_mode="pose_capsule_native",
        status="no_ground_contact",
        unit="m",
        is_metric=True,
        sample_count=600,
        valid_fraction=1.0,
        depth_median=8.0,
        depth_center=8.0,
    )
    state = hooks._WorldAnchorState(
        ts=100.0,
        world_x=0.0,
        world_z=6.0,
        filtered_ts=100.0,
        last_good_world=(0.0, 0.0, 6.0),
        last_good_ts=100.0,
        posture="sitting",
        motion_mode="sit",
        last_non_upright_ts=100.0,
    )
    processor._world_state_by_track[(0, 19)] = state
    now_values = iter((100.0, 100.2, 100.8, 101.5, 102.2))
    monkeypatch.setattr(hooks.time, "time", lambda: next(now_values))

    tracks: list[dict[str, Any]] = []
    for frame_id in range(1, 6):
        track = {"tracker_id": 19, "frame_id": frame_id, "bbox": bbox}
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=keypoints,
            depth_result=depth_result,
        )
        tracks.append(track)

    # The first three frames establish exact-frame box stationarity.  A
    # trusted seated lifecycle can then remain visible past the generic 0.40s
    # hold, without becoming an accepted measurement or trail point.
    assert tracks[2]["world_valid"] is True
    assert tracks[2]["world_source"] == "anchor_hold"
    assert "stationary_bbox_hold" in tracks[2]["world_quality_reason"]
    assert tracks[2]["world_measurement_accepted"] is False
    assert tracks[2]["trail_append_allowed"] is False
    assert tracks[3]["world_valid"] is True
    assert tracks[4]["world_valid"] is False


def test_stationary_anchor_hold_keeps_bounded_process_continuity(
    monkeypatch,
) -> None:
    """A seated hold must not jump back to the older accepted anchor.

    The rejected-frame process state is the only canonical non-authoritative
    continuation.  Replacing it with ``last_good_world`` after a prior
    ``cv_prediction`` creates a backward jump at the media-frame cadence.
    """

    calibration = _anchor_calibration()
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={"models": {"pose": {"kpt_threshold": 0.35}}}
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=SimpleNamespace(
            snapshot=lambda _sensor_id, _camera_id: calibration
        ),
        world_fusion_policy=_depth_only_policy(),
    )
    keypoints = np.zeros((17, 3), dtype=np.float64)
    for index, point in {
        5: (600.0, 200.0),
        6: (680.0, 200.0),
        11: (610.0, 300.0),
        12: (670.0, 300.0),
        13: (650.0, 315.0),
        14: (700.0, 315.0),
    }.items():
        keypoints[index] = [point[0], point[1], 0.98]
    bbox = [560.0, 180.0, 180.0, 190.0]
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=7,
        class_id=0,
        bbox=bbox,
        score=0.95,
        sampling_mode="pose_capsule_native",
        status="no_ground_contact",
        unit="m",
        is_metric=True,
        sample_count=600,
        valid_fraction=1.0,
        depth_median=8.0,
        depth_center=8.0,
    )
    state = hooks._WorldAnchorState(
        ts=100.0,
        world_x=0.0,
        world_z=6.0,
        vel_world_x=4.0,
        vel_world_z=0.0,
        filtered_ts=100.0,
        last_good_world=(0.0, 0.0, 6.0),
        last_good_ts=100.0,
        posture="sitting",
        motion_mode="walk",
    )
    processor._world_state_by_track[(0, 7)] = state

    stationary = iter((False, True))
    monkeypatch.setattr(
        hooks,
        "observe_bbox_stationarity",
        lambda *_args, **_kwargs: next(stationary),
    )
    now = iter((100.10, 100.15))
    monkeypatch.setattr(hooks.time, "time", lambda: next(now))

    outputs: list[dict[str, Any]] = []
    for frame_id in (1, 2):
        if frame_id == 2:
            # Exercise the adversarial path where motion-mode processing sees
            # an idle-like lifecycle with an older lock while this frame's
            # rejected observation has already advanced the bounded process.
            state.motion_mode = "sit"
            state.locked_world = (0.0, 6.0)
        track = {"tracker_id": 7, "frame_id": frame_id, "bbox": bbox}
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=keypoints,
            depth_result=depth_result,
        )
        outputs.append(track)

    first, second = outputs
    assert first["world_source"] == "cv_prediction"
    assert second["world_source"] == "anchor_hold"
    assert second["trail_append_allowed"] is False
    assert second["world_measurement_accepted"] is False
    assert second["world"][0] == pytest.approx(0.60, abs=1e-6)
    # At 50 ms the source transition is bounded at the configured 4 m/s;
    # the old stale-anchor substitution produced an 8 m/s reversal.
    distance = abs(float(second["world"][0]) - float(first["world"][0]))
    assert distance / 0.05 <= 4.0 + 1e-6
    assert second["world"][0] != pytest.approx(
        float(state.last_good_world[0]), abs=1e-6
    )
    assert state.locked_world == pytest.approx((0.60, 6.0), abs=1e-6)


def test_world_filter_diagnostic_uses_bounded_reject_output() -> None:
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={"models": {"pose": {"kpt_threshold": 0.35}}}
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=None,
    )
    state = hooks._WorldAnchorState()
    processor._update_world_state(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        alpha=0.5,
        beta=0.1,
        quality="good",
    )
    state.last_good_world = (0.0, 0.0, 0.0)
    state.last_good_ts = 0.0
    state.vel_world_x = 1.25
    state.vel_world_z = -0.50

    for frame_id, now_ts, expected in (
        (1, 0.2, [0.25, 0.0, -0.10]),
        (2, 0.8, [0.50, 0.0, -0.20]),
        (3, 1.4, [0.50, 0.0, -0.20]),
    ):
        track: dict[str, Any] = {"frame_id": frame_id}
        result = processor._update_track_world_state(
            track,
            state,
            measurement=np.array([10.0, 0.0, 10.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=now_ts,
            alpha=0.5,
            beta=0.1,
            quality="good",
        )
        assert result == pytest.approx(expected)
        assert track["world_filter_prediction"] == pytest.approx(expected)


def test_person_mask_anchor_rejects_bimodal_lower_body_depth() -> None:
    mask = np.ones((40, 40), dtype=bool)
    depth = np.full((40, 40), np.nan, dtype=np.float32)
    depth[35:40, 13:20] = 3.0
    depth[35:40, 20:27] = 6.0

    anchor = hooks._extract_person_depth_anchor(mask, depth, frame_origin=(0, 0))

    assert anchor.anchor_source is None
    assert anchor.anchor_depth_m is None
    assert anchor.anchor_rejection_reason == "lower_body_depth_spread_exceeded"
    assert anchor.anchor_depth_spread_m == 3.0
