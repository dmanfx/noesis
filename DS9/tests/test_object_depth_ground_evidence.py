from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from geometry.homography import project_world_to_image
from noesis.calibration.geometry import pixel_to_world
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
        camera_calibration_sha256="b" * 64,
        intrinsics=intrinsics,
        extrinsics_col_major=list(extrinsics.flatten(order="F")),
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=1.0,
    )


def test_floor_projection_uses_physical_calibration_before_revision_transform() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    raw = _anchor_calibration()
    angle = np.deg2rad(12.0)
    source_to_target = np.eye(4, dtype=np.float64)
    source_to_target[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    source_to_target[:3, 3] = [0.3, -0.5, 0.2]
    raw_extrinsics = np.asarray(raw.extrinsics_col_major, dtype=np.float64).reshape(
        (4, 4),
        order="F",
    )
    target_extrinsics = raw_extrinsics @ np.linalg.inv(source_to_target)
    active = SimpleNamespace(
        camera_id="cam0",
        intrinsics=raw.intrinsics,
        extrinsics_col_major=tuple(target_extrinsics.flatten(order="F")),
        floor_y=0.0,
        image_size=raw.image_size,
        unit_scale=1.0,
        calibration_extrinsics_col_major=tuple(raw.extrinsics_col_major),
        calibration_floor_y=0.0,
        target_from_calibration_col_major=tuple(
            source_to_target.flatten(order="F")
        ),
    )
    u, v = 740.0, 500.0

    raw_hit = pixel_to_world(
        raw.intrinsics,
        list(raw.extrinsics_col_major),
        raw.floor_y,
        raw.unit_scale,
        u,
        v,
    )
    direct_target_hit = pixel_to_world(
        active.intrinsics,
        list(active.extrinsics_col_major),
        active.floor_y,
        active.unit_scale,
        u,
        v,
    )
    projected = processor._project_pixel_to_floor_world(
        active,
        u,
        v,
        flip_u=False,
        flip_v=False,
    )

    assert raw_hit.ok and raw_hit.world_point is not None
    assert direct_target_hit.ok and direct_target_hit.world_point is not None
    assert projected is not None
    expected = (
        source_to_target
        @ np.r_[np.asarray(raw_hit.world_point, dtype=np.float64), 1.0]
    )[:3]
    expected[1] = active.floor_y
    np.testing.assert_allclose(projected, expected, atol=1e-9)
    assert np.linalg.norm(
        (projected - np.asarray(direct_target_hit.world_point))[[0, 2]]
    ) > 0.25

    image_track: dict[str, Any] = {}
    processor._set_track_image_base_from_world(
        image_track,
        calib=active,
        world_point=projected,
        flip_u=False,
        flip_v=False,
    )
    assert image_track["image_base"] == pytest.approx([u, v], abs=1e-9)

    processor.world_fusion_policy = None
    range_track: dict[str, Any] = {}
    assert processor._admit_floor_ray_range(
        "cam0",
        calib=active,
        floor_candidate=projected,
        track=range_track,
        anchor_uv=(u, v),
    )
    _rotation, raw_camera = hooks.parse_extrinsics(raw.extrinsics_col_major)
    raw_delta = np.asarray(raw_hit.world_point, dtype=np.float64) - raw_camera
    assert range_track["world_floor_range_m"] == pytest.approx(
        np.hypot(raw_delta[0], raw_delta[2]),
        abs=1e-9,
    )


def _prime_metric_output_watermark(
    processor: Any,
    calibration: SimpleNamespace,
    state: Any,
    *,
    tracker_id: int,
    media_pts_ns: int,
    filter_ts: float,
) -> None:
    """Model one metric row that crossed the ordered publication queue."""

    if not getattr(calibration, "world_frame_id", None):
        calibration.world_frame_id = "backend_world_m"
    if not getattr(calibration, "world_frame_revision", None):
        calibration.world_frame_revision = "rev-stationary-test"
    if not getattr(calibration, "frame_transform_sha256", None):
        calibration.frame_transform_sha256 = "d" * 64
    if not getattr(calibration, "camera_calibration_sha256", None):
        calibration.camera_calibration_sha256 = "b" * 64
    state.tracker_lifecycle_generation = 1
    frame_id, revision, transform = hooks.world_frame_binding_from_calibration(
        calibration,
        default_frame_id="backend_world_m",
    )
    assert frame_id is not None
    assert revision is not None
    assert transform is not None
    key = processor._world_output_watermark_key(
        0,
        {
            "tracker_id": tracker_id,
            "tracker_lifecycle_generation": 1,
        },
        world_frame_id=frame_id,
        world_frame_revision=revision,
        world_transform_sha256=transform,
        camera_calibration_sha256=calibration.camera_calibration_sha256,
    )
    assert key is not None
    processor._set_world_output_watermark(
        key,
        world_x=state.world_x,
        world_z=state.world_z,
        media_pts_ns=media_pts_ns,
        filter_ts=filter_ts,
        trail_segment_id=state.trail_segment_id,
        metric_authoritative=True,
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


class _HostOnlyDepthDevice:
    """Force the bounded host-ROI fallback with distinct torso/leg ranges."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []
        self.capsule_calls: list[tuple[list[list[float]], int]] = []

    def copy_roi_to_numpy(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        self.calls.append((left, top, width, height))
        depth = np.full((height, width), 9.0, dtype=np.float32)
        # The pose torso capsule is centered around local (50, 69), while the
        # observed legs occupy the lower half of the same skeleton mask.
        depth[35:105, 20:80] = 4.0
        return depth


def _run_pose_depth_sample(
    monkeypatch: Any,
    *,
    contaminate_contact: bool = False,
    ankles_available: bool = True,
    torso_available: bool = True,
    force_host: bool = False,
) -> tuple[dict[str, Any], Any, hooks._ObjectDepthFusionProcessor]:
    pts_us = 2_000_000
    device: Any = (
        _HostOnlyDepthDevice()
        if force_host
        else _MaskedStatsDevice(contaminate_contact=contaminate_contact)
    )
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
    if not torso_available:
        pose_payload["keypoints_roi"][5][2] = 0.0
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


def test_pose_without_observed_ankles_uses_torso_for_range_only(monkeypatch) -> None:
    payload, device, _processor = _run_pose_depth_sample(
        monkeypatch,
        ankles_available=False,
    )

    assert device.calls == []
    assert len(device.capsule_calls) == 1
    capsules, _max_samples = device.capsule_calls[0]
    assert len(capsules) == 1
    assert len(capsules[0]) == 6
    assert payload["status"] == "ok"
    assert payload["anchor_source"] == "pose_torso_support"
    assert payload["anchor_uv"] == pytest.approx([150.0, 168.8])
    assert payload["anchor_depth_m"] == 4.0
    assert payload["evidence_quality"] == "estimated"


def test_host_pose_torso_fallback_samples_only_matching_torso_capsule(monkeypatch) -> None:
    payload, device, _processor = _run_pose_depth_sample(
        monkeypatch,
        ankles_available=False,
        force_host=True,
    )

    assert device.capsule_calls == []
    assert device.calls == [(100, 100, 100, 200)]
    assert payload["sampling_mode"] == "pose_capsule"
    assert payload["status"] == "ok"
    assert payload["anchor_source"] == "pose_torso_support"
    assert payload["anchor_uv"] == pytest.approx([150.0, 168.8])
    assert payload["anchor_depth_m"] == pytest.approx(4.0)
    # The full pose/skeleton mask also covers the 9m legs. The torso anchor
    # must follow the torso UV/capsule and never their different range.
    assert payload["anchor_sample_count"] < payload["sample_count"]


def test_pose_torso_requires_complete_confident_anatomy(monkeypatch) -> None:
    payload, device, _processor = _run_pose_depth_sample(
        monkeypatch,
        ankles_available=False,
        torso_available=False,
    )

    assert device.calls == []
    assert device.capsule_calls == []
    assert payload["status"] == "no_ground_contact"
    assert payload["evidence_quality"] == "rejected"
    assert payload["evidence_reason"] == "pose_ankles_unavailable"
    assert "anchor_source" not in payload


def test_pose_torso_rejects_mixed_foreground_depth(monkeypatch) -> None:
    payload, device, _processor = _run_pose_depth_sample(
        monkeypatch,
        contaminate_contact=True,
        ankles_available=False,
    )

    assert len(device.capsule_calls) == 1
    assert payload["status"] == "ambiguous_depth"
    assert payload["evidence_quality"] == "rejected"
    assert payload["evidence_reason"] == "depth_spread_exceeded"
    assert "anchor_source" not in payload


def test_pose_torso_is_registered_range_not_verified_floor_contact() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    depth = ObjectDepthResult(
        source_id=0,
        frame_id=50,
        object_id=7,
        class_id=0,
        bbox=(100.0, 100.0, 100.0, 200.0),
        score=0.9,
        sampling_mode="pose_capsule_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=40,
        valid_fraction=1.0,
        anchor_uv=(150.0, 168.8),
        anchor_source="pose_torso_support",
        anchor_depth_m=4.0,
        anchor_sample_count=40,
        anchor_valid_fraction=1.0,
        evidence_quality="estimated",
    )

    anchor = processor._resolve_person_depth_anchor(depth)

    assert anchor is not None
    assert anchor.quality == "estimated"
    assert anchor.contact_basis == "depth:pose_torso_support"
    assert processor._anchor_is_verified_ground_contact(anchor) is False


def test_accepted_image_origin_uses_reprojected_metric_foot_for_torso_range() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    torso_anchor = hooks._PoseAnchorCandidate(
        u=150.0,
        v=168.8,
        source="person_mask_floor",
        contact_basis="depth:pose_torso_support",
    )
    track = {
        "_world_current_image_foot_calib": [150.0, 168.8],
        "image_base": [152.0, 292.0],
    }

    assert processor._accepted_image_foot_for_recording(
        track,
        world_source="person_anchor_depth_only",
        anchor=torso_anchor,
    ) == pytest.approx((152.0, 292.0))

    ankle_anchor = hooks._PoseAnchorCandidate(
        u=148.0,
        v=291.0,
        source="pose_ankle_floor",
        contact_basis="pose:ankle_pair",
    )
    track["_world_current_image_foot_calib"] = [148.0, 291.0]
    assert processor._accepted_image_foot_for_recording(
        track,
        world_source="pose_floor_only",
        anchor=ankle_anchor,
    ) == pytest.approx((148.0, 291.0))


def test_pose_torso_motion_anchor_is_current_anatomy_not_floor() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor._pose_anchor_kpt_threshold = 0.35
    keypoints = np.zeros((17, 3), dtype=np.float64)
    for name, point in {
        "left_shoulder": (125.0, 140.0),
        "right_shoulder": (175.0, 140.0),
        "left_hip": (135.0, 205.0),
        "right_hip": (165.0, 205.0),
    }.items():
        keypoints[hooks._POSE_KPT_INDEX[name]] = [point[0], point[1], 0.9]

    anchor = processor._resolve_pose_torso_motion_anchor(
        keypoints,
        bbox=(100.0, 100.0, 100.0, 200.0),
    )

    assert anchor is not None
    assert anchor.source == "pose_torso_motion"
    assert anchor.contact_basis == "pose:torso_motion"
    assert anchor.u == pytest.approx(150.0)
    assert anchor.v == pytest.approx(171.2)
    assert processor._anchor_is_verified_ground_contact(anchor) is False

    keypoints[hooks._POSE_KPT_INDEX["right_hip"], 2] = 0.0
    assert (
        processor._resolve_pose_torso_motion_anchor(
            keypoints,
            bbox=(100.0, 100.0, 100.0, 200.0),
        )
        is None
    )


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
            "tracker_lifecycle_generation": 1,
            "frame_id": frame_id,
            "media_pts_ns": int(round(float(at_s) * 1_000_000_000.0)),
            "confidence": 0.91,
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
        processor._commit_enqueued_world_output_watermarks(
            0,
            [track],
            filter_ts=float(at_s),
        )
        return track

    first = run_frame(1, 100.0, 6.0, 6.0, 1, 100_000_000, False)
    second = run_frame(2, 100.2, 6.8, 6.8, 2, 100_200_000, False)
    state = processor._world_state_by_track[(0, 840)]
    last_good_before_cached = tuple(state.last_good_world or ())
    last_good_ts_before_cached = float(state.last_good_ts)

    # A confident accepted standing row arms the anatomy reference even
    # though standing rows are not allowed to consume the occlusion bridge.
    assert state.last_accepted_motion_anchor is not None
    assert state.last_accepted_motion_basis == "pose:torso_motion"

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
            "last_accepted_pose_projective_origin"
        )
        assert predicted["world_prediction_provenance"]["state_integrated"] is True
        assert predicted["world"] == pytest.approx(
            predicted["world_filter_prediction"]
        )
        assert predicted["depth_measurement_cached"] is True
        assert predicted["depth_used_m"] is None
        assert predicted["trail_break_required"] is False
        assert predicted["trail_append_allowed"] is True
        # Image-motion corroboration uses the independently observed torso
        # basis even though the accepted metric contact remains the ankle
        # pair. The torso witness cannot itself become a floor measurement.
        assert predicted["world_contact_basis"] == "pose:torso_motion"
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

    # The independent pose-compatible origin deliberately outlives bbox-only
    # transport, so the producer can still form this current image candidate
    # immediately before its fixed 2.5s TTL. Its queue-origin proof is already
    # older than the 1.25s canonical filter reset bound, however, so focused
    # local revalidation must not advance the service-owned watermark from it.
    extended = run_frame(6, 102.69, 7.6, 20.0, 5, 100_400_000, True)
    assert extended["world_source"] == "image_motion_prediction"
    assert extended["world_prediction_provenance"]["origin"] == (
        "last_accepted_pose_projective_origin"
    )

    # With no service-admitted image root, later CV rows fail closed instead
    # of renewing authority from the producer-only candidate.
    bridged = run_frame(7, 102.701, 7.7, 20.0, 5, 100_400_000, True)
    assert bridged["world_valid"] is False
    assert bridged.get("world_source") is None
    assert bridged.get("world_prediction_provenance") is None

    continued = run_frame(8, 102.801, 7.8, 20.0, 5, 100_400_000, True)
    assert continued["world_valid"] is False
    assert continued.get("world_source") is None
    assert continued.get("world_prediction_provenance") is None

    fresh = run_frame(9, 102.9, 7.9, 7.9, 9, 102_900_000, False)
    assert fresh["world_measurement_accepted"] is True
    assert fresh["world_source"] == "pose_depth_only"
    assert fresh["world"][2] > cached_two["world"][2]
    assert fresh["trail_break_required"] is False
    assert state.last_good_ts == pytest.approx(102.9)


def test_universal_resolver_stale_depth_does_not_clear_pose_reacquire(
    monkeypatch,
) -> None:
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "test-world-revision"
    calibration.frame_transform_sha256 = "b" * 64
    calibration.camera_calibration_sha256 = "c" * 64
    provider = SimpleNamespace(
        snapshot=lambda _sensor_id, _camera_id: calibration,
        world_snapshot=lambda _sensor_id, _camera_id: calibration,
    )
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
        bev_calibration=provider,
        world_fusion_policy=_depth_only_policy(),
    )
    assert processor._world_resolver_enabled is True
    now = [100.0]
    monkeypatch.setattr(hooks.time, "time", lambda: now[0])
    camera_world = np.array([0.0, 2.2, -6.0], dtype=np.float64)

    def run_frame(
        frame_id: int,
        at_s: float,
        foot_z: float,
        *,
        stale_depth: bool,
    ) -> dict[str, Any]:
        now[0] = float(at_s)
        keypoints, bbox = _world_pose_and_bbox(calibration, foot_z=foot_z)
        current_ts_us = int(round(float(at_s) * 1_000_000.0))
        range_m = float(
            np.linalg.norm(
                np.array([0.0, 0.0, float(foot_z)], dtype=np.float64)
                - camera_world
            )
        )
        depth_result = ObjectDepthResult(
            source_id=0,
            frame_id=frame_id,
            object_id=941,
            class_id=0,
            bbox=bbox,
            score=0.94,
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
            measurement_frame_id=frame_id,
            measurement_ts_us=current_ts_us,
            measurement_age_us=0,
            measurement_cached=False,
            depth_tensor_frame_id=frame_id - 1 if stale_depth else frame_id,
            depth_tensor_ts_us=(
                current_ts_us - 33_333 if stale_depth else current_ts_us
            ),
            depth_tensor_age_frames=1 if stale_depth else 0,
            depth_tensor_age_us=33_333 if stale_depth else 0,
            ts_us=current_ts_us,
        )
        track: dict[str, Any] = {
            "tracker_id": 941,
            "tracker_lifecycle_generation": 1,
            "frame_id": frame_id,
            "source_id": 0,
            "observed_at_us": current_ts_us,
            "confidence": 0.94,
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
        return track

    seed = {}
    for frame_id, at_s in ((1, 100.0), (2, 100.1), (3, 100.2)):
        seed = run_frame(frame_id, at_s, 6.0, stale_depth=False)
    assert seed["world_valid"] is True, seed
    state = processor._world_state_by_track[(0, 941)]
    accepted_world = tuple(state.last_good_world or ())
    # This test isolates independent floor-vs-stale-depth cohort behavior;
    # disable the separately tested learned-height gravity hypothesis so a
    # synthetic one-frame range jump is not classified as an occluder.
    state.height_ref_scene = None
    state.last_full_body_ts = -1.0
    state.lower_body_occluded = False

    rejected = [
        run_frame(frame_id, at_s, foot_z, stale_depth=True)
        for frame_id, at_s, foot_z in (
            (4, 100.3, 10.00),
            (5, 100.4, 10.04),
            (6, 100.5, 10.08),
        )
    ]

    assert [track["world_reacquire_count"] for track in rejected] == [1, 2, 0]
    assert [track["world_reacquired"] for track in rejected] == [False, False, True]
    assert all(
        track["world_resolver_selected_id"] == "floor_ray"
        for track in rejected
    )
    assert all(track.get("depth_registered_m") is None for track in rejected)
    assert rejected[-1]["world_source"] == "pose_floor_only"
    assert rejected[-1]["trail_break_required"] is True
    assert tuple(state.last_good_world or ()) == pytest.approx(rejected[-1]["world"])
    assert tuple(state.last_good_world or ()) != pytest.approx(accepted_world)


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


@pytest.mark.parametrize(
    "depth_status",
    ("no_ground_contact", "depth_not_ready", "error"),
)
def test_unposed_noncontact_depth_status_gets_bounded_bbox_floor_candidate(
    depth_status: str,
) -> None:
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "world-r1"
    calibration.frame_transform_sha256 = "a" * 64
    calibration.camera_calibration_sha256 = "b" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    },
                    "person_admission": {
                        "min_unposed_detection_confidence": 0.75,
                    },
                }
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
    _unused_pose, bbox = _world_pose_and_bbox(calibration, foot_z=6.0)
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=80,
        class_id=0,
        bbox=bbox,
        score=0.8,
        sampling_mode="pose_capsule_native",
        status=depth_status,
        unit="m",
        is_metric=True,
        sample_count=400,
        valid_fraction=1.0,
        depth_median=8.0,
        depth_center=8.0,
    )
    tracks: list[dict[str, Any]] = []
    for frame_id, offset in ((1, 0.0), (2, 2.0), (3, 4.0)):
        current_bbox = list(bbox)
        current_bbox[0] += offset
        track = {
            "tracker_id": 80,
            "frame_id": frame_id,
            "source_id": 0,
            "observed_at_us": frame_id * 100_000,
            "tracker_lifecycle_generation": 1,
            "bbox": current_bbox,
            "image_size": list(calibration.image_size),
            "confidence": 0.80,
        }
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=None,
            depth_result=depth_result,
            world_now_ts=frame_id * 0.1,
        )
        tracks.append(track)

    assert [track["world_valid"] for track in tracks] == [False, False, True]
    assert tracks[-1]["world_source"] == "person_anchor_floor_only"
    assert tracks[-1]["world_resolver_selected_id"] == "floor_ray"
    assert tracks[-1]["world_resolver_contact_basis"] == "bbox:bottom_center"
    assert tracks[-1]["world_quality"] in ("good", "estimated")


def test_torso_depth_does_not_suppress_independent_bbox_floor_hypothesis() -> None:
    """A non-contact depth range must not block a guarded floor candidate."""
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "world-r1"
    calibration.frame_transform_sha256 = "a" * 64
    calibration.camera_calibration_sha256 = "b" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    },
                    "person_admission": {
                        "min_unposed_detection_confidence": 0.75,
                    },
                }
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
    processor.set_world_resolver_diagnostics_enabled(True)
    _unused_pose, bbox = _world_pose_and_bbox(calibration, foot_z=6.0)
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=84,
        class_id=0,
        bbox=bbox,
        score=0.90,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=64,
        valid_fraction=0.90,
        anchor_uv=(bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5),
        anchor_source="torso_core",
        anchor_depth_m=6.0,
        anchor_sample_count=64,
        anchor_valid_fraction=0.90,
        evidence_quality="estimated",
        anchor_depth_spread_m=0.10,
    )
    track = {
        "tracker_id": 84,
        "frame_id": 1,
        "source_id": 0,
        "observed_at_us": 1_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "image_size": list(calibration.image_size),
        "confidence": 0.90,
    }

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        pose_kpts_abs=None,
        depth_result=depth_result,
        world_now_ts=1.0,
    )

    # A single cold bbox-bottom row still has to pass the shared temporal
    # bootstrap.  The contract under test is that the incompatible torso
    # range cannot displace or hide the observed floor-contact hypothesis.
    assert track["world_valid"] is False
    assert track["world_quality_reason"] == "weak_measurement_bootstrap_pending"
    diagnostics = track["world_resolver"]
    candidate_ids = {candidate["id"] for candidate in diagnostics["candidates"]}
    assert "floor_ray" in candidate_ids
    assert "registered_depth" in candidate_ids
    floor_candidate = next(
        candidate
        for candidate in diagnostics["candidates"]
        if candidate["id"] == "floor_ray"
    )
    assert floor_candidate["anchor"] == "bbox_bottom"
    assert diagnostics["selected_id"] == "floor_ray"
    assert track["world_resolver_contact_basis"] == "bbox:bottom_center"


def test_observed_pose_keeps_pose_floor_contact_without_bbox_substitution() -> None:
    """A valid pose contact remains the primary floor hypothesis."""
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "world-r1"
    calibration.frame_transform_sha256 = "a" * 64
    calibration.camera_calibration_sha256 = "b" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    },
                    "person_admission": {
                        "min_unposed_detection_confidence": 0.75,
                    },
                }
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
    processor.set_world_resolver_diagnostics_enabled(True)
    keypoints, bbox = _world_pose_and_bbox(calibration, foot_z=6.0)
    track = {
        "tracker_id": 85,
        "frame_id": 1,
        "source_id": 0,
        "observed_at_us": 1_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "image_size": list(calibration.image_size),
        "confidence": 0.90,
    }

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        pose_kpts_abs=keypoints,
        depth_result=None,
        world_now_ts=1.0,
    )

    assert track["world_valid"] is True
    assert track["world_resolver_contact_basis"] == "pose:ankle_pair"
    assert [candidate["id"] for candidate in track["world_resolver"]["candidates"]] == [
        "floor_ray"
    ]


def test_torso_depth_localizes_seated_pose_without_enabling_bbox_floor() -> None:
    """Typed seated body projection works while bbox-floor stays rejected."""
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "world-r1"
    calibration.frame_transform_sha256 = "a" * 64
    calibration.camera_calibration_sha256 = "b" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    },
                    "person_admission": {
                        "min_unposed_detection_confidence": 0.75,
                    },
                }
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
    processor.set_world_resolver_diagnostics_enabled(True)
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
        object_id=86,
        class_id=0,
        bbox=bbox,
        score=0.90,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=64,
        valid_fraction=0.90,
        anchor_uv=(650.0, 270.0),
        anchor_source="torso_core",
        anchor_depth_m=6.0,
        anchor_sample_count=64,
        anchor_valid_fraction=0.90,
        evidence_quality="estimated",
        anchor_depth_spread_m=0.10,
    )
    track = {
        "tracker_id": 86,
        "frame_id": 1,
        "source_id": 0,
        "observed_at_us": 1_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "image_size": list(calibration.image_size),
        "confidence": 0.90,
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
    assert track["world_support_state"] == "seat"
    resolver = track["world_resolver"]
    assert resolver["selected_kind"] == "registered_depth"
    assert all(candidate["id"] != "floor_ray" for candidate in resolver["candidates"])
    assert "seated_ground_projection" in str(
        track["world_quality_reason"]
    )


def test_tracker_confidence_continues_trusted_bbox_floor_when_detector_is_sentinel() -> None:
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "world-r1"
    calibration.frame_transform_sha256 = "a" * 64
    calibration.camera_calibration_sha256 = "b" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    },
                    "person_admission": {
                        "min_unposed_detection_confidence": 0.75,
                    },
                }
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
    _unused_pose, bbox = _world_pose_and_bbox(calibration, foot_z=6.0)
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=83,
        class_id=0,
        bbox=bbox,
        score=0.8,
        sampling_mode="pose_capsule_native",
        status="no_ground_contact",
        unit="m",
        is_metric=True,
        sample_count=400,
        valid_fraction=1.0,
    )
    track = {
        "tracker_id": 83,
        "frame_id": 1,
        "source_id": 0,
        "observed_at_us": 1_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "image_size": list(calibration.image_size),
        "confidence": -0.1,
        "tracker_confidence": 0.80,
    }
    processor._world_state_by_track[(0, 83)] = hooks._WorldAnchorState(
        ts=0.9,
        world_x=0.0,
        world_z=6.0,
        filtered_ts=0.9,
        last_good_world=(0.0, 0.0, 6.0),
        last_good_ts=0.9,
    )

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        pose_kpts_abs=None,
        depth_result=depth_result,
        world_now_ts=1.0,
    )

    assert track["world_valid"] is True
    assert track["world_resolver_selected_id"] == "floor_ray"
    assert track["world_resolver_contact_basis"] == "bbox:bottom_center"


def test_cold_low_detector_high_tracker_artifact_never_gets_bbox_world() -> None:
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "world-r1"
    calibration.frame_transform_sha256 = "a" * 64
    calibration.camera_calibration_sha256 = "b" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    },
                    "person_admission": {
                        "min_unposed_detection_confidence": 0.50,
                    },
                }
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
    _unused_pose, bbox = _world_pose_and_bbox(calibration, foot_z=6.0)
    bbox = [bbox[0], bbox[1], 39.0, 82.0]
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=830,
        class_id=0,
        bbox=bbox,
        score=0.30,
        sampling_mode="pose_capsule_native",
        status="no_ground_contact",
        unit="m",
        is_metric=True,
        sample_count=100,
        valid_fraction=1.0,
    )

    tracks: list[dict[str, Any]] = []
    for frame_id in range(1, 7):
        track = {
            "tracker_id": 830,
            "frame_id": frame_id,
            "source_id": 0,
            "observed_at_us": frame_id * 100_000,
            "tracker_lifecycle_generation": 1,
            "bbox": list(bbox),
            "image_size": list(calibration.image_size),
            "confidence": 0.30,
            "tracker_confidence": 0.99,
        }
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=None,
            depth_result=depth_result,
            world_now_ts=frame_id * 0.1,
        )
        tracks.append(track)

    assert all(track["world_valid"] is False for track in tracks)
    state = processor._world_state_by_track[(0, 830)]
    assert state.last_good_world is None
    assert state.reacquire_count == 6


def test_partial_pose_without_contact_gets_bbox_floor_candidate_for_confident_upright_box() -> None:
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "world-r1"
    calibration.frame_transform_sha256 = "a" * 64
    calibration.camera_calibration_sha256 = "b" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    },
                    "person_admission": {
                        "min_unposed_detection_confidence": 0.75,
                    },
                }
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
    # Keep pose data present while removing both ankle contacts.  The narrow
    # upright detector box supplies the only current, generic contact cue.
    keypoints[hooks._POSE_KPT_INDEX["left_ankle"], 2] = 0.0
    keypoints[hooks._POSE_KPT_INDEX["right_ankle"], 2] = 0.0
    left, top, width, height = bbox
    bbox = [left + 0.25 * width, top, 0.50 * width, height]
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=81,
        class_id=0,
        bbox=bbox,
        score=0.80,
        sampling_mode="pose_capsule_native",
        status="no_ground_contact",
        unit="m",
        is_metric=True,
        sample_count=400,
        valid_fraction=1.0,
    )
    tracks: list[dict[str, Any]] = []
    for frame_id, offset in ((1, 0.0), (2, 2.0), (3, 4.0)):
        current_bbox = list(bbox)
        current_bbox[0] += offset
        current_keypoints = np.asarray(keypoints, dtype=np.float64).copy()
        current_keypoints[:, 0] += offset
        track = {
            "tracker_id": 81,
            "frame_id": frame_id,
            "source_id": 0,
            "observed_at_us": frame_id * 100_000,
            "tracker_lifecycle_generation": 1,
            "bbox": current_bbox,
            "confidence": 0.80,
            "image_size": list(calibration.image_size),
        }
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=current_keypoints,
            depth_result=depth_result,
            world_now_ts=frame_id * 0.1,
        )
        tracks.append(track)

    assert [track["world_valid"] for track in tracks] == [False, False, True]
    assert tracks[-1]["world_resolver_selected_id"] == "floor_ray"
    assert tracks[-1]["world_resolver_contact_basis"] == "bbox:bottom_center"
    assert tracks[-1]["world_quality"] in ("good", "estimated")


@pytest.mark.parametrize(
    "pose_support",
    ("all_zero", "torso_only", "observed_ankles"),
)
def test_post_occlusion_reacquire_requires_verified_metric_support(
    pose_support: str,
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
                    },
                    "person_admission": {
                        "min_unposed_detection_confidence": 0.75,
                    },
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
    full_pose, bbox = _world_pose_and_bbox(calibration, foot_z=10.0)
    left, top, width, height = bbox
    bbox = [left + 0.25 * width, top, 0.50 * width, height]
    if pose_support == "observed_ankles":
        keypoints = full_pose
    else:
        keypoints = np.zeros((17, 3), dtype=np.float64)
        if pose_support == "torso_only":
            for name in (
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
            ):
                index = hooks._POSE_KPT_INDEX[name]
                keypoints[index] = full_pose[index]
            nose_uv = project_world_to_image(
                (0.0, 0.94 * 1.8, 10.0),
                calibration.intrinsics,
                calibration.extrinsics_col_major,
                calibration.image_size,
                unit_scale=1.0,
            )
            assert nose_uv is not None
            keypoints[hooks._POSE_KPT_INDEX["nose"]] = [
                float(nose_uv[0]),
                float(nose_uv[1]),
                0.98,
            ]

    state = hooks._WorldAnchorState(
        ts=0.9,
        world_x=0.0,
        world_z=6.0,
        filtered_ts=0.9,
        last_good_world=(0.0, 0.0, 6.0),
        last_good_ts=0.9,
        motion_mode="walk",
        posture="standing",
        post_occlusion_reacquire_support_required=True,
    )
    processor._world_state_by_track[(0, 812)] = state
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=812,
        class_id=0,
        bbox=bbox,
        score=0.90,
        sampling_mode="pose_capsule_native",
        status="no_ground_contact",
        unit="m",
        is_metric=True,
        sample_count=400,
        valid_fraction=1.0,
    )

    tracks: list[dict[str, Any]] = []
    for frame_id in (1, 2, 3):
        track = {
            "tracker_id": 812,
            "tracker_lifecycle_generation": 1,
            "frame_id": frame_id,
            "source_id": 0,
            "observed_at_us": frame_id * 100_000,
            "bbox": list(bbox),
            "image_size": list(calibration.image_size),
            "confidence": 0.90,
        }
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=np.asarray(keypoints, dtype=np.float64).copy(),
            depth_result=depth_result,
            world_now_ts=0.9 + frame_id * 0.1,
        )
        tracks.append(track)

    if pose_support in {"observed_ankles", "torso_only"}:
        assert state.post_occlusion_reacquire_support_required is False
        assert state.reacquired is True
        assert tracks[-1]["world_measurement_accepted"] is True
        expected_basis = (
            "pose:ankle_pair"
            if pose_support == "observed_ankles"
            else "pose:upright_body_planes"
        )
        assert tracks[-1]["world_resolver_contact_basis"] == expected_basis
        if pose_support == "torso_only":
            assert tracks[-1]["world_upright_body_reacquire_support"] is True
    else:
        assert state.post_occlusion_reacquire_support_required is True
        assert state.reacquired is False
        assert state.reacquire_count == 0
        assert all(
            track["world_rejection_reason"]
            == "post_occlusion_reacquire_requires_verified_support"
            for track in tracks
        )
        assert all(
            track["world_resolver_contact_basis"] == "bbox:bottom_center"
            for track in tracks
        )


def test_bbox_floor_fallback_rejects_partial_pose_below_configured_confidence() -> None:
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "world-r1"
    calibration.frame_transform_sha256 = "a" * 64
    calibration.camera_calibration_sha256 = "b" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    },
                    "person_admission": {
                        "min_unposed_detection_confidence": 0.75,
                    },
                }
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
    left, top, width, height = bbox
    bbox = [left + 0.25 * width, top, 0.50 * width, height]
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=82,
        class_id=0,
        bbox=bbox,
        score=0.70,
        sampling_mode="pose_capsule_native",
        status="no_ground_contact",
        unit="m",
        is_metric=True,
        sample_count=400,
        valid_fraction=1.0,
    )
    track = {
        "tracker_id": 82,
        "frame_id": 1,
        "source_id": 0,
        "observed_at_us": 1_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "confidence": 0.70,
        "image_size": list(calibration.image_size),
    }

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        pose_kpts_abs=keypoints,
        depth_result=depth_result,
        world_now_ts=1.0,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track.get("world_quality_reason") == "no_valid_measurement_or_process_continuation"


def test_universal_resolver_keeps_cold_upright_body_scale_diagnostic_only(
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

    assert track["world_valid"] is False
    assert track.get("world_source") is None
    assert track["world_quality"] == "invalid"
    assert "world" not in track
    assert track["world_resolver_selected_id"] == "gravity_reconstruction"


def test_moving_inferred_gravity_cannot_steer_metric_or_process_state(
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
    state = hooks._WorldAnchorState(
        ts=0.9,
        height_ref_scene=1.8,
        last_good_world=(0.0, 0.0, 6.0),
        last_good_ts=0.9,
        world_x=0.0,
        world_z=6.0,
        filtered_ts=0.9,
        motion_mode="walk",
        posture="standing",
        last_full_body_ts=0.8,
        # An estimator-local bridge token is insufficient: the queue-visible
        # output below is metric, so final provenance must stay metric-rooted.
        projective_bridge_origin_ts=0.9,
        projective_bridge_process_ts=0.9,
        projective_bridge_rows_remaining=1,
    )
    processor._world_state_by_track[(0, 80)] = state
    gravity_point = [0.0, 0.0, 5.8]
    monkeypatch.setattr(
        processor,
        "_gravity_drop_world",
        lambda *_args, **_kwargs: np.asarray(gravity_point, dtype=np.float64),
    )
    # Isolate the unknown-support gravity hypothesis. Missing ankles alone can
    # still produce the separately supported straight-leg or guarded bbox-floor
    # hypotheses on an established lifecycle; those are tested elsewhere.
    monkeypatch.setattr(
        processor,
        "_resolve_pose_floor_anchor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        processor,
        "_resolve_bbox_floor_anchor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        hooks,
        "observe_coherent_image_motion",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        processor,
        "_projective_continuation_allowed",
        lambda *_args, **_kwargs: True,
    )
    track = {
        "tracker_id": 80,
        "frame_id": 30,
        "source_id": 0,
        "observed_at_us": 1_000_000,
        "media_pts_ns": 1_000_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "image_size": [1280, 720],
        "confidence": 0.95,
    }
    output_key = processor._world_output_watermark_key(
        0,
        track,
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        world_transform_sha256="a" * 64,
        camera_calibration_sha256=calibration.camera_calibration_sha256,
    )
    processor._set_world_output_watermark(
        output_key,
        world_x=0.0,
        world_z=6.0,
        media_pts_ns=900_000_000,
        filter_ts=0.9,
        trail_segment_id=0,
        metric_authoritative=True,
    )

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        pose_kpts_abs=keypoints,
        world_now_ts=1.0,
    )

    # The already-established bounded CV process may keep the dot visible,
    # but its point must come only from the last metric state—not from the
    # current gravity reconstruction.
    assert track["world_valid"] is True
    assert track["world_source"] == "cv_prediction"
    assert track["world_prediction_provenance"]["type"] == "bounded_cv_process"
    assert track["world_prediction_provenance"]["origin"] == "last_metric_process"
    assert track["world_filter_prediction"] == pytest.approx([0.0, 0.0, 6.0])
    assert track["world_measurement_accepted"] is False
    assert tuple(state.last_good_world or ()) == pytest.approx((0.0, 0.0, 6.0))
    assert state.last_good_ts == pytest.approx(0.9)


def test_established_standing_occlusion_uses_gravity_as_bounded_process_only(
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
    state = hooks._WorldAnchorState(
        ts=0.9,
        height_ref_scene=1.8,
        last_good_world=(0.0, 0.0, 6.0),
        last_good_ts=0.9,
        world_x=0.0,
        world_z=6.0,
        filtered_ts=0.9,
        motion_mode="walk",
        posture="standing",
        last_full_body_ts=0.8,
        lower_body_occluded=True,
        lower_body_occlusion_level="feet_ankles",
        lower_body_occlusion_confidence=0.89,
        lower_body_occlusion_reason="ankles_missing,upright_bbox_collapsed",
        last_output_world_x=0.0,
        last_output_world_z=6.0,
        last_output_media_pts_ns=900_000_000,
        last_output_filter_ts=0.9,
        last_output_trail_segment_id=0,
    )
    processor._world_state_by_track[(0, 81)] = state
    gravity_point = [0.0, 0.0, 5.8]
    monkeypatch.setattr(
        processor,
        "_gravity_drop_world",
        lambda *_args, **_kwargs: np.asarray(gravity_point, dtype=np.float64),
    )
    # Isolate exact-current learned-height reconstruction from the independent
    # observed-contact paths.
    monkeypatch.setattr(
        processor,
        "_resolve_pose_floor_anchor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        processor,
        "_resolve_bbox_floor_anchor",
        lambda *_args, **_kwargs: None,
    )
    track = {
        "tracker_id": 81,
        "frame_id": 30,
        "source_id": 0,
        "observed_at_us": 1_000_000,
        "media_pts_ns": 1_000_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "image_size": [1280, 720],
        "confidence": 0.95,
    }
    output_key = processor._world_output_watermark_key(
        0,
        track,
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        world_transform_sha256="a" * 64,
        camera_calibration_sha256=calibration.camera_calibration_sha256,
    )
    processor._set_world_output_watermark(
        output_key,
        world_x=0.0,
        world_z=6.0,
        media_pts_ns=900_000_000,
        filter_ts=0.9,
        trail_segment_id=0,
        metric_authoritative=True,
    )

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        pose_kpts_abs=keypoints,
        world_now_ts=1.0,
    )

    assert track["world_valid"] is True
    assert track["world_source"] == "image_motion_prediction"
    assert track["world_quality"] == "held"
    assert track["world_measurement_accepted"] is False
    assert track["world_filter_prediction"] == pytest.approx(track["world"])
    provenance = track["world_prediction_provenance"]
    assert provenance["type"] == "inferred_ground_process_observation"
    assert provenance["origin"] == "learned_body_height"
    assert provenance["transport"] == "fixed_occlusion_origin_raw_world_delta"
    assert provenance["support_kind"] == "lower_body_occlusion"
    assert provenance["trusted_origin_kind"] == (
        "queue_admitted_metric_world_output"
    )
    assert provenance["raw_evidence_gap_s"] == pytest.approx(0.0)
    transition = provenance["filter_transition"]
    assert transition["origin_kind"] == "queue_admitted_world_output"
    assert transition["origin_world"] == pytest.approx([0.0, 0.0, 6.0])
    assert transition["origin_media_pts_ns"] == 900_000_000
    assert transition["current_media_pts_ns"] == 1_000_000_000
    assert provenance["state_integrated"] is True
    assert provenance["observed_at_us"] == 1_000_000
    assert provenance["tracker_lifecycle_generation"] == 1
    assert provenance["raw_consensus"] == pytest.approx([0.0, 0.0, 5.8])
    assert provenance["raw_consensus_ts_s"] == pytest.approx(1.0)
    assert provenance["raw_consensus_observed_at_us"] == 1_000_000
    assert provenance["raw_consensus_media_pts_ns"] == 1_000_000_000
    assert provenance["raw_origin"] == pytest.approx([0.0, 0.0, 5.8])
    assert provenance["raw_delta"] == pytest.approx([0.0, 0.0, 0.0])
    assert provenance["trusted_world_origin"] == pytest.approx([0.0, 0.0, 6.0])
    assert provenance["process_observation"] == pytest.approx([0.0, 0.0, 6.0])
    assert track["world_inferred_raw_observation"] == pytest.approx(
        [0.0, 0.0, 5.8]
    )
    assert track["world_inferred_process_observation"] == pytest.approx(
        [0.0, 0.0, 6.0]
    )
    assert tuple(state.last_good_world or ()) == pytest.approx((0.0, 0.0, 6.0))
    assert state.last_good_ts == pytest.approx(0.9)

    gravity_point[:] = [0.2, 0.0, 5.6]
    second_track = {
        "tracker_id": 81,
        "frame_id": 33,
        "source_id": 0,
        "observed_at_us": 1_100_000,
        "media_pts_ns": 1_100_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "image_size": [1280, 720],
        "confidence": 0.95,
    }
    processor._augment_track_with_world(
        0,
        "cam0",
        second_track,
        pose_kpts_abs=keypoints,
        world_now_ts=1.1,
    )

    assert second_track["world_valid"] is True
    second_provenance = second_track["world_prediction_provenance"]
    assert second_provenance["raw_origin"] == pytest.approx([0.0, 0.0, 5.8])
    assert second_provenance["raw_consensus"] == pytest.approx([0.2, 0.0, 5.6])
    assert second_provenance["raw_consensus_ts_s"] == pytest.approx(1.1)
    assert second_provenance["raw_consensus_observed_at_us"] == 1_100_000
    assert second_provenance["raw_consensus_media_pts_ns"] == 1_100_000_000
    assert second_provenance["raw_delta"] == pytest.approx([0.2, 0.0, -0.2])
    assert second_provenance["trusted_world_origin"] == pytest.approx(
        [0.0, 0.0, 6.0]
    )
    assert second_provenance["process_observation"] == pytest.approx(
        [0.2, 0.0, 5.8]
    )
    assert tuple(state.last_good_world or ()) == pytest.approx((0.0, 0.0, 6.0))
    assert state.last_good_ts == pytest.approx(0.9)


def test_established_coherent_torso_motion_bridges_rejected_floor_contact(
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
    state = hooks._WorldAnchorState(
        ts=0.9,
        height_ref_scene=1.8,
        last_good_world=(0.0, 0.0, 6.0),
        last_good_ts=0.9,
        world_x=0.0,
        world_z=6.0,
        filtered_ts=0.9,
        motion_mode="walk",
        posture="standing",
        last_full_body_ts=0.9,
        lower_body_occluded=False,
        last_output_world_x=0.0,
        last_output_world_z=6.0,
        last_output_media_pts_ns=900_000_000,
        last_output_filter_ts=0.9,
        last_output_trail_segment_id=0,
    )
    processor._world_state_by_track[(0, 82)] = state
    gravity_samples = iter(
        (
            np.asarray([0.15, 0.0, 5.9]),
            np.asarray([0.20, 0.0, 5.85]),
            np.asarray([0.25, 0.0, 5.8]),
            np.asarray([0.35, 0.0, 5.7]),
            np.asarray([0.45, 0.0, 5.6]),
        )
    )
    monkeypatch.setattr(
        processor,
        "_gravity_drop_world",
        lambda *_args, **_kwargs: next(gravity_samples),
    )
    monkeypatch.setattr(
        processor,
        "_resolve_pose_floor_anchor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        processor,
        "_resolve_bbox_floor_anchor",
        lambda *_args, **_kwargs: None,
    )

    motion_rows = iter((True, False, True, True, False))

    def _coherent_torso_motion(current_state, **_kwargs):
        supported = next(motion_rows)
        current_state.image_motion_supported = supported
        current_state.image_motion_streak = 3 if supported else 0
        current_state.image_motion_contact_basis = (
            "pose:torso_motion" if supported else ""
        )
        return supported

    monkeypatch.setattr(
        hooks,
        "observe_coherent_image_motion",
        _coherent_torso_motion,
    )
    monkeypatch.setattr(hooks, "classify_posture", lambda *_args, **_kwargs: "standing")
    track = {
        "tracker_id": 82,
        "frame_id": 30,
        "source_id": 0,
        "observed_at_us": 1_000_000,
        "media_pts_ns": 1_000_000_000,
        "tracker_lifecycle_generation": 1,
        "bbox": bbox,
        "image_size": [1280, 720],
        "confidence": 0.95,
    }
    output_key = processor._world_output_watermark_key(
        0,
        track,
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        world_transform_sha256="a" * 64,
        camera_calibration_sha256=calibration.camera_calibration_sha256,
    )
    processor._set_world_output_watermark(
        output_key,
        world_x=0.0,
        world_z=6.0,
        media_pts_ns=900_000_000,
        filter_ts=0.9,
        trail_segment_id=0,
        metric_authoritative=True,
    )

    processor._augment_track_with_world(
        0,
        "cam0",
        track,
        pose_kpts_abs=keypoints,
        world_now_ts=1.0,
    )

    assert track["world_valid"] is True
    assert track["world_source"] == "image_motion_prediction", {
        key: track.get(key)
        for key in (
            "world_source",
            "world_quality_reason",
            "world_projective_rejection_reason",
            "world_contact_basis",
            "world_image_motion_supported",
            "world_image_motion_streak",
            "lower_body_occluded",
            "posture",
            "motion_mode",
            "world_observation_range_admitted",
            "world_resolver_contact_basis",
        )
    }
    assert track["lower_body_occluded"] is False
    provenance = track["world_prediction_provenance"]
    assert provenance["type"] == "inferred_ground_process_observation"
    assert provenance["support_kind"] == "coherent_torso_motion"
    assert provenance["transport"] == "fixed_torso_origin_raw_world_delta"
    assert provenance["image_motion_supported"] is True
    assert provenance["image_motion_streak"] == 3
    assert provenance["image_motion_contact_basis"] == "pose:torso_motion"
    assert provenance["detector_confidence"] == pytest.approx(0.95)
    assert provenance["process_observation"] == pytest.approx(
        [0.0, 0.0, 6.0]
    )
    assert tuple(state.last_good_world or ()) == pytest.approx((0.0, 0.0, 6.0))
    assert state.last_good_ts == pytest.approx(0.9)

    first_episode_anchor = state.inferred_ground_continuity_anchor
    assert first_episode_anchor is not None
    processor._reconcile_inferred_ground_with_committed_output(
        state,
        (
            "image_motion_prediction",
            "inferred_ground_process_observation",
        ),
        committed_media_pts_ns=1_000_000_000,
    )
    assert state.inferred_ground_continuity_anchor is first_episode_anchor
    processor._reconcile_inferred_ground_with_committed_output(
        state,
        ("cv_prediction", "bounded_cv_process"),
        committed_media_pts_ns=1_000_000_000,
    )
    assert state.inferred_ground_continuity_anchor is first_episode_anchor
    unsupported_track = {
        **track,
        "frame_id": 32,
        "observed_at_us": 1_033_000,
        "media_pts_ns": 1_033_000_000,
    }
    for key in tuple(unsupported_track):
        if key.startswith("world_"):
            unsupported_track.pop(key, None)
    processor._augment_track_with_world(
        0,
        "cam0",
        unsupported_track,
        pose_kpts_abs=keypoints,
        world_now_ts=1.033,
    )
    assert state.inferred_ground_continuity_anchor is first_episode_anchor

    # A learned-height update from an unsupported callback must not change the
    # frozen height used by the still-live inferred episode.
    state.height_ref_scene = 1.7
    second_track = {
        **track,
        "frame_id": 33,
        "observed_at_us": 1_100_000,
        "media_pts_ns": 1_100_000_000,
    }
    for key in tuple(second_track):
        if key.startswith("world_"):
            second_track.pop(key, None)
    processor._augment_track_with_world(
        0,
        "cam0",
        second_track,
        pose_kpts_abs=keypoints,
        world_now_ts=1.1,
    )

    assert second_track["world_valid"] is True
    assert second_track["world_source"] == "image_motion_prediction"
    assert state.inferred_ground_continuity_anchor is first_episode_anchor
    second_provenance = second_track["world_prediction_provenance"]
    assert second_provenance["support_kind"] == "coherent_torso_motion"
    assert second_provenance["raw_origin"] == pytest.approx([0.15, 0.0, 5.9])
    assert second_provenance["raw_sample"] == pytest.approx([0.25, 0.0, 5.8])
    assert second_provenance["raw_delta"] == pytest.approx([0.1, 0.0, -0.1])
    assert second_provenance["trusted_world_origin"] == pytest.approx(
        [0.0, 0.0, 6.0]
    )

    continuous_track = {
        **second_track,
        "frame_id": 70,
        "observed_at_us": 2_350_000,
        "media_pts_ns": 2_350_000_000,
    }
    for key in tuple(continuous_track):
        if key.startswith("world_"):
            continuous_track.pop(key, None)
    processor._augment_track_with_world(
        0,
        "cam0",
        continuous_track,
        pose_kpts_abs=keypoints,
        world_now_ts=2.35,
    )
    assert continuous_track["world_valid"] is True
    assert state.inferred_ground_continuity_anchor is first_episode_anchor

    unsupported_after_gap = {
        **continuous_track,
        "frame_id": 72,
        "observed_at_us": 3_610_000,
        "media_pts_ns": 3_610_000_000,
    }
    for key in tuple(unsupported_after_gap):
        if key.startswith("world_"):
            unsupported_after_gap.pop(key, None)
    processor._augment_track_with_world(
        0,
        "cam0",
        unsupported_after_gap,
        pose_kpts_abs=keypoints,
        world_now_ts=3.61,
    )
    assert state.inferred_ground_continuity_anchor is None
    assert state.inferred_ground_continuity_blocked is True

    rearm_attempt = {
        **continuous_track,
        "frame_id": 73,
        "observed_at_us": 3_620_000,
        "media_pts_ns": 3_620_000_000,
    }
    for key in tuple(rearm_attempt):
        if key.startswith("world_"):
            rearm_attempt.pop(key, None)
    processor._augment_track_with_world(
        0,
        "cam0",
        rearm_attempt,
        pose_kpts_abs=keypoints,
        world_now_ts=3.62,
    )
    assert state.inferred_ground_continuity_anchor is None
    assert state.inferred_ground_continuity_blocked is True

    # A visible metric successor is the authority that ends the shared
    # producer/service episode and permits a later independent root.
    processor._reconcile_inferred_ground_with_committed_output(
        state,
        ("pose_floor_only", None),
        committed_media_pts_ns=3_700_000_000,
    )
    assert state.inferred_ground_continuity_anchor is None
    assert state.inferred_ground_continuity_blocked is False


def test_upright_presence_hold_evidence_requires_exact_current_support() -> None:
    state = hooks._WorldAnchorState()
    state.posture = "standing"
    state.motion_mode = "walk"
    state.lower_body_occluded = False
    state.bbox_stationary_supported = True
    state.image_motion_contact_basis = "pose:torso_motion"
    track = {
        "frame_id": 3050,
        "media_pts_ns": 101_733_333_333,
        "pose_present": True,
        "lower_body_occluded": False,
        "confidence": 0.8955,
        "tracker_confidence": 0.6634,
        "bbox": [736.875, 382.5, 74.625, 153.75],
        "image_size": [1920, 1080],
        "world_floor_candidate": [16.2482, 0.0, 6.6546],
        "world_floor_admitted": True,
        "world_floor_contact_plausible": True,
        "world_observation_range_admitted": True,
        "world_contact_basis": "pose:torso_motion",
        "world_support_state": "floor",
    }
    prior_output = (16.3403, 6.2349, 101_633_333_333, 101.633, 0)

    evidence = hooks._AnalyticsTelemetryProcessor._upright_presence_output_hold_evidence(
        track,
        state=state,
        prior_output=prior_output,
        floor_y=0.0,
    )

    assert evidence is not None
    assert evidence["kind"] == "pose_confirmed_nonseated_floor_near_output"
    assert evidence["output_distance_m"] == pytest.approx(0.4297, abs=1e-3)

    # A gain-zero presence hold does not infer motion from the box. The
    # current pose, floor support, and bounded candidate/output proximity are
    # the evidence; a moving box and a not-yet-classified motion mode are
    # therefore still eligible.
    state.motion_mode = "unknown"
    state.bbox_stationary_supported = False
    moving_evidence = (
        hooks._AnalyticsTelemetryProcessor._upright_presence_output_hold_evidence(
            track,
            state=state,
            prior_output=prior_output,
            floor_y=0.0,
        )
    )
    assert moving_evidence is not None
    assert moving_evidence["motion_mode"] == "unknown"

    state.motion_mode = "sit"
    assert (
        hooks._AnalyticsTelemetryProcessor._upright_presence_output_hold_evidence(
            track,
            state=state,
            prior_output=prior_output,
            floor_y=0.0,
        )
        is None
    )
    state.motion_mode = "unknown"

    track["confidence"] = 1.5
    assert (
        hooks._AnalyticsTelemetryProcessor._upright_presence_output_hold_evidence(
            track,
            state=state,
            prior_output=prior_output,
            floor_y=0.0,
        )
        is None
    )
    track["confidence"] = 0.8955

    track["world_contact_basis"] = "bbox:bottom_center"
    assert (
        hooks._AnalyticsTelemetryProcessor._upright_presence_output_hold_evidence(
            track,
            state=state,
            prior_output=prior_output,
            floor_y=0.0,
        )
        is None
    )
    track["world_contact_basis"] = "pose:torso_motion"

    assert (
        hooks._AnalyticsTelemetryProcessor._upright_presence_output_hold_evidence(
            track,
            state=state,
            prior_output=(16.3403, 6.2349, 99_000_000_000, 99.0, 0),
            floor_y=0.0,
        )
        is None
    )

    track["pose_present"] = False
    assert (
        hooks._AnalyticsTelemetryProcessor._upright_presence_output_hold_evidence(
            track,
            state=state,
            prior_output=prior_output,
            floor_y=0.0,
        )
        is None
    )


def test_stationary_seated_hold_extends_only_for_trusted_lifecycle() -> None:
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
    _prime_metric_output_watermark(
        processor,
        calibration,
        state,
        tracker_id=19,
        media_pts_ns=99_900_000_000,
        filter_ts=99.9,
    )

    tracks: list[dict[str, Any]] = []
    for frame_id, now_ts in enumerate(
        (100.0, 100.2, 100.8, 101.5, 102.2),
        start=1,
    ):
        track = {
            "tracker_id": 19,
            "frame_id": frame_id,
            "bbox": bbox,
            "media_pts_ns": int(round(now_ts * 1_000_000_000)),
            "tracker_lifecycle_generation": 1,
        }
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=keypoints,
            depth_result=depth_result,
            world_now_ts=now_ts,
        )
        processor._commit_enqueued_world_output_watermarks(
            0,
            [track],
            filter_ts=now_ts,
        )
        tracks.append(track)

    # The first three frames establish exact-frame box stationarity.  A
    # trusted seated lifecycle can then remain visible past the generic 0.40s
    # hold, without becoming an accepted measurement or trail point.
    assert tracks[2]["world_valid"] is True
    assert tracks[2]["world_source"] == "anchor_hold"
    assert tracks[2]["world_prediction_provenance"]["stationary_evidence"] == {
        "version": 1,
        "kind": "bbox_stationary",
        "frame_id": 3,
        "media_pts_ns": 100_800_000_000,
        "posture": "sitting",
    }
    assert tracks[2]["world_measurement_accepted"] is False
    assert tracks[2]["trail_append_allowed"] is False
    assert tracks[3]["world_valid"] is True
    assert tracks[4]["world_valid"] is False


def test_main_ground_path_output_speed_rejection_commits_strict_hold() -> None:
    calibration = _anchor_calibration()
    calibration.world_frame_id = "backend_world_m"
    calibration.world_frame_revision = "rev-output-hold-test"
    calibration.frame_transform_sha256 = "f" * 64
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={"models": {"pose": {"kpt_threshold": 0.35}}}
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_calibration=SimpleNamespace(
            snapshot=lambda _sensor_id, _camera_id: calibration,
            world_snapshot=lambda _sensor_id, _camera_id: calibration,
        ),
    )

    def run(frame_id: int, media_pts_ns: int, bbox_left: float) -> dict[str, Any]:
        track: dict[str, Any] = {
            "tracker_id": 33,
            "tracker_lifecycle_generation": 1,
            "frame_id": frame_id,
            "observed_at_us": media_pts_ns // 1_000,
            "media_pts_ns": media_pts_ns,
            "bbox": [bbox_left, 180.0, 80.0, 180.0],
            "image_size": [1280, 720],
            "confidence": 0.95,
        }
        filter_ts = media_pts_ns / 1_000_000_000.0
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            world_now_ts=filter_ts,
        )
        processor._commit_enqueued_world_output_watermarks(
            0,
            [track],
            filter_ts=filter_ts,
        )
        return track

    metric = run(1, 1_000_000_000, 500.0)
    held = run(2, 1_100_000_000, 600.0)

    assert metric["world_measurement_accepted"] is True
    assert held["world_source"] == "anchor_hold"
    assert held["world_measurement_accepted"] is False
    assert held["world_rejection_reason"] == "physical_output_speed_exceeded"
    assert held["world"] == pytest.approx(metric["world"])
    provenance = held["world_prediction_provenance"]
    assert provenance["type"] == "bounded_output_hold"
    assert provenance["origin"] == "last_published_output"
    assert provenance["filter_transition"]["kind"] == "output_hold"
    assert provenance["filter_transition"]["position_gain"] == pytest.approx(0.0)

    key = processor._world_output_watermark_key(
        0,
        held,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-output-hold-test",
        world_transform_sha256="f" * 64,
    )
    reference = processor._world_output_reference(key)
    assert reference is not None
    assert reference[0:2] == pytest.approx(
        (float(metric["world"][0]), float(metric["world"][2]))
    )
    assert reference[2] == 1_100_000_000


def test_current_coherent_motion_revokes_stationary_hold_in_full_hook() -> None:
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
    base_keypoints = np.zeros((17, 3), dtype=np.float64)
    for index, point in {
        5: (600.0, 200.0),
        6: (680.0, 200.0),
        11: (610.0, 300.0),
        12: (670.0, 300.0),
        13: (650.0, 315.0),
        14: (700.0, 315.0),
    }.items():
        base_keypoints[index] = [point[0], point[1], 0.98]
    base_bbox = [560.0, 180.0, 180.0, 190.0]
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=29,
        class_id=0,
        bbox=base_bbox,
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
        ts=99.5,
        world_x=0.0,
        world_z=6.0,
        filtered_ts=99.5,
        last_good_world=(0.0, 0.0, 6.0),
        last_good_ts=99.5,
        posture="sitting",
        motion_mode="sit",
        last_non_upright_ts=99.5,
    )
    processor._world_state_by_track[(0, 29)] = state

    tracks: list[dict[str, Any]] = []
    for frame_id, now_ts, offset in (
        (1, 100.0, 0.0),
        (2, 100.2, 2.1),
        (3, 100.4, 4.2),
    ):
        bbox = list(base_bbox)
        bbox[0] += offset
        keypoints = np.asarray(base_keypoints, dtype=np.float64).copy()
        keypoints[:, 0] += offset
        track = {
            "tracker_id": 29,
            "frame_id": frame_id,
            "bbox": bbox,
            "confidence": 0.95,
        }
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=keypoints,
            depth_result=depth_result,
            world_now_ts=now_ts,
        )
        tracks.append(track)

    # Each 2.1px step is bbox-stationary jitter, but the exact three-row
    # contact+bbox window proves 4.2px coherent movement. The current motion
    # proof clears stationarity before fallback selection, so the stale local
    # value cannot extend a 0.40s metric hold to the 2.0s seated TTL.
    assert state.image_motion_supported is True
    assert state.bbox_stationary_supported is False
    assert tracks[-1]["world_valid"] is False
    assert "stationary_bbox_hold" not in str(
        tracks[-1].get("world_quality_reason") or ""
    )


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
    _prime_metric_output_watermark(
        processor,
        calibration,
        state,
        tracker_id=7,
        media_pts_ns=100_000_000_000,
        filter_ts=100.0,
    )

    stationary = iter((False, True))
    monkeypatch.setattr(
        hooks,
        "observe_bbox_stationarity",
        lambda *_args, **_kwargs: next(stationary),
    )
    outputs: list[dict[str, Any]] = []
    for frame_id, now_ts in ((1, 100.10), (2, 100.15)):
        if frame_id == 2:
            # Exercise the adversarial path where motion-mode processing sees
            # an idle-like lifecycle with an older lock while this frame's
            # rejected observation has already advanced the bounded process.
            state.motion_mode = "sit"
            state.locked_world = (0.0, 6.0)
        track = {
            "tracker_id": 7,
            "frame_id": frame_id,
            "bbox": bbox,
            "media_pts_ns": int(round(now_ts * 1_000_000_000)),
            "tracker_lifecycle_generation": 1,
        }
        processor._augment_track_with_world(
            0,
            "cam0",
            track,
            pose_kpts_abs=keypoints,
            depth_result=depth_result,
            world_now_ts=now_ts,
        )
        processor._commit_enqueued_world_output_watermarks(
            0,
            [track],
            filter_ts=now_ts,
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
