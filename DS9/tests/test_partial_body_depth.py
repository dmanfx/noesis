from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from noesis.metadata.object_depth import ObjectDepthResult, PoseDepthSample
from noesis.pipelines import hooks
from noesis_core.contracts.base import Matrix3


_BBOX = (100.0, 100.0, 100.0, 200.0)
_MODEL = "depth-anything-v2-metric-hypersim-vits"


def _keypoints() -> np.ndarray:
    return np.zeros((17, 3), dtype=np.float32)


def _good_stats(
    *,
    median: float = 4.0,
    p10: float = 3.95,
    p90: float = 4.05,
    count: int = 80,
    fraction: float = 0.9,
) -> dict[str, Any]:
    return {
        "sample_count": count,
        "valid_fraction": fraction,
        "depth_center": median,
        "depth_median": median,
        "depth_mean": median,
        "depth_p10": p10,
        "depth_p90": p90,
        "depth_min": p10,
        "depth_max": p90,
        "sampled_capsule_area_px": 100,
    }


class _CapsuleStatsDevice:
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[list[float]], int]] = []

    def sample_pose_capsule_stats(
        self,
        capsules: list[list[float]],
        max_samples: int,
    ) -> Mapping[str, Any]:
        self.calls.append((capsules, max_samples))
        return self.responses[len(self.calls) - 1]


def _sample_native(
    device: _CapsuleStatsDevice,
    *,
    body_mask: np.ndarray | None = None,
    contact_mask: np.ndarray | None = None,
    contact_uv: tuple[float, float] | None = None,
    visible_ankles: int = 0,
    contact_capsules: tuple[tuple[float, ...], ...] = (),
    torso_capsule: tuple[float, ...] | None = None,
    torso_uv: tuple[float, float] | None = None,
    shoulder_capsule: tuple[float, ...] | None = None,
    shoulder_uv: tuple[float, float] | None = None,
) -> ObjectDepthResult | None:
    processor = object.__new__(hooks._ObjectDepthFusionProcessor)
    processor.depth_unit = "m"
    processor.depth_is_metric = True
    processor.depth_model_name = _MODEL
    shape = (200, 100)
    return processor._sample_pose_capsule_result_native(
        SimpleNamespace(source_id=0, frame_number=18, buf_pts=1_500_000_000),
        SimpleNamespace(object_id=27, class_id=0, confidence=0.93),
        bbox=_BBOX,
        depth_frame=SimpleNamespace(depth_device_frame=device),
        crop_rect=(100, 100, 200, 300),
        body_mask=np.ones(shape, dtype=bool) if body_mask is None else body_mask,
        contact_mask=(
            np.ones(shape, dtype=bool) if contact_mask is None else contact_mask
        ),
        contact_uv=contact_uv,
        visible_ankles=visible_ankles,
        contact_capsules=contact_capsules,
        torso_capsule=torso_capsule,
        torso_uv=torso_uv,
        shoulder_capsule=shoulder_capsule,
        shoulder_uv=shoulder_uv,
    )


def test_shoulders_alone_support_native_body_depth_without_hips_or_ankles() -> None:
    processor = object.__new__(hooks._ObjectDepthFusionProcessor)
    keypoints = _keypoints()
    keypoints[5] = [120.0, 150.0, 0.95]
    keypoints[6] = [180.0, 150.0, 0.95]

    body_mask, contact_mask, contact_uv, visible_ankles, contacts = (
        processor._pose_capsule_masks(
            keypoints,
            crop_rect=(100, 100, 200, 300),
            bbox=_BBOX,
        )
    )
    torso_capsule, torso_uv = processor._pose_torso_capsule_geometry(
        keypoints,
        bbox=_BBOX,
    )
    shoulder_capsule, shoulder_uv = processor._pose_shoulder_capsule_geometry(
        keypoints,
        bbox=_BBOX,
    )

    assert np.any(body_mask)
    assert not np.any(contact_mask)
    assert contact_uv is None
    assert visible_ankles == 0
    assert contacts == []
    assert torso_capsule is None and torso_uv is None
    assert shoulder_capsule is not None and shoulder_uv is not None

    device = _CapsuleStatsDevice([_good_stats()])
    result = _sample_native(
        device,
        body_mask=body_mask,
        contact_mask=contact_mask,
        contact_uv=contact_uv,
        visible_ankles=visible_ankles,
        contact_capsules=tuple(contacts),
        shoulder_capsule=shoulder_capsule,
        shoulder_uv=tuple(shoulder_uv),
    )

    assert result is not None
    assert result.status == "ok"
    assert result.anchor_source == "pose_shoulder_support"
    assert result.anchor_uv == pytest.approx(shoulder_uv)
    assert len(device.calls) == 1
    assert device.calls[0][0] == [list(shoulder_capsule)]

    telemetry = object.__new__(hooks._AnalyticsTelemetryProcessor)
    anchor = telemetry._resolve_person_depth_anchor(result)
    assert anchor is not None
    assert anchor.source == "person_body_projection"
    assert anchor.contact_basis == "depth:pose_shoulder_support"
    assert telemetry._anchor_is_verified_ground_contact(anchor) is False


def test_full_torso_capsule_accepts_horizontal_pose_geometry() -> None:
    processor = object.__new__(hooks._ObjectDepthFusionProcessor)
    keypoints = _keypoints()
    keypoints[5] = [120.0, 140.0, 0.95]
    keypoints[6] = [140.0, 180.0, 0.95]
    keypoints[11] = [200.0, 140.0, 0.95]
    keypoints[12] = [220.0, 180.0, 0.95]

    capsule, uv = processor._pose_torso_capsule_geometry(
        keypoints,
        bbox=(100.0, 100.0, 160.0, 120.0),
    )

    assert capsule is not None
    assert uv is not None
    assert capsule[2] > capsule[0]
    assert capsule[1] == pytest.approx(capsule[3])
    assert uv[1] == pytest.approx(capsule[1])


def test_rejected_ankle_and_torso_attempts_fall_back_to_shoulder_uv() -> None:
    ankle_capsule = (148.0, 275.0, 150.0, 290.0, 5.0, 8.0)
    torso_capsule = (140.0, 180.0, 150.0, 180.0, 8.0, 8.0)
    shoulder_capsule = (130.0, 150.0, 140.0, 150.0, 6.0, 6.0)
    device = _CapsuleStatsDevice(
        [
            _good_stats(median=4.0, p10=2.0, p90=8.0),
            _good_stats(count=0, fraction=0.0, median=4.0, p10=4.0, p90=4.0),
            _good_stats(median=4.6, p10=4.55, p90=4.65),
        ]
    )

    result = _sample_native(
        device,
        contact_uv=(150.0, 290.0),
        visible_ankles=1,
        contact_capsules=(ankle_capsule,),
        torso_capsule=torso_capsule,
        torso_uv=(145.0, 180.0),
        shoulder_capsule=shoulder_capsule,
        shoulder_uv=(135.0, 150.0),
    )

    assert result is not None
    assert [sample.anchor_source for sample in result.pose_depth_samples] == [
        "pose_ankle_support",
        "pose_torso_support",
        "pose_shoulder_support",
    ]
    ankle, torso, shoulder = result.pose_depth_samples
    assert ankle.evidence_reason == "depth_spread_exceeded"
    assert torso.evidence_reason == "depth_support_count_low"
    assert shoulder.evidence_reason is None
    assert result.anchor_source == "pose_shoulder_support"
    assert result.anchor_uv == pytest.approx((135.0, 150.0))
    assert result.anchor_depth_m == pytest.approx(4.6)
    assert [call[0] for call in device.calls] == [
        [list(ankle_capsule)],
        [list(torso_capsule)],
        [list(shoulder_capsule)],
    ]


def test_good_ankle_and_body_patches_are_both_sampled_with_posture_selection() -> None:
    ankle_capsule = (148.0, 275.0, 150.0, 290.0, 5.0, 8.0)
    torso_capsule = (140.0, 180.0, 150.0, 180.0, 8.0, 8.0)
    shoulder_capsule = (130.0, 150.0, 140.0, 150.0, 6.0, 6.0)
    device = _CapsuleStatsDevice(
        [
            _good_stats(median=5.0, p10=4.98, p90=5.02),
            _good_stats(median=4.4, p10=4.35, p90=4.45),
        ]
    )

    result = _sample_native(
        device,
        contact_uv=(150.0, 290.0),
        visible_ankles=1,
        contact_capsules=(ankle_capsule,),
        torso_capsule=torso_capsule,
        torso_uv=(145.0, 180.0),
        shoulder_capsule=shoulder_capsule,
        shoulder_uv=(135.0, 150.0),
    )

    assert result is not None
    assert len(device.calls) == 2
    assert [sample.anchor_source for sample in result.pose_depth_samples] == [
        "pose_ankle_support",
        "pose_torso_support",
    ]
    assert result.anchor_source == "pose_ankle_support"

    telemetry = object.__new__(hooks._AnalyticsTelemetryProcessor)
    non_upright = telemetry._person_depth_for_posture(result, "sitting")
    standing = telemetry._person_depth_for_posture(result, "standing")
    assert non_upright is not None
    assert non_upright.anchor_source == "pose_torso_support"
    assert non_upright.anchor_uv == pytest.approx((145.0, 180.0))
    assert standing is result
    assert standing.anchor_source == "pose_ankle_support"
    assert [sample.anchor_source for sample in standing.pose_depth_samples] == [
        "pose_ankle_support",
        "pose_torso_support",
    ]


def test_posture_selection_preserves_cached_timing_and_nested_round_trip() -> None:
    ankle = PoseDepthSample(
        anchor_source="pose_ankle_support",
        anchor_uv=(150.0, 290.0),
        sample_count=80,
        valid_fraction=0.9,
        mask_area_px=100,
        depth_median=5.0,
        depth_p10=4.98,
        depth_p90=5.02,
        evidence_quality="good",
    )
    shoulder = PoseDepthSample(
        anchor_source="pose_shoulder_support",
        anchor_uv=(135.0, 150.0),
        sample_count=60,
        valid_fraction=0.8,
        mask_area_px=90,
        depth_median=4.6,
        depth_p10=4.55,
        depth_p90=4.65,
        evidence_quality="estimated",
    )
    result = ObjectDepthResult(
        source_id=0,
        frame_id=20,
        object_id=27,
        class_id=0,
        bbox=_BBOX,
        score=0.93,
        sampling_mode="pose_capsule_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=140,
        valid_fraction=0.87,
        anchor_source="pose_ankle_support",
        anchor_uv=ankle.anchor_uv,
        anchor_depth_m=5.0,
        anchor_sample_count=80,
        anchor_valid_fraction=0.9,
        evidence_quality="good",
        pose_depth_samples=(ankle, shoulder),
        measurement_frame_id=18,
        measurement_ts_us=1_500_000,
        measurement_age_us=250_000,
        measurement_cached=True,
        depth_tensor_frame_id=19,
        depth_tensor_ts_us=1_700_000,
        depth_tensor_age_frames=1,
        depth_tensor_age_us=50_000,
        ts_us=1_750_000,
    )

    telemetry = object.__new__(hooks._AnalyticsTelemetryProcessor)
    selected = telemetry._person_depth_for_posture(result, "lying")

    assert selected is not None
    assert selected is not result
    assert selected.anchor_source == "pose_shoulder_support"
    assert selected.anchor_uv == pytest.approx((135.0, 150.0))
    assert selected.anchor_depth_m == pytest.approx(4.6)
    assert selected.measurement_frame_id == 18
    assert selected.measurement_ts_us == 1_500_000
    assert selected.measurement_age_us == 250_000
    assert selected.measurement_cached is True
    assert selected.depth_tensor_frame_id == 19
    assert selected.depth_tensor_ts_us == 1_700_000
    assert selected.depth_tensor_age_frames == 1
    assert selected.depth_tensor_age_us == 50_000
    assert result.anchor_source == "pose_ankle_support"
    assert result.anchor_uv == pytest.approx((150.0, 290.0))

    restored = ObjectDepthResult.from_json(selected.to_json())
    assert restored == selected
    assert restored.to_dict() == selected.to_dict()
    assert all(isinstance(sample, PoseDepthSample) for sample in restored.pose_depth_samples)


@pytest.mark.parametrize(
    ("posture", "expected_support"),
    (("sitting", "seat"), ("lying", "couch"), ("unknown", "unknown")),
)
def test_registered_shoulder_depth_gets_typed_candidate_without_floor_claim(
    posture: str,
    expected_support: str,
) -> None:
    shoulder = PoseDepthSample(
        anchor_source="pose_shoulder_support",
        anchor_uv=(135.0, 150.0),
        sample_count=60,
        valid_fraction=0.8,
        mask_area_px=90,
        depth_median=4.6,
        depth_p10=4.55,
        depth_p90=4.65,
        evidence_quality="estimated",
    )
    result = ObjectDepthResult(
        source_id=0,
        frame_id=20,
        object_id=27,
        class_id=0,
        bbox=_BBOX,
        score=0.93,
        sampling_mode="pose_capsule_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=140,
        valid_fraction=0.87,
        anchor_source="pose_shoulder_support",
        anchor_uv=shoulder.anchor_uv,
        anchor_depth_m=4.6,
        anchor_sample_count=60,
        anchor_valid_fraction=0.8,
        evidence_quality="estimated",
        pose_depth_samples=(shoulder,),
    )
    telemetry = object.__new__(hooks._AnalyticsTelemetryProcessor)
    telemetry.depth_registration = None
    telemetry.scene_priors = None
    telemetry._depth_candidate_covariance = lambda *_args, **_kwargs: Matrix3(  # type: ignore[method-assign]
        values=(0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04)
    )
    selected = telemetry._person_depth_for_posture(result, posture)
    assert selected is not None
    person_anchor = telemetry._resolve_person_depth_anchor(selected)
    assert person_anchor is not None
    assert person_anchor.source == "person_body_projection"
    assert person_anchor.contact_basis == "depth:pose_shoulder_support"
    assert telemetry._anchor_is_verified_ground_contact(person_anchor) is False

    candidates: list[dict[str, Any]] = []
    telemetry._append_universal_world_candidates(
        candidates,
        sensor_id=0,
        track={"tracker_id": 27, "world_floor_incidence_sin": 0.8},
        camera_id="cam0",
        calib=SimpleNamespace(floor_y=0.0, camera_id="cam0"),
        floor_candidate=None,
        floor_ray_admitted=False,
        anchor_candidate=None,
        pose_uv=None,
        contact_basis=person_anchor.contact_basis,
        posture=posture,
        occlusion_fraction=0.7,
        image_motion_supported=False,
        depth_obs=np.asarray([1.0, 1.0, 2.0], dtype=np.float64),
        depth_weight=0.8,
        depth_observation=hooks._DepthObservationResult(
            np.asarray([1.0, 1.0, 2.0], dtype=np.float64),
            0.8,
            "ok",
            registered_depth_m=4.6,
            registration_status="raw_passthrough",
        ),
        depth_result=selected,
        person_anchor=person_anchor,
        reject_current_geometry=False,
        flip_u=False,
        flip_v=False,
    )

    assert [candidate["kind"] for candidate in candidates] == ["registered_depth"]
    candidate = candidates[0]
    assert candidate["anchor"] == "person_body_projection"
    assert candidate["contact_basis"] == "depth:pose_shoulder_support"
    assert candidate["support_state"] == expected_support
    assert telemetry._resolver_candidate_is_ground_footprint_supported(
        SimpleNamespace(**candidate)
    )
