from __future__ import annotations

from noesis.metadata.object_depth import ObjectDepthResult


def test_object_depth_result_roundtrip() -> None:
    payload = ObjectDepthResult(
        source_id=2,
        frame_id=14,
        object_id=7,
        class_id=0,
        bbox=(10.0, 20.0, 30.0, 40.0),
        score=0.91,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=512,
        valid_fraction=0.82,
        depth_center=1.2,
        depth_median=1.3,
        depth_mean=1.31,
        depth_p10=0.9,
        depth_p90=1.8,
        depth_min=0.7,
        depth_max=2.0,
        mask_area_px=640,
        stable_id=9,
        depth_map_ref="memory://depth/family-room/1234",
        anchor_uv=(25.0, 57.5),
        anchor_source="lower_body_band",
        anchor_depth_m=1.25,
        anchor_sample_count=72,
        anchor_valid_fraction=0.91,
        evidence_quality="good",
        evidence_reason="pose_ankle_support",
        depth_spread_m=0.22,
        anchor_depth_spread_m=0.12,
        measurement_frame_id=14,
        measurement_ts_us=123456000,
        measurement_age_us=789,
        measurement_cached=True,
        world_point=(1.0, 0.0, 3.5),
        world_point_depth=(1.1, 0.2, 3.6),
        world_point_floor=(1.0, 0.0, 3.4),
        projection_method="depth",
        spatial_status="ok",
        spatial_class="person",
        model="depth-anything-v2-metric-hypersim-vits",
        ts_us=123456789,
    )

    restored = ObjectDepthResult.from_json(payload.to_json())
    assert restored == payload
    as_dict = restored.to_dict()
    assert as_dict["type"] == "object_depth"
    assert as_dict["version"] == 2
    assert as_dict["sampling_mode"] == "instance_mask"
    assert as_dict["unit"] == "m"
    assert as_dict["is_metric"] is True
    assert as_dict["projection_method"] == "depth"
    assert as_dict["spatial_status"] == "ok"
    assert as_dict["anchor_uv"] == [25.0, 57.5]
    assert as_dict["anchor_sample_count"] == 72
    assert as_dict["anchor_valid_fraction"] == 0.91
    assert as_dict["evidence_quality"] == "good"
    assert as_dict["anchor_depth_spread_m"] == 0.12
    assert as_dict["measurement_frame_id"] == 14
    assert as_dict["measurement_ts_us"] == 123456000
    assert as_dict["measurement_age_us"] == 789
    assert as_dict["measurement_cached"] is True
    assert as_dict["world_point"] == [1.0, 0.0, 3.5]


def test_object_depth_result_allows_nullable_depth_fields() -> None:
    payload = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=-1,
        class_id=56,
        bbox=(0, 0, 10, 10),
        score=0.5,
        sampling_mode="instance_mask",
        status="no_valid_depth",
        unit="relative",
        is_metric=False,
        sample_count=0,
        valid_fraction=0.0,
    )
    as_dict = payload.to_dict()
    assert "depth_center" not in as_dict
    assert "anchor_uv" not in as_dict
    assert "world_point" not in as_dict
    assert as_dict["status"] == "no_valid_depth"
    assert as_dict["sampling_mode"] == "instance_mask"
