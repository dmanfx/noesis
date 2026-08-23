from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np

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


class _MaskedStatsDevice:
    def __init__(self, *, contaminate_contact: bool = False) -> None:
        self.contaminate_contact = bool(contaminate_contact)
        self.calls: list[tuple[int, int, int, int, int]] = []

    def sample_masked_roi_stats(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        mask: np.ndarray,
        threshold: float,
        max_samples: int,
    ) -> dict[str, float | int]:
        active = int(np.count_nonzero(np.asarray(mask) > float(threshold)))
        self.calls.append((left, top, width, height, active))
        contaminated = self.contaminate_contact and len(self.calls) == 2
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
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        SimpleNamespace(extract_pose_features=lambda _obj: json.dumps(_pose_payload())),
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
    processor.handle_frame_ds8(SimpleNamespace(frame_items=[frame_meta]), frame_meta)
    assert len(attached) == 1
    return attached[0], device, processor


def test_pose_capsule_uses_observed_ankles_for_native_depth_support(monkeypatch) -> None:
    payload, device, _processor = _run_pose_depth_sample(monkeypatch)

    assert len(device.calls) == 2
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

    processor._apply_public_depth_fields(track, None)

    assert track["depth_evidence_quality"] is None
    assert track["depth_evidence_reason"] is None
    assert track["depth_spread_m"] is None
    assert track["depth_anchor_spread_m"] is None
    assert track["depth_measurement_frame_id"] is None
    assert track["depth_measurement_ts_us"] is None
    assert track["depth_measurement_age_us"] is None
    assert track["depth_measurement_cached"] is None


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
