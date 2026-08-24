from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Mapping, Sequence

import numpy as np

import pytest

from geometry.homography import project_world_to_image
from noesis.calibration.world_fusion_policy import (
    CameraWorldFusionProfile,
    WorldFusionPolicy,
)
from noesis.metadata.object_depth import ObjectDepthResult
from noesis.pipelines import ds8_pipeline, hooks, hooks_v3dt_reimpl
from noesis.telemetry.bev import BevPublicationReceipt, CalibrationSnapshot
from noesis.telemetry.publishers import TrackingPublicationReceipt


def _tracking_receipt(
    source_id: int,
    kwargs: Mapping[str, Any],
    *,
    sequence: int = 0,
    submission_id: int = 1,
) -> TrackingPublicationReceipt:
    metadata = dict(kwargs.get("frame_metadata") or {})
    return TrackingPublicationReceipt(
        source_id=int(source_id),
        frame_id=(
            int(metadata["frame_id"])
            if metadata.get("frame_id") is not None
            else None
        ),
        observed_at_us=(
            int(metadata["observed_at_us"])
            if metadata.get("observed_at_us") is not None
            else None
        ),
        tracking_publication_sequence=int(sequence),
        outbound_submission_id=int(submission_id),
        outbound_message_count=1,
    )


@pytest.fixture(autouse=True)
def reset_pipeline_singleton():
    ds8_pipeline._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]
    yield
    ds8_pipeline._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]


def _build_pipeline() -> ds8_pipeline.DS8Pipeline:
    return ds8_pipeline.build_pipeline(Path("config/infer.yaml"))


class _StableIDMgr:
    def __init__(self) -> None:
        self.active_tracks: dict[tuple[int, int], dict[str, Any]] = {}

    def update(
        self,
        *,
        sensor_id: int,
        ds_obj_id: int,
        bbox_ltrbwh: tuple[float, float, float, float],
        ts: float,
        zone: str | None,
        frame_bgr: Any,
        embedding: Any,
        pose_features: Any = None,
        pose_quality: float | None = None,
        world_xy: tuple[float, float] | None = None,
        world_valid: bool = False,
    ) -> int:
        self.active_tracks[(int(sensor_id), int(ds_obj_id))] = {"stable_id": 1}
        return 1

    def remove_missing_tracks(self, sensor_id: int, present_track_ids: list[int], ts: float) -> None:
        return None

    def prune_ghosts(self, ts: float) -> None:
        return None

    def observe_copresence(self, stable_ids: list[int], ts: float) -> None:
        return None


class _TrackingPublisher:
    def publish(
        self,
        _source_id: int,
        _tracks: Sequence[dict[str, Any]],
        **_kwargs: Any,
    ) -> TrackingPublicationReceipt:
        return _tracking_receipt(_source_id, _kwargs)


class _CalibrationProvider:
    def __init__(self, snapshot: CalibrationSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self, source_id: int, camera_id: str) -> CalibrationSnapshot:
        return self._snapshot


class _PoseExt:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self._payload = payload

    def extract_pose_features(self, _obj_meta: Any) -> str | None:
        if self._payload is None:
            return None
        return json.dumps(self._payload)


class _DepthExt:
    def __init__(self, payload: ObjectDepthResult | None) -> None:
        self._payload = payload

    def extract_object_depth(self, _obj_meta: Any) -> str | None:
        if self._payload is None:
            return None
        return self._payload.to_json()


class _DepthRegistrationManager:
    def __init__(self, *, scale: float, camera_id: str = "cam0", registration_id: str = "reg-1") -> None:
        self.scale = float(scale)
        self.camera_id = str(camera_id)
        self.registration_id = str(registration_id)

    def apply(self, *, camera_id: str, raw_depth_m: float) -> tuple[float | None, str, str | None]:
        if str(camera_id) != self.camera_id:
            return None, "camera_mismatch", None
        return float(raw_depth_m) * float(self.scale), "ok", self.registration_id


class _RejectingDepthRegistrationManager:
    def apply(self, *, camera_id: str, raw_depth_m: float) -> tuple[None, str, str]:
        assert camera_id == "cam0"
        assert raw_depth_m > 0.0
        return None, "out_of_domain_or_invalid", "reg-rejected"


def _build_anchor_snapshot() -> CalibrationSnapshot:
    K = np.array(
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
    R_wc = np.column_stack([right, down, forward])
    R_cw = R_wc.T
    t = -R_cw @ camera_world
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R_cw
    E[:3, 3] = t
    return CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=K,
        extrinsics_col_major=list(E.flatten(order="F")),
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=1.0,
    )


def _project_anchor_uv(calib: CalibrationSnapshot, world_xyz: Sequence[float]) -> tuple[float, float]:
    uv = project_world_to_image(
        world_xyz,
        calib.intrinsics,
        calib.extrinsics_col_major,
        calib.image_size,
        unit_scale=1.0,
    )
    assert uv is not None
    return float(uv[0]), float(uv[1])


def _anchor_bbox(
    calib: CalibrationSnapshot,
    *,
    foot_world: Sequence[float],
    height_scene: float = 1.8,
    width_px: float = 80.0,
) -> list[float]:
    u_foot, v_foot = _project_anchor_uv(calib, foot_world)
    _u_head, v_head = _project_anchor_uv(
        calib,
        [float(foot_world[0]), float(foot_world[1]) + float(height_scene), float(foot_world[2])],
    )
    bbox = [float(u_foot - (width_px * 0.5)), float(v_head), float(width_px), float(v_foot - v_head)]
    assert bbox[3] > 0.0
    return bbox


def _build_pose_payload(calib: CalibrationSnapshot, bbox: Sequence[float], *, foot_z: float) -> dict[str, Any]:
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    world_points = {
        "nose": (0.00, 1.68, foot_z),
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
        u, v = _project_anchor_uv(calib, world_xyz)
        idx = hooks._POSE_KPT_INDEX[name]  # type: ignore[attr-defined]
        keypoints[idx] = [float(u), float(v), 0.98]
    return {
        "bbox": [float(x) for x in bbox],
        "keypoints_abs": keypoints,
    }


def _build_anchor_processor(calib: CalibrationSnapshot) -> hooks._AnalyticsTelemetryProcessor:
    pipeline = SimpleNamespace(config={"models": {"pose": {"kpt_threshold": 0.35}}})
    return hooks._AnalyticsTelemetryProcessor(
        pipeline=pipeline,
        tracking_pub=_TrackingPublisher(),
        camera_labels={0: "cam0"},
        sensor_id_map={},
        bev_renderer=None,
        bev_calibration=_CalibrationProvider(calib),
        diagnostics_logger=None,
    )


def test_track_image_size_uses_declared_resolution_not_principal_point() -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.bev_calibration = SimpleNamespace(
        _intrinsics_loader=SimpleNamespace(
            get=lambda _source_id: SimpleNamespace(
                width=1920,
                height=1080,
                cx=937.3184106054618,
                cy=524.9146561743939,
            )
        )
    )
    frame_meta = SimpleNamespace(source_frame_width=1280, source_frame_height=720)

    assert processor._track_image_size(2, frame_meta) == (1920, 1080)  # type: ignore[attr-defined]

    processor.bev_calibration._intrinsics_loader.get = lambda _source_id: SimpleNamespace(
        width=0,
        height=0,
        cx=937.3184106054618,
        cy=524.9146561743939,
    )
    assert processor._track_image_size(2, frame_meta) == (1280, 720)  # type: ignore[attr-defined]


def test_post_mux_frame_raster_is_not_scaled_again_from_raw_intrinsics() -> None:
    """Family's 1920x1080 tracker raster must not be treated as 1280x720."""
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.bev_calibration = SimpleNamespace(
        _intrinsics_loader=SimpleNamespace(
            get=lambda _source_id: SimpleNamespace(width=1280, height=720)
        )
    )
    frame_meta = SimpleNamespace(
        source_frame_width=1280,
        source_frame_height=720,
        frame_width=1920,
        frame_height=1080,
    )

    assert processor._track_image_size(0, frame_meta) == (1920, 1080)
    bbox = [600.0, 300.0, 240.0, 360.0]
    assert processor._scale_bbox_to_image_size(
        bbox,
        processor._track_image_size(0, frame_meta),
        (1920, 1080),
    ) == pytest.approx(bbox)

    # Native metadata producers are allowed to omit the canonical frame
    # dimensions.  The configured post-mux raster still outranks both the raw
    # source size and the raw calibration artifact in that case.
    processor.pipeline.frame_size = (1920, 1080)
    frame_meta_without_canonical_size = SimpleNamespace(
        source_frame_width=1280,
        source_frame_height=720,
    )
    assert processor._track_image_size(
        0,
        frame_meta_without_canonical_size,
    ) == (1920, 1080)


def test_pose_keypoints_scale_into_calibration_image_coordinates() -> None:
    processor = _build_anchor_processor(_build_anchor_snapshot())
    keypoints = np.array(
        [
            [320.0, 180.0, 0.95],
            [100.0, 50.0, 0.80],
        ],
        dtype=np.float64,
    )

    scaled = processor._scale_pose_keypoints_to_image_size(  # type: ignore[attr-defined]
        keypoints,
        (640, 360),
        (1280, 720),
    )

    assert scaled is not None
    assert np.allclose(
        scaled[:, :2],
        [[640.0, 360.0], [200.0, 100.0]],
    )
    assert scaled[:, 2] == pytest.approx([0.95, 0.80])
    assert keypoints[0] == pytest.approx([320.0, 180.0, 0.95])


def _fusion_policy(
    *,
    floor_weight_scale: float,
    depth_weight_scale: float,
    floor_only_allowed: bool,
    floor_ray_max_range_m: float = 22.0,
) -> WorldFusionPolicy:
    profile = CameraWorldFusionProfile(
        camera_id="cam0",
        calibration_fingerprint_sha256="a" * 64,
        registration_id="reg-1",
        floor_weight_scale=float(floor_weight_scale),
        depth_weight_scale=float(depth_weight_scale),
        floor_only_allowed=bool(floor_only_allowed),
        floor_ray_max_range_m=float(floor_ray_max_range_m),
    )
    return WorldFusionPolicy(
        policy_id="test-policy",
        runtime_lane="ds8",
        evidence={},
        cameras={"cam0": profile},
    )


def _anchor_depth_result(
    calib: CalibrationSnapshot,
    bbox: Sequence[float],
    *,
    object_id: int,
) -> ObjectDepthResult:
    anchor_range_m = float(
        np.linalg.norm(
            np.array([0.0, 0.0, 6.0])
            - np.array([0.0, 2.2, -6.0], dtype=np.float64)
        )
    )
    return ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=object_id,
        class_id=0,
        bbox=bbox,
        score=0.91,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=512,
        valid_fraction=0.96,
        anchor_source="lower_body_band",
        anchor_depth_m=anchor_range_m,
        depth_center=anchor_range_m,
        depth_median=anchor_range_m,
    )


def test_pose_anchor_prefers_attached_payload_before_native(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    payload = _build_pose_payload(calib, bbox, foot_z=6.0)
    calls = {"native": 0}

    class _CountingPoseExt(_PoseExt):
        def extract_pose_keypoints(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            calls["native"] += 1
            return payload

    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _CountingPoseExt(payload))
    hooks.reset_core_path_instrumentation()

    keypoints = processor._extract_pose_keypoints_for_anchor(SimpleNamespace(), bbox)  # type: ignore[attr-defined]

    assert keypoints is not None
    assert calls["native"] == 0
    counters = hooks.get_core_path_instrumentation_snapshot().get("counters", {})
    assert int(counters.get("detection_wake.pose_anchor_payload_hit", 0)) == 1


def test_pose_feature_processor_reuses_cache_before_native_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Rect:
        left = 10.0
        top = 20.0
        width = 80.0
        height = 180.0

    class _Obj:
        def __init__(self) -> None:
            self.object_id = 7
            self.class_id = 0
            self.rect_params = _Rect()

    keypoints_roi = [[float(i * 2), float(i * 3), 0.92] for i in range(17)]
    keypoints_abs = [[10.0 + row[0], 20.0 + row[1], row[2]] for row in keypoints_roi]
    pose_payload = {
        "score": 0.87,
        "keypoints_roi": keypoints_roi,
        "keypoints_abs": keypoints_abs,
    }

    class _NativePoseExt:
        calls = 0
        attached: list[dict[str, Any]] = []

        @classmethod
        def extract_pose_keypoints(cls, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            cls.calls += 1
            return pose_payload

        @classmethod
        def attach_pose_features(cls, obj_meta: Any, payload_json: str, _replace_existing: bool = True) -> bool:
            payload = json.loads(payload_json)
            cls.attached.append(payload)
            obj_meta.pose_features = payload
            return True

    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _NativePoseExt)
    pipeline = SimpleNamespace(stable_id_mgr=None)
    processor = hooks.PoseFeatureProcessor(  # type: ignore[attr-defined]
        pipeline=pipeline,
        gie_id=4,
        cache_max_age_frames=8,
        cache_max_bbox_shift=0.35,
    )
    hooks.reset_core_path_instrumentation()

    processor.handle_frame_ds8(SimpleNamespace(source_id=0, frame_number=1, object_items=[_Obj()]))
    processor.handle_frame_ds8(SimpleNamespace(source_id=0, frame_number=2, object_items=[_Obj()]))

    assert _NativePoseExt.calls == 1
    assert len(_NativePoseExt.attached) == 2
    assert _NativePoseExt.attached[1]["pose_cache_reused"] is True
    assert _NativePoseExt.attached[1]["pose_cache_age_frames"] == 1
    counters = hooks.get_core_path_instrumentation_snapshot().get("counters", {})
    assert int(counters.get("detection_wake.pose_feature_native_extract", 0)) == 1
    assert int(counters.get("detection_wake.pose_feature_cache_hit", 0)) == 1
    assert int(counters.get("tensor_host_copies_total.pose", 0)) == 1


def test_ds8_analytics_reid_embedding_budget_defers_extra_people(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOESIS_REID_EMBEDS_PER_FRAME_MAX", "1")
    monkeypatch.setenv("NOESIS_TRACKING_PUBLISH_MAX_HZ", "0")
    pipeline = SimpleNamespace(
        config={"models": {"reid": {"gie_id": 3}}},
        stable_id_mgr=_StableIDMgr(),
    )
    processor = hooks._AnalyticsTelemetryProcessor(  # type: ignore[attr-defined]
        pipeline=pipeline,
        tracking_pub=_TrackingPublisher(),
        camera_labels={0: "camera_0"},
        sensor_id_map={},
        bev_renderer=None,
        bev_calibration=None,
        diagnostics_logger=None,
    )

    class _NativeReidExt:
        calls = 0

        @classmethod
        def extract_reid_embedding(cls, *_args: Any, **_kwargs: Any) -> list[float]:
            cls.calls += 1
            return [1.0] + [0.0] * 511

    class _Rect:
        def __init__(self, left: float) -> None:
            self.left = left
            self.top = 20.0
            self.width = 40.0
            self.height = 80.0

    class _Obj:
        def __init__(self, object_id: int, left: float) -> None:
            self.object_id = object_id
            self.class_id = 0
            self.confidence = 0.91
            self.tracker_confidence = 0.80
            self.rect_params = _Rect(left)
            self.text_params = SimpleNamespace(display_text="", font_params=SimpleNamespace(size=0, name=None))

    frame = SimpleNamespace(
        source_id=0,
        frame_number=1,
        frame_width=1280,
        frame_height=720,
        buf_pts=123456789000,
        object_items=iter([_Obj(1, 10.0), _Obj(2, 100.0)]),
    )
    monkeypatch.setattr(hooks, "noesis_reid_meta_ext", _NativeReidExt)
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", None)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", None)
    hooks.reset_core_path_instrumentation()

    processor.handle_frame_ds8(frame)  # type: ignore[attr-defined]

    assert _NativeReidExt.calls == 1
    counters = hooks.get_core_path_instrumentation_snapshot().get("counters", {})
    assert int(counters.get("detection_wake.reid_emb_due", 0)) == 2
    assert int(counters.get("detection_wake.reid_emb_extracted", 0)) == 1
    assert int(counters.get("detection_wake.reid_emb_budget_skipped", 0)) == 1


def test_identity_v2_receives_one_complete_source_frame_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_REID_EMBEDS_PER_FRAME_MAX", "1")
    monkeypatch.setenv("NOESIS_TRACKING_PUBLISH_MAX_HZ", "0")

    class _IdentityV2Service:
        authoritative = False

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def process_source_frame(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    identity_service = _IdentityV2Service()
    pipeline = SimpleNamespace(
        config={"models": {"reid": {"gie_id": 3}}},
        stable_id_mgr=_StableIDMgr(),
        identity_v2_service=identity_service,
    )
    processor = hooks._AnalyticsTelemetryProcessor(  # type: ignore[attr-defined]
        pipeline=pipeline,
        tracking_pub=_TrackingPublisher(),
        camera_labels={0: "camera_0"},
        sensor_id_map={},
        bev_renderer=None,
        bev_calibration=None,
        diagnostics_logger=None,
    )

    class _NativeReidExt:
        calls = 0

        @classmethod
        def extract_reid_embedding(cls, *_args: Any, **_kwargs: Any) -> list[float]:
            cls.calls += 1
            return [1.0] + [0.0] * 511

    class _Rect:
        def __init__(self, left: float) -> None:
            self.left = left
            self.top = 20.0
            self.width = 40.0
            self.height = 80.0

    class _Obj:
        def __init__(self, object_id: int, left: float) -> None:
            self.object_id = object_id
            self.class_id = 0
            self.confidence = 0.91
            self.tracker_confidence = 0.80
            self.rect_params = _Rect(left)
            self.text_params = SimpleNamespace(
                display_text="", font_params=SimpleNamespace(size=0, name=None)
            )

    frame = SimpleNamespace(
        source_id=0,
        frame_number=1,
        frame_width=1280,
        frame_height=720,
        buf_pts=123456789000,
        object_items=iter([_Obj(1, 10.0), _Obj(2, 100.0)]),
    )
    monkeypatch.setattr(hooks, "noesis_reid_meta_ext", _NativeReidExt)
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", None)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", None)

    processor.handle_frame_ds8(frame)  # type: ignore[attr-defined]

    assert _NativeReidExt.calls == 2
    # The shared SDK-neutral processor invokes the identity service
    # synchronously.  DS9's canonical adapter owns a separate bounded worker;
    # this legacy/shared contract has no worker lifecycle to shut down.
    assert len(identity_service.calls) == 1
    call = identity_service.calls[0]
    assert call["camera_id"] == "camera_0"
    assert call["frame_id"] == 1
    assert {row.tracker_id for row in call["primitives"]} == {"1", "2"}
    assert all(row.embedding is not None for row in call["primitives"])


def test_authoritative_identity_skips_legacy_mutation_and_defers_sid_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_REID_EMBEDS_PER_FRAME_MAX", "1")
    monkeypatch.setenv("NOESIS_TRACKING_PUBLISH_MAX_HZ", "0")
    events: list[tuple[Any, ...]] = []

    class _LegacyManager(_StableIDMgr):
        def __init__(self) -> None:
            super().__init__()
            # Deliberately stale state proves authoritative OSD clearing does
            # not fall back to the legacy lookup path.
            self.active_tracks[(0, 7)] = {"stable_id": 999}

        def update(self, **kwargs: Any) -> int:
            events.append(("legacy_update", kwargs["ds_obj_id"]))
            return super().update(**kwargs)

        def remove_missing_tracks(self, sensor_id: int, present_track_ids: list[int], ts: float) -> None:
            events.append(("legacy_remove", sensor_id))

        def prune_ghosts(self, ts: float) -> None:
            events.append(("legacy_prune",))

        def observe_copresence(self, stable_ids: list[int], ts: float) -> None:
            events.append(("legacy_copresence", tuple(stable_ids)))

    class _AuthoritativeIdentity:
        authoritative = True

        def process_source_frame(self, **kwargs: Any) -> None:
            events.append(("identity_v2_resolve",))
            for primitive in kwargs["primitives"]:
                primitive.public_track.update(
                    {
                        "stable_id": 42,
                        "identity_state": "resident",
                        "identity_kind": "resident",
                        "resident_uuid": "resident-42",
                        "display_name": "Alice",
                        "visitor_generation": None,
                    }
                )

    class _CapturingPublisher:
        def __init__(self) -> None:
            self.tracks: list[dict[str, Any]] = []

        def publish(
            self,
            _source_id: int,
            tracks: Sequence[dict[str, Any]],
            **_kwargs: Any,
        ) -> TrackingPublicationReceipt:
            self.tracks = [dict(track) for track in tracks]
            return _tracking_receipt(_source_id, _kwargs)

    class _NativeReidExt:
        @staticmethod
        def extract_reid_embedding(*_args: Any, **_kwargs: Any) -> list[float]:
            return [1.0] + [0.0] * 511

    class _Rect:
        left = 10.0
        top = 20.0
        width = 40.0
        height = 80.0

    obj = SimpleNamespace(
        object_id=7,
        class_id=0,
        confidence=0.91,
        tracker_confidence=0.80,
        rect_params=_Rect(),
        text_params=SimpleNamespace(
            display_text="",
            font_params=SimpleNamespace(size=0, name=None),
        ),
        label="Person",
        obj_label="Person",
    )
    manager = _LegacyManager()
    publisher = _CapturingPublisher()
    pipeline = SimpleNamespace(
        config={"models": {"reid": {"gie_id": 3}}},
        stable_id_mgr=manager,
        identity_v2_service=_AuthoritativeIdentity(),
    )
    processor = hooks._AnalyticsTelemetryProcessor(  # type: ignore[attr-defined]
        pipeline=pipeline,
        tracking_pub=publisher,
        camera_labels={0: "camera_0"},
        sensor_id_map={},
        bev_renderer=None,
        bev_calibration=None,
        diagnostics_logger=None,
        osd_label_processor=hooks._OsdLabelProcessor(  # type: ignore[attr-defined]
            stable_id_mgr=manager
        ),
    )
    monkeypatch.setattr(
        processor,
        "_build_track_dict_ds8",
        lambda _obj, _camera: {
            "track_id": 7,
            "bbox": [10.0, 20.0, 40.0, 80.0],
            "center": [30.0, 60.0],
            "class_id": 0,
            "confidence": 0.91,
            "tracker_confidence": 0.80,
            "analytics": {"lcStatus": {"door": 1}},
            "zone": "Kitchen",
        },
    )
    monkeypatch.setattr(
        processor,
        "_update_dwell_time",
        lambda _sensor, sid, _zone, _now: events.append(("dwell", sid)) or 12.5,
    )
    monkeypatch.setattr(
        processor,
        "_record_transition",
        lambda **kwargs: events.append(("transition", kwargs["stable_id"])),
    )
    monkeypatch.setattr(hooks, "noesis_reid_meta_ext", _NativeReidExt)
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", None)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", None)

    processor.handle_frame_ds8(  # type: ignore[attr-defined]
        SimpleNamespace(
            source_id=0,
            frame_number=1,
            frame_width=1280,
            frame_height=720,
            buf_pts=123456789000,
            object_items=iter((obj,)),
        )
    )

    assert [event[0] for event in events] == [
        "identity_v2_resolve",
        "dwell",
        "transition",
    ]
    assert manager.active_tracks[(0, 7)]["stable_id"] == 999
    assert publisher.tracks[0]["stable_id"] == 42
    assert publisher.tracks[0]["dwell_time"] == 12.5
    assert publisher.tracks[0]["identity_kind"] == "resident"
    # Whole-frame assignment happens after the one-shot metadata walk, so the
    # same-frame OSD is intentionally identity-neutral instead of stale/wrong.
    assert obj.text_params.display_text.startswith("#XX")
    assert "#42" not in obj.text_params.display_text
    assert "#999" not in obj.text_params.display_text


def test_analytics_hook_publishes_tracking_payload(monkeypatch: pytest.MonkeyPatch):
    pipeline = _build_pipeline()
    pipeline.stable_id_mgr = _StableIDMgr()  # type: ignore[attr-defined]
    depth_payload = ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=5,
        class_id=0,
        bbox=[10.0, 20.0, 30.0, 40.0],
        score=0.88,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=128,
        valid_fraction=0.90,
        anchor_source="lower_body_band",
        anchor_depth_m=7.52,
        depth_center=7.40,
        depth_median=7.61,
    )
    monkeypatch.setattr(hooks, "noesis_depth_meta_ext", _DepthExt(depth_payload))

    published: List[tuple[int, List[dict[str, Any]]]] = []

    class StubTrackingPublisher:
        def publish(
            self,
            source_id: int,
            tracks: List[dict[str, Any]],
            **_kwargs: Any,
        ) -> TrackingPublicationReceipt:
            published.append((source_id, list(tracks)))
            return _tracking_receipt(source_id, _kwargs)

    hooks.attach_analytics_telemetry_hook(
        pipeline,
        tracking_pub=StubTrackingPublisher(),
        camera_labels={0: "camera_0"},
    )
    component = pipeline.components["analytics"]
    processor = component.config["_analytics_processor"]
    assert getattr(pipeline, "analytics_telemetry_processor", None) is processor

    class BaseMeta:
        def __init__(self) -> None:
            self.meta_type = "NVIDIA.DSANALYTICSOBJ.USER_META"

    class DummyAnalyticsObjInfo:
        def __init__(self) -> None:
            self.dirStatus = {"north": 1}
            self.lcStatus = {"door": 1}
            self.ocStatus = {"room": 1}
            self.roiStatus = {"zoneA": 1}

    class DummyUserMeta:
        def __init__(self) -> None:
            self.base_meta = BaseMeta()
            self.user_meta_data = DummyAnalyticsObjInfo()

    class DummyRect:
        left = 10.0
        top = 20.0
        width = 30.0
        height = 40.0

    class DummyObjMeta:
        def __init__(self) -> None:
            self.object_id = 5
            self.class_id = 0
            self.confidence = 0.88
            self.rect_params = DummyRect()
            self.tracker_confidence = 0.42
            self.obj_user_meta_list = [DummyUserMeta()]

    class DummyFrameMeta:
        def __init__(self) -> None:
            self.obj_meta_list = [DummyObjMeta()]
            self.source_id = 0
            self.frame_num = 123

    processor.handle_frame(DummyFrameMeta())  # type: ignore[attr-defined]

    assert published, "Tracking telemetry publisher was not invoked"
    source_id, tracks = published[0]
    assert source_id == 0
    assert len(tracks) == 1
    track = tracks[0]
    assert track["stable_id"] == 1
    assert track["camera_id"] == "camera_0"
    assert track["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert track["center"] == [25.0, 40.0]
    assert track["class_id"] == 0
    assert track["frame_id"] == 123
    assert track["zone"] == "room"
    assert track["zone_source"] == "nvdsanalytics_roi"
    assert track["zone_authoritative"] is True
    assert track["dwell_time"] == 0.0
    assert track["depth_status"] == "ok"
    assert track["depth_anchor_source"] == "lower_body_band"
    assert track["depth_anchor_m"] == pytest.approx(7.52)
    assert track["depth_used_m"] == pytest.approx(7.52)
    assert track["depth_center_m"] == pytest.approx(7.40)
    assert track["depth_median_m"] == pytest.approx(7.61)
    assert track["depth_sample_count"] == 128
    assert track["depth_valid_fraction"] == pytest.approx(0.90)
    analytics = track.get("analytics")
    assert analytics is not None
    assert analytics["ocStatus"]["room"] == 1
    assert analytics["roiStatus"]["zoneA"] == 1
    active = processor.get_active_track_map(0)  # type: ignore[attr-defined]
    assert 5 in active
    assert active[5]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_analytics_hook_publishes_advancing_empty_frame_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_TRACKING_EMPTY_HEARTBEAT_HZ", "2")
    pipeline = _build_pipeline()
    pipeline.stable_id_mgr = _StableIDMgr()  # type: ignore[attr-defined]
    published: list[tuple[int, list[dict[str, Any]], dict[str, Any]]] = []

    class StubTrackingPublisher:
        def publish(
            self,
            source_id: int,
            tracks: Sequence[dict[str, Any]],
            *,
            frame_metadata: Mapping[str, Any] | None = None,
        ) -> TrackingPublicationReceipt:
            published.append(
                (
                    int(source_id),
                    [dict(track) for track in tracks],
                    dict(frame_metadata or {}),
                )
            )
            return _tracking_receipt(
                source_id,
                {"frame_metadata": frame_metadata},
            )

    hooks.attach_analytics_telemetry_hook(
        pipeline,
        tracking_pub=StubTrackingPublisher(),
        camera_labels={0: "camera_0"},
    )
    processor = pipeline.components["analytics"].config["_analytics_processor"]
    now = [100.0]
    monkeypatch.setattr(hooks.time, "time", lambda: now[0])

    def empty_frame(frame_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            source_id=0,
            frame_number=frame_id,
            frame_width=1280,
            frame_height=720,
            buf_pts=frame_id * 1_000,
            object_items=[],
        )

    processor.handle_frame_ds8(empty_frame(10))  # type: ignore[attr-defined]
    now[0] = 100.1
    processor.handle_frame_ds8(empty_frame(11))  # type: ignore[attr-defined]
    now[0] = 100.6
    processor.handle_frame_ds8(empty_frame(12))  # type: ignore[attr-defined]

    assert [(source_id, tracks) for source_id, tracks, _meta in published] == [
        (0, []),
        (0, []),
    ]
    assert [meta["frame_id"] for _source, _tracks, meta in published] == [10, 12]
    assert [meta["media_pts_ns"] for _source, _tracks, meta in published] == [
        10_000,
        12_000,
    ]
    assert processor.get_active_track_map(0) == {}  # type: ignore[attr-defined]


def test_ds8_tracking_and_bev_use_one_pair_safe_cadence_above_15fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_REID_TEST_MODE", "1")
    monkeypatch.setenv("NOESIS_TRACKING_PUBLISH_MAX_HZ", "15")
    monkeypatch.setenv("NOESIS_BEV_PUBLISH_MAX_HZ", "12")
    pipeline = _build_pipeline()
    pipeline.stable_id_mgr = _StableIDMgr()  # type: ignore[attr-defined]
    events: list[tuple[str, int]] = []
    tracking_attempts: list[int] = []
    failed_tracking_frames: set[int] = set()

    class PairTrackingPublisher:
        def __init__(self) -> None:
            self.sequence = 0
            self.submission_id = 0

        def publish(
            self,
            _source_id: int,
            _tracks: Sequence[dict[str, Any]],
            *,
            frame_metadata: Mapping[str, Any] | None = None,
        ) -> TrackingPublicationReceipt:
            assert frame_metadata is not None
            frame_id = int(frame_metadata["frame_id"])
            tracking_attempts.append(frame_id)
            if frame_id in failed_tracking_frames:
                raise RuntimeError("tracking_boundary_failed")
            events.append(("tracking", frame_id))
            self.submission_id += 2
            receipt = _tracking_receipt(
                _source_id,
                {"frame_metadata": frame_metadata},
                sequence=self.sequence,
                submission_id=self.submission_id,
            )
            self.sequence += 1
            return receipt

    class PairBevRenderer:
        def render_and_publish(self, **kwargs: Any) -> BevPublicationReceipt:
            events.append(("bev", int(kwargs["frame_id"])))
            return BevPublicationReceipt(
                status="admitted",
                camera_id=str(kwargs["camera_id"]),
                source_id=int(kwargs["source_id"]),
                frame_id=int(kwargs["frame_id"]),
                observed_at_us=int(kwargs["observed_at_us"]),
                tracking_publication_sequence=int(
                    kwargs["tracking_publication_sequence"]
                ),
                tracking_outbound_submission_id=int(
                    kwargs["tracking_outbound_submission_id"]
                ),
                outbound_submission_id=int(
                    kwargs["tracking_outbound_submission_id"]
                )
                + 1,
            )

        def record_input_failure(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("paired BEV unexpectedly failed before rendering")

    hooks.attach_analytics_telemetry_hook(
        pipeline,
        tracking_pub=PairTrackingPublisher(),
        camera_labels={0: "camera_0"},
        bev_renderer=PairBevRenderer(),
        bev_calibration=_CalibrationProvider(_build_anchor_snapshot()),
    )
    processor = pipeline.components["analytics"].config["_analytics_processor"]
    now = [100.0]
    monkeypatch.setattr(hooks.time, "time", lambda: now[0])

    def empty_frame(frame_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            source_id=0,
            frame_number=frame_id,
            frame_width=1280,
            frame_height=720,
            buf_pts=frame_id * 1_000,
            object_items=[],
        )

    frame_times: dict[int, float] = {}
    for frame_id in range(13):
        now[0] = 100.0 + (frame_id / 30.0)
        frame_times[frame_id] = now[0]
        processor.handle_frame_ds8(empty_frame(frame_id))  # type: ignore[attr-defined]

    assert len(events) >= 4
    assert len(events) % 2 == 0
    assert all(
        events[index][0] == "tracking"
        and events[index + 1] == ("bev", events[index][1])
        for index in range(0, len(events), 2)
    )
    paired_frames = [events[index][1] for index in range(0, len(events), 2)]
    assert all(
        frame_times[right] - frame_times[left] >= (1.0 / 12.0) - 1e-9
        for left, right in zip(paired_frames, paired_frames[1:])
    )

    monkeypatch.delenv("NOESIS_REID_TEST_MODE")
    now[0] += 0.01
    processor.handle_frame_ds8(empty_frame(100))  # type: ignore[attr-defined]
    monkeypatch.setenv("NOESIS_REID_TEST_MODE", "1")
    now[0] += 0.01
    processor.handle_frame_ds8(empty_frame(101))  # type: ignore[attr-defined]

    assert events[-4:] == [
        ("tracking", 100),
        ("bev", 100),
        ("tracking", 101),
        ("bev", 101),
    ]

    before_failure = list(events)
    failed_tracking_frames.add(102)
    monkeypatch.delenv("NOESIS_REID_TEST_MODE")
    now[0] += 0.01
    processor.handle_frame_ds8(empty_frame(102))  # type: ignore[attr-defined]

    assert tracking_attempts[-1] == 102
    assert events == before_failure


def test_osd_label_processor_keeps_person_label_compact_when_depth_is_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hooks,
        "noesis_depth_meta_ext",
        _DepthExt(
            ObjectDepthResult(
                source_id=0,
                frame_id=1,
                object_id=7,
                class_id=0,
                bbox=[10.0, 20.0, 30.0, 40.0],
                score=0.92,
                sampling_mode="instance_mask",
                status="ok",
                unit="m",
                is_metric=True,
                sample_count=256,
                valid_fraction=0.95,
                anchor_source="lower_body_band",
                anchor_depth_m=6.25,
                depth_center=6.10,
                depth_median=6.30,
            )
        ),
    )

    text_params = SimpleNamespace(display_text="", font_params=SimpleNamespace(size=0, name=None))
    obj_meta = SimpleNamespace(
        label="Person",
        obj_label="Person",
        class_id=0,
        object_id=7,
        confidence=0.92,
        text_params=text_params,
    )
    processor = hooks._OsdLabelProcessor(decimals=2, font_size=None, font_name=None)  # type: ignore[attr-defined]

    processor._apply_label(obj_meta, sensor_id=0, stable_id_override=11)  # type: ignore[attr-defined]

    assert text_params.display_text == "#11 0.92"
    assert "z=" not in text_params.display_text
    assert obj_meta.obj_label == "#11 0.92"


def test_analytics_hook_falls_back_to_camera_when_zone_missing():
    pipeline = _build_pipeline()
    pipeline.stable_id_mgr = _StableIDMgr()  # type: ignore[attr-defined]

    published: List[tuple[int, List[dict[str, Any]]]] = []

    class StubTrackingPublisher:
        def publish(
            self,
            source_id: int,
            tracks: List[dict[str, Any]],
            **_kwargs: Any,
        ) -> TrackingPublicationReceipt:
            published.append((source_id, list(tracks)))
            return _tracking_receipt(source_id, _kwargs)

    hooks.attach_analytics_telemetry_hook(
        pipeline,
        tracking_pub=StubTrackingPublisher(),
        camera_labels={0: "camera_0"},
    )
    component = pipeline.components["analytics"]
    processor = component.config["_analytics_processor"]

    class DummyRect:
        left = 10.0
        top = 20.0
        width = 30.0
        height = 40.0

    class DummyObjMeta:
        def __init__(self) -> None:
            self.object_id = 5
            self.class_id = 0
            self.confidence = 0.88
            self.rect_params = DummyRect()
            self.tracker_confidence = 0.42
            self.obj_user_meta_list = []

    class DummyFrameMeta:
        def __init__(self) -> None:
            self.obj_meta_list = [DummyObjMeta()]
            self.source_id = 0
            self.frame_num = 123

    processor.handle_frame(DummyFrameMeta())  # type: ignore[attr-defined]

    assert published, "Tracking telemetry publisher was not invoked"
    _source_id, tracks = published[0]
    assert len(tracks) == 1
    track = tracks[0]
    assert track["camera_id"] == "camera_0"
    assert track["zone"] == "Camera 0"
    assert track["zone_source"] == "camera_default"
    assert track["zone_authoritative"] is False


def test_analytics_hook_occupancy_grace_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NOESIS_OCCUPANCY_GRACE_S", "1.0")
    pipeline = _build_pipeline()
    pipeline.stable_id_mgr = _StableIDMgr()  # type: ignore[attr-defined]

    published: List[tuple[int, List[dict[str, Any]]]] = []

    class StubTrackingPublisher:
        def publish(
            self,
            source_id: int,
            tracks: List[dict[str, Any]],
            **_kwargs: Any,
        ) -> TrackingPublicationReceipt:
            published.append((source_id, list(tracks)))
            return _tracking_receipt(source_id, _kwargs)

    hooks.attach_analytics_telemetry_hook(
        pipeline,
        tracking_pub=StubTrackingPublisher(),
        camera_labels={0: "camera_0"},
    )
    component = pipeline.components["analytics"]
    processor = component.config["_analytics_processor"]

    class DummyRect:
        left = 10.0
        top = 20.0
        width = 30.0
        height = 40.0

    class DummyObjMeta:
        def __init__(self) -> None:
            self.object_id = 5
            self.class_id = 0
            self.confidence = 0.88
            self.rect_params = DummyRect()
            self.tracker_confidence = 0.42
            self.obj_user_meta_list = []

    class DummyFrameMeta:
        def __init__(self, objects: list[Any]) -> None:
            self.obj_meta_list = objects
            self.source_id = 0
            self.frame_num = 123

    # Frame with one person.
    monkeypatch.setattr(hooks.time, "time", lambda: 100.0)
    processor.handle_frame(DummyFrameMeta([DummyObjMeta()]))  # type: ignore[attr-defined]
    stats = processor.get_tracking_stats(0)  # type: ignore[attr-defined]
    assert stats["occupancy"] == {"Camera 0": 1}

    # Brief occlusion: no detections, but occupancy should remain during grace.
    monkeypatch.setattr(hooks.time, "time", lambda: 100.4)
    processor.handle_frame(DummyFrameMeta([]))  # type: ignore[attr-defined]
    stats = processor.get_tracking_stats(0)  # type: ignore[attr-defined]
    assert stats["occupancy"] == {"Camera 0": 1}

    # Past grace window: occupancy should drop to empty.
    monkeypatch.setattr(hooks.time, "time", lambda: 101.5)
    processor.handle_frame(DummyFrameMeta([]))  # type: ignore[attr-defined]
    stats = processor.get_tracking_stats(0)  # type: ignore[attr-defined]
    assert stats["occupancy"] == {}


def test_pose_anchor_is_used_as_primary_world_source(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    payload = _build_pose_payload(calib, bbox, foot_z=6.0)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(payload))

    track = {"tracker_id": 7, "bbox": bbox}
    processor._augment_track_with_world(0, "cam0", track, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]

    assert track["world_valid"] is True
    assert track["world_source"] == "pose_floor_only"
    assert track["world_quality"] == "good"
    assert track["world_quality_reason"] == "anchor=pose_ankle_floor,depth=depth_meta_missing"
    assert np.allclose(track["world"], [0.0, 0.0, 6.0], atol=0.15)


def test_pose_depth_fusion_updates_world_from_concurrent_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    payload = _build_pose_payload(calib, bbox, foot_z=6.0)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(payload))

    # DAv2 metric depth is interpreted as camera-to-anchor range, not world-space z.
    anchor_range_m = float(np.linalg.norm(np.array([0.0, 0.0, 6.0]) - np.array([0.0, 2.2, -6.0])))
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=7,
        class_id=0,
        bbox=bbox,
        score=0.91,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=512,
        valid_fraction=0.96,
        anchor_source="lower_body_band",
        anchor_depth_m=anchor_range_m,
        depth_center=anchor_range_m,
        depth_median=anchor_range_m,
    )

    track = {"tracker_id": 7, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=depth_result,
    )

    assert track["world_valid"] is True
    assert track["world_source"] == "pose_depth_fused"
    assert track["world_quality"] == "good"
    assert "depth_anchor=lower_body_band" in track["world_quality_reason"]
    assert np.allclose(track["world"], [0.0, 0.0, 6.0], atol=0.20)


def test_floor_only_policy_ignores_depth_without_losing_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.depth_registration = _DepthRegistrationManager(scale=0.5)
    processor.world_fusion_policy = _fusion_policy(
        floor_weight_scale=1.0,
        depth_weight_scale=0.0,
        floor_only_allowed=True,
    )
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        _PoseExt(_build_pose_payload(calib, bbox, foot_z=6.0)),
    )

    track = {"tracker_id": 801, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=_anchor_depth_result(calib, bbox, object_id=801),
    )

    assert track["world_valid"] is True
    assert track["world_source"] == "pose_floor_only"
    assert track["world"] == pytest.approx([0.0, 0.0, 6.0], abs=0.20)
    assert track["world_depth_candidate"] is not None
    assert track["world_floor_weight_effective"] == pytest.approx(1.0)
    assert track["world_depth_weight_effective"] == pytest.approx(0.0)
    assert track["world_fusion_policy_id"] == "test-policy"


def test_registered_depth_only_policy_excludes_floor_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.depth_registration = _DepthRegistrationManager(scale=0.5)
    processor.world_fusion_policy = _fusion_policy(
        floor_weight_scale=0.0,
        depth_weight_scale=1.0,
        floor_only_allowed=False,
    )
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        _PoseExt(_build_pose_payload(calib, bbox, foot_z=6.0)),
    )

    track = {"tracker_id": 802, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=_anchor_depth_result(calib, bbox, object_id=802),
    )

    assert track["world_valid"] is True
    assert track["world_source"] == "pose_depth_only"
    assert track["world_floor_weight_effective"] == pytest.approx(0.0)
    assert track["world_depth_weight_effective"] > 0.0
    assert track["depth_registration_status"] == "ok"
    assert track["world"][0] == pytest.approx(track["world_depth_candidate"][0])
    assert track["world"][1] == pytest.approx(0.0)
    assert track["world"][2] == pytest.approx(track["world_depth_candidate"][2])


def test_depth_required_policy_does_not_fall_back_to_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.world_fusion_policy = _fusion_policy(
        floor_weight_scale=0.0,
        depth_weight_scale=1.0,
        floor_only_allowed=False,
    )
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        _PoseExt(_build_pose_payload(calib, bbox, foot_z=6.0)),
    )

    track = {"tracker_id": 803, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=None,
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "fusion_policy_requires_registered_depth"


def test_floor_ray_range_rejection_does_not_seed_world_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.world_fusion_policy = _fusion_policy(
        floor_weight_scale=1.0,
        depth_weight_scale=0.0,
        floor_only_allowed=True,
    )
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 30.0])
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        _PoseExt(_build_pose_payload(calib, bbox, foot_z=30.0)),
    )

    track = {"tracker_id": 804, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=None,
    )

    assert track["world_valid"] is False
    assert track["world_quality_reason"] == "floor_ray_range_exceeded"
    assert track["world_floor_admitted"] is False
    assert track["world_floor_range_limit_m"] == pytest.approx(22.0)
    assert track["world_floor_range_m"] > 22.0
    assert track["world_floor_weight_effective"] == pytest.approx(0.0)
    state = processor._world_state_by_track[(0, 804)]  # type: ignore[attr-defined]
    assert state.last_good_world is None
    assert state.height_ref_scene is None


def test_depth_observation_range_rejection_does_not_seed_world_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A depth point cannot bypass the calibrated first-sample range gate."""
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.world_fusion_policy = _fusion_policy(
        floor_weight_scale=0.0,
        depth_weight_scale=1.0,
        floor_only_allowed=False,
    )
    # Keep the image/floor candidate in range, but make the depth projection
    # itself absurd.  This isolates the bypass that used to let a depth-only
    # first sample seed the filter outside the calibrated envelope.
    foot_world = [0.0, 0.0, 6.0]
    bbox = _anchor_bbox(calib, foot_world=foot_world)
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        _PoseExt(_build_pose_payload(calib, bbox, foot_z=6.0)),
    )
    raw_depth_m = 36.0
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=806,
        class_id=0,
        bbox=bbox,
        score=0.91,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=512,
        valid_fraction=0.96,
        anchor_source="lower_body_band",
        anchor_depth_m=raw_depth_m,
        depth_center=raw_depth_m,
        depth_median=raw_depth_m,
    )

    track = {"tracker_id": 806, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=depth_result,
    )

    assert track["world_valid"] is False
    assert track["world_quality_reason"] in {
        "world_observation_range_exceeded",
        "world_measurement_unavailable",
    }
    assert track["world_depth_rejection_reason"] == "world_observation_range_exceeded"
    assert track["world_observation_range_admitted"] is False
    assert track["world_observation_range_limit_m"] == pytest.approx(22.0)
    assert track["world_observation_range_m"] > 22.0
    state = processor._world_state_by_track[(0, 806)]  # type: ignore[attr-defined]
    assert state.last_good_world is None
    assert state.world_x is None


def test_bbox3d_preseed_range_rejection_does_not_seed_world_state() -> None:
    """A pre-seeded tracker world point must use the same metric gate."""
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.world_fusion_policy = _fusion_policy(
        floor_weight_scale=1.0,
        depth_weight_scale=1.0,
        floor_only_allowed=True,
    )
    track = {
        "tracker_id": 807,
        "bbox": _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0]),
        "bbox3d": {"x": 0.0, "y": 0.0, "z": 30.0},
        "world": [0.0, 0.0, 30.0],
        "world_valid": True,
        "world_source": "bbox3d",
        "world_frame": "backend_world_m",
    }

    processor._refine_seeded_world_with_ground_state(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        world_source_label="bbox3d",
    )

    assert track["world_valid"] is False
    assert "world" not in track
    assert track["world_quality_reason"] == "world_observation_range_exceeded"
    assert track["world_observation_range_admitted"] is False
    state = processor._world_state_by_track[(0, 807)]  # type: ignore[attr-defined]
    assert state.last_good_world is None
    assert state.world_x is None


def test_registered_depth_fuses_when_shared_floor_ray_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.world_fusion_policy = _fusion_policy(
        floor_weight_scale=1.0,
        depth_weight_scale=1.0,
        floor_only_allowed=True,
    )
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 30.0])
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        _PoseExt(_build_pose_payload(calib, bbox, foot_z=6.0)),
    )

    track = {"tracker_id": 805, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=_anchor_depth_result(calib, bbox, object_id=805),
    )

    assert track["world_valid"] is True
    # The canonical DS9 adapter has an additional near-horizon bbox-contact
    # gate, covered by DS9/tests/test_family_floor_contact_geometry.py.  The
    # shared SDK-neutral hook does not apply that adapter-only gate here, and
    # this synthetic candidate has healthy ray incidence, so both observations
    # are correctly fused on this path.
    assert track["world_source"] == "pose_depth_fused"
    assert track["world_floor_admitted"] is True
    assert track["world_floor_weight_effective"] == pytest.approx(1.0)
    assert track["world_depth_weight_effective"] > 0.0


def test_pose_depth_fusion_uses_registered_depth_for_world_update() -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.depth_registration = _DepthRegistrationManager(scale=0.5)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    pose_payload = _build_pose_payload(calib, bbox, foot_z=6.0)
    pose_kpts_abs = np.asarray(pose_payload["keypoints_abs"], dtype=np.float32)
    anchor_range_m = float(np.linalg.norm(np.array([0.0, 0.0, 6.0]) - np.array([0.0, 2.2, -6.0])))
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=7,
        class_id=0,
        bbox=bbox,
        score=0.91,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=512,
        valid_fraction=0.96,
        anchor_source="lower_body_band",
        anchor_depth_m=anchor_range_m,
        depth_center=anchor_range_m,
        depth_median=anchor_range_m,
    )
    pose_u, pose_v = _project_anchor_uv(calib, [0.0, 0.0, 6.0])
    pose_anchor = hooks._PoseAnchorCandidate(u=pose_u, v=pose_v, source="pose_ankle_floor")

    depth_observation = processor._depth_observation_from_anchor(  # type: ignore[attr-defined]
        calib=calib,
        anchor=pose_anchor,
        depth_result=depth_result,
        flip_u=False,
        flip_v=False,
    )
    assert depth_observation.reason == "ok"
    assert depth_observation.raw_depth_m == pytest.approx(anchor_range_m)
    assert depth_observation.registered_depth_m == pytest.approx(anchor_range_m * 0.5)
    assert depth_observation.registration_status == "ok"
    assert depth_observation.registration_id == "reg-1"

    track = {"tracker_id": 7, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        pose_kpts_abs=pose_kpts_abs,
        depth_result=depth_result,
    )

    assert track["depth_anchor_m"] == pytest.approx(anchor_range_m)
    assert track["depth_registered_m"] == pytest.approx(anchor_range_m * 0.5)
    assert track["depth_used_m"] == pytest.approx(anchor_range_m * 0.5)
    assert track["depth_registration_status"] == "ok"
    assert track["depth_registration_id"] == "reg-1"


def test_bev_footpoint_ignores_unregistered_depth_for_live_tracking_diagnostics() -> None:
    processor = _build_anchor_processor(_build_anchor_snapshot())
    track = {
        "track_id": 7,
        "stable_id": 11,
        "class_id": 0,
        "bbox": [100.0, 100.0, 80.0, 180.0],
        "image_foot": [140.0, 280.0],
        "depth_used_m": 12.0,
        "depth_anchor_m": 12.0,
        "depth_registration_status": "out_of_domain_or_invalid",
    }

    footpoint = processor._footpoint_from_track(track, (1280, 720))  # type: ignore[attr-defined]

    assert footpoint is not None
    assert footpoint.depth_m is None
    assert footpoint.depth_source is None


def test_bev_footpoint_carries_registered_depth_for_live_tracking_diagnostics() -> None:
    processor = _build_anchor_processor(_build_anchor_snapshot())
    track = {
        "track_id": 7,
        "stable_id": 11,
        "class_id": 0,
        "bbox": [100.0, 100.0, 80.0, 180.0],
        "image_foot": [140.0, 280.0],
        "depth_status": "ok",
        "depth_registered_m": 6.0,
        "depth_used_m": 6.0,
        "depth_anchor_m": 12.0,
        "depth_registration_status": "ok",
    }

    footpoint = processor._footpoint_from_track(track, (1280, 720))  # type: ignore[attr-defined]

    assert footpoint is not None
    assert footpoint.depth_m == pytest.approx(6.0)
    assert footpoint.depth_source == "depth_registered_m"


@pytest.mark.parametrize(
    "processor_type",
    (
        hooks._AnalyticsTelemetryProcessor,
        hooks_v3dt_reimpl._AnalyticsTelemetryProcessor,
    ),
    ids=("ds8", "v3dt"),
)
@pytest.mark.parametrize(
    "depth_fields",
    (
        {
            "depth_status": "no_depth",
            "depth_registration_status": "ok",
            "depth_registered_m": 6.0,
            "depth_used_m": 6.0,
        },
        {
            "depth_status": "ok",
            "depth_registration_status": "ok",
            "depth_registered_m": 6.0,
            "depth_used_m": 12.0,
        },
    ),
    ids=("invalid-depth-status", "registered-used-mismatch"),
)
def test_bev_footpoint_rejects_depth_canonical_observations_reject(
    processor_type: type,
    depth_fields: Mapping[str, object],
) -> None:
    processor = object.__new__(processor_type)
    processor._bev_class_ids_ready = True  # type: ignore[attr-defined]
    processor._bev_class_ids = frozenset({0})  # type: ignore[attr-defined]
    track = {
        "track_id": 7,
        "stable_id": 11,
        "class_id": 0,
        "bbox": [100.0, 100.0, 80.0, 180.0],
        "image_foot": [140.0, 280.0],
        **depth_fields,
    }

    footpoint = processor._footpoint_from_track(track, (1280, 720))  # type: ignore[attr-defined]

    assert footpoint is not None
    assert footpoint.depth_m is None
    assert footpoint.depth_source is None


def test_bev_footpoint_scales_track_pixels_to_calibration_image_space() -> None:
    processor = _build_anchor_processor(_build_anchor_snapshot())
    track = {
        "track_id": 8,
        "stable_id": 15,
        "class_id": 0,
        "bbox": [738.0, 362.1797180175781, 67.5, 193.0078125],
        "image_foot": [769.5635681152344, 537.9977416992188],
        "image_base": [808.2943385017778, 517.858788801387],
        "image_size": [1828, 1122],
        "depth_registration_status": "out_of_domain_or_invalid",
    }

    footpoint = processor._footpoint_from_track(  # type: ignore[attr-defined]
        track,
        (1828, 1122),
        target_image_size=(1920, 1080),
    )

    assert footpoint is not None
    sx = 1920.0 / 1828.0
    sy = 1080.0 / 1122.0
    assert footpoint.image_size == (1920, 1080)
    assert footpoint.u == pytest.approx(769.5635681152344 * sx)
    assert footpoint.v == pytest.approx(537.9977416992188 * sy)
    assert footpoint.bbox == pytest.approx(
        (738.0 * sx, 362.1797180175781 * sy, 67.5 * sx, 193.0078125 * sy)
    )
    assert footpoint.debug["track_image_size"] == [1828, 1122]
    assert footpoint.debug["bev_image_size"] == [1920, 1080]
    assert footpoint.debug["image_scale"] == pytest.approx([sx, sy])

    candidates = {item["name"]: item for item in footpoint.debug["image_candidates"]}
    assert candidates["image_foot"]["u"] == pytest.approx(track["image_foot"][0] * sx)
    assert candidates["image_foot"]["v"] == pytest.approx(track["image_foot"][1] * sy)
    assert candidates["bbox_bottom_center"]["u"] == pytest.approx((738.0 + 67.5 * 0.5) * sx)
    assert candidates["bbox_bottom_center"]["v"] == pytest.approx((362.1797180175781 + 193.0078125) * sy)


def test_v3dt_footpoint_callers_bind_calibration_target_image_size() -> None:
    source = Path("noesis/pipelines/hooks_v3dt_reimpl.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    calls_by_name: dict[str, list[ast.Call]] = {
        "_footpoint_from_track": [],
        "_footpoints_from_tracks": [],
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in calls_by_name:
            calls_by_name[node.func.attr].append(node)

    for name, calls in calls_by_name.items():
        assert calls, name
        assert all(
            any(keyword.arg == "target_image_size" for keyword in call.keywords)
            for call in calls
        ), name


def test_v3dt_footpoint_collection_scales_to_calibration_target() -> None:
    processor = object.__new__(hooks_v3dt_reimpl._AnalyticsTelemetryProcessor)
    processor._bev_class_ids_ready = True  # type: ignore[attr-defined]
    processor._bev_class_ids = frozenset({0})  # type: ignore[attr-defined]
    track = {
        "track_id": 8,
        "stable_id": 15,
        "class_id": 0,
        "bbox": [50.0, 5.0, 20.0, 40.0],
        "image_foot": [60.0, 45.0],
        "image_size": [100, 50],
    }

    footpoints = processor._footpoints_from_tracks(  # type: ignore[attr-defined]
        [track],
        (100, 50),
        target_image_size=(200, 100),
    )

    assert len(footpoints) == 1
    assert footpoints[0].u == pytest.approx(120.0)
    assert footpoints[0].v == pytest.approx(90.0)
    assert footpoints[0].bbox == pytest.approx((100.0, 10.0, 40.0, 80.0))
    assert footpoints[0].image_size == (200, 100)


def test_all_ds8_ds9_v3dt_handlers_emit_bev_only_after_tracking_success() -> None:
    for path in (
        Path("noesis/pipelines/hooks.py"),
        Path("DS9/noesis/pipelines/hooks.py"),
        Path("noesis/pipelines/hooks_v3dt_reimpl.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_publish_bev"
        ]
        expected_calls = 1 if path == Path("DS9/noesis/pipelines/hooks.py") else 2
        assert len(calls) == expected_calls, path
        for call in calls:
            assert any(
                keyword.arg == "paired_with_tracking"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in call.keywords
            ), path
            ancestor = parents.get(call)
            guarded = False
            worker_owned = False
            while ancestor is not None:
                if (
                    path == Path("DS9/noesis/pipelines/hooks.py")
                    and isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and ancestor.name == "_publish_tracking_work"
                ):
                    worker_owned = True
                if (
                    isinstance(ancestor, ast.If)
                    and isinstance(ancestor.test, ast.Name)
                    and ancestor.test.id == "tracking_published"
                ):
                    guarded = True
                    break
                ancestor = parents.get(ancestor)
            assert guarded or worker_owned, path


def test_pose_depth_fusion_prefers_anchor_band_support_over_whole_mask_support(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    payload = _build_pose_payload(calib, bbox, foot_z=6.0)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(payload))

    anchor_range_m = float(np.linalg.norm(np.array([0.0, 0.0, 6.0]) - np.array([0.0, 2.2, -6.0])))
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=7,
        class_id=0,
        bbox=bbox,
        score=0.91,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=20,
        valid_fraction=0.20,
        anchor_source="lower_body_band",
        anchor_depth_m=anchor_range_m,
        anchor_sample_count=48,
        anchor_valid_fraction=0.80,
        depth_center=anchor_range_m,
        depth_median=anchor_range_m,
    )

    track = {"tracker_id": 7, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=depth_result,
    )

    assert track["world_valid"] is True
    assert track["world_source"] == "pose_depth_fused"
    assert "depth_samples=48" in track["world_quality_reason"]
    assert "depth_valid=0.80" in track["world_quality_reason"]
    assert np.allclose(track["world"], [0.0, 0.0, 6.0], atol=0.20)


def test_gravity_drop_reuses_height_lock_when_pose_feet_disappear(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox_full = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    payload = _build_pose_payload(calib, bbox_full, foot_z=6.0)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(payload))

    track_full = {"tracker_id": 7, "bbox": bbox_full}
    processor._augment_track_with_world(0, "cam0", track_full, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]

    state = processor._world_state_by_track[(0, 7)]  # type: ignore[attr-defined]
    assert state.height_ref_scene is not None

    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(None))
    bbox_partial = [float(bbox_full[0]), float(bbox_full[1]), float(bbox_full[2]), float(bbox_full[3] * 0.30)]
    track_partial = {"tracker_id": 7, "bbox": bbox_partial}
    processor._augment_track_with_world(0, "cam0", track_partial, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]

    assert track_partial["world_valid"] is True
    assert track_partial["world_source"] == "gravity_drop"
    assert track_partial["world_quality"] == "estimated"
    assert track_partial["lower_body_occluded"] is True
    assert track_partial["lower_body_occlusion_level"] == "waist_hips"
    assert track_partial["world_quality_reason"].startswith(
        "lower_body_occlusion=waist_hips"
    )
    assert np.allclose(track_partial["world"], [0.0, 0.0, 6.0], atol=0.25)


def test_gravity_drop_range_rejection_cannot_seed_or_replace_world_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.world_fusion_policy = _fusion_policy(
        floor_weight_scale=1.0,
        depth_weight_scale=0.0,
        floor_only_allowed=True,
    )
    bbox_full = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        _PoseExt(_build_pose_payload(calib, bbox_full, foot_z=6.0)),
    )
    track_full = {"tracker_id": 808, "bbox": bbox_full}
    processor._augment_track_with_world(0, "cam0", track_full, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]
    state = processor._world_state_by_track[(0, 808)]  # type: ignore[attr-defined]
    assert state.last_good_world is not None

    def _absurd_gravity(_self: Any, *_args: Any, **_kwargs: Any) -> np.ndarray:
        return np.asarray([0.0, 0.0, 30.0], dtype=np.float64)

    monkeypatch.setattr(hooks._AnalyticsTelemetryProcessor, "_gravity_drop_world", _absurd_gravity)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(None))
    bbox_partial = [
        float(bbox_full[0]),
        float(bbox_full[1]),
        float(bbox_full[2]),
        float(bbox_full[3]) * 0.30,
    ]
    track_partial = {"tracker_id": 808, "bbox": bbox_partial}
    processor._augment_track_with_world(0, "cam0", track_partial, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]

    assert track_partial["world_valid"] is True
    # The rejected candidate must not alter the canonical state.  Depending on
    # the active bounded-fallback policy, the prior state is surfaced as an
    # explicit hold or as a CV prediction; both are valid here.
    assert track_partial["world_source"] in {"anchor_hold", "cv_prediction"}
    assert track_partial["world_observation_range_admitted"] is False
    assert track_partial["world"] == pytest.approx([0.0, 0.0, 6.0], abs=0.25)
    assert state.last_good_world == pytest.approx((0.0, 0.0, 6.0), abs=0.25)


@pytest.mark.parametrize(
    ("hidden_groups", "expected_level", "bbox_ratio"),
    (
        (("ankles",), "feet_ankles", 0.78),
        (("ankles", "knees"), "knees", 0.62),
        (("ankles", "knees", "hips"), "waist_hips", 0.45),
    ),
)
def test_partial_lower_body_pose_and_valid_counter_edge_depth_use_gravity_drop(
    monkeypatch: pytest.MonkeyPatch,
    hidden_groups: tuple[str, ...],
    expected_level: str,
    bbox_ratio: float,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox_full = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    full_payload = _build_pose_payload(calib, bbox_full, foot_z=6.0)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(full_payload))

    track_full = {"tracker_id": 70, "bbox": bbox_full}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track_full,
        obj_meta=SimpleNamespace(),
    )
    state = processor._world_state_by_track[(0, 70)]  # type: ignore[attr-defined]
    assert state.height_ref_scene == pytest.approx(1.8, abs=0.12)

    partial_payload = _build_pose_payload(calib, bbox_full, foot_z=6.0)
    names_by_group = {
        "ankles": ("left_ankle", "right_ankle"),
        "knees": ("left_knee", "right_knee"),
        "hips": ("left_hip", "right_hip"),
    }
    for group in hidden_groups:
        for name in names_by_group[group]:
            partial_payload["keypoints_abs"][hooks._POSE_KPT_INDEX[name]] = [  # type: ignore[attr-defined]
                0.0,
                0.0,
                0.0,
            ]
    bbox_partial = [
        float(bbox_full[0]),
        float(bbox_full[1]),
        float(bbox_full[2]),
        float(bbox_full[3]) * float(bbox_ratio),
    ]
    partial_payload["bbox"] = list(bbox_partial)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(partial_payload))

    counter_edge_uv = (
        float(bbox_partial[0]) + float(bbox_partial[2]) * 0.5,
        float(bbox_partial[1]) + float(bbox_partial[3]),
    )
    wrong_range_m = float(
        np.linalg.norm(
            np.array([0.0, 1.0, 9.0])
            - np.array([0.0, 2.2, -6.0], dtype=np.float64)
        )
    )
    counter_edge_depth = ObjectDepthResult(
        source_id=0,
        frame_id=124,
        object_id=70,
        class_id=0,
        bbox=bbox_partial,
        score=0.96,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=256,
        valid_fraction=0.94,
        anchor_uv=counter_edge_uv,
        anchor_source="lower_body_band",
        anchor_depth_m=wrong_range_m,
        anchor_sample_count=96,
        anchor_valid_fraction=0.90,
        depth_center=wrong_range_m,
        depth_median=wrong_range_m,
    )

    track_partial = {"tracker_id": 70, "bbox": bbox_partial}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track_partial,
        obj_meta=SimpleNamespace(),
        depth_result=counter_edge_depth,
    )

    assert track_partial["world_valid"] is True
    assert track_partial["world_source"] == "gravity_drop"
    assert track_partial["lower_body_occluded"] is True
    assert track_partial["lower_body_occlusion_level"] == expected_level
    assert track_partial["world_quality_reason"].startswith(
        f"lower_body_occlusion={expected_level}"
    )
    assert np.allclose(track_partial["world"], [0.0, 0.0, 6.0], atol=0.30)
    assert track_partial["image_foot"] == pytest.approx(
        track_partial["image_base"],
        abs=1e-6,
    )
    assert float(track_partial["image_foot"][1]) > (
        float(bbox_partial[1]) + float(bbox_partial[3]) + 20.0
    )


def test_world_is_left_invalid_without_pose_or_height_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", None)

    track = {"tracker_id": 11, "bbox": bbox}
    processor._augment_track_with_world(0, "cam0", track, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]

    assert track["world_valid"] is False
    assert "world_source" not in track
    assert "world" not in track
    assert track["world_quality"] == "invalid"
    assert track["world_quality_reason"] == "pose_meta_missing,depth_meta_missing,height_lock_missing"


def test_person_anchor_without_pose_can_publish_world_from_anchor_uv_and_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", None)

    anchor_uv = _project_anchor_uv(calib, [0.0, 0.0, 6.0])
    anchor_range_m = float(np.linalg.norm(np.array([0.0, 0.0, 6.0]) - np.array([0.0, 2.2, -6.0])))
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=11,
        class_id=0,
        bbox=bbox,
        score=0.97,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=192,
        valid_fraction=0.95,
        anchor_uv=anchor_uv,
        anchor_source="lower_body_band",
        anchor_depth_m=anchor_range_m,
        anchor_sample_count=96,
        anchor_valid_fraction=0.92,
        depth_center=anchor_range_m,
        depth_median=anchor_range_m,
    )

    track = {"tracker_id": 11, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=depth_result,
    )

    assert track["world_valid"] is True
    assert track["world_source"] == "person_anchor_depth_fused"
    assert track["world_quality"] == "good"
    assert "anchor=person_mask_floor" in track["world_quality_reason"]
    assert "depth_anchor=lower_body_band" in track["world_quality_reason"]
    assert np.allclose(track["world"], [0.0, 0.0, 6.0], atol=0.20)


def test_rejected_registered_depth_falls_back_when_floor_policy_allows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.depth_registration = _RejectingDepthRegistrationManager()
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", None)

    anchor_range_m = float(
        np.linalg.norm(np.array([0.0, 0.0, 6.0]) - np.array([0.0, 2.2, -6.0]))
    )
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=11,
        class_id=0,
        bbox=bbox,
        score=0.97,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=192,
        valid_fraction=0.95,
        anchor_uv=_project_anchor_uv(calib, [0.0, 0.0, 6.0]),
        anchor_source="lower_body_band",
        anchor_depth_m=anchor_range_m,
        anchor_sample_count=96,
        anchor_valid_fraction=0.92,
        depth_center=anchor_range_m,
        depth_median=anchor_range_m,
    )

    track = {"tracker_id": 11, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=depth_result,
    )

    assert track["world_valid"] is True
    assert track["world_quality"] == "good"
    assert track["world_source"] == "person_anchor_floor_only"
    assert track["world"] == pytest.approx([0.0, 0.0, 6.0], abs=0.20)
    assert track["depth_anchor_m"] == pytest.approx(anchor_range_m)
    assert track["depth_registered_m"] is None
    assert track["depth_used_m"] is None
    assert track["depth_registration_status"] == "out_of_domain_or_invalid"
    state = processor._world_state_by_track[(0, 11)]  # type: ignore[attr-defined]
    assert state.height_ref_scene is None
    assert state.last_good_world == pytest.approx([0.0, 0.0, 6.0], abs=0.20)


def test_rejected_registered_depth_fails_closed_when_policy_requires_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    processor.depth_registration = _RejectingDepthRegistrationManager()
    processor.world_fusion_policy = _fusion_policy(
        floor_weight_scale=0.0,
        depth_weight_scale=1.0,
        floor_only_allowed=False,
    )
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", None)

    anchor_range_m = float(
        np.linalg.norm(np.array([0.0, 0.0, 6.0]) - np.array([0.0, 2.2, -6.0]))
    )
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=12,
        class_id=0,
        bbox=bbox,
        score=0.97,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=192,
        valid_fraction=0.95,
        anchor_uv=_project_anchor_uv(calib, [0.0, 0.0, 6.0]),
        anchor_source="lower_body_band",
        anchor_depth_m=anchor_range_m,
        anchor_sample_count=96,
        anchor_valid_fraction=0.92,
        depth_center=anchor_range_m,
        depth_median=anchor_range_m,
    )

    track = {"tracker_id": 12, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=depth_result,
    )

    assert track["world_valid"] is False
    assert track["world_quality"] == "invalid"
    assert "world" not in track
    assert "world_source" not in track
    assert track["depth_registration_status"] == "out_of_domain_or_invalid"
    state = processor._world_state_by_track[(0, 12)]  # type: ignore[attr-defined]
    assert state.height_ref_scene is None
    assert state.last_good_world is None


def test_person_anchor_without_pose_can_publish_floor_only_from_anchor_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", None)

    track = {"tracker_id": 12, "bbox": bbox}
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=123,
        object_id=12,
        class_id=0,
        bbox=bbox,
        score=0.95,
        sampling_mode="instance_mask",
        status="no_valid_depth",
        unit="m",
        is_metric=True,
        sample_count=0,
        valid_fraction=0.0,
        anchor_uv=_project_anchor_uv(calib, [0.0, 0.0, 6.0]),
        anchor_source=None,
        anchor_depth_m=None,
        depth_center=None,
        depth_median=None,
    )
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        track,
        obj_meta=SimpleNamespace(),
        depth_result=depth_result,
    )

    assert track["world_valid"] is True
    assert track["world_source"] == "person_anchor_floor_only"
    assert track["world_quality"] == "estimated"
    assert track["world_quality_reason"] == "anchor=person_mask_floor,depth=depth_status_no_valid_depth"
    assert np.allclose(track["world"], [0.0, 0.0, 6.0], atol=0.20)


def test_anchor_hold_reuses_last_good_world_when_leg_anchor_disappears(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    payload = _build_pose_payload(calib, bbox, foot_z=6.0)
    for ankle_name in ("left_ankle", "right_ankle"):
        idx = hooks._POSE_KPT_INDEX[ankle_name]  # type: ignore[attr-defined]
        payload["keypoints_abs"][idx] = [0.0, 0.0, 0.0]
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(payload))

    track = {"tracker_id": 17, "bbox": bbox}
    processor._augment_track_with_world(0, "cam0", track, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]

    assert track["world_valid"] is True
    assert track["world_source"] == "pose_floor_only"
    assert track["world_quality_reason"] == "anchor=pose_leg_floor,depth=depth_meta_missing"
    state = processor._world_state_by_track[(0, 17)]  # type: ignore[attr-defined]
    assert state.height_ref_scene is None
    assert state.last_good_world is not None

    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(None))
    bbox_partial = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3] * 0.45)]
    held_track = {"tracker_id": 17, "bbox": bbox_partial}
    processor._augment_track_with_world(0, "cam0", held_track, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]

    assert held_track["world_valid"] is True
    assert held_track["world_source"] == "anchor_hold"
    assert held_track["world_quality"] == "estimated"
    assert held_track["world_quality_reason"] == "pose_keypoints_unusable,depth_meta_missing,height_lock_missing"
    assert np.allclose(held_track["world"], [0.0, 0.0, 6.0], atol=0.25)


def test_world_state_update_rejects_physically_impossible_measurement() -> None:
    processor = _build_anchor_processor(_build_anchor_snapshot())
    state = hooks._WorldAnchorState(  # type: ignore[attr-defined]
        world_x=10.0,
        world_z=20.0,
        vel_world_x=0.0,
        vel_world_z=0.0,
        filtered_ts=10.0,  # valid prior timestamp (not the unset sentinel -1)
        motion_mode="walk",
    )

    updated = processor._update_world_state(  # type: ignore[attr-defined]
        state,
        measurement=np.array([12.0, 0.0, 22.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.5,
        alpha=0.5,
        beta=0.25,
        quality="good",
    )

    # The 2.83m innovation exceeds 0.75m + 4m/s * 0.5s.  It must be
    # quarantined, not clipped into a plausible-looking intermediate point.
    assert np.allclose(updated, [10.0, 0.0, 20.0])
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "physical_innovation_exceeded"
    # Reacquisition is deliberately evidence-gated.  A direct filter caller
    # did not provide a stable contact basis plus exact-frame image motion, so
    # this impossible sample is quarantined without starting a reacquire run.
    assert state.reacquire_count == 0


def test_world_augmentation_quarantines_impossible_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox_first = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    payload_first = _build_pose_payload(calib, bbox_first, foot_z=6.0)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(payload_first))
    monkeypatch.setattr(hooks.time, "time", lambda: 10.0)

    first = {"tracker_id": 71, "bbox": bbox_first}
    processor._augment_track_with_world(0, "cam0", first, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]
    assert first["world_valid"] is True
    accepted_world = list(first["world"])

    bbox_far = list(bbox_first)
    monkeypatch.setattr(
        processor,
        "_project_pixel_to_floor_world",
        lambda *_args, **_kwargs: np.array([8.0, 0.0, 6.0], dtype=np.float64),
    )
    monkeypatch.setattr(hooks.time, "time", lambda: 10.1)
    rejected = {"tracker_id": 71, "bbox": bbox_far}
    processor._augment_track_with_world(0, "cam0", rejected, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]

    assert rejected["world_valid"] is True
    assert rejected["world"] == pytest.approx(accepted_world)
    assert rejected["world_source"] == "anchor_hold"
    assert rejected["world_quality"] == "estimated"
    assert rejected["world_measurement_accepted"] is False
    assert rejected["world_rejection_reason"] == "physical_innovation_exceeded"
    assert rejected["world_quality_reason"] == "physical_innovation_exceeded"
    state = processor._world_state_by_track[(0, 71)]  # type: ignore[attr-defined]
    assert list(state.last_good_world or ()) == pytest.approx(accepted_world)
    assert state.last_good_ts == pytest.approx(10.0)


def test_source_hysteresis_rejects_before_filter_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib = _build_anchor_snapshot()
    processor = _build_anchor_processor(calib)
    bbox = _anchor_bbox(calib, foot_world=[0.0, 0.0, 6.0])
    payload = _build_pose_payload(calib, bbox, foot_z=6.0)
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", _PoseExt(payload))
    monkeypatch.setattr(hooks.time, "time", lambda: 20.0)
    floor_track = {"tracker_id": 72, "bbox": bbox}
    processor._augment_track_with_world(0, "cam0", floor_track, obj_meta=SimpleNamespace())  # type: ignore[attr-defined]
    accepted_world = list(floor_track["world"])

    anchor_range_m = float(
        np.linalg.norm(np.array([0.0, 0.0, 6.0]) - np.array([0.0, 2.2, -6.0]))
    )
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=124,
        object_id=72,
        class_id=0,
        bbox=bbox,
        score=0.91,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=512,
        valid_fraction=0.96,
        anchor_source="lower_body_band",
        anchor_depth_m=anchor_range_m,
        depth_center=anchor_range_m,
        depth_median=anchor_range_m,
    )
    monkeypatch.setattr(hooks.time, "time", lambda: 20.1)
    switched = {"tracker_id": 72, "bbox": bbox}
    processor._augment_track_with_world(  # type: ignore[attr-defined]
        0,
        "cam0",
        switched,
        obj_meta=SimpleNamespace(),
        depth_result=depth_result,
    )

    assert switched["world_source"] == "anchor_hold"
    assert switched["world_quality_reason"] == "source_hysteresis_rejected_current_measurement"
    assert switched["world"] == pytest.approx(accepted_world)
    state = processor._world_state_by_track[(0, 72)]  # type: ignore[attr-defined]
    assert list(state.last_good_world or ()) == pytest.approx(accepted_world)
    assert state.last_good_ts == pytest.approx(20.0)
