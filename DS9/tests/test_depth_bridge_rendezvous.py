from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


def test_ds9_depth_bridge_rendezvous_and_counters_match_canonical_contract() -> None:
    script = textwrap.dedent(
        """
        import os
        import json
        from pathlib import Path
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        assert Path(hooks.__file__).resolve().is_relative_to(
            (Path.cwd() / "DS9").resolve()
        )

        def frame(source_id, frame_id, pts_us, device=None):
            return hooks._AlignedDepthFrame(
                key=(source_id, frame_id, pts_us),
                source_id=source_id,
                frame_id=frame_id,
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

        hooks.reset_core_path_instrumentation()
        store = hooks._AlignedDepthFrameStore()
        lagged = frame(0, 40, 1_000_000)
        exact = frame(0, 41, 1_033_333)
        store.put(lagged)
        resolved = [store.resolve(
            source_id=0,
            frame_id=41,
            pts_us=1_033_333,
            max_age_frames=1,
            wait_ms=150,
        )]
        # The optional exact-frame wait is compatibility-only.  Resolution is
        # query-only and must return the ready prior frame immediately.
        assert resolved == [(lagged, 1, 33.333)]
        store.put(exact)
        assert store.resolve(
            source_id=0,
            frame_id=41,
            pts_us=1_033_333,
            max_age_frames=1,
            wait_ms=150,
        ) == (exact, 0, 0.0)

        class PendingDevice:
            def __init__(self, ready):
                self.ready = bool(ready)
            def is_ready(self):
                return self.ready

        pending = frame(0, 42, 1_066_666, PendingDevice(False))
        store.put(pending)
        assert store.resolve(
            source_id=0,
            frame_id=42,
            pts_us=1_066_666,
            max_age_frames=2,
        ) == (exact, 1, 33.333)
        assert store.resolve(
            source_id=0,
            frame_id=43,
            pts_us=1_099_999,
            max_age_frames=2,
        ) == (exact, 2, 66.666)

        processor = hooks._ObjectDepthFusionProcessor(
            depth_store=store,
            fallback_frame_size=(1920, 1080),
            depth_model_name="depth-anything-v2-metric-hypersim-vits",
            depth_unit="m",
            depth_is_metric=True,
            depth_every_n_frames=2,
        )
        os.environ.pop("NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS", None)
        assert processor._exact_frame_wait_ms() == 0.0
        os.environ["NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS"] = "999999"
        assert processor._exact_frame_wait_ms() == 250.0
        batch_meta = SimpleNamespace(frame_items=[])

        hooks.noesis_depth_meta_ext = SimpleNamespace(
            attach_object_depth=lambda *_args, **_kwargs: False
        )
        assert processor._attach_object_depth_payload(
            batch_meta, SimpleNamespace(), {"status": "ok"}
        ) is False

        class StatsDevice:
            def __init__(self):
                self.calls = []

            def sample_roi_stats(self, _left, _top, width, height, _max_samples):
                self.calls.append((_left, _top, width, height))
                area = int(width) * int(height)
                return {
                    "roi_area_px": area,
                    "sample_count": area,
                    "valid_fraction": 1.0,
                    "depth_center": 3.5,
                    "depth_median": 3.5,
                    "depth_mean": 3.5,
                    "depth_p10": 3.5,
                    "depth_p90": 3.5,
                    "depth_min": 3.5,
                    "depth_max": 3.5,
                }

        stats_device = StatsDevice()
        store.put(frame(0, 50, 2_000_000, stats_device))
        attached_payloads = []
        hooks.noesis_depth_meta_ext = SimpleNamespace(
            extract_object_mask=lambda _obj: None,
            attach_object_depth=lambda _batch, _obj, payload, _replace: (
                attached_payloads.append(json.loads(payload)) or True
            ),
        )
        processor.handle_frame_ds8(
            batch_meta,
            SimpleNamespace(
                source_id=0,
                frame_number=50,
                frame_width=1920,
                frame_height=1080,
                buf_pts=2_000_000_000,
                object_items=[
                    SimpleNamespace(
                        object_id=7,
                        class_id=0,
                        confidence=0.9,
                        rect_params=SimpleNamespace(
                            left=10.0,
                            top=20.0,
                            width=30.0,
                            height=40.0,
                        ),
                    )
                ],
            )
        )
        assert stats_device.calls == [(20, 50, 10, 10)]
        assert len(attached_payloads) == 1
        bbox_payload = attached_payloads[0]
        assert bbox_payload["sampling_mode"] == "bbox_core_native"
        assert bbox_payload["status"] == "no_ground_contact"
        assert bbox_payload["evidence_quality"] == "rejected"
        assert bbox_payload["evidence_reason"] == "bbox_only_without_person_contact_support"
        assert "anchor_uv" not in bbox_payload
        assert "anchor_depth_m" not in bbox_payload
        assert bbox_payload["depth_tensor_frame_id"] == 50
        assert bbox_payload["depth_tensor_ts_us"] == 2_000_000
        assert bbox_payload["depth_tensor_age_frames"] == 0
        assert bbox_payload["depth_tensor_age_us"] == 0

        processor.handle_frame_ds8(
            batch_meta,
            SimpleNamespace(
                source_id=0,
                frame_number=51,
                frame_width=1920,
                frame_height=1080,
                buf_pts=2_033_333_000,
                object_items=[
                    SimpleNamespace(
                        object_id=8,
                        class_id=0,
                        confidence=0.9,
                        rect_params=SimpleNamespace(
                            left=10.0,
                            top=20.0,
                            width=30.0,
                            height=40.0,
                        ),
                    )
                ],
            )
        )
        lagged_payload = attached_payloads[1]
        assert lagged_payload["frame_id"] == 51
        assert lagged_payload["measurement_frame_id"] == 51
        assert lagged_payload["depth_tensor_frame_id"] == 50
        assert lagged_payload["depth_tensor_ts_us"] == 2_000_000
        assert lagged_payload["depth_tensor_age_frames"] == 1
        assert lagged_payload["depth_tensor_age_us"] == 33_333
        assert hooks._AnalyticsTelemetryProcessor._depth_measurement_is_current(
            hooks.ObjectDepthResult.from_dict(lagged_payload),
            track={
                "source_id": 0,
                "tracker_id": 8,
                "frame_id": 51,
                "media_pts_ns": 2_033_333_000,
            },
        ) is False

        device = SimpleNamespace(depth_width=518, depth_height=294)
        hooks.noesis_depth_tracking_tensor_ext = SimpleNamespace(
            capture_aligned_depth_frame=lambda *_args: device
        )
        capture = hooks._DepthTrackingFrameProcessor(
            depth_store=store,
            depth_gie_id=5,
            fallback_frame_size=(1920, 1080),
            depth_model_name="depth-anything-v2-metric-hypersim-vits",
            depth_unit="m",
            depth_is_metric=True,
        )
        capture.handle_frame_ds8(
            SimpleNamespace(
                source_id=0,
                frame_number=42,
                frame_width=1920,
                frame_height=1080,
                buf_pts=1_066_666_000,
            )
        )

        counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
        assert counters["depth_bridge_wait_bypassed_total"] == 2
        assert counters["depth_bridge_pending_exact_total"] == 1
        assert counters["depth_bridge_pending_lagged_skipped_total"] == 1
        assert counters["depth_bridge_exact_resolve_total"] == 2
        assert counters["depth_bridge_lagged_resolve_total"] == 4
        assert counters["object_depth_attach_failure_total"] == 1
        assert counters["object_depth_attach_failure_total.native_rejected"] == 1
        assert counters["object_depth_gpu_roi_copies_total"] == 2
        assert counters["object_depth_attach_total"] == 2
        assert counters["object_depth_status_total.no_ground_contact"] == 2
        assert counters["depth_tracking_device_frames_total"] == 1
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(DS9_ROOT), str(REPO_ROOT)))
    result = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_pending_depth_miss_retries_next_frame_without_negative_cache(monkeypatch) -> None:
    from noesis.pipelines import hooks

    class StatsDevice:
        def sample_roi_stats(self, _left, _top, width, height, _max_samples):
            value = 3.5
            area = int(width) * int(height)
            return {
                "roi_area_px": area,
                "sample_count": area,
                "valid_fraction": 1.0,
                "depth_center": value,
                "depth_median": value,
                "depth_mean": value,
                "depth_p10": value,
                "depth_p90": value,
                "depth_min": value,
                "depth_max": value,
            }

    store = hooks._AlignedDepthFrameStore()
    processor = hooks._ObjectDepthFusionProcessor(
        depth_store=store,
        fallback_frame_size=(1920, 1080),
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        depth_every_n_frames=2,
    )
    attached = []
    monkeypatch.setattr(
        hooks,
        "noesis_depth_meta_ext",
        SimpleNamespace(
            extract_object_mask=lambda _obj: None,
            attach_object_depth=lambda _batch, _obj, payload, _replace: (
                attached.append(payload) or True
            ),
        ),
    )

    def metadata(frame_number: int) -> SimpleNamespace:
        return SimpleNamespace(
            source_id=0,
            frame_number=frame_number,
            frame_width=1920,
            frame_height=1080,
            buf_pts=frame_number * 1_000_000_000,
            object_items=[
                SimpleNamespace(
                    object_id=7,
                    class_id=0,
                    confidence=0.9,
                    rect_params=SimpleNamespace(
                        left=10.0,
                        top=20.0,
                        width=30.0,
                        height=40.0,
                    ),
                )
            ],
        )

    hooks.reset_core_path_instrumentation()
    batch = SimpleNamespace(frame_items=[])
    processor.handle_frame_ds8(batch, metadata(1))
    assert attached == []
    assert processor._result_cache == {}
    assert (0, 7) in processor._depth_retry_tracks

    ready = hooks._AlignedDepthFrame(
        key=(0, 2, 2_000_000),
        source_id=0,
        frame_id=2,
        pts_us=2_000_000,
        depth_map=None,
        valid_mask=None,
        frame_w=1920,
        frame_h=1080,
        depth_w=518,
        depth_h=294,
        unit="m",
        is_metric=True,
        model_name="depth-anything-v2-metric-hypersim-vits",
        depth_device_frame=StatsDevice(),
    )
    store.put(ready)
    processor.handle_frame_ds8(batch, metadata(2))
    assert len(attached) == 1
    assert (0, 7) not in processor._depth_retry_tracks
    assert (0, 7) in processor._result_cache
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters["detection_wake.object_depth_pending_skip_total"] == 1


def test_pending_depth_retry_preserves_a_valid_cached_measurement(monkeypatch) -> None:
    from noesis.pipelines import hooks

    class PendingDevice:
        def is_ready(self):
            return False

    store = hooks._AlignedDepthFrameStore()
    store.put(
        hooks._AlignedDepthFrame(
            key=(0, 2, 2_000_000),
            source_id=0,
            frame_id=2,
            pts_us=2_000_000,
            depth_map=None,
            valid_mask=None,
            frame_w=1920,
            frame_h=1080,
            depth_w=518,
            depth_h=294,
            unit="m",
            is_metric=True,
            model_name="depth-anything-v2-metric-hypersim-vits",
            depth_device_frame=PendingDevice(),
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
    processor._result_cache[(0, 7)] = {
        "source_id": 0,
        "frame_id": 1,
        "object_id": 7,
        "class_id": 0,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "score": 0.9,
        "sampling_mode": "pose_capsule_native",
        "status": "ok",
        "unit": "m",
        "is_metric": True,
        "sample_count": 32,
        "valid_fraction": 1.0,
        "depth_median": 3.0,
        "mask_area_px": 32,
        "model": "depth-anything-v2-metric-hypersim-vits",
        "ts_us": 1_700_000,
        "measurement_frame_id": 1,
        "measurement_ts_us": 1_700_000,
        "measurement_age_us": 0,
        "measurement_cached": False,
        "_sample_ts_us": 1_700_000,
    }
    processor._depth_retry_tracks.add((0, 7))
    attached = []
    monkeypatch.setattr(
        hooks,
        "noesis_depth_meta_ext",
        SimpleNamespace(
            attach_object_depth=lambda _batch, _obj, payload, _replace: (
                attached.append(json.loads(payload)) or True
            )
        ),
    )
    frame = SimpleNamespace(
        source_id=0,
        frame_number=2,
        frame_width=1920,
        frame_height=1080,
        buf_pts=2_000_000_000,
        object_items=[
            SimpleNamespace(
                object_id=7,
                class_id=0,
                confidence=0.9,
                rect_params=SimpleNamespace(
                    left=10.0,
                    top=20.0,
                    width=30.0,
                    height=40.0,
                ),
            )
        ],
    )

    hooks.reset_core_path_instrumentation()
    processor.handle_frame_ds8(SimpleNamespace(frame_items=[]), frame)

    assert len(attached) == 1
    assert attached[0]["measurement_cached"] is True
    assert attached[0]["measurement_frame_id"] == 1
    assert attached[0]["measurement_age_us"] == 300_000
    assert (0, 7) in processor._depth_retry_tracks
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters["detection_wake.object_depth_pending_skip_total"] == 1
    assert counters["detection_wake.object_depth_pending_cache_hit"] == 1
