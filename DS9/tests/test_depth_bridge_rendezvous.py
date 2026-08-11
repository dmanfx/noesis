from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


def test_ds9_depth_bridge_rendezvous_and_counters_match_canonical_contract() -> None:
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path
        from types import SimpleNamespace
        import threading
        import time

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
        started = threading.Event()
        resolved = []

        def resolve():
            started.set()
            resolved.append(
                store.resolve(
                    source_id=0,
                    frame_id=41,
                    pts_us=1_033_333,
                    max_age_frames=1,
                    wait_ms=150,
                )
            )

        thread = threading.Thread(target=resolve, daemon=True)
        thread.start()
        assert started.wait(1.0)
        time.sleep(0.02)
        store.put(exact)
        thread.join(1.0)
        assert not thread.is_alive()
        assert resolved == [(exact, 0, 0.0)]

        processor = hooks._ObjectDepthFusionProcessor(
            depth_store=store,
            fallback_frame_size=(1920, 1080),
            depth_model_name="depth-anything-v2-metric-hypersim-vits",
            depth_unit="m",
            depth_is_metric=True,
            depth_every_n_frames=2,
        )
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
            def sample_roi_stats(self, _left, _top, width, height, _max_samples):
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

        store.put(frame(0, 50, 2_000_000, StatsDevice()))
        hooks.noesis_depth_meta_ext = SimpleNamespace(
            extract_object_mask=lambda _obj: None,
            attach_object_depth=lambda *_args, **_kwargs: True,
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
        assert counters["depth_bridge_wait_total"] == 1
        assert counters["depth_bridge_exact_resolve_total"] == 2
        assert counters.get("depth_bridge_wait_timeout_total", 0) == 0
        assert counters["object_depth_attach_failure_total"] == 1
        assert counters["object_depth_attach_failure_total.native_rejected"] == 1
        assert counters["object_depth_gpu_roi_copies_total"] == 1
        assert counters["object_depth_attach_total"] == 1
        assert counters["object_depth_status_total.ok"] == 1
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
