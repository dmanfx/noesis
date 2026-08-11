from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


def _run_isolated(script: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(DS9_ROOT), str(REPO_ROOT)))
    result = subprocess.run(
        [sys.executable, "-P", "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"isolated DS9 assertion failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_ds9_mapanything_wait_idle_is_bounded_and_non_closing() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace
        import numpy as np
        import pytest

        from noesis.pipelines import hooks

        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(4, 3)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=1,
        )
        processed = []
        processor.handle_numpy_arrays = (
            lambda **kwargs: processed.append(int(kwargs["frame_id"])) or object()
        )
        job = hooks._MapAnythingNativeJob(
            source_id=0,
            frame_id=1,
            pts_ns=1,
            captured_at_us=1,
            tensors={"depth": np.ones((3, 4), dtype=np.float32)},
        )
        processor._enqueue_async_job(job)
        receipt = processor.wait_idle(timeout_s=2.0)
        assert processed == [1]
        assert receipt.accepting is True
        assert processor._async_accepting is True

        with processor._async_idle:
            processor._async_active_captures = 1
        with pytest.raises(TimeoutError, match="did not become idle"):
            processor.wait_idle(timeout_s=0.01)
        assert processor._async_accepting is True
        with processor._async_idle:
            processor._async_active_captures = 0
            processor._async_idle.notify_all()
        processor.shutdown(wait=True, timeout_s=2.0)
        """
    )


def test_ds9_operator_uses_only_the_exact_native_capture_path() -> None:
    _run_isolated(
        """
        from pathlib import Path
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        assert Path(hooks.__file__).resolve().is_relative_to(
            (Path.cwd() / "DS9").resolve()
        )

        class Processor:
            gie_id = 2
            pipeline = SimpleNamespace(depth_enabled=True)

            def __init__(self):
                self.native_calls = []
                self.wrapper_calls = []

            def handle_native_frame_ds9(self, frame):
                self.native_calls.append(frame)
                return object()

            def handle_nvds_tensor_ds8(self, *_args):
                self.wrapper_calls.append(_args)
                raise AssertionError("Service Maker tensor wrapper must not be consumed")

        processor = Processor()
        operator = hooks._MapAnythingOperator.__new__(hooks._MapAnythingOperator)
        operator._processor = processor
        operator._frames_seen = 0
        operator._matched_frames = 0
        frame = SimpleNamespace(
            frame_number=7,
            tensor_items=[SimpleNamespace(unique_id=2)],
        )

        hooks.reset_core_path_instrumentation()
        operator.handle_metadata(SimpleNamespace(frame_items=[frame]))

        assert processor.native_calls == [frame]
        assert processor.wrapper_calls == []
        assert operator._matched_frames == 1
        counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
        assert counters["mapanything_exact_native_capture_frames_total"] == 1
        assert counters.get("tensor_gie_mismatch_drops_total.mapanything", 0) == 0
        """
    )


def test_ds9_operator_fails_closed_without_exact_uid_capture() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        class Processor:
            gie_id = 2
            pipeline = SimpleNamespace(depth_enabled=True)

            def handle_native_frame_ds9(self, _frame):
                raise RuntimeError("missing exact native tensor UID 2")

        operator = hooks._MapAnythingOperator.__new__(hooks._MapAnythingOperator)
        operator._processor = Processor()
        operator._frames_seen = 0
        operator._matched_frames = 0

        hooks.reset_core_path_instrumentation()
        try:
            operator.handle_metadata(
                SimpleNamespace(frame_items=[SimpleNamespace(frame_number=8)])
            )
        except RuntimeError as exc:
            assert "missing exact native tensor UID 2" in str(exc)
        else:
            raise AssertionError("missing MapAnything UID did not fail closed")

        counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
        assert counters["mapanything_exact_native_capture_failures_total"] == 1
        assert operator._matched_frames == 0
        """
    )


def test_ds9_processor_binds_uid_batch_and_exact_layers() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        import numpy as np

        from noesis.pipelines import hooks

        calls = []

        def capture(frame, gie_id, expected_height, expected_width):
            calls.append((frame, gie_id, expected_height, expected_width))
            return {
                "depth": np.full((1, 294, 518), 2.0, dtype=np.float32),
                "conf": np.ones((1, 294, 518), dtype=np.float32),
                "mask": np.ones((1, 294, 518), dtype=np.float32),
            }

        pipeline = SimpleNamespace(depth_enabled=True, frame_size=(3, 2))
        processor = hooks.MapAnythingProcessor(
            pipeline=pipeline,
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            native_capture=capture,
        )
        processor._start_async_worker = lambda: None
        frame = SimpleNamespace(
            batch_id=1,
            pad_index=2,
            frame_num=22,
            buf_pts=33,
        )
        hooks.reset_core_path_instrumentation()
        result = processor.handle_native_frame_ds9(frame)

        assert result is True
        assert calls == [(frame, 2, 294, 518)]
        job = processor._async_queue.get_nowait()
        assert job.source_id == 2
        assert job.frame_id == 22
        assert job.pts_ns == 33
        assert set(job.tensors) == {"depth", "conf", "mask"}
        assert all(value.shape == (294, 518) for value in job.tensors.values())
        processor._async_queue.task_done()
        snapshot = hooks.get_core_path_instrumentation_snapshot()
        counters = snapshot["counters"]
        assert counters["tensor_host_copies_total.mapanything"] == 3
        assert counters["tensor_boundary_copy_bytes_total.mapanything"] == 3 * 294 * 518 * 4
        assert counters["mapanything_probe_local_d2h_frames_total"] == 1
        assert counters["mapanything_probe_local_d2h_bytes_total"] == 3 * 294 * 518 * 4
        assert hooks._MAPANYTHING_HOST_PAYLOAD_BYTES == 1_827_504
        timing = snapshot["serialization_prep"][
            "mapanything.exact_native_tensor_to_host"
        ]
        assert timing["count"] == 1
        assert timing["last_payload_bytes"] == 1_827_504
        assert timing["last_ns"] >= 0

        bad = hooks.MapAnythingProcessor(
            pipeline=pipeline,
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            native_capture=lambda *_args: {
                "depth": np.ones((294, 518), dtype=np.float32),
                "conf": np.ones((294, 518), dtype=np.float32),
            },
        )
        try:
            bad.handle_native_frame_ds9(frame)
        except RuntimeError as exc:
            assert "exactly depth/conf/mask" in str(exc)
        else:
            raise AssertionError("partial MapAnything layers did not fail closed")
        """
    )


def test_non_inference_and_unarmed_batches_never_extract_rgb_surface() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace
        import numpy as np

        from noesis.pipelines import hooks

        class Buffer:
            def __init__(self):
                self.extract_calls = 0

            def extract(self, _batch_id):
                self.extract_calls += 1
                raise AssertionError("unarmed/non-inference RGB must not extract")

        class Provider:
            def __init__(self):
                self.capture_arm_calls = 0

            def capture_arm(self, **_kwargs):
                self.capture_arm_calls += 1
                return None

        provider = Provider()
        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(4, 3)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=1,
            camera_labels={0: "cam0"},
            rgb_provider=provider,
            rgb_width=4,
            rgb_height=3,
            native_capture=lambda *_args: None,
        )
        frame = SimpleNamespace(
            batch_id=0,
            pad_index=0,
            frame_num=41,
            buf_pts=11_000_000,
        )
        buffer = Buffer()
        assert processor.handle_native_buffer_frame_ds9(buffer, frame) is False
        assert provider.capture_arm_calls == 0
        assert buffer.extract_calls == 0

        processor.native_capture = lambda *_args: {
            "depth": np.ones((1, 294, 518), dtype=np.float32),
            "conf": np.ones((1, 294, 518), dtype=np.float32),
            "mask": np.ones((1, 294, 518), dtype=np.float32),
        }
        processor._start_async_worker = lambda: None
        assert processor.handle_native_buffer_frame_ds9(buffer, frame) is True
        assert provider.capture_arm_calls == 1
        assert buffer.extract_calls == 0
        processor._async_queue.get_nowait()
        processor._async_queue.task_done()
        """
    )


def test_buffer_operator_uses_one_exact_buffer_for_metadata_and_surface() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        buffer = SimpleNamespace(
            batch_meta=SimpleNamespace(
                frame_items=[
                    SimpleNamespace(frame_number=1),
                    SimpleNamespace(frame_number=2),
                ]
            )
        )

        class Processor:
            gie_id = 2
            pipeline = SimpleNamespace(depth_enabled=True)

            def __init__(self):
                self.calls = []

            def handle_native_buffer_frame_ds9(self, observed_buffer, frame):
                assert observed_buffer is buffer
                self.calls.append(frame.frame_number)
                return frame.frame_number == 2

        processor = Processor()
        operator = hooks._MapAnythingBufferOperator.__new__(
            hooks._MapAnythingBufferOperator
        )
        operator._processor = processor
        operator._frames_seen = 0
        operator._matched_frames = 0
        assert operator.handle_buffer(buffer) is True
        assert processor.calls == [1, 2]
        assert operator._frames_seen == 2
        assert operator._matched_frames == 1
        """
    )


def test_buffer_operator_reports_failure_without_raising_across_pybind() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        class Processor:
            gie_id = 2
            pipeline = SimpleNamespace(depth_enabled=True)

            def __init__(self):
                self.failures = []

            def handle_native_buffer_frame_ds9(self, _buffer, _frame):
                raise RuntimeError("exact capture failed")

            def report_capture_failure(self, exc):
                self.failures.append(exc)

        processor = Processor()
        operator = hooks._MapAnythingBufferOperator.__new__(
            hooks._MapAnythingBufferOperator
        )
        operator._processor = processor
        operator._frames_seen = 0
        operator._matched_frames = 0
        buffer = SimpleNamespace(
            batch_meta=SimpleNamespace(
                frame_items=[SimpleNamespace(frame_number=7)]
            )
        )

        assert operator.handle_buffer(buffer) is True
        assert len(processor.failures) == 1
        assert str(processor.failures[0]) == "exact capture failed"
        assert operator._matched_frames == 0
        """
    )


def test_armed_cuda_rgb_surface_yields_exact_rgb_digest_and_identity() -> None:
    _run_isolated(
        """
        import hashlib
        from types import SimpleNamespace

        import numpy as np
        import torch

        from noesis.capture_event_rgb_provider import PipelineRgbFrameProvider
        from noesis.pipelines import hooks
        from noesis_core.capture_event_fusion import RawDepthSnapshot

        assert torch.cuda.is_available()
        rgb_cpu = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb_cpu[..., 0] = 11
        rgb_cpu[..., 1] = 22
        rgb_cpu[..., 2] = 33
        rgb = torch.as_tensor(rgb_cpu, device="cuda")

        class Buffer:
            extract_calls = 0

            def extract(self, batch_id):
                assert batch_id == 0
                self.extract_calls += 1
                return rgb

        provider = PipelineRgbFrameProvider(
            camera_sources={0: "cam0"},
            max_width=4,
            max_height=3,
            max_frame_bytes=36,
            max_total_bytes=36,
        )
        arm = provider.arm("cam0")
        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(4, 3)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=1,
            camera_labels={0: "cam0"},
            rgb_provider=provider,
            rgb_width=4,
            rgb_height=3,
        )
        buffer = Buffer()
        processor._capture_armed_rgb(
            buffer=buffer,
            batch_id=0,
            source_id=0,
            frame_id=42,
            source_media_pts_ns=12_000_000,
        )
        row = RawDepthSnapshot(
            camera_id="cam0",
            storage_key="cam0",
            timestamp_us=1,
            snapshot_id="raw",
            artifact_ref="depth:raw",
            content_sha256="0" * 64,
            sequence=1,
            manifest_sha256="1" * 64,
            source_id=0,
            source_frame_number=42,
            source_media_pts_ns=12_000_000,
        )
        frame = provider.provide("cam0", cohort=(row,))
        assert frame is not None
        assert buffer.extract_calls == 1
        assert (frame.source_id, frame.batch_id, frame.frame_id) == (0, 0, 42)
        assert frame.source_media_pts_ns == 12_000_000
        np.testing.assert_array_equal(frame.pixels, rgb_cpu)
        assert frame.content_sha256 == hashlib.sha256(
            rgb_cpu.tobytes(order="C")
        ).hexdigest()
        provider.disarm(arm)
        assert provider.health_snapshot()["armed"] is False
        """
    )


def test_ds9_processor_uses_hr0_profile_shape_and_payload_contract() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        import numpy as np

        from noesis.pipelines import hooks

        calls = []

        def capture(frame, gie_id, expected_height, expected_width):
            calls.append((frame, gie_id, expected_height, expected_width))
            return {
                name: np.ones((1, 378, 672), dtype=np.float32)
                for name in ("depth", "conf", "mask")
            }

        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(1920, 1080)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            profile_name="hr0_378x672_b3_fp32",
            output_height=378,
            output_width=672,
            native_capture=capture,
        )
        processor._start_async_worker = lambda: None
        frame = SimpleNamespace(
            batch_id=0,
            pad_index=0,
            frame_num=23,
            buf_pts=34,
        )
        hooks.reset_core_path_instrumentation()
        assert processor.handle_native_frame_ds9(frame) is True
        assert calls == [(frame, 2, 378, 672)]
        assert processor.host_payload_bytes == 3_048_192
        job = processor._async_queue.get_nowait()
        assert all(value.shape == (378, 672) for value in job.tensors.values())
        processor._async_queue.task_done()
        counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
        assert counters["tensor_boundary_copy_bytes_total.mapanything"] == 3_048_192
        assert counters["mapanything_probe_local_d2h_bytes_total"] == 3_048_192
        """
    )


def test_ds9_postprocess_attachment_resolves_explicit_hr0_profile() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        component = SimpleNamespace(name="mapanything_fullframe", config={})
        pipeline = SimpleNamespace(
            config={
                "models": {
                    "mapanything": {
                        "name": "mapanything_fullframe",
                        "profile": "hr0_378x672_b3_fp32",
                        "engine": (
                            "DS9/models/engines/candidates/"
                            "mapanything_images_378x672_b3_fp32.plan"
                        ),
                        "config-file-path": (
                            "DS9/pipelines/"
                            "config_infer_secondary_mapanything_hr0.ini"
                        ),
                        "gie_id": 2,
                        "batch_size": 3,
                    }
                }
            },
            components={"mapanything_fullframe": component},
            camera_labels={0: "camera-0"},
            ds_pipeline=None,
        )
        hooks.attach_mapanything_postprocess_hook(
            pipeline,
            storage=SimpleNamespace(),
        )
        processor = pipeline.mapanything_processor
        assert processor.profile_name == "hr0_378x672_b3_fp32"
        assert (processor.output_height, processor.output_width) == (378, 672)
        assert processor.host_payload_bytes == 3_048_192
        assert component.config["_mapanything_processor"] is processor
        """
    )


def test_exact_rgb_postprocess_attaches_after_inference_conversion() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        from noesis.capture_event_rgb_provider import PipelineRgbFrameProvider
        from noesis.pipelines import hooks

        class DsPipeline:
            def __init__(self):
                self.attachments = []

            def attach(self, name, probe):
                self.attachments.append((name, probe))

        mapanything = SimpleNamespace(name="mapanything_fullframe", config={})
        rgb_caps = SimpleNamespace(name="mapanything_rgb_caps", config={})
        streammux = SimpleNamespace(
            name="streammux",
            config={"width": 1920, "height": 1080},
        )
        pipeline = SimpleNamespace(
            config={
                "models": {
                    "mapanything": {
                        "name": "mapanything_fullframe",
                        "profile": "hr0_378x672_b3_fp32",
                        "engine": (
                            "DS9/models/engines/candidates/"
                            "mapanything_images_378x672_b3_fp32.plan"
                        ),
                        "config-file-path": (
                            "DS9/pipelines/"
                            "config_infer_secondary_mapanything_hr0.ini"
                        ),
                        "gie_id": 2,
                        "batch_size": 3,
                    }
                }
            },
            components={
                "mapanything_fullframe": mapanything,
                "mapanything_rgb_caps": rgb_caps,
                "streammux": streammux,
            },
            camera_labels={0: "camera-0"},
            ds_pipeline=DsPipeline(),
        )
        provider = PipelineRgbFrameProvider(camera_sources={0: "camera-0"})
        hooks.BufferOperator = object
        hooks.Probe = lambda _name, operator: operator
        hooks._MapAnythingBufferOperator = lambda processor: processor
        hooks.noesis_depth_tracking_tensor_ext = SimpleNamespace(
            capture_mapanything_tensor_layers_exact=lambda *_args: {},
        )

        hooks.attach_mapanything_postprocess_hook(
            pipeline,
            storage=SimpleNamespace(),
            rgb_provider=provider,
        )

        assert [name for name, _probe in pipeline.ds_pipeline.attachments] == [
            "mapanything_rgb_caps"
        ]
        assert mapanything.config["_mapanything_processor"] is pipeline.mapanything_processor
        provider.close()
        """
    )


def test_ds9_postprocess_requires_native_probe_types_only_for_real_pipeline() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        import pytest

        from noesis.pipelines import hooks
        from noesis_core.servicemaker_shutdown import (
            synthetic_stub_lifecycle_evidence,
        )

        def make_pipeline(*, ds_pipeline, lifecycle_evidence=None):
            component = SimpleNamespace(
                name="mapanything_fullframe",
                config={},
            )
            return SimpleNamespace(
                config={
                    "models": {
                        "mapanything": {
                            "name": "mapanything_fullframe",
                            "gie_id": 2,
                            "batch_size": 3,
                        }
                    }
                },
                components={"mapanything_fullframe": component},
                camera_labels={0: "camera-0"},
                ds_pipeline=ds_pipeline,
                lifecycle_evidence=lifecycle_evidence,
            )

        hooks.BufferOperator = None
        hooks.Probe = None

        real = make_pipeline(ds_pipeline=SimpleNamespace())
        with pytest.raises(RuntimeError, match="requires pyservicemaker"):
            hooks.attach_mapanything_postprocess_hook(
                real,
                storage=SimpleNamespace(),
            )

        synthetic = make_pipeline(
            ds_pipeline=SimpleNamespace(),
            lifecycle_evidence=synthetic_stub_lifecycle_evidence(),
        )
        hooks.attach_mapanything_postprocess_hook(
            synthetic,
            storage=SimpleNamespace(),
        )
        assert synthetic.mapanything_processor is not None

        no_pipeline = make_pipeline(ds_pipeline=None)
        hooks.attach_mapanything_postprocess_hook(
            no_pipeline,
            storage=SimpleNamespace(),
        )
        assert no_pipeline.mapanything_processor is not None
        """
    )


def test_ds9_rejects_coexisting_generic_native_capture() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        hooks.noesis_depth_tracking_tensor_ext = SimpleNamespace(
            capture_mapanything_tensor_layers_exact=lambda *_args: {},
            capture_tensor_layers=lambda *_args: {},
        )
        try:
            hooks._require_ds9_mapanything_native_capture()
        except RuntimeError as exc:
            assert "retired generic tensor capture" in str(exc)
        else:
            raise AssertionError("coexisting generic capture was accepted")

        hooks.noesis_depth_tracking_tensor_ext = SimpleNamespace(
            capture_mapanything_tensor_layers_exact=lambda *_args: {},
        )
        selected = hooks._require_ds9_mapanything_native_capture()
        assert callable(selected)
        """
    )


def test_ds9_postprocess_is_bounded_async_and_fail_closed() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        import numpy as np

        from noesis.pipelines import hooks

        tensors = {
            "depth": np.full((294, 518), 2.0, dtype=np.float32),
            "conf": np.ones((294, 518), dtype=np.float32),
            "mask": np.ones((294, 518), dtype=np.float32),
        }
        pipeline = SimpleNamespace(depth_enabled=True, frame_size=(3, 2))
        processor = hooks.MapAnythingProcessor(
            pipeline=pipeline,
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            native_capture=lambda *_args: tensors,
        )
        processed = []

        def emit(**kwargs):
            processed.append(kwargs)
            return object()

        processor.handle_numpy_arrays = emit
        frame = SimpleNamespace(
            batch_id=0,
            pad_index=0,
            frame_num=1,
            buf_pts=2,
        )
        assert processor.handle_native_frame_ds9(frame) is True
        pipeline.depth_enabled = False
        processor._async_queue.join()
        assert len(processed) == 1
        assert processed[0]["captured_while_enabled"] is True
        processor.shutdown(wait=True, timeout_s=2.0)
        assert processor.async_shutdown_quiesced() is True
        assert processor._async_thread is not None
        assert processor._async_thread.is_alive() is False

        blocked = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(3, 2)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            native_capture=lambda *_args: tensors,
        )
        blocked._start_async_worker = lambda: None
        blocked._async_queue = __import__("queue").Queue(maxsize=1)
        blocked._async_queue.put_nowait(object())
        try:
            blocked.handle_native_frame_ds9(frame)
        except RuntimeError as exc:
            assert "bounded async postprocess queue is full" in str(exc)
        else:
            raise AssertionError("full async queue was silently degraded")

        poisoned = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(3, 2)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            native_capture=lambda *_args: tensors,
        )
        poisoned._async_failure = "fixture failure"
        try:
            poisoned.handle_native_frame_ds9(frame)
        except RuntimeError as exc:
            assert "async postprocess is poisoned" in str(exc)
        else:
            raise AssertionError("poisoned async worker was reused")
        """
    )


def test_ds9_depth_publication_waits_for_durable_storage_commit() -> None:
    _run_isolated(
        """
        import os
        import time
        from types import SimpleNamespace

        import numpy as np

        from noesis.pipelines import hooks

        # The retired disable flag must not create an uncommitted memory:// ref.
        os.environ["NOESIS_DEPTH_STORE_ENABLED"] = "0"
        os.environ["NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S"] = "0.25"
        events = []

        class Handle:
            path = "/planned/depth.zarr"

            def wait(self, timeout=None):
                events.append(("wait", timeout))
                return SimpleNamespace(path="/committed/depth.zarr")

        class Storage:
            def store(self, *_args, **_kwargs):
                events.append("store")
                return Handle()

        class Publisher:
            def publish(self, result):
                events.append(("publish", result.depth_map_ref))

        pipeline = SimpleNamespace(
            depth_enabled=True,
            frame_size=(4, 3),
            record_depth_frame=lambda _now: events.append("record_depth_frame"),
        )
        processor = hooks.MapAnythingProcessor(
            pipeline=pipeline,
            storage=Storage(),
            depth_pub=Publisher(),
            gie_id=2,
            batch_size=3,
        )
        result = processor.handle_numpy_arrays(
            source_id=0,
            frame_id=1,
            pts_ns=time.time_ns(),
            tensors={
                "depth": np.ones((3, 4), dtype=np.float32),
                "conf": np.ones((3, 4), dtype=np.float32),
                "mask": np.ones((3, 4), dtype=np.uint8),
            },
            captured_while_enabled=True,
        )
        assert result is not None
        assert result.depth_map_ref == "/committed/depth.zarr"
        assert "/planned/depth.zarr" not in str(events)
        assert events == [
            "store",
            ("wait", 0.25),
            "record_depth_frame",
            ("publish", "/committed/depth.zarr"),
        ]

        os.environ["NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S"] = "0.01"

        class TimedOutHandle:
            path = "/planned/never-public.zarr"

            def wait(self, timeout=None):
                assert timeout == 0.1
                raise TimeoutError("durable DS9 depth commit timed out")

        published = []
        failures = []
        failed_pipeline = SimpleNamespace(
            depth_enabled=True,
            frame_size=(4, 3),
            record_depth_frame=lambda _now: (_ for _ in ()).throw(
                AssertionError("uncommitted depth was recorded")
            ),
        )
        failed = hooks.MapAnythingProcessor(
            pipeline=failed_pipeline,
            storage=SimpleNamespace(
                store=lambda *_args, **_kwargs: TimedOutHandle()
            ),
            depth_pub=SimpleNamespace(publish=published.append),
            gie_id=2,
            batch_size=3,
            failure_callback=failures.append,
        )
        failed._enqueue_async_job(
            hooks._MapAnythingNativeJob(
                source_id=0,
                frame_id=2,
                pts_ns=time.time_ns(),
                captured_at_us=time.time_ns() // 1_000,
                tensors={
                    "depth": np.ones((3, 4), dtype=np.float32),
                    "conf": np.ones((3, 4), dtype=np.float32),
                    "mask": np.ones((3, 4), dtype=np.uint8),
                },
            )
        )
        try:
            failed.shutdown(wait=True, timeout_s=2.0)
        except RuntimeError as exc:
            assert "async postprocess is poisoned" in str(exc)
            assert "durable DS9 depth commit timed out" in str(exc)
        else:
            raise AssertionError("failed durable commit did not poison DS9 worker")
        assert published == []
        assert len(failures) == 1
        assert isinstance(failures[0], TimeoutError)
        assert failed.async_shutdown_quiesced() is True
        """
    )


def test_ds9_attachment_failures_abort_startup() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        from noesis.pipelines import hooks

        class DsPipeline:
            def __init__(self, *, fail_attach=False):
                self.fail_attach = fail_attach

            def attach(self, _name, _probe):
                if self.fail_attach:
                    raise RuntimeError("fixture attach failure")

        def make_pipeline(*, fail_attach=False):
            component = SimpleNamespace(
                name="mapanything_fullframe",
                config={},
            )
            return SimpleNamespace(
                config={
                    "models": {
                        "mapanything": {
                            "name": "mapanything_fullframe",
                            "gie_id": 2,
                            "batch_size": 3,
                        }
                    }
                },
                components={"mapanything_fullframe": component},
                camera_labels={0: "camera-0"},
                ds_pipeline=DsPipeline(fail_attach=fail_attach),
            )

        hooks.BatchMetadataOperator = object()
        hooks.Probe = lambda _name, operator: operator
        hooks.noesis_depth_tracking_tensor_ext = None
        missing = make_pipeline()
        try:
            hooks.attach_mapanything_postprocess_hook(
                missing,
                storage=SimpleNamespace(),
            )
        except RuntimeError as exc:
            assert "exact native post-process attachment failed" in str(exc)
            assert "owned noesis_depth_tracking_tensor_ext" in str(exc.__cause__)
        else:
            raise AssertionError("missing exact native bridge did not abort startup")
        assert missing.mapanything_processor is not None

        hooks.noesis_depth_tracking_tensor_ext = SimpleNamespace(
            capture_mapanything_tensor_layers_exact=lambda *_args: {},
        )
        rejected = make_pipeline(fail_attach=True)
        try:
            hooks.attach_mapanything_postprocess_hook(
                rejected,
                storage=SimpleNamespace(),
            )
        except RuntimeError as exc:
            assert "exact native post-process attachment failed" in str(exc)
            assert "fixture attach failure" in str(exc.__cause__)
        else:
            raise AssertionError("probe attach failure did not abort startup")

        runtime_source = (
            __import__("pathlib").Path.cwd()
            / "DS9"
            / "noesis"
            / "ds9_runtime_core.py"
        ).read_text(encoding="utf-8")
        attach_guard = runtime_source[
            runtime_source.index("# Attach MapAnything postprocess") :
            runtime_source.index("    try:\\n        hooks.attach_pose_feature_hook")
        ]
        assert 'logger.exception("Error while evaluating MapAnything postprocess attachment")' in attach_guard
        assert "    except StartupResourceRegistrationError:\\n        raise\\n" in attach_guard
        assert "        _mapanything_failed(exc)\\n" in attach_guard
        assert '        return _abort_startup("mapanything_hook_failed")\\n' in attach_guard
        """
    )


def test_ds9_poisoned_final_job_is_joined_and_surfaced_at_teardown() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        import numpy as np

        from noesis.pipelines import hooks

        tensors = {
            "depth": np.full((1, 294, 518), 2.0, dtype=np.float32),
            "conf": np.ones((1, 294, 518), dtype=np.float32),
            "mask": np.ones((1, 294, 518), dtype=np.float32),
        }
        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(518, 294)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            native_capture=lambda *_args: tensors,
        )

        def fail_final_job(**_kwargs):
            raise RuntimeError("final enabled frame failed")

        processor.handle_numpy_arrays = fail_final_job
        frame = SimpleNamespace(
            batch_id=0,
            pad_index=0,
            frame_num=77,
            buf_pts=88,
        )
        hooks.reset_core_path_instrumentation()
        assert processor.handle_native_frame_ds9(frame) is True
        try:
            processor.shutdown(wait=True, timeout_s=2.0)
        except RuntimeError as exc:
            assert "async postprocess is poisoned" in str(exc)
            assert "final enabled frame failed" in str(exc)
        else:
            raise AssertionError("final-job poison stayed latent at teardown")

        assert processor.async_shutdown_quiesced() is True
        assert processor._async_thread is not None
        assert processor._async_thread.daemon is False
        assert processor._async_thread.is_alive() is False
        assert processor._async_stopped.is_set() is True
        assert processor._async_unfinished_task_count() == 0
        counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
        assert counters["mapanything_async_postprocess_failures_total"] == 1
        assert counters["mapanything_async_shutdown_total"] == 1
        """
    )


def test_ds9_mapanything_snapshot_preserves_exact_source_frame_identity() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        import numpy as np

        from noesis.pipelines import hooks

        observed = {}

        class Handle:
            def wait(self, timeout=None):
                observed["timeout"] = timeout
                return SimpleNamespace(path="/committed/depth.zarr")

        class Storage:
            def store(self, *args, **kwargs):
                observed["args"] = args
                observed["kwargs"] = kwargs
                return Handle()

        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(
                depth_enabled=True,
                frame_size=(4, 3),
                record_depth_frame=lambda _now: None,
            ),
            storage=Storage(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            camera_labels={0: "living-room"},
        )
        result = processor.handle_numpy_arrays(
            source_id=0,
            frame_id=321,
            pts_ns=456_000_000,
            tensors={
                "depth": np.ones((3, 4), dtype=np.float32),
                "conf": np.ones((3, 4), dtype=np.float32),
                "mask": np.ones((3, 4), dtype=np.uint8),
            },
            captured_while_enabled=True,
        )
        assert result is not None
        assert observed["kwargs"]["attrs"] == {
            "source_frame_contract": "noesis.mapanything.source_frame.v1",
            "source_id": 0,
            "source_frame_number": 321,
            "source_media_pts_ns": 456_000_000,
            "storage_timestamp_basis": "wall_clock_fallback",
            "stored_mask_policy": "finite_positive_prediction",
            "strict_calibration_mask_policy": "finite_positive_model_non_ambiguous_calibrated_fov",
            "strict_calibration_valid_fraction": 1.0,
            "model_ambiguous_confidence_scale": 0.5,
            "outside_calibrated_fov_confidence_scale": 0.25,
        }
        """
    )


def test_ds9_native_capture_timestamp_precedes_async_processing_delay() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        import numpy as np

        from noesis.pipelines import hooks

        observed = {}
        now_ns = [1_234_567_000]
        hooks.time.time_ns = lambda: now_ns[0]

        class Handle:
            def wait(self, timeout=None):
                return SimpleNamespace(path="/committed/depth.zarr")

        class Storage:
            def store(self, camera_id, timestamp_us, *_args, **kwargs):
                observed["camera_id"] = camera_id
                observed["timestamp_us"] = timestamp_us
                observed["attrs"] = kwargs["attrs"]
                return Handle()

        tensors = {
            name: np.ones((294, 518), dtype=np.float32)
            for name in ("depth", "conf", "mask")
        }

        def capture(*_args):
            # Simulate native capture and queued processing completing much later
            # than admission at the probe callback.
            now_ns[0] = 9_876_543_000
            return tensors

        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(
                depth_enabled=True,
                frame_size=(518, 294),
                record_depth_frame=lambda _now: None,
            ),
            storage=Storage(),
            depth_pub=None,
            gie_id=2,
            batch_size=1,
            camera_labels={0: "living-room"},
            native_capture=capture,
        )
        frame = SimpleNamespace(
            batch_id=0,
            pad_index=0,
            frame_num=321,
            buf_pts=456_000_000,
        )
        assert processor.handle_native_frame_ds9(frame) is True
        processor.wait_idle(timeout_s=2.0)
        processor.shutdown(wait=True, timeout_s=2.0)

        assert observed == {
            "camera_id": "living-room",
            "timestamp_us": 1_234_567,
            "attrs": {
                "source_frame_contract": "noesis.mapanything.source_frame.v1",
                "source_id": 0,
                    "source_frame_number": 321,
                    "source_media_pts_ns": 456_000_000,
                    "storage_timestamp_basis": "capture_wall_clock",
                    "stored_mask_policy": "finite_positive_prediction",
                    "strict_calibration_mask_policy": "finite_positive_model_non_ambiguous_calibrated_fov",
                    "strict_calibration_valid_fraction": 1.0,
                    "model_ambiguous_confidence_scale": 0.5,
                    "outside_calibrated_fov_confidence_scale": 0.25,
                },
            }
        """
    )


def test_ds9_native_capture_requires_present_source_frame_evidence() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        import numpy as np
        import pytest

        from noesis.pipelines import hooks

        capture_calls = []

        def capture(*_args):
            capture_calls.append(_args)
            return {
                "depth": np.ones((294, 518), dtype=np.float32),
                "conf": np.ones((294, 518), dtype=np.float32),
                "mask": np.ones((294, 518), dtype=np.float32),
            }

        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(518, 294)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            native_capture=capture,
        )
        processor._start_async_worker = lambda: None

        with pytest.raises(RuntimeError, match="source frame number"):
            processor.handle_native_frame_ds9(
                SimpleNamespace(batch_id=0, pad_index=0, buf_pts=0)
            )
        with pytest.raises(RuntimeError, match="source media PTS"):
            processor.handle_native_frame_ds9(
                SimpleNamespace(batch_id=0, pad_index=0, frame_num=0)
            )
        with pytest.raises(RuntimeError, match="valid non-negative"):
            processor.handle_native_frame_ds9(
                SimpleNamespace(
                    batch_id=0,
                    pad_index=0,
                    frame_num=0,
                    buf_pts=(1 << 64) - 1,
                )
            )
        assert capture_calls == []
        assert processor._async_thread is None
        assert processor._async_unfinished_task_count() == 0

        assert processor.handle_native_frame_ds9(
            SimpleNamespace(
                batch_id=0,
                pad_index=0,
                frame_num=0,
                buf_pts=0,
            )
        ) is True
        job = processor._async_queue.get_nowait()
        assert job.frame_id == 0
        assert job.pts_ns == 0
        processor._async_queue.task_done()
        """
    )


def test_ds9_shutdown_rejects_new_capture_without_starting_a_worker() -> None:
    _run_isolated(
        """
        from types import SimpleNamespace

        import numpy as np

        from noesis.pipelines import hooks

        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(518, 294)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            native_capture=lambda *_args: {
                "depth": np.ones((1, 294, 518), dtype=np.float32),
                "conf": np.ones((1, 294, 518), dtype=np.float32),
                "mask": np.ones((1, 294, 518), dtype=np.float32),
            },
        )
        processor.shutdown(wait=True, timeout_s=1.0)
        assert processor.async_shutdown_quiesced() is True
        assert processor._async_thread is None

        frame = SimpleNamespace(
            batch_id=0,
            pad_index=0,
            frame_num=1,
            buf_pts=2,
        )
        try:
            processor.handle_native_frame_ds9(frame)
        except RuntimeError as exc:
            assert "shutting down" in str(exc)
        else:
            raise AssertionError("capture admission reopened after shutdown")
        assert processor._async_thread is None
        """
    )


def test_ds9_shutdown_waits_for_admitted_native_capture_before_fifo_stop() -> None:
    _run_isolated(
        """
        import threading
        import time
        from types import SimpleNamespace

        import numpy as np

        from noesis.pipelines import hooks

        capture_started = threading.Event()
        release_capture = threading.Event()
        failures = []
        processed = []

        def capture(*_args):
            capture_started.set()
            if not release_capture.wait(timeout=2.0):
                raise TimeoutError("test did not release admitted DS9 capture")
            return {
                "depth": np.ones((294, 518), dtype=np.float32),
                "conf": np.ones((294, 518), dtype=np.float32),
                "mask": np.ones((294, 518), dtype=np.float32),
            }

        processor = hooks.MapAnythingProcessor(
            pipeline=SimpleNamespace(depth_enabled=True, frame_size=(518, 294)),
            storage=SimpleNamespace(),
            depth_pub=None,
            gie_id=2,
            batch_size=3,
            native_capture=capture,
        )
        processor.handle_numpy_arrays = (
            lambda **kwargs: processed.append(kwargs) or object()
        )
        frame = SimpleNamespace(
            batch_id=0,
            pad_index=0,
            frame_num=11,
            buf_pts=12,
        )

        def run_capture():
            try:
                processor.handle_native_frame_ds9(frame)
            except BaseException as exc:
                failures.append(exc)

        def run_shutdown():
            try:
                processor.shutdown(wait=True, timeout_s=2.0)
            except BaseException as exc:
                failures.append(exc)

        capture_thread = threading.Thread(target=run_capture)
        shutdown_thread = threading.Thread(target=run_shutdown)
        capture_thread.start()
        assert capture_started.wait(timeout=1.0)
        shutdown_thread.start()
        deadline = time.monotonic() + 1.0
        while processor._async_accepting and time.monotonic() < deadline:
            time.sleep(0.001)
        assert processor._async_accepting is False
        assert shutdown_thread.is_alive()
        assert processor._async_active_captures == 1
        assert processor._async_stop_enqueued is False

        release_capture.set()
        capture_thread.join(timeout=2.0)
        shutdown_thread.join(timeout=2.0)
        assert failures == []
        assert capture_thread.is_alive() is False
        assert shutdown_thread.is_alive() is False
        assert len(processed) == 1
        assert processor.async_shutdown_quiesced() is True
        """
    )
