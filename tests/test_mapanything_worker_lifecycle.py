from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from noesis.pipelines import hooks
from noesis.pipelines import hooks_v3dt_reimpl


HOOK_MODULES = (hooks, hooks_v3dt_reimpl)


def _processor(module):
    processor = module.MapAnythingProcessor(
        pipeline=SimpleNamespace(depth_enabled=True, frame_size=(4, 3)),
        storage=SimpleNamespace(),
        depth_pub=None,
        gie_id=2,
    )
    processor._to_numpy = lambda tensor: tensor
    return processor


def _job(module, frame_id: int = 1):
    depth = np.ones((1, 3, 4), dtype=np.float32)
    return module._MapAnythingJob(
        source_id=0,
        frame_id=frame_id,
        pts_ns=frame_id,
        batch_id=0,
        depth=depth,
        confidence=None,
        mask=None,
    )


def _idle_job(module, frame_id: int = 1):
    return _job(module, frame_id)


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_mapanything_wait_idle_drains_without_closing_admission(module) -> None:
    processor = _processor(module)
    processed: list[int] = []
    processor.handle_numpy_arrays = (
        lambda **kwargs: processed.append(int(kwargs["frame_id"])) or object()
    )

    processor._enqueue_async_job(_idle_job(module, 1))
    receipt = processor.wait_idle(timeout_s=2.0)

    assert processed == [1]
    assert receipt.active_captures == 0
    assert receipt.unfinished_tasks == 0
    assert receipt.accepting is True
    assert processor._async_accepting is True

    processor._enqueue_async_job(_idle_job(module, 2))
    processor.shutdown(wait=True, timeout_s=2.0)
    assert processed == [1, 2]


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_mapanything_wait_idle_is_bounded_and_non_closing(module) -> None:
    processor = _processor(module)
    with processor._async_idle:
        processor._async_active_captures = 1

    with pytest.raises(TimeoutError, match="did not become idle"):
        processor.wait_idle(timeout_s=0.01)

    assert processor._async_accepting is True
    with processor._async_idle:
        processor._async_active_captures = 0
        processor._async_idle.notify_all()
    receipt = processor.wait_idle(timeout_s=0.1)
    assert receipt.accepting is True
    processor.shutdown(wait=True, timeout_s=1.0)


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_mapanything_worker_drains_owned_job_after_depth_gate_closes(module) -> None:
    processor = _processor(module)
    processed: list[dict[str, object]] = []

    def emit(**kwargs):
        processed.append(kwargs)
        return object()

    processor.handle_numpy_arrays = emit
    assert processor._enqueue_async_job(_job(module)) is True
    processor.pipeline.depth_enabled = False
    processor.shutdown(wait=True, timeout_s=2.0)

    assert len(processed) == 1
    assert processed[0]["captured_while_enabled"] is True
    assert processor.async_shutdown_quiesced() is True
    assert processor._async_thread is not None
    assert processor._async_thread.daemon is False
    assert processor._async_thread.is_alive() is False


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_mapanything_worker_surfaces_final_job_poison_after_join(module) -> None:
    processor = _processor(module)

    def fail(**_kwargs):
        raise RuntimeError("final DS8 MapAnything job failed")

    processor.handle_numpy_arrays = fail
    assert processor._enqueue_async_job(_job(module)) is True

    with pytest.raises(RuntimeError, match="async postprocess is poisoned"):
        processor.shutdown(wait=True, timeout_s=2.0)

    assert processor.async_shutdown_quiesced() is True
    assert processor._async_thread is not None
    assert processor._async_thread.is_alive() is False
    assert processor._async_unfinished_task_count() == 0


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_mapanything_worker_never_reopens_admission_after_shutdown(module) -> None:
    processor = _processor(module)
    processor.shutdown(wait=True, timeout_s=1.0)

    with pytest.raises(RuntimeError, match="shutting down"):
        processor._enqueue_async_job(_job(module))

    assert processor.async_shutdown_quiesced() is True
    assert processor._async_thread is None


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_mapanything_shutdown_waits_for_admitted_capture_before_fifo_stop(
    module,
) -> None:
    processor = _processor(module)
    capture_started = threading.Event()
    release_capture = threading.Event()
    processed: list[dict[str, object]] = []
    failures: list[BaseException] = []

    class _BlockingTensor:
        def clone(self):
            capture_started.set()
            if not release_capture.wait(timeout=2.0):
                raise TimeoutError("test did not release admitted capture")
            return np.ones((1, 3, 4), dtype=np.float32)

    class _TensorMeta:
        @staticmethod
        def get_layers():
            return {"depth": _BlockingTensor()}

    processor.handle_numpy_arrays = (
        lambda **kwargs: processed.append(kwargs) or object()
    )
    frame_meta = SimpleNamespace(
        pad_index=0,
        frame_num=7,
        buf_pts=8,
        batch_id=0,
    )

    def _capture() -> None:
        try:
            processor.handle_nvds_tensor_ds8(frame_meta, _TensorMeta())
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def _shutdown() -> None:
        try:
            processor.shutdown(wait=True, timeout_s=2.0)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    capture_thread = threading.Thread(target=_capture, name="test-map-capture")
    shutdown_thread = threading.Thread(target=_shutdown, name="test-map-shutdown")
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
    assert processed[0]["captured_while_enabled"] is True
    assert processor.async_shutdown_quiesced() is True


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_mapanything_queue_saturation_fails_loudly(module) -> None:
    processor = _processor(module)
    processor._async_queue = queue.Queue(maxsize=1)
    processor._async_queue.put_nowait(object())
    processor._start_async_worker_locked = lambda: None

    with pytest.raises(RuntimeError, match="bounded async postprocess queue is full"):
        processor._enqueue_async_job(_job(module))

    assert processor._dropped_jobs == 1
    processor._async_queue.get_nowait()
    processor._async_queue.task_done()


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_depth_result_is_published_only_after_exact_storage_commit(
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The retired disable flag must not create an uncommitted memory:// ref.
    monkeypatch.setenv("NOESIS_DEPTH_STORE_ENABLED", "0")
    monkeypatch.setenv("NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S", "0.25")
    events: list[object] = []

    class _Handle:
        path = "/planned/depth.zarr"

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            return SimpleNamespace(path="/committed/depth.zarr")

    class _Storage:
        @staticmethod
        def store(*_args, **_kwargs):
            events.append("store")
            return _Handle()

    class _Publisher:
        @staticmethod
        def publish(result):
            events.append(("publish", result.depth_map_ref))

    pipeline = SimpleNamespace(
        depth_enabled=True,
        frame_size=(4, 3),
        record_depth_frame=lambda _now: events.append("record_depth_frame"),
    )
    processor = module.MapAnythingProcessor(
        pipeline=pipeline,
        storage=_Storage(),
        depth_pub=_Publisher(),
        gie_id=2,
    )
    result = processor.handle_numpy_arrays(
        source_id=0,
        frame_id=1,
        pts_ns=time.time_ns(),
        tensors={
            "depth": np.ones((3, 4), dtype=np.float32),
            "confidence": np.ones((3, 4), dtype=np.float32),
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


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_storage_commit_timeout_poisons_worker_and_publishes_nothing(
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_DEPTH_STORE_ENABLED", "0")
    monkeypatch.setenv("NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S", "0.01")
    published: list[object] = []
    depth_frames: list[float] = []
    failures: list[BaseException] = []

    class _Handle:
        path = "/planned/depth.zarr"

        @staticmethod
        def wait(timeout=None):
            assert timeout == 0.1
            raise TimeoutError("durable depth commit timed out")

    storage = SimpleNamespace(store=lambda *_args, **_kwargs: _Handle())
    pipeline = SimpleNamespace(
        depth_enabled=True,
        frame_size=(4, 3),
        record_depth_frame=depth_frames.append,
    )
    processor = module.MapAnythingProcessor(
        pipeline=pipeline,
        storage=storage,
        depth_pub=SimpleNamespace(publish=published.append),
        gie_id=2,
        failure_callback=failures.append,
    )
    processor._to_numpy = lambda tensor: tensor

    assert processor._enqueue_async_job(_job(module)) is True
    with pytest.raises(RuntimeError, match="async postprocess is poisoned"):
        processor.shutdown(wait=True, timeout_s=2.0)

    assert published == []
    assert depth_frames == []
    assert len(failures) == 1
    assert isinstance(failures[0], TimeoutError)
    assert processor.async_shutdown_quiesced() is True
    assert "durable depth commit timed out" in str(processor._async_failure)


@pytest.mark.parametrize("module", HOOK_MODULES, ids=lambda module: module.__name__)
def test_storage_commit_timeout_configuration_is_strict_and_shared(
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S", "nan")
    with pytest.raises(ValueError, match="must be a finite number"):
        _processor(module)

    monkeypatch.setenv("NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S", "999")
    processor = _processor(module)
    assert processor._storage_commit_timeout_s == 60.0
