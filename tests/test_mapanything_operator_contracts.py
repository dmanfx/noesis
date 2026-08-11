from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from noesis.pipelines import hooks


REPO_ROOT = Path(__file__).resolve().parents[1]
IS_DS9_ADAPTER = Path(hooks.__file__).resolve().is_relative_to(
    (REPO_ROOT / "DS9").resolve()
)


class _FakeTensorOutput:
    def __init__(self, unique_id: int) -> None:
        self.unique_id = int(unique_id)


class _FakeTensorItem:
    def __init__(self, unique_id: int) -> None:
        self._meta = _FakeTensorOutput(unique_id)

    def as_tensor_output(self) -> _FakeTensorOutput:
        return self._meta


class _FakeFrameMeta:
    def __init__(self, ids: list[int], frame_number: int = 1) -> None:
        self.tensor_items = [_FakeTensorItem(i) for i in ids]
        self.frame_number = int(frame_number)


class _FakeBatchMeta:
    def __init__(self, frames: list[_FakeFrameMeta]) -> None:
        self.frame_items = list(frames)


class _ProcessorStub:
    def __init__(self, gie_id: int) -> None:
        self.gie_id = int(gie_id)
        self.pipeline = SimpleNamespace(depth_enabled=True)
        self.calls: list[tuple[Any, Any]] = []
        self.native_calls: list[tuple[Any, bool]] = []
        self.native_result: object | None = None

    def handle_nvds_tensor_ds9(self, frame_meta: Any, tensor_meta: Any) -> None:
        self.calls.append((frame_meta, tensor_meta))

    def handle_nvds_tensor_ds8(self, frame_meta: Any, tensor_meta: Any) -> None:
        self.calls.append((frame_meta, tensor_meta))

    def handle_native_frame_ds8(self, frame_meta: Any) -> object | None:
        self.native_calls.append((frame_meta, False))
        return self.native_result

    def handle_native_frame_ds9(self, frame_meta: Any) -> object:
        self.native_calls.append((frame_meta, True))
        if self.native_result is None:
            raise RuntimeError("missing exact native tensor UID")
        return self.native_result


def _operator(processor: _ProcessorStub) -> Any:
    """Construct the probe shell without requiring the DeepStream base class."""

    operator_type = hooks._MapAnythingOperator  # type: ignore[attr-defined]
    operator = operator_type.__new__(operator_type)
    operator._processor = processor
    operator._frames_seen = 0
    operator._matched_frames = 0
    operator._warned_no_tensors = False
    operator._warned_no_match = False
    operator._warned_native_probe = False
    return operator


def test_mapanything_operator_drops_mismatched_gie_ids(monkeypatch) -> None:
    hooks.reset_core_path_instrumentation()
    monkeypatch.setattr(hooks, "noesis_depth_tracking_tensor_ext", None)
    processor = _ProcessorStub(gie_id=5)
    operator = _operator(processor)

    if IS_DS9_ADAPTER:
        try:
            operator.handle_metadata(_FakeBatchMeta([_FakeFrameMeta([1, 2, 3])]))
        except RuntimeError as exc:
            assert "missing exact native tensor UID" in str(exc)
        else:
            raise AssertionError("DS9 missing UID did not fail closed")
    else:
        operator.handle_metadata(_FakeBatchMeta([_FakeFrameMeta([1, 2, 3])]))

    assert processor.calls == []
    snap = hooks.get_core_path_instrumentation_snapshot()
    counters = snap.get("counters", {})
    if IS_DS9_ADAPTER:
        assert int(counters.get("mapanything_exact_native_capture_failures_total", 0)) == 1
        assert int(counters.get("tensor_gie_mismatch_drops_total.mapanything", 0)) == 0
    else:
        assert int(counters.get("tensor_gie_mismatch_drops_total.mapanything", 0)) == 1


def test_mapanything_operator_processes_matching_gie_id() -> None:
    hooks.reset_core_path_instrumentation()
    processor = _ProcessorStub(gie_id=9)
    if IS_DS9_ADAPTER:
        processor.native_result = object()
    operator = _operator(processor)

    operator.handle_metadata(_FakeBatchMeta([_FakeFrameMeta([7, 9, 11])]))

    if IS_DS9_ADAPTER:
        assert processor.calls == []
        assert len(processor.native_calls) == 1
        assert processor.native_calls[0][1] is True
    else:
        assert len(processor.calls) == 1
        _, tensor_meta = processor.calls[0]
        assert int(getattr(tensor_meta, "unique_id", -1)) == 9


def test_mapanything_operator_uses_native_tensor_path_on_service_maker_gie_mismatch(monkeypatch) -> None:
    hooks.reset_core_path_instrumentation()
    monkeypatch.setattr(hooks, "noesis_depth_tracking_tensor_ext", object())
    processor = _ProcessorStub(gie_id=2)
    processor.native_result = object()
    operator = _operator(processor)
    frame = _FakeFrameMeta([5, 5])

    operator.handle_metadata(_FakeBatchMeta([frame]))

    assert processor.calls == []
    assert processor.native_calls == [(frame, IS_DS9_ADAPTER)]
    assert operator._matched_frames == 1  # type: ignore[attr-defined]
    snap = hooks.get_core_path_instrumentation_snapshot()
    counters = snap.get("counters", {})
    # Native capture satisfied the frame. DS8 retains its historical wrapper
    # diagnostic; DS9 never uses wrapper IDs as the ownership selector.
    assert int(counters.get("tensor_gie_mismatch_drops_total.mapanything", 0)) == 0


def test_mapanything_postprocess_is_async_only() -> None:
    processor = hooks.MapAnythingProcessor(
        pipeline=SimpleNamespace(depth_enabled=True, frame_size=(1920, 1080)),
        storage=SimpleNamespace(),
        depth_pub=None,
        gie_id=4,
    )
    assert processor._async_enabled is True  # type: ignore[attr-defined]
