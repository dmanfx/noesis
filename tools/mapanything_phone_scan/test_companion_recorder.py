from __future__ import annotations

import json
import queue
import sys
import time
import types
from pathlib import Path

import pytest

from tools.mapanything_phone_scan.companion_capture import GstCompressedRecorder


class _Fraction:
    numerator = 30
    denominator = 1


class _Structure:
    def __init__(self, name: str, values: dict[str, object] | None = None) -> None:
        self._name = name
        self.values = values or {}

    def get_name(self) -> str:
        return self._name

    def get_string(self, key: str) -> str | None:
        value = self.values.get(key)
        return value if isinstance(value, str) else None

    def get_value(self, key: str) -> object | None:
        return self.values.get(key)


class _Caps:
    def __init__(self, structure: _Structure, text: str) -> None:
        self.structure = structure
        self.text = text

    def get_size(self) -> int:
        return 1

    def get_structure(self, _index: int) -> _Structure:
        return self.structure

    def to_string(self) -> str:
        return self.text


class _Pad:
    def __init__(self, caps: _Caps | None = None) -> None:
        self.caps = caps
        self.linked = False
        self.probe = None

    def is_linked(self) -> bool:
        return self.linked

    def link(self, _sink: "_Pad") -> bool:
        self.linked = True
        return True

    def get_current_caps(self) -> _Caps | None:
        return self.caps

    def query_caps(self, _filter: object) -> _Caps | None:
        return self.caps

    def add_probe(self, _probe_type: object, callback: object) -> int:
        self.probe = callback
        return 1


class _Element:
    def __init__(self, name: str, parser_caps: _Caps | None = None) -> None:
        self.name = name
        self.properties: dict[str, object] = {}
        self.callbacks: dict[str, object] = {}
        self.sink_pad = _Pad()
        self.src_pad = _Pad(parser_caps)

    def set_property(self, key: str, value: object) -> None:
        self.properties[key] = value

    def find_property(self, key: str) -> object:
        return object() if key in {"add-reference-timestamp-meta", "offset-to-zero", "config-interval"} else None

    def link(self, _other: "_Element") -> bool:
        return True

    def connect(self, signal: str, callback: object) -> None:
        self.callbacks[signal] = callback

    def get_static_pad(self, name: str) -> _Pad:
        return self.sink_pad if name == "sink" else self.src_pad


class _Clock:
    def get_time(self) -> int:
        return 10_000


class _Message:
    def __init__(self, message_type: object) -> None:
        self.type = message_type


class _Bus:
    def __init__(self) -> None:
        self.messages: queue.Queue[_Message] = queue.Queue()

    def timed_pop_filtered(self, _timeout: int, _types: object) -> _Message | None:
        try:
            return self.messages.get(timeout=0.05)
        except queue.Empty:
            return None


class _Pipeline:
    def __init__(self, bus: _Bus) -> None:
        self.bus = bus
        self.state = None
        self.elements: list[_Element] = []

    def add(self, element: _Element) -> None:
        self.elements.append(element)

    def get_bus(self) -> _Bus:
        return self.bus

    def get_clock(self) -> _Clock:
        return _Clock()

    def get_base_time(self) -> int:
        return 1_000

    def set_state(self, state: object) -> str:
        self.state = state
        return "async"

    def send_event(self, event: object) -> bool:
        del event
        self.bus.messages.put(_Message(_Gst.MessageType.EOS))
        return True


class _PipelineFactory:
    current: _Pipeline | None = None

    @staticmethod
    def new(_name: str) -> _Pipeline:
        _PipelineFactory.current = _Pipeline(_Bus())
        return _PipelineFactory.current


class _ElementFactory:
    created: dict[str, _Element] = {}

    @classmethod
    def make(cls, element_name: str, _instance_name: str) -> _Element:
        parser_caps = _Caps(
            _Structure(
                "video/x-h264",
                {
                    "media": "video",
                    "encoding-name": "H264",
                    "width": 1920,
                    "height": 1080,
                    "framerate": _Fraction(),
                    "stream-format": "byte-stream",
                    "alignment": "au",
                },
            ),
            "video/x-h264, width=(int)1920, height=(int)1080, framerate=(fraction)30/1",
        )
        element = _Element(element_name, parser_caps if element_name == "h264parse" else None)
        cls.created[element_name] = element
        return element


class _Event:
    @staticmethod
    def new_eos() -> object:
        return object()


class _Segment:
    format = "time"
    start = 100
    stop = 10_000
    time = 0
    base = 7
    offset = 0
    position = 100
    rate = 1.0
    applied_rate = 1.0

    def to_running_time(self, _format: object, pts: int) -> int:
        return pts - self.start + self.base


class _Info:
    def __init__(self, *, event: object | None = None, buffer: object | None = None) -> None:
        self.event = event
        self.buffer = buffer

    def get_event(self) -> object | None:
        return self.event

    def get_buffer(self) -> object | None:
        return self.buffer


class _SegmentEvent:
    type = "segment"

    @staticmethod
    def parse_segment() -> _Segment:
        return _Segment()


class _ReferenceCaps:
    def to_string(self) -> str:
        return "timestamp/x-rtcp, clock-rate=(int)90000"


class _ReferenceMeta:
    timestamp = 4_000
    duration = 33
    reference = _ReferenceCaps()


class _Buffer:
    pts = 133
    dts = 120
    duration = 33

    def __init__(self, *, keyframe: bool = True, reference: bool = True) -> None:
        self.flags = 0 if keyframe else 1
        self.reference = reference
        self.reference_args: list[object] = []

    def get_flags(self) -> int:
        return self.flags

    def get_reference_timestamp_meta(self, caps: object) -> _ReferenceMeta | None:
        self.reference_args.append(caps)
        return _ReferenceMeta() if self.reference else None


class _GstModule:
    CLOCK_TIME_NONE = 2**64 - 1
    MSECOND = 1
    Pipeline = _PipelineFactory
    ElementFactory = _ElementFactory
    Event = _Event
    Format = types.SimpleNamespace(TIME="time")
    State = types.SimpleNamespace(PLAYING="playing", NULL="null")
    StateChangeReturn = types.SimpleNamespace(FAILURE="failure")
    PadProbeType = types.SimpleNamespace(BUFFER=1, EVENT_DOWNSTREAM=2)
    PadProbeReturn = types.SimpleNamespace(OK="ok")
    BufferFlags = types.SimpleNamespace(DELTA_UNIT=1)
    EventType = types.SimpleNamespace(SEGMENT="segment")
    MessageType = types.SimpleNamespace(ERROR=1, EOS=2, STATE_CHANGED=4)

    @staticmethod
    def init(_args: object) -> None:
        return None


_Gst = _GstModule()


class _BrokenHandle:
    def write(self, _payload: str) -> None:
        raise OSError("writer failed for rtsp://user:secret@camera.example/live")

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _BrokenPacketPath:
    name = "packet_timing.jsonl"

    def open(self, *_args: object, **_kwargs: object) -> _BrokenHandle:
        return _BrokenHandle()

    def is_file(self) -> bool:
        return False


def _install_fake_gi(monkeypatch: pytest.MonkeyPatch) -> None:
    gi = types.ModuleType("gi")
    gi.require_version = lambda _name, _version: None
    repository = types.ModuleType("gi.repository")
    repository.Gst = _Gst
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)
    _ElementFactory.created.clear()


def _wait_for(predicate: object, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_recorder_accepts_only_video_h264_and_preserves_parser_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_gi(monkeypatch)
    recorder = GstCompressedRecorder("rtsp://camera.example/live", tmp_path, stale_timeout_s=5.0)
    recorder.start()
    source = _ElementFactory.created["rtspsrc"]
    audio_pad = _Pad(_Caps(_Structure("application/x-rtp", {"media": "audio", "encoding-name": "OPUS"}), "audio"))
    video_pad = _Pad(_Caps(_Structure("application/x-rtp", {"media": "video", "encoding-name": "H264"}), "video"))
    source.callbacks["pad-added"](source, audio_pad)
    source.callbacks["pad-added"](source, video_pad)
    assert audio_pad.linked is False
    assert video_pad.linked is True

    parser_pad = _ElementFactory.created["h264parse"].src_pad
    assert parser_pad.probe is not None
    parser_pad.probe(parser_pad, _Info(event=_SegmentEvent()))
    buffer = _Buffer()
    parser_pad.probe(parser_pad, _Info(buffer=buffer))
    _wait_for(lambda: recorder.status()["packet_count"] == 1)
    status = recorder.status()
    assert status["encoded_ready"] is True
    assert status["partial"] is False
    recorder.stop("user")
    row = json.loads((tmp_path / "packet_timing.jsonl").read_text().splitlines()[0])
    assert row["pts_ns"] == 133
    assert row["running_time_ns"] == 40
    assert row["pipeline_clock"]["base_time_ns"] == 1000
    assert row["parser_caps"]["width"] == 1920
    assert row["parser_caps"]["height"] == 1080
    assert row["parser_caps"]["framerate"] == {"numerator": 30, "denominator": 1}
    assert row["keyframe"] is True
    assert row["reference_timestamp_meta"]["caps"].startswith("timestamp/x-rtcp")
    assert buffer.reference_args == [None]


def test_recorder_stale_video_sets_partial_and_cleans_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_gi(monkeypatch)
    recorder = GstCompressedRecorder("rtsp://camera.example/live", tmp_path, stale_timeout_s=0.05)
    recorder.start()
    _wait_for(lambda: recorder.status()["error"] == "video_stale_timeout")
    status = recorder.status()
    assert status["status"] == "failed"
    assert status["partial"] is True
    assert _PipelineFactory.current is not None
    assert _PipelineFactory.current.state == _Gst.State.NULL
    recorder.stop("user")


def test_recorder_packet_limit_surfaces_failure_without_overflowing_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_gi(monkeypatch)
    recorder = GstCompressedRecorder("rtsp://camera.example/live", tmp_path, max_packet_records=1)
    recorder._gst = _Gst
    recorder._record_packet(_Buffer(reference=False))
    recorder._record_packet(_Buffer(reference=False))
    status = recorder.status()
    assert status["packet_count"] == 1
    assert status["dropped_packet_records"] == 1
    assert status["partial"] is True
    assert status["error"] == "max_packet_records"
    recorder.stop("user")


def test_recorder_packet_writer_failure_is_redacted_and_stops_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_gi(monkeypatch)
    recorder = GstCompressedRecorder("rtsp://camera.example/live", tmp_path)
    recorder.packet_path = _BrokenPacketPath()  # type: ignore[assignment]
    recorder.start()
    recorder._record_packet(_Buffer(reference=False))
    _wait_for(lambda: recorder.status()["error"] is not None)
    status = recorder.status()
    assert status["status"] == "failed"
    assert status["partial"] is True
    assert str(status["error"]).startswith("packet_writer_failure:")
    assert "user:secret" not in str(status["error"])
    assert _PipelineFactory.current is not None
    _wait_for(lambda: _PipelineFactory.current.state == _Gst.State.NULL)
    recorder.stop("user")


def test_recorder_packet_fifo_overflow_is_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_gi(monkeypatch)
    recorder = GstCompressedRecorder("rtsp://camera.example/live", tmp_path)
    recorder._gst = _Gst
    recorder._packet_queue = queue.Queue(maxsize=1)
    recorder._record_packet(_Buffer(reference=False))
    recorder._record_packet(_Buffer(reference=False))
    status = recorder.status()
    assert status["packet_count"] == 1
    assert status["dropped_packet_records"] == 1
    assert status["error"] == "packet_timing_queue_overflow"
    recorder.stop("user")
