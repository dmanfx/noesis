from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

import tools.mapanything_phone_scan.static_capture_sources as sources
from tools.mapanything_phone_scan.static_capture_sources import (
    CanonicalTrackingRecorder,
    ObserverLimits,
    StaticCaptureObserver,
    StaticCaptureSourceError,
    build_static_capture_provenance,
    canonical_timing_provenance,
    discover_active_runtime_pid,
    list_active_camera_sources,
    resolve_active_source_authority,
)


def _configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dewarp = tmp_path / "dewarp.txt"
    dewarp.write_text(
        "[property]\noutput-width=1920\noutput-height=1080\n[surface0]\n",
        encoding="utf-8",
    )
    pipeline = tmp_path / "infer.yaml"
    pipeline.write_text(
        "version: 1\n"
        "sources:\n"
        "  - uri_secret: living-room\n"
        "    dewarper:\n"
        "      enable: true\n"
        f"      config-file: {dewarp.name}\n"
        "  - uri_secret: kitchen\n"
        "    dewarper:\n"
        "      enable: false\n",
        encoding="utf-8",
    )
    cameras = tmp_path / "cameras.yaml"
    cameras.write_text(
        "version: 1\n"
        "cameras:\n"
        "  0:\n"
        "    name: living-room\n"
        "    model: g3_rectified\n"
        "  1:\n"
        "    name: kitchen\n"
        "    model: g3_rectified\n",
        encoding="utf-8",
    )
    return pipeline, cameras, dewarp


def _authority(tmp_path: Path):
    pipeline, cameras, _ = _configs(tmp_path)
    return resolve_active_source_authority(
        "living-room",
        pipeline_config=pipeline,
        cameras_config=cameras,
        camera_registry={
            "living-room": "rtsps://user:password@example.test/live",
            "kitchen": "rtsp://example.test/kitchen",
        },
    )


def test_source_authority_uses_selected_active_config_and_hides_uri(tmp_path: Path) -> None:
    authority = _authority(tmp_path)

    assert authority.source_id == 0
    assert authority.camera_id == "living-room"
    assert authority.private_uri.startswith("rtsps://")
    public = authority.public_snapshot()
    rendered = json.dumps(public, sort_keys=True)
    assert "user:password" not in rendered
    assert "example.test" not in rendered
    assert public["source_provenance"] == "camera-secret:living-room"
    assert public["dewarper"]["raw_rtsp_pixels"]["coordinate_space"] == "camera_raw_rtsp_pixels"
    assert public["dewarper"]["canonical_tracker_pixels"]["coordinate_space"] == "post_dewarper_streammux_pixels"
    assert public["dewarper"]["canonical_tracker_pixels"]["output_resolution_px"] == [1920, 1080]
    assert "user:password" not in repr(authority)


def test_runtime_pid_discovery_is_strict_and_process_config_can_be_selected(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "1234").mkdir(parents=True)
    (proc / "1234" / "cmdline").write_bytes(
        b"python\0/home/mayor/Noesis_Devel/DS9/noesis/ds9_runtime.py\0"
        b"--pipeline-config\0/tmp/effective.yaml\0"
    )
    assert discover_active_runtime_pid(proc_root=proc) == 1234

    (proc / "5678").mkdir()
    (proc / "5678" / "cmdline").write_bytes(
        b"python\0/home/mayor/Noesis_Devel/DS9/noesis/ds9_runtime.py\0"
    )
    with pytest.raises(StaticCaptureSourceError, match="multiple"):
        discover_active_runtime_pid(proc_root=proc)


def test_source_resolution_rejects_missing_private_uri_without_fallback(tmp_path: Path) -> None:
    pipeline, cameras, _ = _configs(tmp_path)
    with pytest.raises(StaticCaptureSourceError, match="private camera credential"):
        resolve_active_source_authority(
            "living-room",
            pipeline_config=pipeline,
            cameras_config=cameras,
            camera_registry={},
        )


def test_camera_inventory_is_public_and_marks_unavailable_sources(tmp_path: Path) -> None:
    pipeline, cameras, _ = _configs(tmp_path)
    inventory = list_active_camera_sources(
        pipeline_config=pipeline,
        cameras_config=cameras,
        camera_registry={"living-room": "rtsp://example.test/living"},
    )
    assert inventory["available"] is True
    assert [row["camera_id"] for row in inventory["cameras"]] == ["living-room", "kitchen"]
    assert inventory["cameras"][0]["available"] is True
    assert inventory["cameras"][1]["available"] is False
    assert "uri" not in json.dumps(inventory)


def _message(kind: str, source_id: int, frame_id: int) -> dict[str, object]:
    return {
        "type": kind,
        "source_id": source_id,
        "frame_id": frame_id,
        "cohort": {
            "source_id": source_id,
            "frame_id": frame_id,
            "observed_at_us": 1_700_000_000_000_000 + frame_id,
            "tracking_publication_sequence": frame_id,
        },
        "tracks": [],
    }


def test_observer_filters_source_and_preserves_exact_fifo_envelopes(tmp_path: Path) -> None:
    output = tmp_path / "tracking.jsonl"
    observer = StaticCaptureObserver(output, selected_source_id=0)
    observer.start()
    assert observer.observe(_message("tracking", 1, 99)) is False
    first = _message("tracking", 0, 4)
    second = _message("world_snapshot", 0, 5)
    bev = {
        "type": "bev-frame",
        "sourceId": 0,
        "cohort": {"source_id": 0, "frame_id": 5},
        "displayBounds": {"xMin": -1, "xMax": 1},
    }
    assert observer.observe(first, received_monotonic_ns=11, received_unix_ns=22, received_utc="2026-01-01T00:00:00Z")
    assert observer.observe(second, received_monotonic_ns=12, received_unix_ns=23, received_utc="2026-01-01T00:00:01Z")
    assert observer.observe(bev, received_monotonic_ns=13, received_unix_ns=24, received_utc="2026-01-01T00:00:02Z")
    status = observer.stop()
    assert status["partial"] is False
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["observer_sequence"] for row in rows] == [0, 1, 2]
    assert [row["message_type"] for row in rows] == ["tracking", "world_snapshot", "bev-frame"]
    assert rows[0]["received_monotonic_ns"] == "11"
    assert rows[0]["message"] == first
    assert rows[2]["message"] == bev


def test_observer_keeps_initial_evidence_without_source_join(tmp_path: Path) -> None:
    output = tmp_path / "tracking.jsonl"
    observer = StaticCaptureObserver(
        output,
        selected_source_id=0,
        initial_provenance={
            "calibration_bundle": {"frame": "backend_world_m", "token": "drop-me"},
            "runtime": {"run_id": "run-1"},
        },
    )
    observer.start()
    assert observer.observe({"type": "calibration-bundle", "payload": {"frame": "backend_world_m"}})
    assert observer.observe({"type": "stats", "sources": {"0": {"frames": 4}}})
    observer.stop()
    snapshot = observer.snapshot()
    assert snapshot["initial_provenance"]["calibration_bundle"]["frame"] == "backend_world_m"
    assert "token" not in json.dumps(snapshot)
    assert snapshot["timing"]["media_pts_ns"]["camera_epoch_verified"] is False


def test_observer_record_limit_marks_partial_without_raising(tmp_path: Path) -> None:
    observer = StaticCaptureObserver(
        tmp_path / "tracking.jsonl",
        selected_source_id=0,
        limits=ObserverLimits(max_records=1, max_queue_items=4, max_queue_bytes=4096, max_file_bytes=4096),
    )
    observer.start()
    assert observer.observe(_message("tracking", 0, 1))
    assert observer.observe(_message("tracking", 0, 2)) is False
    status = observer.stop()
    assert status["partial"] is True
    assert "observer_record_limit_exceeded" in status["partial_reasons"]


class _BrokenWriter:
    def write(self, _payload: bytes) -> None:
        raise OSError("disk full")

    def flush(self) -> None:
        return None


def test_observer_writer_failure_is_observer_local_and_visible(tmp_path: Path) -> None:
    observer = StaticCaptureObserver(
        tmp_path / "unused.jsonl",
        selected_source_id=0,
        writer=_BrokenWriter(),
    )
    observer.start()
    assert observer.observe(_message("tracking", 0, 1))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and observer.status()["writer_error"] is None:
        time.sleep(0.01)
    status = observer.stop("writer_failure")
    assert status["partial"] is True
    assert status["writer_error"] == "OSError"


class _BlockedWriter:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def write(self, _payload: bytes) -> None:
        self.started.set()
        self.release.wait(timeout=5.0)

    def flush(self) -> None:
        return None


def test_blocked_writer_does_not_hold_observer_state_lock(tmp_path: Path) -> None:
    writer = _BlockedWriter()
    observer = StaticCaptureObserver(
        tmp_path / "unused.jsonl",
        selected_source_id=0,
        limits=ObserverLimits(
            max_queue_items=4,
            max_queue_bytes=4096,
            max_file_bytes=4096,
            writer_join_timeout_s=0.05,
        ),
        writer=writer,
    )
    observer.start()
    assert observer.observe(_message("tracking", 0, 1))
    assert writer.started.wait(timeout=1.0)
    started = time.monotonic()
    observer.status()
    assert time.monotonic() - started < 0.2
    started = time.monotonic()
    stopped = observer.stop("writer_failure")
    assert time.monotonic() - started < 0.5
    assert stopped["partial"] is True
    assert "observer_writer_join_timeout" in stopped["partial_reasons"]
    writer.release.set()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and observer.status()["queue_items"]:
        time.sleep(0.01)
    assert observer.status()["bytes_queued"] == 0


def test_observer_disconnect_marks_partial_and_stops_acceptance(tmp_path: Path) -> None:
    observer = StaticCaptureObserver(tmp_path / "tracking.jsonl", selected_source_id=0)
    observer.start()
    assert observer.handle_disconnect()["partial"] is True
    assert observer.observe(_message("tracking", 0, 1)) is False
    assert observer.status()["state"] == "stopped"


def test_provenance_separates_canonical_and_presentation_frames(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    provenance = build_static_capture_provenance(
        authority,
        calibration_bundle={"world_frame": {"frame_id": "backend_world_m", "revision": "r1"}},
        coordinate_frame={
            "canonical_tracking_world": "backend_world_m",
            "presentation_scene_frame": "menon_scene",
            "authority": "canonical_tracking_world",
        },
    )
    assert provenance["coordinate_frame"]["canonical_tracking_world"] == "backend_world_m"
    assert provenance["coordinate_frame"]["presentation_scene_frame"] == "menon_scene"
    assert provenance["coordinate_frame"]["authority"] == "canonical_tracking_world"
    assert canonical_timing_provenance()["observed_at_us"]["status"] == "estimated"


class _FakeRuntimeWebSocket:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages = [json.dumps(message, separators=(",", ":")) for message in messages]
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _tb):
        self.closed = True

    async def recv(self):
        if self.messages:
            return self.messages.pop(0)
        while not self.closed:
            await asyncio.sleep(0.01)
        raise RuntimeError("fake websocket closed")


def test_canonical_recorder_captures_authenticated_selected_tracking_and_rest_start_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path)
    websocket = _FakeRuntimeWebSocket(
        [
            {"type": "calibration-bundle", "data": {"world_frame": "backend_world_m"}},
            {"type": "stats", "payload": {"runtime_epoch": "e1"}},
            {"type": "tracking", "source_id": 1, "camera_id": "kitchen", "tracks": []},
            {
                "type": "tracking",
                "source_id": 0,
                "camera_id": "living-room",
                "tracks": [],
                "tracking_publication_sequence": 7,
                "cohort": {"source_id": 0, "tracking_publication_sequence": 7},
            },
        ]
    )
    monkeypatch.setattr(sources, "_load_runtime_auth", lambda _authority: object())
    monkeypatch.setattr(sources, "_runtime_endpoints", lambda _authority: ("ws://fake", "http://fake"))
    monkeypatch.setattr(sources, "_connect_authenticated_websocket", lambda _url, _auth: websocket)
    monkeypatch.setattr(
        sources,
        "_fetch_authenticated_rest_json",
        lambda url, _auth: {"endpoint": url.rsplit("/", 1)[-1], "frame": "backend_world_m"},
    )

    recorder = CanonicalTrackingRecorder(authority, tmp_path / "capture", 32)
    recorder.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not recorder.status()["tracking_ready"]:
        time.sleep(0.01)
    status = recorder.status()
    assert status["status"] == "recording"
    assert status["tracking_ready"] is True
    assert status["record_count"] == 1
    assert status["tracked_count"] == 0
    assert status["last_tracking_publication_sequence"] == 7
    assert status["calibration_seen"] is True
    stopped = recorder.stop("user")
    assert stopped["status"] == "stopped"
    assert stopped["partial"] is False

    rows = [json.loads(line) for line in (tmp_path / "capture" / "tracking.ndjson").read_text().splitlines()]
    assert [row["message_type"] for row in rows] == ["calibration-bundle", "stats", "tracking"]
    provenance = json.loads((tmp_path / "capture" / "provenance.json").read_text())
    assert provenance["coordinate_frame"]["canonical_tracking_world"] == "backend_world_m"
    assert provenance["coordinate_frame"]["presentation_scene_frame"] == "menon_scene"
    assert "example.test" not in json.dumps(provenance)
    assert all((tmp_path / "capture" / name).is_file() for name in status["artifacts"].values())


def test_canonical_recorder_stops_partial_when_selected_tracking_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path)
    websocket = _FakeRuntimeWebSocket([])
    monkeypatch.setattr(sources, "TRACKING_STALE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(sources, "_load_runtime_auth", lambda _authority: object())
    monkeypatch.setattr(sources, "_runtime_endpoints", lambda _authority: ("ws://fake", "http://fake"))
    monkeypatch.setattr(sources, "_connect_authenticated_websocket", lambda _url, _auth: websocket)
    monkeypatch.setattr(sources, "_fetch_authenticated_rest_json", lambda _url, _auth: {"ok": True})

    recorder = CanonicalTrackingRecorder(authority, tmp_path / "stale", 8)
    recorder.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not recorder.status()["partial"]:
        time.sleep(0.01)
    status = recorder.status()
    assert status["partial"] is True
    assert status["status"] == "failed"
    assert status["error"] == "tracking_stale_timeout"
    recorder.stop("user")
