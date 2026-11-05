"""Optional bounded metadata handoff; no SDK references escape the callback."""
from __future__ import annotations

import math
import queue
import threading
import time
from urllib.parse import quote

from noesis_core.contracts.object_observations import (
    ObjectImageBox, ObjectImageObservation, ObjectObservationFrame,
    TTL_MS, MAX_CAMERAS, MAX_FRAME_ROWS, MAX_BYTES,
)


def _integer(value, minimum=0, maximum=2**53 - 1):
    return type(value) is int and minimum <= value <= maximum


def _text(value, limit=160):
    return isinstance(value, str) and 0 < len(value) <= limit and not any(ord(c) < 32 or ord(c) == 127 or 0xD800 <= ord(c) <= 0xDFFF for c in value)


class _FrameCapture:
    def __init__(self, owner, frame, camera_id, source_id, source_epoch, frame_id, temporal):
        self.owner = owner
        self.generation = owner._drop_generation
        self.accepted = owner._monotonic()
        self.camera_id, self.source_id, self.source_epoch, self.frame_id = camera_id, source_id, source_epoch, frame_id
        self.width, self.height = frame.pipeline_width, frame.pipeline_height
        self.source_width, self.source_height = frame.source_width, frame.source_height
        self.temporal = (temporal["observed_at_us"], temporal["captured_at_us"], temporal["capture_time_status"], temporal["media_pts_ns"])
        self.rows = []
        self.track_rows = []
        self.untracked = 0
        self.terminations = ()
        self.termination_available = False
        self.rooms = {}
        self.inspected = 0
        self.omitted = self.invalid = 0
        self.submitted = False

    def capture(self, obj, ordinal, room_resolver=None):
        """At most 32 scalar copies in the existing single object iteration."""
        try:
            class_id = obj.class_id
            if type(class_id) is int and class_id == 0:
                return
            if self.inspected >= MAX_FRAME_ROWS:
                self.omitted += 1
                return
            self.inspected += 1
            if not _integer(class_id, 1, 79) or not _integer(ordinal):
                self.invalid += 1
                return
            confidence, label, rect = obj.confidence, obj.label, obj.rect_params
            if not _text(label):
                self.invalid += 1
                return
            bbox = (rect.left, rect.top, rect.width, rect.height)
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox):
                self.invalid += 1
                return
            left, top, width, height = bbox
            if left < 0 or top < 0 or width <= 0 or height <= 0 or left + width > self.width or top + height > self.height:
                self.invalid += 1
                return
            native = getattr(obj, "object_id", None)
            valid_confidence = type(confidence) in (int, float) and math.isfinite(confidence) and 0 < confidence <= 1
            tracker = getattr(obj, "tracker_confidence", None)
            tracker = float(tracker) if type(tracker) in (int, float) and math.isfinite(tracker) and 0 <= tracker <= 1 else None
            if type(native) is int and 0 <= native < 2**64 - 1:
                self.track_rows.append((ordinal, str(native), class_id, str(label), float(confidence) if valid_confidence else None, tracker, tuple(float(v) for v in bbox)))
                room = room_resolver(obj) if room_resolver is not None else None
                if _text(room): self.rooms[str(native)] = room
            else:
                self.untracked += 1
            if not valid_confidence:
                self.invalid += 1
                return
            self.rows.append((ordinal, class_id, str(label), float(confidence), tuple(float(v) for v in bbox)))
        except Exception:
            self.invalid += 1

    def capture_lifecycle(self, evidence):
        # The native extractor already binds this to a complete-buffer source
        # barrier; copy only bounded scalars, never SDK metadata.
        if not isinstance(evidence, dict) or evidence.get("lifecycle_unavailable"):
            return
        if any(type(evidence.get(k)) is not int or evidence[k] != v for k,v in
               (("source_id", self.source_id), ("frame_id", self.frame_id), ("media_pts_ns", self.temporal[3]))):
            return
        rows = evidence.get("object_terminated")
        if (evidence.get("terminated_available") is not True or evidence.get("termination_boundary") != "complete_buffer_source_barrier"
            or not isinstance(rows, (list,tuple)) or len(rows)>256): return
        if any(not isinstance(r,dict) or not _integer(r.get("tracker_id"),0,2**64-2) or not _integer(r.get("class_id"),1,79) for r in rows): return
        if len({r["tracker_id"] for r in rows})!=len(rows): return
        self.terminations=tuple((str(r["tracker_id"]),r["class_id"]) for r in rows)
        self.termination_available=True

    def submit(self):
        if self.submitted:
            return False
        self.submitted = True
        return self.owner._submit(self)


class ObjectObservationService:
    def __init__(self, run_id, *, track_service=None, monotonic=time.monotonic, clock_us=lambda: time.time_ns() // 1000):
        if not _text(run_id):
            raise ValueError("invalid object-observation run")
        self.run_id = run_id
        self.track_service = track_service
        self._monotonic, self._clock_us = monotonic, clock_us
        self._queue = queue.Queue(maxsize=16)
        self._lock = threading.Lock()
        self._frames = {}
        self._closed = False
        self._stop = threading.Event()
        self._drop_generation = self._stored_generation = self._dropped = 0
        self._worker = threading.Thread(target=self._run, name="noesis-object-observations", daemon=True)
        self._worker.start()

    def _drop(self):
        self._dropped += 1
        self._drop_generation += 1
        if self.track_service is not None:
            self.track_service.note_drop(self._drop_generation)

    def begin_frame(self, frame, *, camera_id, source_id, source_epoch, frame_id, temporal):
        try:
            if self._closed:
                return None
            dims = (frame.pipeline_width, frame.pipeline_height, frame.source_width, frame.source_height)
            if not _text(camera_id) or not all(_integer(v, 1, 32768) for v in dims) or not all(_integer(v) for v in (source_id, source_epoch, frame_id)):
                self._drop(); return None
            if not all(_integer(temporal[key], 1) for key in ("observed_at_us", "captured_at_us")) or temporal["capture_time_status"] != "estimated" or not _integer(temporal["media_pts_ns"]):
                self._drop(); return None
            return _FrameCapture(self, frame, camera_id, source_id, source_epoch, frame_id, temporal)
        except Exception:
            self._drop(); return None

    def _submit(self, capture):
        if self._closed:
            return False
        try:
            self._queue.put_nowait(capture)
            return True
        except queue.Full:
            self._drop()
            return False

    def _run(self):
        while not self._stop.is_set():
            try:
                capture = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._store(capture)
            except (ValueError, TypeError, OverflowError):
                self._drop()
            finally:
                self._queue.task_done()

    def _store(self, capture):
        if self._closed or capture.generation != self._drop_generation or self._monotonic() >= capture.accepted + TTL_MS / 1000:
            return
        observed, captured, status, pts = capture.temporal
        prefix = ":".join(quote(str(v), safe="") for v in (self.run_id, capture.camera_id, capture.source_id, capture.source_epoch, capture.frame_id))
        rows = tuple(ObjectImageObservation(observation_id=f"{prefix}:{ordinal}", ordinal=ordinal, class_id=class_id, label=label,
            confidence=confidence, bbox=ObjectImageBox(left=bbox[0], top=bbox[1], width=bbox[2], height=bbox[3]))
            for ordinal, class_id, label, confidence, bbox in capture.rows)
        frame = ObjectObservationFrame(camera_id=capture.camera_id, source_id=capture.source_id, source_epoch=capture.source_epoch,
            frame_id=capture.frame_id, observed_at_us=observed, captured_at_us=captured, capture_time_status=status, media_pts_ns=pts,
            image_width=capture.width, image_height=capture.height, source_image_width=capture.source_width, source_image_height=capture.source_height,
            valid_until_us=observed + TTL_MS * 1000, omitted_count=capture.omitted, invalid_count=capture.invalid, observations=rows)
        size = len(frame.model_dump_json().encode("utf-8"))
        with self._lock:
            if self._closed or capture.generation != self._drop_generation:
                return
            self._frames = {key: row for key, row in self._frames.items() if row[0] == capture.generation and row[1] > self._monotonic()}
            previous = self._frames.get(capture.camera_id)
            if previous and (frame.source_epoch, frame.frame_id) <= (previous[2].source_epoch, previous[2].frame_id):
                return
            other = {key: row for key, row in self._frames.items() if key != capture.camera_id}
            if len(other) >= MAX_CAMERAS or size + sum(row[3] for row in other.values()) > MAX_BYTES - 4096:
                self._drop(); return
            other[capture.camera_id] = (capture.generation, capture.accepted + TTL_MS / 1000, frame, size)
            self._frames = other
            self._stored_generation = capture.generation
        if self.track_service is not None:
            self.track_service.ingest(capture)

    def snapshot(self):
        at_us = self._clock_us()
        valid_clock = _integer(at_us, 1)
        payload = {"contract": "noesis.object-observations", "version": 1, "status": "unknown", "reason": "no_input",
            "run_id": self.run_id, "generated_at_us": at_us if valid_clock else None, "ttl_ms": TTL_MS, "dropped_frames": self._dropped, "cameras": []}
        if not valid_clock:
            payload["reason"] = "invalid_clock"; return payload
        with self._lock:
            generation = self._drop_generation
            if self._closed:
                payload["reason"] = "closed"; return payload
            if generation != self._stored_generation:
                payload["reason"] = "queue_dropped"; return payload
            frames = tuple(row[2] for row in self._frames.values() if row[0] == generation and self._monotonic() < row[1])
            if not frames:
                payload["reason"] = "expired" if self._frames else "no_input"; return payload
            if any(not frame.observed_at_us <= at_us < frame.valid_until_us for frame in frames):
                payload["reason"] = "invalid_clock"; return payload
        payload["cameras"] = [frame.model_dump(mode="json") for frame in frames]
        if generation != self._drop_generation or self._closed:
            payload["reason"] = "owner_changed"; payload["cameras"] = []; return payload
        payload["status"] = "partial" if any(frame.omitted_count or frame.invalid_count for frame in frames) else "available"
        payload["reason"] = "current_camera_image_observations"
        return payload

    def close(self):
        self._closed = True
        self._stop.set()
        self._worker.join(timeout=2)
        if self._worker.is_alive():
            raise RuntimeError("object observation worker did not close")
        with self._lock:
            self._frames.clear()
        while True:
            try:
                self._queue.get_nowait(); self._queue.task_done()
            except queue.Empty:
                break


def bind_object_observation_owner(pipeline, app, service):
    """Startup transaction binding, with exact-owner-only cleanup."""
    pipeline.object_observation_service = service
    if app is not None:
        app.state.object_observation_service = service


def clear_object_observation_owner(pipeline, app, service):
    if getattr(pipeline, "object_observation_service", None) is service:
        del pipeline.object_observation_service
    if app is not None and getattr(app.state, "object_observation_service", None) is service:
        del app.state.object_observation_service


def install_object_observation_owner(transaction, pipeline, app, run_id, *, track_service=None):
    service = transaction.acquire("object_observations", lambda: ObjectObservationService(run_id, track_service=track_service), lambda owned: owned.close())
    transaction.bind("object_observation_binding", lambda: bind_object_observation_owner(pipeline, app, service),
        lambda: clear_object_observation_owner(pipeline, app, service))
    return service


def close_object_observation_owner(pipeline, app, service):
    service.close()
    clear_object_observation_owner(pipeline, app, service)
