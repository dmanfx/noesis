"""Request-gated DS9 capture-event RGB frames from the active pipeline.

The provider never opens a camera and is dormant outside one controller-owned
manual capture.  An offered frame is admitted only under the exact arm token
for its canonical camera and numeric pipeline source.
"""

from __future__ import annotations

import hashlib
import hmac
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Mapping, Sequence

import numpy as np

from noesis_core.capture_event_fusion import RawDepthSnapshot, TimestampedRgbFrame

_MAX_CONFIGURED_CAMERAS = 64
_MAX_CAMERA_ID_BYTES = 128
_MAX_DIMENSION = 16_384
_MAX_FRAME_BYTES_LIMIT = 256 * 1024 * 1024
_MAX_TOTAL_BYTES_LIMIT = 1024 * 1024 * 1024


def _bounded_positive_int(value: object, *, label: str, upper: int) -> int:
    if type(value) is not int or not 1 <= value <= upper:  # noqa: E721
        raise ValueError(f"{label} must be an integer in [1, {upper}]")
    return value


def _configured_camera_sources(
    camera_sources: Mapping[int, str],
) -> tuple[tuple[int, str], ...]:
    if not isinstance(camera_sources, Mapping):
        raise TypeError("camera_sources must map numeric source IDs to camera IDs")
    if not camera_sources:
        raise ValueError("at least one configured camera source is required")
    if len(camera_sources) > _MAX_CONFIGURED_CAMERAS:
        raise ValueError(
            f"configured camera count must not exceed {_MAX_CONFIGURED_CAMERAS}"
        )

    normalized: list[tuple[int, str]] = []
    for source_id, camera_id in camera_sources.items():
        if type(source_id) is not int or source_id < 0:  # noqa: E721
            raise ValueError("configured source IDs must be non-negative integers")
        if not isinstance(camera_id, str):
            raise TypeError("configured camera IDs must be strings")
        camera = camera_id.strip()
        if not camera:
            raise ValueError("configured camera IDs must be non-empty")
        if camera != camera_id:
            raise ValueError("configured camera IDs must not contain outer whitespace")
        if len(camera.encode("utf-8")) > _MAX_CAMERA_ID_BYTES:
            raise ValueError(
                f"configured camera IDs must be at most {_MAX_CAMERA_ID_BYTES} UTF-8 bytes"
            )
        normalized.append((source_id, camera))
    normalized.sort()
    cameras = tuple(camera for _source, camera in normalized)
    if len(set(cameras)) != len(cameras):
        raise ValueError("configured camera IDs must be unique")
    return tuple(normalized)


@dataclass(frozen=True)
class RgbCaptureArm:
    """Opaque transaction token shared by controller and exact buffer probe."""

    camera_id: str
    source_id: int
    generation: int


class PipelineRgbFrameProvider:
    """Bounded, exact-identity RGB retention for one armed manual capture."""

    def __init__(
        self,
        *,
        camera_sources: Mapping[int, str],
        frames_per_camera: int = 8,
        max_width: int = 4096,
        max_height: int = 4096,
        max_frame_bytes: int = 64 * 1024 * 1024,
        max_total_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        configured = _configured_camera_sources(camera_sources)
        self._frames_per_camera = _bounded_positive_int(
            frames_per_camera,
            label="frames_per_camera",
            upper=64,
        )
        self._max_width = _bounded_positive_int(
            max_width,
            label="max_width",
            upper=_MAX_DIMENSION,
        )
        self._max_height = _bounded_positive_int(
            max_height,
            label="max_height",
            upper=_MAX_DIMENSION,
        )
        self._max_frame_bytes = _bounded_positive_int(
            max_frame_bytes,
            label="max_frame_bytes",
            upper=_MAX_FRAME_BYTES_LIMIT,
        )
        self._max_total_bytes = _bounded_positive_int(
            max_total_bytes,
            label="max_total_bytes",
            upper=_MAX_TOTAL_BYTES_LIMIT,
        )
        if self._max_frame_bytes > self._max_total_bytes:
            raise ValueError("max_frame_bytes must not exceed max_total_bytes")

        self._configured_camera_sources = configured
        self._source_to_camera = dict(configured)
        self._camera_to_source = {
            camera: source for source, camera in configured
        }
        self._frames: dict[str, deque[TimestampedRgbFrame]] = {}
        self._retained_bytes = 0
        self._evicted_frames = 0
        self._generation = 0
        self._active_arm: RgbCaptureArm | None = None
        self._arm_total = 0
        self._disarm_total = 0
        self._offer_total = 0
        self._unarmed_capture_checks = 0
        self._closed = False
        self._lock = RLock()

    def arm(self, camera_id: str) -> RgbCaptureArm:
        camera = str(camera_id or "").strip()
        with self._lock:
            if self._closed:
                raise RuntimeError("pipeline_rgb_provider_closed")
            source_id = self._camera_to_source.get(camera)
            if source_id is None:
                raise ValueError("pipeline RGB camera is not configured")
            if self._active_arm is not None:
                raise RuntimeError("pipeline_rgb_provider_already_armed")
            self._frames.clear()
            self._retained_bytes = 0
            self._generation += 1
            arm = RgbCaptureArm(
                camera_id=camera,
                source_id=source_id,
                generation=self._generation,
            )
            self._active_arm = arm
            self._arm_total += 1
            return arm

    def capture_arm(
        self,
        *,
        source_id: int,
        camera_id: str,
    ) -> RgbCaptureArm | None:
        """Return the matching live arm before any RGB conversion or D2H."""

        with self._lock:
            if self._closed:
                raise RuntimeError("pipeline_rgb_provider_closed")
            arm = self._active_arm
            if (
                arm is None
                or arm.source_id != source_id
                or arm.camera_id != camera_id
            ):
                self._unarmed_capture_checks += 1
                return None
            return arm

    def offer(self, frame: TimestampedRgbFrame, *, arm: RgbCaptureArm) -> None:
        if not isinstance(frame, TimestampedRgbFrame):
            raise TypeError("frame must be TimestampedRgbFrame")
        if not isinstance(arm, RgbCaptureArm):
            raise TypeError("arm must be RgbCaptureArm")
        expected_camera = self._source_to_camera.get(frame.source_id)
        if expected_camera is None or expected_camera != frame.camera_id:
            raise ValueError("pipeline RGB source/camera binding is not configured")
        if arm.camera_id != frame.camera_id or arm.source_id != frame.source_id:
            raise ValueError("pipeline RGB frame does not match its capture arm")
        if frame.width > self._max_width or frame.height > self._max_height:
            raise ValueError("pipeline RGB dimensions exceed configured bounds")
        expected_bytes = frame.width * frame.height * 3
        if expected_bytes > self._max_frame_bytes:
            raise ValueError("pipeline RGB frame exceeds max_frame_bytes")
        if not isinstance(frame.pixels, np.ndarray):
            raise TypeError("pipeline RGB pixels must be a numpy array")
        if frame.pixels.dtype != np.uint8:
            raise ValueError("pipeline RGB pixels must have dtype uint8")
        if frame.pixels.shape != (frame.height, frame.width, 3):
            raise ValueError("pipeline RGB pixels must have shape HxWx3")
        if frame.pixels.nbytes != expected_bytes:
            raise ValueError("pipeline RGB pixel byte size does not match dimensions")

        canonical_bytes = frame.pixels.tobytes(order="C")
        observed_digest = hashlib.sha256(canonical_bytes).hexdigest()
        if not hmac.compare_digest(observed_digest, frame.content_sha256):
            raise ValueError("pipeline RGB content_sha256 does not match pixels")
        owned_pixels = np.frombuffer(canonical_bytes, dtype=np.uint8).reshape(
            frame.height,
            frame.width,
            3,
        )
        retained = TimestampedRgbFrame(
            camera_id=frame.camera_id,
            source_id=frame.source_id,
            batch_id=frame.batch_id,
            captured_at_us=frame.captured_at_us,
            frame_id=frame.frame_id,
            source_media_pts_ns=frame.source_media_pts_ns,
            width=frame.width,
            height=frame.height,
            content_sha256=observed_digest,
            pixels=owned_pixels,
            color_space=frame.color_space,
        )

        with self._lock:
            if self._closed:
                raise RuntimeError("pipeline_rgb_provider_closed")
            if self._active_arm != arm:
                raise RuntimeError("pipeline_rgb_provider_arm_expired")
            rows = self._frames.setdefault(frame.camera_id, deque())
            identity = (
                retained.source_id,
                retained.frame_id,
                retained.source_media_pts_ns,
            )
            if any(
                (row.source_id, row.frame_id, row.source_media_pts_ns) == identity
                for row in rows
            ):
                raise ValueError("pipeline RGB exact frame identity is duplicated")
            if len(rows) == self._frames_per_camera:
                self._evict_oldest_from_camera(frame.camera_id)
            while self._retained_bytes + expected_bytes > self._max_total_bytes:
                self._evict_globally_oldest()
            rows = self._frames.setdefault(frame.camera_id, deque())
            rows.append(retained)
            self._retained_bytes += expected_bytes
            self._offer_total += 1

    def _evict_oldest_from_camera(self, camera_id: str) -> None:
        rows = self._frames[camera_id]
        evicted = rows.popleft()
        self._retained_bytes -= evicted.width * evicted.height * 3
        self._evicted_frames += 1
        if not rows:
            del self._frames[camera_id]

    def _evict_globally_oldest(self) -> None:
        if not self._frames:
            raise RuntimeError("pipeline_rgb_provider_capacity_invariant_failed")
        camera_id = min(
            self._frames,
            key=lambda camera: (
                self._frames[camera][0].captured_at_us,
                camera,
                self._frames[camera][0].frame_id,
            ),
        )
        self._evict_oldest_from_camera(camera_id)

    def provide(
        self,
        camera_id: str,
        *,
        cohort: Sequence[RawDepthSnapshot],
    ) -> TimestampedRgbFrame | None:
        camera = str(camera_id or "").strip()
        expected_source_id = self._camera_to_source.get(camera)
        if expected_source_id is None:
            raise ValueError("pipeline RGB camera is not configured")
        if not isinstance(cohort, Sequence) or not cohort:
            raise ValueError("pipeline RGB cohort must be a non-empty sequence")
        identities: set[tuple[int, int, int]] = set()
        for row in cohort:
            if not isinstance(row, RawDepthSnapshot) or row.camera_id != camera:
                raise ValueError("pipeline RGB cohort scope is invalid")
            if row.source_id is None:
                raise ValueError("pipeline RGB cohort source identity is unavailable")
            if int(row.source_id) != expected_source_id:
                raise ValueError(
                    "pipeline RGB cohort source identity does not match the "
                    "configured camera source"
                )
            identities.add(
                (
                    int(row.source_id),
                    int(row.source_frame_number),
                    int(row.source_media_pts_ns),
                )
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("pipeline_rgb_provider_closed")
            arm = self._active_arm
            if arm is None or arm.camera_id != camera:
                raise RuntimeError("pipeline_rgb_provider_not_armed")
            rows = tuple(self._frames.get(camera, ()))
        matching = tuple(
            frame
            for frame in rows
            if (
                frame.source_id,
                frame.frame_id,
                frame.source_media_pts_ns,
            )
            in identities
        )
        if not matching:
            return None
        return max(
            matching,
            key=lambda frame: (
                frame.source_media_pts_ns,
                frame.frame_id,
                frame.captured_at_us,
            ),
        )

    def disarm(self, arm: RgbCaptureArm) -> None:
        if not isinstance(arm, RgbCaptureArm):
            raise TypeError("arm must be RgbCaptureArm")
        with self._lock:
            if self._closed:
                return
            if self._active_arm != arm:
                raise RuntimeError("pipeline_rgb_provider_arm_expired")
            self._active_arm = None
            self._frames.clear()
            self._retained_bytes = 0
            self._disarm_total += 1

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._active_arm = None
            self._frames.clear()
            self._retained_bytes = 0

    def health_snapshot(self) -> dict[str, object]:
        with self._lock:
            arm = self._active_arm
            return {
                "closed": self._closed,
                "armed": arm is not None,
                "armed_camera_id": arm.camera_id if arm is not None else None,
                "armed_source_id": arm.source_id if arm is not None else None,
                "configured_camera_count": len(self._configured_camera_sources),
                "camera_count": len(self._frames),
                "frame_count": sum(len(rows) for rows in self._frames.values()),
                "retained_bytes": self._retained_bytes,
                "max_total_bytes": self._max_total_bytes,
                "evicted_frames": self._evicted_frames,
                "arm_total": self._arm_total,
                "disarm_total": self._disarm_total,
                "offer_total": self._offer_total,
                "unarmed_capture_checks": self._unarmed_capture_checks,
            }


__all__ = ["PipelineRgbFrameProvider", "RgbCaptureArm"]
