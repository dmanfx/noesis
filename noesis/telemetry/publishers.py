from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence

from noesis.metadata.depth_result import DepthResult
from noesis_core.tracking_continuity import (
    TRACKING_CONTINUITY_CONTRACT,
    TRACKING_CONTINUITY_CONTRACT_VERSION,
)

logger = logging.getLogger(__name__)

_TRACKING_FRAME_FIELDS = (
    "frame_id",
    "captured_at_us",
    "observed_at_us",
    "capture_time_status",
    "media_pts_ns",
    "tracker_lifecycle_tombstones",
)


def _response_model_timing(ws_server: Any, started_ns: int) -> Any:
    timer = getattr(ws_server, "response_model_timing_since", None)
    return timer(started_ns) if callable(timer) else None


def _broadcast_sync(
    ws_server: Any,
    message: Mapping[str, Any],
    response_model_timing: Any,
) -> None:
    if response_model_timing is None:
        ws_server.broadcast_sync(message)
    else:
        ws_server.broadcast_sync(
            message,
            response_model_timing=response_model_timing,
        )


def _admit_broadcast_batch_sync(
    ws_server: Any,
    messages: Sequence[Mapping[str, Any]],
    response_model_timing: Any,
) -> Any:
    admit = getattr(ws_server, "admit_broadcast_batch_sync", None)
    if not callable(admit):
        raise AttributeError(
            "Tracking telemetry requires "
            "WebSocketServer.admit_broadcast_batch_sync"
        )
    if response_model_timing is None:
        admission = admit(messages)
    else:
        admission = admit(
            messages,
            response_model_timing=response_model_timing,
        )
    receipt = getattr(admission, "receipt", None)
    submission_id = getattr(receipt, "submission_id", None)
    message_count = getattr(receipt, "message_count", None)
    if (
        isinstance(submission_id, bool)
        or not isinstance(submission_id, int)
        or submission_id <= 0
        or isinstance(message_count, bool)
        or not isinstance(message_count, int)
        or message_count != len(messages)
    ):
        raise RuntimeError(
            "WebSocket batch admission returned an invalid receipt"
        )
    if not callable(getattr(admission, "commit_then_release", None)):
        raise RuntimeError(
            "WebSocket batch admission omitted its authority gate"
        )
    return admission


def _record_response_model_error(
    ws_server: Any,
    *,
    route: str,
    message_type: str,
    error: BaseException,
) -> None:
    recorder = getattr(ws_server, "record_response_model_error", None)
    if callable(recorder):
        recorder(route=route, message_type=message_type, error=error)


class DepthTelemetryPublisher:
    """Bridge depth bursts to websocket clients."""

    def __init__(
        self,
        ws_server: Any,
        failure_callback: Optional[Any] = None,
    ) -> None:
        self._ws = ws_server
        self._failure_callback = failure_callback

    def _report_failure(self, error: BaseException) -> None:
        if callable(self._failure_callback):
            self._failure_callback(error)

    def publish(self, result: DepthResult) -> None:
        response_model_started_ns = time.perf_counter_ns()
        try:
            message = {
                "type": "depth_result",
                "payload": result.to_public_dict(),
            }
            response_model_timing = _response_model_timing(
                self._ws,
                response_model_started_ns,
            )
        except Exception as exc:
            _record_response_model_error(
                self._ws,
                route="broadcast",
                message_type="depth_result",
                error=exc,
            )
            self._report_failure(exc)
            logger.exception("Failed to assemble DepthResult telemetry")
            raise
        try:
            if not callable(getattr(self._ws, "broadcast_sync", None)):
                raise AttributeError(
                    "Depth telemetry requires WebSocketServer.broadcast_sync"
                )
            _broadcast_sync(self._ws, message, response_model_timing)
        except Exception as exc:
            self._report_failure(exc)
            logger.exception("Failed to publish DepthResult telemetry")
            raise


class TrackingPublicationReceipt(NamedTuple):
    """Committed canonical tracking/global-world sender-admission receipt."""

    source_id: int
    frame_id: Optional[int]
    observed_at_us: Optional[int]
    tracking_publication_sequence: int
    outbound_submission_id: int
    outbound_message_count: int


class TrackingPublicationPoisoned(RuntimeError):
    """An admitted cohort could not complete authority commit and release."""


class TrackingTelemetryPublisher:
    """Broadcast tracking metadata (tracks, occupancy, transitions) to clients."""

    def __init__(
        self,
        ws_server: Any,
        metadata_getter: Optional[Any] = None,
        world_service: Optional[Any] = None,
        health_monitor: Optional[Any] = None,
        failure_callback: Optional[Any] = None,
    ) -> None:
        self._ws = ws_server
        self._metadata_getter = metadata_getter
        self._world_service = world_service
        self._health_monitor = health_monitor
        self._failure_callback = failure_callback
        self._publication_lock = threading.Lock()
        self._next_publication_sequence_by_source: dict[int, int] = {}
        self._terminal_failure: BaseException | None = None

    def _report_failure(self, error: BaseException) -> None:
        if callable(self._failure_callback):
            self._failure_callback(error)

    def _raise_if_poisoned(self) -> None:
        if self._terminal_failure is not None:
            raise TrackingPublicationPoisoned(
                "tracking publisher is poisoned after an admitted cohort "
                "failed authoritative commit or gated release"
            ) from self._terminal_failure

    def publish(
        self,
        source_id: int,
        tracks: Iterable[Mapping[str, Any]],
        *,
        frame_metadata: Optional[Mapping[str, Any]] = None,
    ) -> TrackingPublicationReceipt:
        with self._publication_lock:
            self._raise_if_poisoned()
        track_list = tracks if isinstance(tracks, list) else list(tracks)
        extra: Mapping[str, Any] = {}
        if callable(self._metadata_getter):
            try:
                candidate = self._metadata_getter(int(source_id), track_list)
            except Exception as exc:
                logger.exception("Tracking telemetry metadata getter failed")
                self._report_failure(exc)
                raise
            if candidate is not None and not isinstance(candidate, Mapping):
                exc = TypeError("tracking metadata getter must return a mapping")
                self._report_failure(exc)
                raise exc
            if isinstance(candidate, Mapping):
                extra = candidate
        if isinstance(frame_metadata, Mapping):
            frame_fields = {
                key: frame_metadata[key]
                for key in _TRACKING_FRAME_FIELDS
                if key in frame_metadata
            }
            extra = {**dict(extra), **frame_fields}
        else:
            frame_fields = {}
        source = int(source_id)
        response_model_started_ns = time.perf_counter_ns()
        world_preparation = None
        world_discard = None
        outbound_admitted = False
        world_authority_committed = False
        try:
            with self._publication_lock:
                self._raise_if_poisoned()
                sequence = self._next_publication_sequence_by_source.get(
                    source,
                    0,
                )
                publication = None
                if self._world_service is not None:
                    prepare = getattr(self._world_service, "prepare", None)
                    commit = getattr(self._world_service, "commit", None)
                    world_discard = getattr(self._world_service, "discard", None)
                    if (
                        not callable(prepare)
                        or not callable(commit)
                        or not callable(world_discard)
                    ):
                        raise AttributeError(
                            "canonical world service requires prepare/commit/discard"
                        )
                    world_preparation = prepare(
                        source,
                        track_list,
                        metadata=extra,
                    )
                    publication = getattr(world_preparation, "publication", None)
                    if publication is None:
                        raise RuntimeError(
                            "canonical world preparation omitted publication"
                        )

                payload: dict[str, Any] = {
                    "type": "tracking",
                    "source_id": source,
                    "track_count": len(track_list),
                    "tracks": track_list,
                    **dict(extra),
                    **frame_fields,
                    "tracking_continuity_contract": (
                        TRACKING_CONTINUITY_CONTRACT
                    ),
                    "tracking_continuity_contract_version": (
                        TRACKING_CONTINUITY_CONTRACT_VERSION
                    ),
                    "tracking_publication_sequence": sequence,
                }
                payload.setdefault("tracker_lifecycle_tombstones", [])
                frame_id_raw = payload.get("frame_id")
                observed_at_raw = payload.get("observed_at_us")
                frame_id = (
                    int(frame_id_raw)
                    if isinstance(frame_id_raw, int)
                    and not isinstance(frame_id_raw, bool)
                    else None
                )
                observed_at_us = (
                    int(observed_at_raw)
                    if isinstance(observed_at_raw, int)
                    and not isinstance(observed_at_raw, bool)
                    else None
                )
                cohort = {
                    "source_id": source,
                    "frame_id": frame_id,
                    "observed_at_us": observed_at_us,
                    "tracking_publication_sequence": sequence,
                }
                payload["cohort"] = dict(cohort)
                batch: list[Mapping[str, Any]] = [payload]
                if publication is not None:
                    payload["observation_contract"] = (
                        "noesis.observation.person"
                    )
                    payload["observation_contract_version"] = 1
                    payload["observations"] = [
                        item.model_dump(mode="json")
                        for item in publication.observations
                    ]
                    snapshot_payload = publication.snapshot.model_dump(
                        mode="json"
                    )
                    payload["world_snapshot"] = snapshot_payload
                    event_payloads = [
                        item.model_dump(mode="json")
                        for item in publication.events
                    ]
                    payload["world_events"] = event_payloads
                    batch.append(
                        {
                            "type": "world_snapshot",
                            "payload": snapshot_payload,
                            "cohort": dict(cohort),
                            **cohort,
                        }
                    )
                    batch.extend(
                        {
                            "type": "world_event",
                            "payload": event_payload,
                            "cohort": dict(cohort),
                            **cohort,
                        }
                        for event_payload in event_payloads
                    )

                response_model_timing = _response_model_timing(
                    self._ws,
                    response_model_started_ns,
                )
                outbound = _admit_broadcast_batch_sync(
                    self._ws,
                    batch,
                    response_model_timing,
                )
                outbound_admitted = True
                try:
                    def _commit_world_authority() -> Any:
                        nonlocal world_authority_committed
                        if world_preparation is None:
                            return None
                        committed = self._world_service.commit(
                            world_preparation
                        )
                        world_authority_committed = True
                        if committed != publication:
                            raise RuntimeError(
                                "canonical world commit changed the admitted cohort"
                            )
                        return committed

                    # The exact frozen bytes are already bounded, but their
                    # event-loop task remains behind a one-shot authority gate.
                    # A failed journal/world commit aborts without beginning
                    # any client delivery; a successful commit releases it.
                    outbound.commit_then_release(_commit_world_authority)
                    self._next_publication_sequence_by_source[source] = (
                        sequence + 1
                    )
                    outbound_receipt = outbound.receipt
                    receipt = TrackingPublicationReceipt(
                        source_id=source,
                        frame_id=frame_id,
                        observed_at_us=observed_at_us,
                        tracking_publication_sequence=sequence,
                        outbound_submission_id=int(
                            outbound_receipt.submission_id
                        ),
                        outbound_message_count=int(
                            outbound_receipt.message_count
                        ),
                    )
                except Exception as exc:
                    self._terminal_failure = exc
                    raise

                if publication is not None and self._health_monitor is not None:
                    try:
                        evidence = {
                            "source_id": source,
                            "observation_count": len(publication.observations),
                            "entity_count": len(publication.snapshot.entities),
                            "conflict_count": sum(
                                1
                                for entity in publication.snapshot.entities
                                if entity.conflict
                            ),
                            # Aggregate world time may regress when the freshest
                            # entity disappears; publication time is progress.
                            "world_observed_end_us": (
                                publication.snapshot.observed_end_us
                            ),
                            "outbound_submission_id": (
                                receipt.outbound_submission_id
                            ),
                            "tracking_publication_sequence": sequence,
                        }
                        for capability in (
                            "tracking_observations",
                            "global_world",
                        ):
                            self._health_monitor.record_success(
                                capability,
                                producer_run_id=(
                                    publication.snapshot.producer.run_id
                                ),
                                sequence=publication.snapshot.sequence,
                                observed_at_us=(
                                    publication.snapshot.published_at_us
                                ),
                                checked_at_us=(
                                    publication.snapshot.published_at_us
                                ),
                                contract_compatible=True,
                                evidence=evidence,
                            )
                    except Exception as exc:
                        logger.exception(
                            "Canonical tracking batch succeeded but health receipt failed"
                        )
                        self._report_failure(exc)
                return receipt
        except Exception as exc:
            if (
                world_preparation is not None
                and not world_authority_committed
                and callable(world_discard)
            ):
                try:
                    world_discard(world_preparation)
                except Exception as discard_exc:
                    logger.exception(
                        "Failed to discard uncommitted canonical world candidate"
                    )
                    self._report_failure(discard_exc)
            if outbound_admitted and self._terminal_failure is None:
                self._terminal_failure = exc
            _record_response_model_error(
                self._ws,
                route="broadcast",
                message_type="tracking",
                error=exc,
            )
            logger.exception(
                "Failed release-gated tracking/world publication for source %s",
                source_id,
            )
            self._report_failure(exc)
            raise


def bind_occupancy_publisher(pipeline: Any, occupancy_publisher: Optional[Any]) -> None:
    """Expose occupancy publisher to DS8 pipeline probes."""
    setattr(pipeline, "occupancy_publisher", occupancy_publisher)
    logger.info(
        "Occupancy publisher %s bound to pipeline",
        type(occupancy_publisher).__name__ if occupancy_publisher else "None",
    )
