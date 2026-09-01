from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

from noesis_core.analytics_zones import is_exact_zone_label
from noesis_core.contracts.base import ArtifactFingerprint, Matrix3, ProducerRef, Vector3
from noesis_core.contracts.identity import IdentityKind, SubjectRef, TrackletRef
from noesis_core.contracts.observation import (
    ObservationEnvelope,
    PersonObservation,
    WorldObservationDiagnostics,
    WorldPositionObservation,
    ZoneSource,
)
from noesis_core.contracts.world import EntityLifecycle, WorldEntity, WorldEvent, WorldSnapshot
from noesis_core.contracts.world_measurement import WorldMeasurementCandidateDiagnostic
from noesis_core.depth_contract import usable_registered_depth_m
from noesis_core.journal import (
    AsyncContractJournal,
    AsyncJournalAdmissionReceipt,
    ContractJournal,
)
from noesis_core.world import GlobalWorldFusion


_WORLD_DIAGNOSTIC_TRACK_FIELDS = frozenset(
    {
        "world_estimator_evaluated",
        "world_floor_candidate",
        "world_floor_range_m",
        "world_floor_range_limit_m",
        "world_floor_incidence_sin",
        "world_floor_admitted",
        "world_floor_rejection_reason",
        "world_depth_candidate",
        "world_prefilter_measurement",
        "world_filter_prediction",
        "world_measurement_accepted",
        "world_rejection_reason",
        "world_innovation_m",
        "world_innovation_limit_m",
        "world_fusion_policy_id",
        "world_floor_weight_scale",
        "world_depth_weight_scale",
        "world_floor_weight_effective",
        "world_depth_weight_effective",
        "depth_status",
        "depth_anchor_m",
        "depth_registered_m",
        "depth_used_m",
        "depth_registration_status",
        "depth_registration_id",
        "world_resolver",
    }
)


@dataclass(frozen=True)
class WorldArtifacts:
    calibration: ArtifactFingerprint
    model: ArtifactFingerprint
    config: ArtifactFingerprint
    # Private producer/consumer authority binding. ``calibration`` fingerprints
    # the complete resolved snapshot for the public observation contract, while
    # this is the exact raw camera-calibration file digest stamped on the world
    # coordinate by the DS9 estimator. They are intentionally distinct: a
    # queued coordinate must never be relabeled with a newer provider snapshot.
    camera_calibration_sha256: str | None = None


@dataclass(frozen=True)
class WorldPublication:
    observations: tuple[ObservationEnvelope, ...]
    snapshot: WorldSnapshot
    events: tuple[WorldEvent, ...]


@dataclass(frozen=True)
class PreparedWorldPublication:
    """Opaque owner-bound receipt for one non-authoritative world candidate."""

    base_revision: int
    publication: WorldPublication
    _service_token: object = field(repr=False, compare=False)
    _candidate_id: int = field(repr=False, compare=False)
    # The world candidate is model-dumped once while preparing the durable
    # journal cohort.  Keep that exact JSON-native tuple on the owner-bound
    # receipt so the canonical tracking publisher can reuse it instead of
    # serializing the immutable models a second time before admission.
    _serialized_payloads: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )

    @property
    def serialized_payloads(self) -> tuple[Mapping[str, Any], ...]:
        """Return the prepared JSON payloads in journal cohort order."""

        return self._serialized_payloads


@dataclass
class _PreparedWorldState:
    """Service-private mutable state that must never cross the boundary."""

    publication: WorldPublication
    fusion: GlobalWorldFusion
    sequence_by_source: dict[int, int]
    event_sequence: int
    previous_entities: dict[str, WorldEntity]
    canonical_track_outputs: dict[
        tuple[int, str, int, int, str, str, str, int],
        "_CanonicalTrackOutput",
    ]
    journal_payloads: tuple[Mapping[str, Any], ...]
    recorded_at_us: int
    source_id: int
    artifact_metadata: Mapping[str, Any]
    calibration_artifact_sha256: str
    camera_calibration_sha256: str | None
    camera_calibration_binding_required: bool


@dataclass(frozen=True)
class _InferredGroundEpisodeRoot:
    """Immutable producer proof root accepted and then owned by the service."""

    raw_origin: tuple[float, float, float]
    raw_origin_media_pts_ns: int
    raw_origin_observed_at_us: int
    raw_origin_ts_s: float
    trusted_world_origin: tuple[float, float, float]
    trusted_origin_media_pts_ns: int
    trusted_origin_filter_ts_s: float
    lifecycle_generation: int
    trail_segment_id: int
    height_ref_scene: float
    world_frame: str
    world_frame_revision: str
    world_transform_sha256: str


@dataclass(frozen=True)
class _CanonicalOutputOrigin:
    """One proven earlier queue output retained for asynchronous validation."""

    position: tuple[float, float, float]
    media_pts_ns: int
    trail_segment_id: int
    committed_world_source: str
    committed_provenance_type: str | None = None
    metric_position: tuple[float, float, float] | None = None
    metric_media_pts_ns: int | None = None
    metric_trail_segment_id: int | None = None
    projective_bridge_root_media_pts_ns: int | None = None
    kinematic_position: tuple[float, float, float] | None = None
    kinematic_media_pts_ns: int | None = None
    kinematic_trail_segment_id: int | None = None
    bbox_geometry: tuple[float, float, float, float] | None = None
    kinematic_bbox_geometry: tuple[float, float, float, float] | None = None
    inferred_ground_episode_root: _InferredGroundEpisodeRoot | None = None


@dataclass(frozen=True)
class _CanonicalTrackOutput:
    """Last committed source-local point used by projective continuity."""

    position: tuple[float, float, float]
    media_pts_ns: int
    trail_segment_id: int
    committed_world_source: str
    committed_provenance_type: str | None = None
    metric_position: tuple[float, float, float] | None = None
    metric_media_pts_ns: int | None = None
    metric_trail_segment_id: int | None = None
    projective_bridge_root_media_pts_ns: int | None = None
    # A gain-zero output hold is a current publication of an older uncertain
    # coordinate, not evidence that the person was physically stationary at
    # the hold's media PTS. Preserve the last motion-bearing output separately
    # so a later accurate recovery receives the real elapsed motion budget.
    kinematic_position: tuple[float, float, float] | None = None
    kinematic_media_pts_ns: int | None = None
    kinematic_trail_segment_id: int | None = None
    bbox_geometry: tuple[float, float, float, float] | None = None
    kinematic_bbox_geometry: tuple[float, float, float, float] | None = None
    inferred_ground_episode_root: _InferredGroundEpisodeRoot | None = None
    prior_outputs: tuple[_CanonicalOutputOrigin, ...] = ()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def fingerprint_payload(role: str, payload: Any, *, version: str | None = None) -> ArtifactFingerprint:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return ArtifactFingerprint(role=role, sha256=hashlib.sha256(encoded).hexdigest(), version=version)


class CanonicalWorldService:
    """Normalize primitive tracking rows and own canonical global world state."""

    MAX_PREPARED_CANDIDATES = 16
    MAX_CANONICAL_TRACK_OUTPUTS = 4096
    # The estimator runs in the media callback while strict world publication
    # drains through an ordered non-blocking worker. A current proof can
    # therefore bind a slightly older output by the time the service consumes
    # it. Retain only exact outputs this service actually committed, with a
    # hard per-lifecycle cap; the current candidate is still gated against the
    # latest output below.
    MAX_CANONICAL_TRACK_OUTPUT_HISTORY = 128
    IMAGE_MOTION_COMMITTED_ORIGIN_HISTORY_HORIZON_S = 1.25
    # Image-motion filter-transition proof contract v1. These are the
    # canonical human filter's physical bounds, not producer-declared tuning.
    IMAGE_MOTION_TRANSITION_VERSION = 1
    IMAGE_MOTION_MAX_SPEED_MPS = 4.0
    IMAGE_MOTION_MAX_JUMP_M = 0.75
    IMAGE_MOTION_RESET_AFTER_S = 1.25
    INFERRED_PROCESS_RESTART_MAX_AGE_S = IMAGE_MOTION_RESET_AFTER_S
    IMAGE_MOTION_PROJECTIVE_SLACK_M = 0.35
    # DS9's filter clock is an epoch-shaped float anchored to integer media
    # PTS.  Subtracting adjacent values near 1.8e9 seconds can differ from the
    # exact integer-PTS delta by a few float ULPs.  One microsecond still binds
    # the physical gate to the same media interval (4 micrometres at 4 m/s)
    # while rejecting any operationally meaningful producer-declared slack.
    IMAGE_MOTION_GATE_DT_ABS_TOL_S = 1e-6
    PROJECTIVE_BRIDGE_IMAGE_PROVENANCE_TYPES = frozenset(
        {
            "bbox_affine_floor_projection",
            "inferred_ground_process_observation",
        }
    )
    INFERRED_PROCESS_CONSENSUS_HORIZON_S = 0.40
    # A learned-height continuation outside an explicit occlusion episode is
    # admitted only when the current detector row and three exact-cohort torso
    # observations independently agree that the established person is moving.
    # These are service-owned proof thresholds, not producer-declared tuning.
    INFERRED_PROCESS_MIN_IMAGE_MOTION_STREAK = 3
    INFERRED_PROCESS_MIN_DETECTOR_CONFIDENCE = 0.65
    # Four 10 Hz publication intervals bridge an isolated metric dropout. A
    # projective episode receives one additional, non-renewing tail after the
    # image-origin history window: this covers detector/reacquisition cadence
    # without letting CV descendants create a new root. Every descendant is
    # still tied to one service-committed image root and bounded to 4 m/s.
    BOUNDED_PROCESS_HORIZON_S = 0.40
    PROJECTIVE_PROCESS_BRIDGE_HORIZON_S = (
        IMAGE_MOTION_COMMITTED_ORIGIN_HISTORY_HORIZON_S
        + BOUNDED_PROCESS_HORIZON_S
    )
    STATIONARY_ANCHOR_HOLD_HORIZON_S = 2.0
    # A standing person can remain visually present while the current floor
    # candidate is rejected by the kinematic output gate.  Extend only an
    # exact latest-output hold backed by a current pose, a non-seated upright
    # silhouette, torso motion, and a plausible floor candidate close to that
    # public output. These fixed service-owned thresholds keep the lane camera
    # agnostic and prevent producer-declared evidence from widening it.
    UPRIGHT_PRESENCE_HOLD_HORIZON_S = 2.0
    UPRIGHT_PRESENCE_MIN_DETECTOR_CONFIDENCE = 0.70
    UPRIGHT_PRESENCE_MIN_TRACKER_CONFIDENCE = 0.50
    UPRIGHT_PRESENCE_MAX_OUTPUT_DISTANCE_M = 1.25
    UPRIGHT_PRESENCE_MIN_BBOX_HEIGHT_FRACTION = 48.0 / 1080.0
    UPRIGHT_PRESENCE_MAX_BBOX_ASPECT = 0.85
    # A current detector/tracker row may retain the last accurate coordinate
    # through one short localization hole. This lane is rooted at the last
    # motion-bearing output (not the previous hold), so repeated rows cannot
    # renew it. Bbox limits match the source-local lifecycle continuity rule.
    CURRENT_PRESENCE_HOLD_HORIZON_S = 1.0
    CURRENT_PRESENCE_MIN_DETECTOR_CONFIDENCE = 0.25
    CURRENT_PRESENCE_MIN_TRACKER_CONFIDENCE = 0.45
    CURRENT_PRESENCE_STRONG_DETECTOR_CONFIDENCE = 0.50
    CURRENT_PRESENCE_MAX_BBOX_SIZE_RATIO = 1.8
    CURRENT_PRESENCE_MAX_CENTER_DISPLACEMENT_NORM = 0.65
    # Match the producer's fixed media-clock tolerance at the hold boundary.
    # This is numeric/cadence slack only; it is never accumulated per row.
    BOUNDED_PROCESS_HORIZON_EPSILON_S = 0.005

    def __init__(
        self,
        *,
        producer: ProducerRef,
        artifacts: WorldArtifacts | Callable[[int, Mapping[str, Any]], WorldArtifacts],
        fusion: GlobalWorldFusion | None = None,
        clock_us: Callable[[], int] | None = None,
        journal: ContractJournal | AsyncContractJournal | None = None,
    ) -> None:
        self.producer = producer
        self._artifacts = artifacts
        self._fusion = (
            fusion.fork()
            if fusion is not None
            else GlobalWorldFusion(producer)
        )
        self._clock_us = clock_us or (lambda: int(time.time() * 1_000_000))
        self._journal = journal
        self._sequence_by_source: dict[int, int] = {}
        self._event_sequence = 0
        self._previous_entities: dict[str, WorldEntity] = {}
        self._canonical_track_outputs: dict[
            tuple[int, str, int, int, str, str, str, int],
            _CanonicalTrackOutput,
        ] = {}
        self._lock = RLock()
        self._closed = False
        self._revision = 0
        self._current_snapshot: WorldSnapshot | None = None
        self._service_token = object()
        self._next_candidate_id = 1
        self._prepared_candidates: dict[
            int,
            tuple[PreparedWorldPublication, _PreparedWorldState],
        ] = {}
        self._persistence_rejected_cohorts = 0
        self._persistence_last_error: str | None = None

    def persistence_health(self) -> dict[str, Any]:
        """Return bounded optional-persistence health without touching media."""

        with self._lock:
            journal = self._journal
            rejected = int(self._persistence_rejected_cohorts)
            last_error = self._persistence_last_error
        if journal is None:
            return {
                "status": "disabled",
                "rejected_cohorts": rejected,
                "last_error": last_error,
            }
        snapshot_getter = getattr(journal, "health_snapshot", None)
        snapshot = (
            dict(snapshot_getter())
            if callable(snapshot_getter)
            else {"status": "synchronous"}
        )
        if rejected or last_error:
            snapshot["status"] = "degraded"
        snapshot["rejected_cohorts"] = rejected
        snapshot["last_error"] = last_error or snapshot.get("last_error")
        return snapshot

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def current_snapshot(self) -> WorldSnapshot | None:
        """Return the last committed immutable snapshot without mutation."""

        with self._lock:
            return self._current_snapshot

    def publish(
        self,
        source_id: int,
        tracks: list[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
    ) -> WorldPublication:
        """Compatibility transaction that prepares and commits under one lock."""

        with self._lock:
            prepared = self._prepare_locked(
                source_id,
                tracks,
                metadata=metadata,
            )
            try:
                return self._commit_locked(prepared)
            except Exception:
                self._prepared_candidates.pop(prepared._candidate_id, None)
                raise

    def prepare(
        self,
        source_id: int,
        tracks: list[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
    ) -> PreparedWorldPublication:
        """Build an exact candidate without mutating fusion, counters, or journal."""

        with self._lock:
            return self._prepare_locked(
                source_id,
                tracks,
                metadata=metadata,
            )

    def commit(
        self,
        prepared: PreparedWorldPublication,
    ) -> WorldPublication:
        """Commit one still-current candidate after outbound admission."""

        with self._lock:
            return self._commit_locked(prepared)

    def discard(self, prepared: PreparedWorldPublication) -> None:
        """Forget an unadmitted candidate without changing authority state."""

        with self._lock:
            self._validate_prepared_owner(prepared)
            entry = self._prepared_candidates.get(prepared._candidate_id)
            if entry is None:
                return
            receipt, _state = entry
            if receipt is not prepared:
                raise RuntimeError("canonical world preparation identity mismatch")
            self._prepared_candidates.pop(prepared._candidate_id, None)

    def _prepare_locked(
        self,
        source_id: int,
        tracks: list[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
    ) -> PreparedWorldPublication:
        if self._closed:
            raise RuntimeError("canonical world service is closed")
        if len(self._prepared_candidates) >= int(
            self.MAX_PREPARED_CANDIDATES
        ):
            raise RuntimeError(
                "canonical world preparation capacity is exhausted"
            )
        source = int(source_id)
        artifacts = (
            self._artifacts(source, metadata)
            if callable(self._artifacts)
            else self._artifacts
        )
        now_us = max(1, int(self._clock_us()))
        working_fusion = self._fusion.fork()
        working_sequences = dict(self._sequence_by_source)
        working_track_outputs = dict(self._canonical_track_outputs)
        source_media_pts: list[int] = []
        for value in (
            metadata.get("media_pts_ns"),
            *(track.get("media_pts_ns") for track in tracks),
        ):
            try:
                parsed_pts = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if parsed_pts > 0:
                source_media_pts.append(parsed_pts)
        current_source_media_pts_ns = (
            max(source_media_pts) if source_media_pts else None
        )
        # Expire source-local authority before it can validate this cohort.
        # Pruning only after observation construction admitted one extra stale
        # continuation at the end of a long gap.
        self._prune_expired_track_outputs(
            working_track_outputs,
            source_id=source,
            current_media_pts_ns=current_source_media_pts_ns,
        )
        observations: list[ObservationEnvelope] = []
        pending_track_outputs: dict[
            str,
            tuple[
                tuple[int, str, int, int, str, str, str, int],
                _CanonicalTrackOutput,
            ],
        ] = {}
        for track in tracks:
            continuity_key = self._canonical_track_output_key(
                source,
                track,
                metadata=metadata,
                calibration_sha256=(
                    self._sha256_text(
                        artifacts.camera_calibration_sha256
                    )
                    or artifacts.calibration.sha256
                ),
            )
            expected_continuity_origin = None
            if continuity_key is not None:
                self._prune_superseded_track_outputs(
                    working_track_outputs,
                    continuity_key,
                )
                # Refresh an exact-current origin in insertion order so the
                # hard cap evicts inactive history before a live lifecycle.
                expected_continuity_origin = working_track_outputs.pop(
                    continuity_key,
                    None,
                )
                if expected_continuity_origin is not None:
                    working_track_outputs[continuity_key] = (
                        expected_continuity_origin
                    )
            observation = self._observation(
                source,
                track,
                metadata=metadata,
                artifacts=artifacts,
                published_at_us=now_us,
                sequence_by_source=working_sequences,
                expected_image_motion_origin=expected_continuity_origin,
            )
            if observation is None:
                continue
            observations.append(observation)
            if continuity_key is not None and observation.payload.world is not None:
                position = observation.payload.world.position
                try:
                    media_pts_ns = int(observation.media_pts_ns)
                    trail_segment_id = int(track.get("trail_segment_id", 0))
                except (TypeError, ValueError, OverflowError):
                    media_pts_ns = -1
                    trail_segment_id = -1
                if media_pts_ns >= 0 and trail_segment_id >= 0:
                    previous_output = working_track_outputs.get(
                        continuity_key
                    )
                    current_position = (
                        float(position.x),
                        float(position.y),
                        float(position.z),
                    )
                    current_bbox_geometry = self._bbox_geometry(
                        track.get("bbox")
                    )
                    world_source = str(
                        track.get("world_source") or ""
                    ).strip().lower()
                    provenance = track.get("world_prediction_provenance")
                    committed_provenance_type = (
                        str(provenance.get("type") or "").strip()
                        if isinstance(provenance, Mapping)
                        else ""
                    )
                    gain_zero_output_hold = bool(
                        world_source == "anchor_hold"
                        and committed_provenance_type
                        == "bounded_output_hold"
                    )
                    projective_bridge_root_media_pts_ns = (
                        self._projective_bridge_root_for_admitted_output(
                            track,
                            expected_origin=previous_output,
                            current_media_pts_ns=media_pts_ns,
                        )
                    )
                    inferred_ground_episode_root = (
                        self._inferred_ground_episode_root_for_admitted_output(
                            track,
                            expected_origin=previous_output,
                            current_media_pts_ns=media_pts_ns,
                        )
                    )
                    metric_authoritative = bool(
                        world_source
                        not in {
                            "anchor_hold",
                            "cv_prediction",
                            "image_motion_prediction",
                        }
                        and str(
                            track.get("world_quality") or "estimated"
                        ).strip().lower()
                        != "held"
                        and track.get("world_measurement_accepted") is not False
                    )
                    prior_outputs: tuple[_CanonicalOutputOrigin, ...] = ()
                    if (
                        previous_output is not None
                        and int(previous_output.trail_segment_id)
                        == int(trail_segment_id)
                    ):
                        # Queue delay can make a valid descendant refer to an
                        # older committed image root after a newer descendant
                        # has already committed. Retain history for the full
                        # nonrenewing projective episode; individual validators
                        # still apply their narrower lane-specific horizons.
                        history_horizon_ns = int(
                            round(
                                (
                                    self.PROJECTIVE_PROCESS_BRIDGE_HORIZON_S
                                    + self.BOUNDED_PROCESS_HORIZON_EPSILON_S
                                )
                                * 1_000_000_000.0
                            )
                        )
                        prior_outputs = (
                            _CanonicalOutputOrigin(
                                position=tuple(previous_output.position),
                                media_pts_ns=int(previous_output.media_pts_ns),
                                trail_segment_id=int(
                                    previous_output.trail_segment_id
                                ),
                                committed_world_source=(
                                    previous_output.committed_world_source
                                ),
                                committed_provenance_type=(
                                    previous_output.committed_provenance_type
                                ),
                                metric_position=previous_output.metric_position,
                                metric_media_pts_ns=(
                                    previous_output.metric_media_pts_ns
                                ),
                                metric_trail_segment_id=(
                                    previous_output.metric_trail_segment_id
                                ),
                                projective_bridge_root_media_pts_ns=(
                                    previous_output.projective_bridge_root_media_pts_ns
                                ),
                                kinematic_position=(
                                    previous_output.kinematic_position
                                ),
                                kinematic_media_pts_ns=(
                                    previous_output.kinematic_media_pts_ns
                                ),
                                kinematic_trail_segment_id=(
                                    previous_output.kinematic_trail_segment_id
                                ),
                                bbox_geometry=(
                                    previous_output.bbox_geometry
                                ),
                                kinematic_bbox_geometry=(
                                    previous_output.kinematic_bbox_geometry
                                ),
                                inferred_ground_episode_root=(
                                    previous_output.inferred_ground_episode_root
                                ),
                            ),
                            *tuple(previous_output.prior_outputs),
                        )
                        prior_outputs = tuple(
                            origin
                            for origin in prior_outputs
                            if 0
                            <= media_pts_ns - int(origin.media_pts_ns)
                            <= history_horizon_ns
                        )[: int(self.MAX_CANONICAL_TRACK_OUTPUT_HISTORY)]
                    pending_track_outputs[observation.observation_id] = (
                        continuity_key,
                        _CanonicalTrackOutput(
                            position=current_position,
                            media_pts_ns=media_pts_ns,
                            trail_segment_id=trail_segment_id,
                            committed_world_source=world_source,
                            committed_provenance_type=(
                                committed_provenance_type or None
                            ),
                            metric_position=(
                                current_position
                                if metric_authoritative
                                else (
                                    previous_output.metric_position
                                    if previous_output is not None
                                    else None
                                )
                            ),
                            metric_media_pts_ns=(
                                media_pts_ns
                                if metric_authoritative
                                else (
                                    previous_output.metric_media_pts_ns
                                    if previous_output is not None
                                    else None
                                )
                            ),
                            metric_trail_segment_id=(
                                trail_segment_id
                                if metric_authoritative
                                else (
                                    previous_output.metric_trail_segment_id
                                    if previous_output is not None
                                    else None
                                )
                            ),
                            projective_bridge_root_media_pts_ns=(
                                projective_bridge_root_media_pts_ns
                            ),
                            kinematic_position=(
                                (
                                    previous_output.kinematic_position
                                    or previous_output.position
                                )
                                if gain_zero_output_hold
                                and previous_output is not None
                                else current_position
                            ),
                            kinematic_media_pts_ns=(
                                (
                                    previous_output.kinematic_media_pts_ns
                                    if previous_output.kinematic_media_pts_ns
                                    is not None
                                    else previous_output.media_pts_ns
                                )
                                if gain_zero_output_hold
                                and previous_output is not None
                                else media_pts_ns
                            ),
                            kinematic_trail_segment_id=(
                                (
                                    previous_output.kinematic_trail_segment_id
                                    if previous_output.kinematic_trail_segment_id
                                    is not None
                                    else previous_output.trail_segment_id
                                )
                                if gain_zero_output_hold
                                and previous_output is not None
                                else trail_segment_id
                            ),
                            bbox_geometry=current_bbox_geometry,
                            kinematic_bbox_geometry=(
                                (
                                    previous_output.kinematic_bbox_geometry
                                    or previous_output.bbox_geometry
                                )
                                if gain_zero_output_hold
                                and previous_output is not None
                                else current_bbox_geometry
                            ),
                            inferred_ground_episode_root=(
                                inferred_ground_episode_root
                            ),
                            prior_outputs=prior_outputs,
                        ),
                    )
            subject = self._subject(observation, track)
            # Trail episodes are producer-internal continuity authority, not
            # part of the public observation schema. Pass the exact accepted
            # row's break/segment context privately so global fusion does not
            # mistake a deliberate reacquisition reset for a same-episode
            # teleport. Fusion consumes each source/lifecycle/segment break
            # once and continues gating ordinary camera handoffs.
            working_fusion.ingest(
                observation,
                subject,
                trail_segment_id=self._nonnegative_int(
                    track.get("trail_segment_id")
                ),
                trail_break_required=(
                    track.get("trail_break_required") is True
                ),
                tracker_lifecycle_generation=self._nonnegative_int(
                    track.get("tracker_lifecycle_generation")
                ),
                source_epoch=self._nonnegative_int(
                    track.get("source_epoch")
                    if track.get("source_epoch") is not None
                    else metadata.get("source_epoch")
                ),
            )
        if not tracks:
            camera_id = str(metadata.get("camera_id") or "").strip()
            if not camera_id:
                raise ValueError(
                    "empty source publication requires metadata.camera_id"
                )
            working_fusion.clear_source(
                camera_id,
                source,
                self.producer.run_id,
            )
        self._prune_expired_track_outputs(
            working_track_outputs,
            source_id=source,
            current_media_pts_ns=current_source_media_pts_ns,
        )
        snapshot = working_fusion.snapshot(published_at_us=now_us)
        source_evidence_by_observation_id = {
            source.observation_id: source
            for entity in snapshot.entities
            for source in entity.sources
        }
        # Global fusion may intentionally ignore a held continuation when a
        # fresh metric camera owns the entity. That source-local row remains a
        # valid process origin, but a positional/registration/velocity
        # rejection must never become a committed origin for a later proof.
        cache_admitted_observation_ids = {
            observation_id
            for observation_id in pending_track_outputs
            if (
                (
                    evidence := source_evidence_by_observation_id.get(observation_id)
                )
                is not None
                and (
                    evidence.accepted is True
                    or evidence.rejection_reason
                    == "non_authoritative_held_continuation"
                )
            )
        }
        for (
            observation_id,
            (continuity_key, output),
        ) in pending_track_outputs.items():
            if observation_id in cache_admitted_observation_ids:
                working_track_outputs[continuity_key] = output
        self._bound_track_outputs(working_track_outputs)
        events, next_event_sequence, next_previous_entities = (
            self._events_for_state(
                snapshot,
                previous_entities=self._previous_entities,
                event_sequence=self._event_sequence,
            )
        )
        publication = WorldPublication(
            observations=tuple(observations),
            snapshot=snapshot,
            events=events,
        )
        journal_payloads = (
            *(item.model_dump(mode="json") for item in observations),
            snapshot.model_dump(mode="json"),
            *(item.model_dump(mode="json") for item in events),
        )
        candidate_id = int(self._next_candidate_id)
        self._next_candidate_id += 1
        receipt = PreparedWorldPublication(
            base_revision=int(self._revision),
            # Public contracts and the containing dataclass are immutable;
            # mutable fusion/counter state remains service-private.
            publication=publication,
            _service_token=self._service_token,
            _candidate_id=candidate_id,
            _serialized_payloads=journal_payloads,
        )
        state = _PreparedWorldState(
            publication=publication,
            fusion=working_fusion,
            sequence_by_source=working_sequences,
            event_sequence=next_event_sequence,
            previous_entities=next_previous_entities,
            canonical_track_outputs=working_track_outputs,
            journal_payloads=journal_payloads,
            recorded_at_us=int(snapshot.published_at_us),
            source_id=source,
            artifact_metadata=dict(metadata),
            calibration_artifact_sha256=str(
                artifacts.calibration.sha256
            ),
            camera_calibration_sha256=self._sha256_text(
                artifacts.camera_calibration_sha256
            ),
            camera_calibration_binding_required=(
                artifacts.camera_calibration_sha256 is not None
            ),
        )
        self._prepared_candidates[candidate_id] = (receipt, state)
        return receipt

    def _commit_locked(
        self,
        prepared: PreparedWorldPublication,
    ) -> WorldPublication:
        if self._closed:
            raise RuntimeError("canonical world service is closed")
        self._validate_prepared_owner(prepared)
        entry = self._prepared_candidates.get(prepared._candidate_id)
        if entry is None or entry[0] is not prepared:
            raise RuntimeError("canonical world preparation is unknown or consumed")
        _receipt, state = entry
        if int(prepared.base_revision) != int(self._revision):
            raise RuntimeError(
                "canonical world preparation is stale relative to service state"
            )
        if state.publication.snapshot.producer != self.producer:
            raise RuntimeError("canonical world preparation producer mismatch")
        if callable(self._artifacts):
            current_artifacts = self._artifacts(
                int(state.source_id),
                state.artifact_metadata,
            )
            current_camera_calibration_sha256 = self._sha256_text(
                current_artifacts.camera_calibration_sha256
            )
            if state.camera_calibration_binding_required and (
                current_camera_calibration_sha256 is None
                or current_camera_calibration_sha256
                != state.camera_calibration_sha256
            ):
                raise RuntimeError(
                    "canonical world preparation camera calibration authority "
                    "changed before commit"
                )
            if (
                str(current_artifacts.calibration.sha256)
                != str(state.calibration_artifact_sha256)
            ):
                raise RuntimeError(
                    "canonical world preparation calibration artifact changed "
                    "before commit"
                )
        if self._journal is not None:
            try:
                expected_records = len(state.journal_payloads)
                journal_capacity = getattr(
                    self._journal,
                    "max_records",
                    None,
                )
                if (
                    isinstance(journal_capacity, int)
                    and not isinstance(journal_capacity, bool)
                    and journal_capacity < expected_records
                ):
                    raise RuntimeError(
                        "canonical world journal retention cannot hold the "
                        f"complete cohort: required={expected_records} "
                        f"max_records={journal_capacity}"
                    )
                appended = self._journal.append_many(
                    state.journal_payloads,
                    recorded_at_us=state.recorded_at_us,
                )
                if isinstance(self._journal, AsyncContractJournal):
                    if (
                        not isinstance(appended, AsyncJournalAdmissionReceipt)
                        or int(appended.payload_count) != expected_records
                    ):
                        raise RuntimeError(
                            "canonical world persistence admission did not confirm "
                            f"the exact payload count: expected={expected_records}"
                        )
                elif len(appended) != expected_records:
                    raise RuntimeError(
                        "canonical world journal did not confirm the exact "
                        f"append count: expected={expected_records} "
                        f"actual={len(appended)}"
                    )
            except Exception as exc:
                if isinstance(self._journal, AsyncContractJournal):
                    # The journal is reconstructable persistence, not live
                    # spatial authority.  Record its bounded failure and keep
                    # the exact in-memory tracking/world/BEV cohort moving.
                    self._persistence_rejected_cohorts += 1
                    self._persistence_last_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    self._prepared_candidates.pop(
                        prepared._candidate_id,
                        None,
                    )
                    raise
        self._fusion = state.fusion
        self._sequence_by_source = state.sequence_by_source
        self._event_sequence = int(state.event_sequence)
        self._previous_entities = state.previous_entities
        self._canonical_track_outputs = state.canonical_track_outputs
        self._current_snapshot = state.publication.snapshot
        self._revision += 1
        self._prepared_candidates.clear()
        return state.publication

    def _validate_prepared_owner(
        self,
        prepared: PreparedWorldPublication,
    ) -> None:
        if not isinstance(prepared, PreparedWorldPublication):
            raise TypeError("canonical world operation requires a prepared publication")
        if prepared._service_token is not self._service_token:
            raise RuntimeError("canonical world preparation belongs to another service")

    def close(self) -> None:
        # Serialize close with publication so an in-flight frame either reaches
        # the journal completely or finishes before the journal is closed.
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._prepared_candidates.clear()
            close = getattr(self._journal, "close", None)
            close_error: BaseException | None = None
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    close_error = exc
            if isinstance(self._journal, AsyncContractJournal):
                durable_close = getattr(self._journal.journal, "close", None)
                if callable(durable_close):
                    try:
                        durable_close()
                    except BaseException as exc:
                        if close_error is None:
                            close_error = exc
            if close_error is not None:
                raise close_error

    def _events_for_state(
        self,
        snapshot: WorldSnapshot,
        *,
        previous_entities: Mapping[str, WorldEntity],
        event_sequence: int,
    ) -> tuple[tuple[WorldEvent, ...], int, dict[str, WorldEntity]]:
        current = {entity.entity_id: entity for entity in snapshot.entities}
        pending: list[tuple[str, WorldEntity, str | None]] = []
        for entity_id, entity in sorted(current.items()):
            previous = previous_entities.get(entity_id)
            if previous is None:
                pending.append(("appeared", entity, None))
            else:
                if (
                    previous.lifecycle != EntityLifecycle.HELD
                    and entity.lifecycle == EntityLifecycle.HELD
                ):
                    pending.append(("held", entity, "producer_evidence_stale"))
                elif (
                    previous.lifecycle == EntityLifecycle.HELD
                    and entity.lifecycle == EntityLifecycle.PRESENT
                ):
                    pending.append(("resumed", entity, "fresh_producer_evidence"))
                if not previous.conflict and entity.conflict:
                    pending.append(("conflict_started", entity, entity.conflict_reason))
                elif previous.conflict and not entity.conflict:
                    pending.append(("conflict_cleared", entity, "camera_evidence_reconciled"))
        for entity_id, previous in sorted(previous_entities.items()):
            if entity_id not in current:
                pending.append(("lost", previous, "producer_evidence_expired"))

        events: list[WorldEvent] = []
        for event_type, entity, reason in pending:
            sequence = int(event_sequence)
            event_sequence += 1
            events.append(
                WorldEvent(
                    contract="noesis.world.event",
                    contract_version=1,
                    event_id=f"{self.producer.run_id}:event:{sequence}",
                    producer=self.producer,
                    sequence=sequence,
                    event_type=event_type,  # type: ignore[arg-type]
                    entity_id=entity.entity_id,
                    subject=entity.subject,
                    observed_at_us=min(entity.observed_at_us, snapshot.published_at_us),
                    published_at_us=snapshot.published_at_us,
                    frame="backend_world_m",
                    units="meters",
                    world_frame="backend_world_m",
                    world_frame_revision=entity.world_frame_revision,
                    world_transform_sha256=entity.world_transform_sha256,
                    calibration_revision=entity.calibration_revision,
                    position=entity.position,
                    reason=reason,
                )
            )
        return tuple(events), int(event_sequence), current

    @staticmethod
    def _next_sequence(
        source_id: int,
        sequence_by_source: dict[int, int],
    ) -> int:
        value = sequence_by_source.get(source_id, 0)
        sequence_by_source[source_id] = value + 1
        return value

    def _canonical_track_output_key(
        self,
        source_id: int,
        track: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
        calibration_sha256: str,
    ) -> tuple[int, str, int, int, str, str, str, int] | None:
        """Resolve one epoch-, revision-, and lifecycle-bound output key."""

        try:
            tracker_id = int(track.get("tracker_id", track.get("track_id")))
            lifecycle_generation = int(
                track.get("tracker_lifecycle_generation")
            )
        except (TypeError, ValueError, OverflowError):
            return None
        camera_id = str(
            track.get("camera_id")
            or metadata.get("camera_id")
            or f"camera_{int(source_id)}"
        ).strip()
        world_frame_revision = self._required_revision(
            track.get("world_frame_revision")
        )
        world_transform_sha256 = self._sha256_text(
            track.get("world_transform_sha256")
        )
        calibration_artifact_sha256 = self._sha256_text(
            calibration_sha256
        )
        source_epoch = self._nonnegative_int(
            track.get("source_epoch")
            if track.get("source_epoch") is not None
            else metadata.get("source_epoch")
        )
        if source_epoch is None:
            source_epoch = 0
        if (
            tracker_id < 0
            or lifecycle_generation <= 0
            or not camera_id
            or world_frame_revision is None
            or world_transform_sha256 is None
            or calibration_artifact_sha256 is None
        ):
            return None
        return (
            int(source_id),
            camera_id,
            tracker_id,
            lifecycle_generation,
            world_frame_revision,
            world_transform_sha256,
            calibration_artifact_sha256,
            int(source_epoch),
        )

    @staticmethod
    def _prune_superseded_track_outputs(
        outputs: dict[
            tuple[int, str, int, int, str, str, str, int],
            _CanonicalTrackOutput,
        ],
        current_key: tuple[int, str, int, int, str, str, str, int],
    ) -> None:
        """Drop older lifecycle/revision bindings for one source-local ID."""

        source_id, camera_id, tracker_id = current_key[:3]
        for key in tuple(outputs):
            if (
                key != current_key
                and key[0] == source_id
                and key[1] == camera_id
                and key[2] == tracker_id
            ):
                outputs.pop(key, None)

    def _bound_track_outputs(
        self,
        outputs: dict[
            tuple[int, str, int, int, str, str, str, int],
            _CanonicalTrackOutput,
        ],
    ) -> None:
        """Apply a deterministic global cap to compact continuity origins."""

        capacity = max(1, int(self.MAX_CANONICAL_TRACK_OUTPUTS))
        while len(outputs) > capacity:
            oldest_key = next(iter(outputs))
            outputs.pop(oldest_key, None)

    @classmethod
    def _prune_expired_track_outputs(
        cls,
        outputs: dict[
            tuple[int, str, int, int, str, str, str, int],
            _CanonicalTrackOutput,
        ],
        *,
        source_id: int,
        current_media_pts_ns: int | None,
    ) -> None:
        """Drop origins too old for every canonical continuation lane."""

        if current_media_pts_ns is None or current_media_pts_ns <= 0:
            return
        short_horizon_ns = int(
            round(
                (
                    cls.BOUNDED_PROCESS_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        stationary_horizon_ns = int(
            round(
                (
                    cls.STATIONARY_ANCHOR_HOLD_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        upright_presence_horizon_ns = int(
            round(
                (
                    cls.UPRIGHT_PRESENCE_HOLD_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        committed_origin_history_horizon_ns = int(
            round(
                cls.IMAGE_MOTION_COMMITTED_ORIGIN_HISTORY_HORIZON_S
                * 1_000_000_000.0
            )
        )
        for key, output in tuple(outputs.items()):
            if key[0] != int(source_id):
                continue
            output_age_ns = current_media_pts_ns - int(output.media_pts_ns)
            metric_age_ns = (
                current_media_pts_ns - int(output.metric_media_pts_ns)
                if output.metric_media_pts_ns is not None
                else None
            )
            if output_age_ns < 0 or (
                metric_age_ns is not None and metric_age_ns < 0
            ):
                # A reordered source cohort is handled by the producer's epoch
                # contract. Do not let this cache invent a reset independently.
                continue
            recent_projective_origin = bool(
                output.committed_world_source == "image_motion_prediction"
                and output_age_ns <= short_horizon_ns
            )
            image_motion_validation_origin = bool(
                output_age_ns <= committed_origin_history_horizon_ns
            )
            upright_presence_validation_origin = bool(
                output_age_ns <= upright_presence_horizon_ns
            )
            stationary_metric_origin = bool(
                metric_age_ns is not None
                and metric_age_ns <= stationary_horizon_ns
            )
            if not (
                recent_projective_origin
                or image_motion_validation_origin
                or upright_presence_validation_origin
                or stationary_metric_origin
            ):
                outputs.pop(key, None)

    def _observation(
        self,
        source_id: int,
        track: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
        artifacts: WorldArtifacts,
        published_at_us: int,
        sequence_by_source: dict[int, int],
        expected_image_motion_origin: _CanonicalTrackOutput | None,
    ) -> ObservationEnvelope | None:
        try:
            tracker_id = int(track.get("tracker_id", track.get("track_id")))
            frame_id = int(track.get("frame_id", 0))
        except Exception:
            return None
        if tracker_id < 0 or frame_id < 0:
            return None
        camera_id = str(track.get("camera_id") or metadata.get("camera_id") or f"camera_{source_id}").strip()
        if not camera_id:
            return None
        observed_at_us = self._positive_int(track.get("observed_at_us")) or published_at_us
        published_at_us = max(published_at_us, observed_at_us)
        capture_status = str(track.get("capture_time_status") or "estimated")
        if capture_status not in {"synced", "estimated", "unavailable"}:
            capture_status = "unavailable"
        captured_at_us = None
        if capture_status != "unavailable":
            captured_at_us = (
                self._positive_int(track.get("captured_at_us"))
                or self._positive_int(metadata.get("captured_at_us"))
                or observed_at_us
            )
        media_pts_ns = self._nonnegative_int(track.get("media_pts_ns"))

        bbox = track.get("bbox")
        image_size = track.get("image_size") or metadata.get("image_size") or metadata.get("frame_size")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        if not isinstance(image_size, (list, tuple)) or len(image_size) < 2:
            return None
        try:
            bbox_xywh = tuple(float(value) for value in bbox[:4])
            size = (int(image_size[0]), int(image_size[1]))
        except Exception:
            return None
        if not all(math.isfinite(value) for value in bbox_xywh) or size[0] <= 0 or size[1] <= 0:
            return None

        calibration_revision = track.get("calibration_revision")
        if calibration_revision is None:
            calibration_revision = artifacts.calibration.version
        raw_camera_calibration_sha256 = (
            artifacts.camera_calibration_sha256
        )
        camera_calibration_binding_required = bool(
            raw_camera_calibration_sha256 is not None
        )
        if camera_calibration_binding_required:
            expected_camera_calibration_sha256 = self._sha256_text(
                raw_camera_calibration_sha256
            )
            camera_calibration_matches = bool(
                expected_camera_calibration_sha256 is not None
                and self._sha256_text(
                    track.get("world_calibration_sha256")
                )
                == expected_camera_calibration_sha256
            )
        else:
            camera_calibration_matches = True
        world_observation = (
            self._world_observation(
                track,
                calibration_revision=calibration_revision,
                expected_image_motion_origin=expected_image_motion_origin,
            )
            if camera_calibration_matches
            else None
        )
        world_diagnostics = (
            self._world_diagnostics(
                track,
                world_observation=world_observation,
                expected_continuity_origin=expected_image_motion_origin,
            )
            if camera_calibration_matches
            else WorldObservationDiagnostics(
                first_divergence_reason=(
                    "camera_calibration_sha256_mismatch"
                )
            )
        )
        coordinate_frame = "backend_world_m" if world_observation is not None else "image_px"
        units = "meters" if world_observation is not None else "pixels"
        tracklet = TrackletRef(
            run_id=self.producer.run_id,
            camera_id=camera_id,
            source_id=source_id,
            tracker_id=tracker_id,
            frame_id=frame_id,
            observed_at_us=observed_at_us,
        )
        embedding_provenance = self._embedding_provenance(track)
        payload = PersonObservation(
            tracklet=tracklet,
            detection_confidence=self._confidence(track.get("confidence")),
            tracker_confidence=self._confidence(track.get("tracker_confidence")),
            bbox_xywh=bbox_xywh,  # type: ignore[arg-type]
            image_size=size,
            zone=self._exact_zone(track.get("zone")),
            zone_source=self._zone_source(track.get("zone_source")),
            zone_authoritative=track.get("zone_authoritative") is True,
            world=world_observation,
            world_diagnostics=world_diagnostics,
            **embedding_provenance,
            pose_present=bool(track.get("pose_present", False)),
            depth_present=self._depth_present(track),
            occluded=bool(track.get("occluded", False)),
        )
        sequence = self._next_sequence(source_id, sequence_by_source)
        observation_id = f"{self.producer.run_id}:{source_id}:{sequence}:{tracker_id}"
        return ObservationEnvelope(
            contract="noesis.observation.person",
            contract_version=1,
            observation_id=observation_id,
            producer=self.producer,
            sequence=sequence,
            captured_at_us=captured_at_us,
            observed_at_us=observed_at_us,
            published_at_us=published_at_us,
            capture_time_status=capture_status,  # type: ignore[arg-type]
            media_pts_ns=media_pts_ns,
            coordinate_frame=coordinate_frame,  # type: ignore[arg-type]
            units=units,  # type: ignore[arg-type]
            calibration=artifacts.calibration,
            model=artifacts.model,
            config=artifacts.config,
            payload=payload,
        )

    @staticmethod
    def _embedding_provenance(track: Mapping[str, Any]) -> dict[str, Any]:
        fields = (
            "embedding_sequence",
            "embedding_model_sha256",
            "embedding_dimension",
        )
        present = tuple(track.get(field) is not None for field in fields)
        if any(present) and not all(present):
            populated = ", ".join(
                field for field, is_present in zip(fields, present) if is_present
            )
            raise ValueError(
                "person track contains partial embedding provenance; "
                f"populated fields: {populated}"
            )
        if not any(present):
            return {}
        return {field: track[field] for field in fields}

    @staticmethod
    def _depth_present(track: Mapping[str, Any]) -> bool:
        return usable_registered_depth_m(track) is not None

    @staticmethod
    def _inferred_ground_episode_root_from_provenance(
        provenance: Mapping[str, Any],
    ) -> _InferredGroundEpisodeRoot | None:
        """Parse the immutable root fields without accepting producer authority."""

        raw_origin = provenance.get("raw_origin")
        trusted_world_origin = provenance.get("trusted_world_origin")
        if (
            not isinstance(raw_origin, (list, tuple))
            or len(raw_origin) < 3
            or not isinstance(trusted_world_origin, (list, tuple))
            or len(trusted_world_origin) < 3
        ):
            return None
        world_frame = provenance.get("world_frame")
        world_frame_revision = provenance.get("world_frame_revision")
        world_transform_sha256 = provenance.get("world_transform_sha256")
        if not all(
            isinstance(value, str) and bool(value)
            for value in (
                world_frame,
                world_frame_revision,
                world_transform_sha256,
            )
        ):
            return None
        try:
            root = _InferredGroundEpisodeRoot(
                raw_origin=tuple(float(value) for value in raw_origin[:3]),
                raw_origin_media_pts_ns=int(
                    provenance.get("raw_origin_media_pts_ns")
                ),
                raw_origin_observed_at_us=int(
                    provenance.get("origin_observed_at_us")
                ),
                raw_origin_ts_s=float(provenance.get("raw_origin_ts_s")),
                trusted_world_origin=tuple(
                    float(value) for value in trusted_world_origin[:3]
                ),
                trusted_origin_media_pts_ns=int(
                    provenance.get("trusted_origin_media_pts_ns")
                ),
                trusted_origin_filter_ts_s=float(
                    provenance.get("trusted_origin_filter_ts_s")
                ),
                lifecycle_generation=int(
                    provenance.get("tracker_lifecycle_generation")
                ),
                trail_segment_id=int(
                    provenance.get("origin_trail_segment_id")
                ),
                height_ref_scene=float(provenance.get("height_ref_scene")),
                world_frame=world_frame,
                world_frame_revision=world_frame_revision,
                world_transform_sha256=world_transform_sha256,
            )
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(
            math.isfinite(value)
            for value in (
                *root.raw_origin,
                root.raw_origin_ts_s,
                *root.trusted_world_origin,
                root.trusted_origin_filter_ts_s,
                root.height_ref_scene,
            )
        ):
            return None
        return root

    @classmethod
    def _inferred_ground_episode_root_for_admitted_output(
        cls,
        track: Mapping[str, Any],
        *,
        expected_origin: _CanonicalTrackOutput | None,
        current_media_pts_ns: int,
    ) -> _InferredGroundEpisodeRoot | None:
        """Establish one inferred root or retain it through bounded descendants."""

        provenance = track.get("world_prediction_provenance")
        if not isinstance(provenance, Mapping):
            return None
        provenance_type = str(provenance.get("type") or "")
        source = str(track.get("world_source") or "").strip().lower()
        established_root = (
            expected_origin.inferred_ground_episode_root
            if expected_origin is not None
            else None
        )
        if (
            source == "image_motion_prediction"
            and provenance_type == "inferred_ground_process_observation"
        ):
            current_root = cls._inferred_ground_episode_root_from_provenance(
                provenance
            )
            if current_root is None:
                return None
            if established_root is not None and current_root != established_root:
                return None
            return established_root or current_root
        if (
            expected_origin is None
            or established_root is None
            or source not in {"cv_prediction", "anchor_hold"}
            or provenance_type not in {
                "bounded_cv_process",
                "bounded_output_hold",
            }
        ):
            return None
        # The immutable root belongs to one *active lower-body occlusion*
        # episode, not to the tracker lifecycle indefinitely.  A bounded CV
        # row or exact hold can bridge a brief estimator miss while that same
        # standing occlusion is still visible.  Once the public row says the
        # lower body is no longer occluded (or posture/motion leaves the
        # upright lane), terminate the service-owned root.  This lets a later
        # independently observed occlusion establish a fresh immutable root
        # instead of being rejected as an attempted renewal of an episode
        # that has already ended.
        occlusion_level = str(
            track.get("lower_body_occlusion_level") or ""
        ).strip()
        if not (
            track.get("lower_body_occluded") is True
            and occlusion_level
            and occlusion_level != "none"
            and str(track.get("posture") or "").strip().lower()
            == "standing"
            and str(track.get("motion_mode") or "").strip().lower()
            not in {"sit", "lie"}
        ):
            return None
        try:
            lifecycle_generation = int(
                track.get("tracker_lifecycle_generation")
            )
            trail_segment_id = int(track.get("trail_segment_id"))
        except (TypeError, ValueError, OverflowError):
            return None
        maximum_age_ns = int(
            round(cls.INFERRED_PROCESS_RESTART_MAX_AGE_S * 1_000_000_000.0)
        )
        if not (
            0 < current_media_pts_ns - int(expected_origin.media_pts_ns)
            <= maximum_age_ns
            and lifecycle_generation == established_root.lifecycle_generation
            and trail_segment_id == established_root.trail_segment_id
            and trail_segment_id == int(expected_origin.trail_segment_id)
            and track.get("world_frame") == established_root.world_frame
            and track.get("world_frame_revision")
            == established_root.world_frame_revision
            and track.get("world_transform_sha256")
            == established_root.world_transform_sha256
        ):
            return None
        return established_root

    @classmethod
    def _matched_committed_output_origin(
        cls,
        expected_origin: _CanonicalTrackOutput,
        *,
        origin_point: tuple[float, float, float],
        origin_media_pts_ns: int,
        origin_trail_segment_id: int,
        current_media_pts_ns: int,
        maximum_age_s: float,
    ) -> _CanonicalOutputOrigin | None:
        """Bind a proof to one exact output this service committed.

        The media estimator and ordered publication worker advance on separate
        threads, so a proof created from a committed output can arrive after a
        few newer outputs have also committed.  The bounded history contains
        service-owned facts only; producer-supplied coordinates cannot create
        an origin.
        """

        candidates = (
            _CanonicalOutputOrigin(
                position=expected_origin.position,
                media_pts_ns=int(expected_origin.media_pts_ns),
                trail_segment_id=int(expected_origin.trail_segment_id),
                committed_world_source=expected_origin.committed_world_source,
                committed_provenance_type=(
                    expected_origin.committed_provenance_type
                ),
                metric_position=expected_origin.metric_position,
                metric_media_pts_ns=expected_origin.metric_media_pts_ns,
                metric_trail_segment_id=(
                    expected_origin.metric_trail_segment_id
                ),
                projective_bridge_root_media_pts_ns=(
                    expected_origin.projective_bridge_root_media_pts_ns
                ),
                kinematic_position=expected_origin.kinematic_position,
                kinematic_media_pts_ns=(
                    expected_origin.kinematic_media_pts_ns
                ),
                kinematic_trail_segment_id=(
                    expected_origin.kinematic_trail_segment_id
                ),
                bbox_geometry=expected_origin.bbox_geometry,
                kinematic_bbox_geometry=(
                    expected_origin.kinematic_bbox_geometry
                ),
                inferred_ground_episode_root=(
                    expected_origin.inferred_ground_episode_root
                ),
            ),
            *expected_origin.prior_outputs,
        )
        history_horizon_ns = int(
            round(max(0.0, float(maximum_age_s)) * 1_000_000_000.0)
        )
        for candidate in candidates:
            if (
                0 < current_media_pts_ns - int(candidate.media_pts_ns)
                <= history_horizon_ns
                and origin_media_pts_ns == int(candidate.media_pts_ns)
                and origin_trail_segment_id == int(candidate.trail_segment_id)
                and all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                    for left, right in zip(origin_point, candidate.position)
                )
            ):
                return candidate
        return None

    @classmethod
    def _matches_committed_output_origin(
        cls,
        expected_origin: _CanonicalTrackOutput,
        *,
        origin_point: tuple[float, float, float],
        origin_media_pts_ns: int,
        origin_trail_segment_id: int,
        current_media_pts_ns: int,
    ) -> bool:
        return (
            cls._matched_committed_output_origin(
                expected_origin,
                origin_point=origin_point,
                origin_media_pts_ns=origin_media_pts_ns,
                origin_trail_segment_id=origin_trail_segment_id,
                current_media_pts_ns=current_media_pts_ns,
                maximum_age_s=(
                    cls.IMAGE_MOTION_COMMITTED_ORIGIN_HISTORY_HORIZON_S
                ),
            )
            is not None
        )

    @classmethod
    def _matched_inferred_ground_restart_origin(
        cls,
        expected_origin: _CanonicalTrackOutput,
        provenance: Mapping[str, Any],
        *,
        current_media_pts_ns: int,
    ) -> _CanonicalOutputOrigin | None:
        """Resolve the exact recent service output named by a restart proof."""

        transition = provenance.get("filter_transition")
        if not isinstance(transition, Mapping):
            return None
        origin_world = transition.get("origin_world")
        if not isinstance(origin_world, (list, tuple)) or len(origin_world) < 3:
            return None
        try:
            origin_point = tuple(float(value) for value in origin_world[:3])
            origin_media_pts_ns = int(
                transition.get("origin_media_pts_ns")
            )
            origin_trail_segment_id = int(
                transition.get("origin_trail_segment_id")
            )
        except (TypeError, ValueError, OverflowError):
            return None
        return cls._matched_committed_output_origin(
            expected_origin,
            origin_point=origin_point,
            origin_media_pts_ns=origin_media_pts_ns,
            origin_trail_segment_id=origin_trail_segment_id,
            current_media_pts_ns=current_media_pts_ns,
            maximum_age_s=cls.INFERRED_PROCESS_RESTART_MAX_AGE_S,
        )

    @classmethod
    def _is_projective_image_output(
        cls,
        origin: _CanonicalOutputOrigin | _CanonicalTrackOutput,
    ) -> bool:
        return bool(
            origin.committed_world_source == "image_motion_prediction"
            and origin.committed_provenance_type
            in cls.PROJECTIVE_BRIDGE_IMAGE_PROVENANCE_TYPES
        )

    @classmethod
    def _active_projective_bridge_root(
        cls,
        latest_origin: _CanonicalTrackOutput,
        *,
        current_media_pts_ns: int,
    ) -> int | None:
        """Return the latest service-owned immutable projective episode root.

        A newly admitted image-motion output establishes a root. Bounded CV
        descendants and exact gain-zero holds may inherit that same root, but
        never replace it with their own publication PTS or resurrect a
        historical image after a metric/ordinary output terminated the lane.
        """

        if cls._is_projective_image_output(latest_origin):
            root_media_pts_ns = int(latest_origin.media_pts_ns)
        elif latest_origin.projective_bridge_root_media_pts_ns is not None:
            root_media_pts_ns = int(
                latest_origin.projective_bridge_root_media_pts_ns
            )
        else:
            return None
        horizon_ns = int(
            round(
                (
                    cls.PROJECTIVE_PROCESS_BRIDGE_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        if not (
            0 < root_media_pts_ns < current_media_pts_ns
            and current_media_pts_ns - root_media_pts_ns <= horizon_ns
        ):
            return None
        return root_media_pts_ns

    @classmethod
    def _projective_bridge_root_for_admitted_output(
        cls,
        track: Mapping[str, Any],
        *,
        expected_origin: _CanonicalTrackOutput | None,
        current_media_pts_ns: int,
    ) -> int | None:
        """Retain service-owned nonrenewing lineage for an admitted output."""

        source = str(track.get("world_source") or "").strip().lower()
        provenance = track.get("world_prediction_provenance")
        provenance_type = (
            str(provenance.get("type") or "").strip()
            if isinstance(provenance, Mapping)
            else ""
        )
        if (
            source == "image_motion_prediction"
            and provenance_type in cls.PROJECTIVE_BRIDGE_IMAGE_PROVENANCE_TYPES
        ):
            return int(current_media_pts_ns)
        if (
            expected_origin is None
            or source not in {"cv_prediction", "anchor_hold"}
            or provenance_type
            not in {"bounded_cv_process", "bounded_output_hold"}
        ):
            return None
        return cls._active_projective_bridge_root(
            expected_origin,
            current_media_pts_ns=current_media_pts_ns,
        )

    @classmethod
    def _latest_output_step_is_physical(
        cls,
        expected_origin: _CanonicalTrackOutput,
        *,
        world_point: tuple[float, float, float],
        current_media_pts_ns: int,
        max_speed_mps: float | None = None,
    ) -> bool:
        """Prevent a valid older-origin proof from jumping past latest state."""

        kinematic_position = (
            expected_origin.kinematic_position
            or expected_origin.position
        )
        latest_media_pts_ns = int(
            expected_origin.kinematic_media_pts_ns
            if expected_origin.kinematic_media_pts_ns is not None
            else expected_origin.media_pts_ns
        )
        if current_media_pts_ns <= latest_media_pts_ns:
            return False
        latest_dt_s = (
            current_media_pts_ns - latest_media_pts_ns
        ) / 1_000_000_000.0
        latest_gate_dt_s = min(
            latest_dt_s,
            cls.IMAGE_MOTION_RESET_AFTER_S,
        )
        latest_step_m = math.hypot(
            world_point[0] - kinematic_position[0],
            world_point[2] - kinematic_position[2],
        )
        speed_limit_mps = (
            cls.IMAGE_MOTION_MAX_SPEED_MPS
            if max_speed_mps is None
            else float(max_speed_mps)
        )
        return (
            math.isfinite(speed_limit_mps)
            and 0.0 < speed_limit_mps <= cls.IMAGE_MOTION_MAX_SPEED_MPS
            and math.isfinite(latest_step_m)
            and latest_step_m
            <= speed_limit_mps * latest_gate_dt_s + 1e-9
        )

    @classmethod
    def _latest_visible_output_step_is_physical(
        cls,
        expected_origin: _CanonicalTrackOutput,
        *,
        world_point: tuple[float, float, float],
        current_media_pts_ns: int,
        max_speed_mps: float | None = None,
    ) -> bool:
        """Bound motion against the most recent actually displayed point.

        Gain-zero holds deliberately retain an older kinematic clock so a
        later observation can still be evaluated against the real motion
        interval. That proof budget must not become a one-frame display
        teleport. The producer reduces its canonical filter gain to satisfy
        this second bound; the service independently verifies the emitted
        coordinate from its own latest committed output.
        """

        latest_media_pts_ns = int(expected_origin.media_pts_ns)
        if current_media_pts_ns <= latest_media_pts_ns:
            return False
        visible_dt_s = (
            current_media_pts_ns - latest_media_pts_ns
        ) / 1_000_000_000.0
        visible_gate_dt_s = min(
            visible_dt_s,
            cls.IMAGE_MOTION_RESET_AFTER_S,
        )
        visible_step_m = math.hypot(
            world_point[0] - expected_origin.position[0],
            world_point[2] - expected_origin.position[2],
        )
        speed_limit_mps = (
            cls.IMAGE_MOTION_MAX_SPEED_MPS
            if max_speed_mps is None
            else float(max_speed_mps)
        )
        return (
            math.isfinite(speed_limit_mps)
            and 0.0 < speed_limit_mps <= cls.IMAGE_MOTION_MAX_SPEED_MPS
            and math.isfinite(visible_step_m)
            and visible_step_m
            <= speed_limit_mps * visible_gate_dt_s + 1e-9
        )

    @classmethod
    def _image_motion_filter_transition_is_canonical(
        cls,
        track: Mapping[str, Any],
        provenance: Mapping[str, Any],
        *,
        world_point: tuple[float, float, float],
        expected_origin: _CanonicalTrackOutput | None,
    ) -> bool:
        """Verify the exact queue-origin -> weak-filter posterior algebra."""

        if expected_origin is None:
            return False
        process_observation = provenance.get("process_observation")
        transition = provenance.get("filter_transition")
        if (
            not isinstance(process_observation, (list, tuple))
            or len(process_observation) < 3
            or not isinstance(transition, Mapping)
        ):
            return False
        origin_world = transition.get("origin_world")
        position_base = transition.get("position_base")
        if (
            not isinstance(origin_world, (list, tuple))
            or len(origin_world) < 3
            or not isinstance(position_base, (list, tuple))
            or len(position_base) < 3
        ):
            return False
        try:
            process_point = tuple(
                float(value) for value in process_observation[:3]
            )
            origin_point = tuple(float(value) for value in origin_world[:3])
            base_point = tuple(float(value) for value in position_base[:3])
            version = int(transition.get("version"))
            origin_media_pts_ns = int(
                transition.get("origin_media_pts_ns")
            )
            current_media_pts_ns = int(
                transition.get("current_media_pts_ns")
            )
            track_media_pts_ns = int(track.get("media_pts_ns"))
            origin_trail_segment_id = int(
                transition.get("origin_trail_segment_id")
            )
            track_trail_segment_id = int(track.get("trail_segment_id"))
            gate_dt_s = float(transition.get("gate_dt_s"))
            position_gain = float(transition.get("position_gain"))
            max_speed_mps = float(transition.get("max_speed_mps"))
            max_jump_m = float(transition.get("max_jump_m"))
            reset_after_s = float(transition.get("reset_after_s"))
        except (TypeError, ValueError, OverflowError):
            return False
        filter_point = process_point
        innovation_limit_applied = bool(
            transition.get("innovation_limit_applied") is True
        )
        raw_position_target = transition.get("raw_position_target")
        filter_observation = transition.get("filter_observation")
        innovation_scale = math.nan
        proof_raw_innovation_m = math.nan
        proof_innovation_limit_m = math.nan
        if innovation_limit_applied:
            if (
                not isinstance(raw_position_target, (list, tuple))
                or len(raw_position_target) < 3
                or not isinstance(filter_observation, (list, tuple))
                or len(filter_observation) < 3
            ):
                return False
            try:
                proof_raw_target = tuple(
                    float(value) for value in raw_position_target[:3]
                )
                filter_point = tuple(
                    float(value) for value in filter_observation[:3]
                )
                innovation_scale = float(transition.get("innovation_scale"))
                proof_raw_innovation_m = float(
                    transition.get("raw_innovation_m")
                )
                proof_innovation_limit_m = float(
                    transition.get("innovation_limit_m")
                )
            except (TypeError, ValueError, OverflowError):
                return False
            if not all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(proof_raw_target, process_point)
            ):
                return False
        elif (
            raw_position_target is not None
            or filter_observation is not None
            or transition.get("innovation_scale") is not None
            or transition.get("raw_innovation_m") is not None
            or transition.get("innovation_limit_m") is not None
        ):
            # The alternate target is meaningful only as one complete robust
            # innovation proof.  Partial fields must not change the posterior
            # algebra selected by the producer.
            return False

        vectors = (
            world_point,
            process_point,
            filter_point,
            origin_point,
            base_point,
        )
        if not all(
            math.isfinite(value)
            for point in vectors
            for value in point
        ):
            return False
        if (
            version != cls.IMAGE_MOTION_TRANSITION_VERSION
            or transition.get("origin_kind")
            != "queue_admitted_world_output"
            or not math.isclose(
                max_speed_mps,
                cls.IMAGE_MOTION_MAX_SPEED_MPS,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                max_jump_m,
                cls.IMAGE_MOTION_MAX_JUMP_M,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                reset_after_s,
                cls.IMAGE_MOTION_RESET_AFTER_S,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            return False
        if (
            origin_media_pts_ns <= 0
            or current_media_pts_ns <= origin_media_pts_ns
            or current_media_pts_ns != track_media_pts_ns
            or origin_trail_segment_id != track_trail_segment_id
        ):
            return False
        if not cls._matches_committed_output_origin(
            expected_origin,
            origin_point=origin_point,
            origin_media_pts_ns=origin_media_pts_ns,
            origin_trail_segment_id=origin_trail_segment_id,
            current_media_pts_ns=current_media_pts_ns,
        ):
            return False
        # Every vector describes the same calibrated ground plane.
        if not all(
            math.isclose(
                point[1],
                world_point[1],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for point in (process_point, origin_point, base_point)
        ):
            return False

        media_dt_s = (
            current_media_pts_ns - origin_media_pts_ns
        ) / 1_000_000_000.0
        expected_gate_dt_s = min(
            media_dt_s,
            cls.IMAGE_MOTION_RESET_AFTER_S,
        )
        if (
            not math.isfinite(gate_dt_s)
            or gate_dt_s <= 0.0
            or not math.isclose(
                gate_dt_s,
                expected_gate_dt_s,
                rel_tol=0.0,
                abs_tol=cls.IMAGE_MOTION_GATE_DT_ABS_TOL_S,
            )
        ):
            return False

        base_step_m = math.hypot(
            base_point[0] - origin_point[0],
            base_point[2] - origin_point[2],
        )
        raw_process_innovation_m = math.hypot(
            process_point[0] - base_point[0],
            process_point[2] - base_point[2],
        )
        filter_innovation_m = math.hypot(
            filter_point[0] - base_point[0],
            filter_point[2] - base_point[2],
        )
        innovation_limit_m = (
            cls.IMAGE_MOTION_MAX_JUMP_M
            + cls.IMAGE_MOTION_MAX_SPEED_MPS * gate_dt_s
        )
        posterior_step_m = math.hypot(
            world_point[0] - origin_point[0],
            world_point[2] - origin_point[2],
        )
        if innovation_limit_applied:
            expected_scale = innovation_limit_m / raw_process_innovation_m
            expected_filter_point = (
                base_point[0]
                + expected_scale * (process_point[0] - base_point[0]),
                base_point[1],
                base_point[2]
                + expected_scale * (process_point[2] - base_point[2]),
            )
            if (
                raw_process_innovation_m <= innovation_limit_m + 1e-9
                or not (0.0 < innovation_scale < 1.0)
                or not math.isclose(
                    proof_raw_innovation_m,
                    raw_process_innovation_m,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    proof_innovation_limit_m,
                    innovation_limit_m,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    innovation_scale,
                    expected_scale,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                    for left, right in zip(
                        filter_point,
                        expected_filter_point,
                    )
                )
            ):
                return False
        if (
            base_step_m
            > cls.IMAGE_MOTION_MAX_SPEED_MPS * gate_dt_s + 1e-9
            or filter_innovation_m > innovation_limit_m + 1e-9
            or posterior_step_m
            > cls.IMAGE_MOTION_MAX_SPEED_MPS * gate_dt_s + 1e-9
        ):
            return False
        if not cls._latest_output_step_is_physical(
            expected_origin,
            world_point=world_point,
            current_media_pts_ns=current_media_pts_ns,
        ):
            return False
        if not cls._latest_visible_output_step_is_physical(
            expected_origin,
            world_point=world_point,
            current_media_pts_ns=current_media_pts_ns,
        ):
            return False

        transition_kind = str(transition.get("kind") or "")
        if transition_kind == "innovation_update":
            if not (0.0 < position_gain <= 1.0):
                return False
        elif transition_kind == "deadzone_hold":
            if not math.isclose(
                position_gain,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        else:
            return False
        expected_posterior = (
            base_point[0]
            + position_gain * (filter_point[0] - base_point[0]),
            base_point[1],
            base_point[2]
            + position_gain * (filter_point[2] - base_point[2]),
        )
        return all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(world_point, expected_posterior)
        )

    @classmethod
    def _image_motion_observation_is_canonical(
        cls,
        track: Mapping[str, Any],
        *,
        expected_origin: _CanonicalTrackOutput | None = None,
    ) -> bool:
        """Validate one current calibrated image-foot continuation.

        This is not fresh metric evidence, but it is a current Noesis-owned
        floor coordinate. Admit it only with the complete fixed-origin
        projective proof emitted by the DS9 estimator so the canonical held
        snapshot consumed by Menon is exactly the same point as tracking/BEV.
        """

        if str(track.get("world_quality") or "").strip().lower() != "held":
            return False
        if track.get("world_measurement_accepted") is not False:
            return False
        provenance = track.get("world_prediction_provenance")
        if not isinstance(provenance, Mapping):
            return False
        if provenance.get("non_authoritative") is not True:
            return False
        if provenance.get("state_integrated") is not True:
            return False
        world = track.get("world")
        posterior = track.get("world_filter_prediction")
        if (
            not isinstance(world, (list, tuple))
            or len(world) < 3
            or not isinstance(posterior, (list, tuple))
            or len(posterior) < 3
        ):
            return False
        try:
            world_point = tuple(float(value) for value in world[:3])
            posterior_point = tuple(float(value) for value in posterior[:3])
        except (TypeError, ValueError, OverflowError):
            return False
        if not (
            all(math.isfinite(value) for value in world_point + posterior_point)
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(world_point, posterior_point)
            )
        ):
            return False
        if not cls._image_motion_filter_transition_is_canonical(
            track,
            provenance,
            world_point=world_point,
            expected_origin=expected_origin,
        ):
            return False
        provenance_type = str(provenance.get("type") or "")
        if provenance_type == "bbox_affine_floor_projection":
            if provenance.get("origin") not in {
                "last_accepted_image_foot",
                "last_accepted_pose_projective_origin",
            }:
                return False
            if provenance.get("transport") not in {
                "bbox_affine",
                "pose_torso_translation",
            }:
                return False
            image_foot = provenance.get("image_foot")
            projective_origin_world = provenance.get(
                "projective_origin_world"
            )
            process_observation = provenance.get("process_observation")
            if not isinstance(image_foot, (list, tuple)) or len(image_foot) != 2:
                return False
            if (
                not isinstance(projective_origin_world, (list, tuple))
                or len(projective_origin_world) < 3
                or not isinstance(process_observation, (list, tuple))
                or len(process_observation) < 3
            ):
                return False
            try:
                values = tuple(float(value) for value in image_foot)
                projective_origin_point = tuple(
                    float(value) for value in projective_origin_world[:3]
                )
                process_point = tuple(
                    float(value) for value in process_observation[:3]
                )
                age_s = float(provenance.get("age_s"))
                world_delta_m = float(provenance.get("world_delta_m"))
                world_delta_limit_m = float(
                    provenance.get("world_delta_limit_m")
                )
            except (TypeError, ValueError, OverflowError):
                return False
            expected_world_delta_m = math.hypot(
                process_point[0] - projective_origin_point[0],
                process_point[2] - projective_origin_point[2],
            )
            expected_world_delta_limit_m = (
                cls.IMAGE_MOTION_MAX_SPEED_MPS * age_s
                + cls.IMAGE_MOTION_PROJECTIVE_SLACK_M
            )
            return bool(
                all(
                    math.isfinite(value)
                    for value in (
                        *values,
                        *projective_origin_point,
                        *process_point,
                    )
                )
                and math.isfinite(age_s)
                and 0.0 <= age_s <= 5.0
                and math.isfinite(world_delta_m)
                and math.isfinite(world_delta_limit_m)
                and math.isclose(
                    world_delta_m,
                    expected_world_delta_m,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    world_delta_limit_m,
                    expected_world_delta_limit_m,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and 0.0 <= world_delta_m <= world_delta_limit_m + 1e-9
            )

        if provenance_type != "inferred_ground_process_observation":
            return False
        # Learned body height is useful only as current-cohort process evidence
        # for an already-established standing lifecycle. It is authorized by
        # either explicit lower-body occlusion or three coherent exact-current
        # torso observations on a confident detector/tracker row. It never
        # becomes a metric measurement. Bind the proof to the current cohort
        # and require the DS9 filter posterior to equal the published
        # tracking/BEV point exactly before Menon sees it.
        if provenance.get("origin") != "learned_body_height":
            return False
        support_kind = str(provenance.get("support_kind") or "")
        support_is_explicit_occlusion = support_kind == "lower_body_occlusion"
        support_is_coherent_torso = support_kind == "coherent_torso_motion"
        if support_is_explicit_occlusion:
            if (
                provenance.get("transport")
                != "fixed_occlusion_origin_raw_world_delta"
            ):
                return False
        elif support_is_coherent_torso:
            if (
                provenance.get("transport")
                != "fixed_torso_origin_raw_world_delta"
            ):
                return False
        else:
            return False
        if (
            provenance.get("trusted_origin_kind")
            != "queue_admitted_metric_world_output"
        ):
            return False
        if provenance.get("resolver_candidate_id") != "gravity_reconstruction":
            return False
        if provenance.get("raw_consensus_method") != "xz_medoid":
            return False
        occlusion_level = str(provenance.get("occlusion_level") or "")
        if support_is_explicit_occlusion:
            if track.get("lower_body_occluded") is not True:
                return False
            if (
                not occlusion_level
                or occlusion_level == "none"
                or occlusion_level
                != str(track.get("lower_body_occlusion_level") or "")
            ):
                return False
        else:
            try:
                image_motion_streak = int(
                    provenance.get("image_motion_streak")
                )
                track_image_motion_streak = int(
                    track.get("world_image_motion_streak")
                )
                detector_confidence = float(
                    provenance.get("detector_confidence")
                )
                track_detector_confidence = max(
                    float(track.get("confidence", 0.0) or 0.0),
                    float(track.get("tracker_confidence", 0.0) or 0.0),
                )
            except (TypeError, ValueError, OverflowError):
                return False
            track_detector_confidence = min(1.0, track_detector_confidence)
            if not (
                track.get("lower_body_occluded") is False
                and occlusion_level == "none"
                and str(track.get("lower_body_occlusion_level") or "")
                == "none"
                and provenance.get("image_motion_supported") is True
                and track.get("world_image_motion_supported") is True
                and provenance.get("image_motion_contact_basis")
                == "pose:torso_motion"
                and track.get("world_contact_basis") == "pose:torso_motion"
                and image_motion_streak
                >= cls.INFERRED_PROCESS_MIN_IMAGE_MOTION_STREAK
                and image_motion_streak == track_image_motion_streak
                and math.isfinite(detector_confidence)
                and math.isfinite(track_detector_confidence)
                and detector_confidence
                >= cls.INFERRED_PROCESS_MIN_DETECTOR_CONFIDENCE
                and math.isclose(
                    detector_confidence,
                    track_detector_confidence,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                return False
        if str(track.get("posture") or "") != "standing":
            return False
        if str(track.get("motion_mode") or "") in {"sit", "lie"}:
            return False
        if track.get("world_observation_range_admitted") is not True:
            return False
        raw_sample = provenance.get("raw_sample")
        raw_consensus = provenance.get("raw_consensus")
        raw_origin = provenance.get("raw_origin")
        raw_delta = provenance.get("raw_delta")
        trusted_world_origin = provenance.get("trusted_world_origin")
        process_observation = provenance.get("process_observation")
        track_raw = track.get("world_inferred_raw_observation")
        track_process = track.get("world_inferred_process_observation")
        if (
            any(
                not isinstance(value, (list, tuple)) or len(value) < 3
                for value in (
                    raw_sample,
                    raw_consensus,
                    raw_origin,
                    raw_delta,
                    trusted_world_origin,
                    process_observation,
                    track_raw,
                    track_process,
                )
            )
        ):
            return False
        episode_root = cls._inferred_ground_episode_root_from_provenance(
            provenance
        )
        if episode_root is None:
            return False
        established_episode_root = (
            expected_origin.inferred_ground_episode_root
            if expected_origin is not None
            else None
        )
        if (
            established_episode_root is not None
            and episode_root != established_episode_root
        ):
            return False
        try:
            raw_sample_point = tuple(float(value) for value in raw_sample[:3])
            raw_consensus_point = tuple(
                float(value) for value in raw_consensus[:3]
            )
            raw_origin_point = tuple(float(value) for value in raw_origin[:3])
            raw_delta_point = tuple(float(value) for value in raw_delta[:3])
            trusted_origin_point = tuple(
                float(value) for value in trusted_world_origin[:3]
            )
            process_point = tuple(float(value) for value in process_observation[:3])
            track_raw_point = tuple(float(value) for value in track_raw[:3])
            track_process_point = tuple(float(value) for value in track_process[:3])
            age_s = float(provenance.get("age_s"))
            raw_origin_ts_s = float(provenance.get("raw_origin_ts_s"))
            raw_consensus_ts_s = float(
                provenance.get("raw_consensus_ts_s")
            )
            current_ts_s = float(provenance.get("current_ts_s"))
            trusted_origin_filter_ts_s = float(
                provenance.get("trusted_origin_filter_ts_s")
            )
            height_ref_scene = float(provenance.get("height_ref_scene"))
            occlusion_confidence = float(
                provenance.get("occlusion_confidence")
            )
            track_occlusion_confidence = float(
                track.get("lower_body_occlusion_confidence")
            )
            observed_at_us = int(provenance.get("observed_at_us"))
            origin_observed_at_us = int(
                provenance.get("origin_observed_at_us")
            )
            raw_consensus_observed_at_us = int(
                provenance.get("raw_consensus_observed_at_us")
            )
            raw_origin_media_pts_ns = int(
                provenance.get("raw_origin_media_pts_ns")
            )
            raw_consensus_media_pts_ns = int(
                provenance.get("raw_consensus_media_pts_ns")
            )
            trusted_origin_media_pts_ns = int(
                provenance.get("trusted_origin_media_pts_ns")
            )
            current_media_pts_ns = int(
                provenance.get("current_media_pts_ns")
            )
            raw_consensus_count = int(
                provenance.get("raw_consensus_count")
            )
            raw_consensus_span_s = float(
                provenance.get("raw_consensus_span_s")
            )
            raw_evidence_gap_s = float(
                provenance.get("raw_evidence_gap_s")
            )
            lifecycle_generation = int(
                provenance.get("tracker_lifecycle_generation")
            )
            origin_trail_segment_id = int(
                provenance.get("origin_trail_segment_id")
            )
            track_observed_at_us = int(track.get("observed_at_us"))
            track_media_pts_ns = int(track.get("media_pts_ns"))
            track_lifecycle_generation = int(
                track.get("tracker_lifecycle_generation")
            )
            track_trail_segment_id = int(track.get("trail_segment_id"))
        except (TypeError, ValueError, OverflowError):
            return False
        proof_vectors = (
            raw_sample_point,
            raw_consensus_point,
            raw_origin_point,
            raw_delta_point,
            trusted_origin_point,
            process_point,
            track_raw_point,
            track_process_point,
            world_point,
            posterior_point,
        )
        raw_delta_matches = all(
            math.isclose(
                raw_delta_point[index],
                raw_consensus_point[index] - raw_origin_point[index],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for index in range(3)
        )
        process_algebra_matches = all(
            math.isclose(
                process_point[index],
                trusted_origin_point[index] + raw_delta_point[index],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for index in range(3)
        )
        expected_metric_origin_matches = bool(
            expected_origin is not None
            and expected_origin.metric_position is not None
            and expected_origin.metric_media_pts_ns is not None
            and expected_origin.metric_trail_segment_id is not None
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(
                    trusted_origin_point,
                    expected_origin.metric_position,
                )
            )
            and trusted_origin_media_pts_ns
            == int(expected_origin.metric_media_pts_ns)
            and origin_trail_segment_id
            == int(expected_origin.metric_trail_segment_id)
        )
        recent_consensus_evidence = bool(
            math.isfinite(raw_evidence_gap_s)
            and 0.0
            <= raw_evidence_gap_s
            <= cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S + 1e-9
        )
        restart_origin = (
            cls._matched_inferred_ground_restart_origin(
                expected_origin,
                provenance,
                current_media_pts_ns=current_media_pts_ns,
            )
            if (
                expected_origin is not None
                and raw_evidence_gap_s
                > cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S + 1e-9
            )
            else None
        )
        restart_age_ns = int(
            round(
                cls.INFERRED_PROCESS_RESTART_MAX_AGE_S
                * 1_000_000_000.0
            )
        )
        exact_root_restart = bool(
            expected_origin is not None
            and established_episode_root is not None
            and episode_root == established_episode_root
            and restart_origin is not None
            and restart_origin.inferred_ground_episode_root == episode_root
            and math.isfinite(raw_evidence_gap_s)
            and cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S
            < raw_evidence_gap_s
            <= cls.INFERRED_PROCESS_RESTART_MAX_AGE_S + 1e-9
            and 0
            < current_media_pts_ns - int(expected_origin.media_pts_ns)
            <= restart_age_ns
        )
        return bool(
            all(
                math.isfinite(value)
                for point in proof_vectors
                for value in point
            )
            and raw_delta_matches
            and process_algebra_matches
            and expected_metric_origin_matches
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(raw_consensus_point, track_raw_point)
            )
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(process_point, track_process_point)
            )
            and math.isfinite(age_s)
            and age_s >= 0.0
            and math.isfinite(raw_origin_ts_s)
            and math.isfinite(raw_consensus_ts_s)
            and math.isfinite(current_ts_s)
            and math.isfinite(trusted_origin_filter_ts_s)
            and raw_origin_ts_s >= trusted_origin_filter_ts_s
            and raw_origin_ts_s <= raw_consensus_ts_s
            and raw_consensus_ts_s <= current_ts_s
            and current_ts_s >= raw_origin_ts_s
            and math.isclose(
                age_s,
                current_ts_s - raw_origin_ts_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and raw_evidence_gap_s <= age_s + 1e-9
            and math.isfinite(height_ref_scene)
            and height_ref_scene > 0.0
            and (
                (
                    support_is_explicit_occlusion
                    and math.isfinite(occlusion_confidence)
                    and 0.75 <= occlusion_confidence <= 1.0
                    and math.isfinite(track_occlusion_confidence)
                    and math.isclose(
                        occlusion_confidence,
                        track_occlusion_confidence,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
                or (
                    support_is_coherent_torso
                    and math.isfinite(occlusion_confidence)
                    and math.isclose(
                        occlusion_confidence,
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and math.isfinite(track_occlusion_confidence)
                    and math.isclose(
                        track_occlusion_confidence,
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
            )
            and origin_observed_at_us > 0
            and raw_consensus_observed_at_us > 0
            and observed_at_us > 0
            and origin_observed_at_us <= raw_consensus_observed_at_us
            and raw_consensus_observed_at_us <= observed_at_us
            and observed_at_us == track_observed_at_us
            and raw_origin_media_pts_ns > 0
            and raw_consensus_media_pts_ns > 0
            and trusted_origin_media_pts_ns > 0
            and current_media_pts_ns > 0
            and trusted_origin_media_pts_ns <= raw_origin_media_pts_ns
            and raw_origin_media_pts_ns <= raw_consensus_media_pts_ns
            and raw_consensus_media_pts_ns <= current_media_pts_ns
            and current_media_pts_ns == track_media_pts_ns
            and 1 <= raw_consensus_count <= 5
            and math.isfinite(raw_consensus_span_s)
            and 0.0
            <= raw_consensus_span_s
            <= cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S + 1e-9
            and (recent_consensus_evidence or exact_root_restart)
            and current_ts_s - raw_consensus_ts_s
            <= cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S + 1e-9
            and lifecycle_generation > 0
            and lifecycle_generation == track_lifecycle_generation
            and origin_trail_segment_id >= 0
            and origin_trail_segment_id == track_trail_segment_id
            and provenance.get("world_frame") == track.get("world_frame")
            and provenance.get("world_frame_revision")
            == track.get("world_frame_revision")
            and provenance.get("world_transform_sha256")
            == track.get("world_transform_sha256")
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(world_point, posterior_point)
            )
        )

    @classmethod
    def _current_presence_evidence_is_canonical(
        cls,
        track: Mapping[str, Any],
        evidence: Any,
        *,
        expected_origin: _CanonicalTrackOutput,
        current_media_pts_ns: int,
    ) -> bool:
        """Validate one fixed-root current-person bbox continuity proof."""

        if not isinstance(evidence, Mapping):
            return False
        root_bbox = expected_origin.kinematic_bbox_geometry
        root_media_pts_ns = expected_origin.kinematic_media_pts_ns
        root_segment = expected_origin.kinematic_trail_segment_id
        root_position = expected_origin.kinematic_position
        track_bbox = cls._bbox_geometry(track.get("bbox"))
        evidence_root_bbox = cls._bbox_geometry(
            evidence.get("kinematic_origin_bbox")
        )
        evidence_current_bbox = cls._bbox_geometry(
            evidence.get("current_bbox")
        )
        if (
            root_bbox is None
            or root_media_pts_ns is None
            or root_segment is None
            or root_position is None
            or track_bbox is None
            or evidence_root_bbox is None
            or evidence_current_bbox is None
        ):
            return False
        try:
            class_id = int(track.get("class_id"))
            frame_id = int(track.get("frame_id"))
            evidence_frame_id = int(evidence.get("frame_id"))
            evidence_media_pts_ns = int(evidence.get("media_pts_ns"))
            evidence_root_media_pts_ns = int(
                evidence.get("kinematic_origin_media_pts_ns")
            )
            evidence_root_segment = int(
                evidence.get("kinematic_origin_trail_segment_id")
            )
            detector_confidence = float(track.get("confidence"))
            tracker_confidence = float(track.get("tracker_confidence"))
            evidence_detector_confidence = float(
                evidence.get("detector_confidence")
            )
            evidence_tracker_confidence = float(
                evidence.get("tracker_confidence")
            )
            evidence_size_ratio = float(evidence.get("bbox_size_ratio"))
            evidence_displacement_norm = float(
                evidence.get("bbox_center_displacement_norm")
            )
            root_left, root_top, root_width, root_height = (
                float(value) for value in root_bbox
            )
            current_left, current_top, current_width, current_height = (
                float(value) for value in track_bbox
            )
            size_ratio = max(
                root_width / current_width,
                current_width / root_width,
                root_height / current_height,
                current_height / root_height,
            )
            root_center = (
                root_left + 0.5 * root_width,
                root_top + 0.5 * root_height,
            )
            current_center = (
                current_left + 0.5 * current_width,
                current_top + 0.5 * current_height,
            )
            displacement_px = math.hypot(
                current_center[0] - root_center[0],
                current_center[1] - root_center[1],
            )
            bbox_scale_px = 0.5 * (
                math.hypot(root_width, root_height)
                + math.hypot(current_width, current_height)
            )
            displacement_norm = displacement_px / bbox_scale_px
        except (
            TypeError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
        ):
            return False
        horizon_ns = int(
            round(
                (
                    cls.CURRENT_PRESENCE_HOLD_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        pose_present = track.get("pose_present") is True
        return bool(
            evidence.get("version") == 1
            and evidence.get("kind")
            == "same_lifecycle_bbox_from_kinematic_output"
            and class_id == 0
            and frame_id >= 0
            and evidence_frame_id == frame_id
            and evidence_media_pts_ns == current_media_pts_ns
            and evidence_root_media_pts_ns == int(root_media_pts_ns)
            and evidence_root_segment == int(root_segment)
            and 0
            < current_media_pts_ns - int(root_media_pts_ns)
            <= horizon_ns
            and evidence.get("pose_present") is pose_present
            and all(
                math.isfinite(value)
                for value in (
                    detector_confidence,
                    tracker_confidence,
                    evidence_detector_confidence,
                    evidence_tracker_confidence,
                    size_ratio,
                    displacement_norm,
                    evidence_size_ratio,
                    evidence_displacement_norm,
                    bbox_scale_px,
                )
            )
            and 0.0 <= detector_confidence <= 1.0
            and detector_confidence
            >= cls.CURRENT_PRESENCE_MIN_DETECTOR_CONFIDENCE
            and tracker_confidence
            >= cls.CURRENT_PRESENCE_MIN_TRACKER_CONFIDENCE
            and (
                pose_present
                or detector_confidence
                >= cls.CURRENT_PRESENCE_STRONG_DETECTOR_CONFIDENCE
            )
            and math.isclose(
                evidence_detector_confidence,
                detector_confidence,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                evidence_tracker_confidence,
                tracker_confidence,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(track_bbox, evidence_current_bbox)
            )
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(root_bbox, evidence_root_bbox)
            )
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(
                    root_position,
                    expected_origin.position,
                )
            )
            and size_ratio <= cls.CURRENT_PRESENCE_MAX_BBOX_SIZE_RATIO
            and displacement_norm
            <= cls.CURRENT_PRESENCE_MAX_CENTER_DISPLACEMENT_NORM
            and math.isclose(
                evidence_size_ratio,
                size_ratio,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                evidence_displacement_norm,
                displacement_norm,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

    @classmethod
    def _bounded_process_observation_is_canonical(
        cls,
        track: Mapping[str, Any],
        *,
        expected_origin: _CanonicalTrackOutput | None = None,
    ) -> bool:
        """Validate one exact bounded process continuation.

        CV predictions and stationary anchor holds are not new metric camera
        measurements.  They are nevertheless the current Noesis-owned world
        point after the DS9 physical filter has integrated and bounded the
        process state.  Admit only the complete proof emitted by that filter,
        including exact agreement between its posterior diagnostic and the
        coordinate carried by the tracking row.  This gives Menon the same
        held point as tracking/BEV without allowing an arbitrary display-only
        payload to masquerade as canonical world state.
        """

        if expected_origin is None:
            return False
        source = str(track.get("world_source") or "").strip().lower()
        if source not in {"cv_prediction", "anchor_hold"}:
            return False
        if str(track.get("world_quality") or "").strip().lower() != "held":
            return False
        if track.get("world_measurement_accepted") is not False:
            return False
        provenance = track.get("world_prediction_provenance")
        if not isinstance(provenance, Mapping):
            return False
        provenance_type = provenance.get("type")
        provenance_origin = provenance.get("origin")
        if provenance_type == "bounded_cv_process":
            if provenance_origin not in {
                "recent_projective_process",
                "last_metric_process",
            }:
                return False
        elif provenance_type == "bounded_output_hold":
            if source != "anchor_hold" or provenance_origin != "last_published_output":
                return False
        else:
            return False
        if provenance.get("non_authoritative") is not True:
            return False
        if provenance.get("state_integrated") is not True:
            return False
        reason = provenance.get("reason")
        quality_reason = track.get("world_quality_reason")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > 240
            or not isinstance(quality_reason, str)
            or not quality_reason.strip()
            or len(quality_reason) > 240
            or reason != quality_reason
        ):
            return False
        world = track.get("world")
        posterior = track.get("world_filter_prediction")
        process_observation = provenance.get("process_observation")
        transition = provenance.get("filter_transition")
        if (
            not isinstance(world, (list, tuple))
            or len(world) < 3
            or not isinstance(posterior, (list, tuple))
            or len(posterior) < 3
            or not isinstance(process_observation, (list, tuple))
            or len(process_observation) < 3
            or not isinstance(transition, Mapping)
        ):
            return False
        origin_world = transition.get("origin_world")
        metric_origin_world = transition.get("metric_origin_world")
        position_base = transition.get("position_base")
        if (
            not isinstance(origin_world, (list, tuple))
            or len(origin_world) < 3
            or not isinstance(metric_origin_world, (list, tuple))
            or len(metric_origin_world) < 3
            or not isinstance(position_base, (list, tuple))
            or len(position_base) < 3
        ):
            return False
        try:
            world_point = tuple(float(value) for value in world[:3])
            posterior_point = tuple(float(value) for value in posterior[:3])
            process_point = tuple(
                float(value) for value in process_observation[:3]
            )
            origin_point = tuple(float(value) for value in origin_world[:3])
            metric_origin_point = tuple(
                float(value) for value in metric_origin_world[:3]
            )
            base_point = tuple(float(value) for value in position_base[:3])
            version = int(transition.get("version"))
            origin_media_pts_ns = int(transition.get("origin_media_pts_ns"))
            current_media_pts_ns = int(transition.get("current_media_pts_ns"))
            track_media_pts_ns = int(track.get("media_pts_ns"))
            origin_trail_segment_id = int(
                transition.get("origin_trail_segment_id")
            )
            metric_origin_media_pts_ns = int(
                transition.get("metric_origin_media_pts_ns")
            )
            metric_origin_trail_segment_id = int(
                transition.get("metric_origin_trail_segment_id")
            )
            track_trail_segment_id = int(track.get("trail_segment_id"))
            lifecycle_generation = int(
                transition.get("tracker_lifecycle_generation")
            )
            track_lifecycle_generation = int(
                track.get("tracker_lifecycle_generation")
            )
            gate_dt_s = float(transition.get("gate_dt_s"))
            position_gain = float(transition.get("position_gain"))
            max_speed_mps = float(transition.get("max_speed_mps"))
            reset_after_s = float(transition.get("reset_after_s"))
        except (TypeError, ValueError, OverflowError):
            return False
        vectors = (
            world_point,
            posterior_point,
            process_point,
            origin_point,
            metric_origin_point,
            base_point,
        )
        if not all(math.isfinite(value) for point in vectors for value in point):
            return False
        if (
            version != 1
            or transition.get("origin_kind")
            != "queue_admitted_world_output"
            or transition.get("metric_origin_kind")
            != "queue_admitted_metric_world_output"
            or not math.isfinite(max_speed_mps)
            or not (0.0 < max_speed_mps <= cls.IMAGE_MOTION_MAX_SPEED_MPS)
            or not math.isclose(
                reset_after_s,
                cls.IMAGE_MOTION_RESET_AFTER_S,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            return False
        expected_metric_position = expected_origin.metric_position
        expected_metric_pts = expected_origin.metric_media_pts_ns
        expected_metric_segment = expected_origin.metric_trail_segment_id
        if (
            expected_metric_position is None
            or expected_metric_pts is None
            or expected_metric_segment is None
        ):
            return False
        stationary_evidence = provenance.get("stationary_evidence")
        # Stationary evidence is emitted from the tracking posture state and
        # binds that exact public field. ``world_posture`` is a resolver
        # diagnostic and may legitimately remain ``unknown`` while tracking
        # has classified the same row as sitting/lying; it must neither
        # suppress nor grant the longer exact-hold authority.
        posture = str(
            track.get("posture")
            or track.get("world_posture")
            or ""
        ).strip().lower()
        stationary_anchor_hold = bool(
            source == "anchor_hold"
            and provenance_type == "bounded_output_hold"
            and provenance_origin == "last_published_output"
            and isinstance(stationary_evidence, Mapping)
            and stationary_evidence.get("version") == 1
            and stationary_evidence.get("kind") == "bbox_stationary"
            and stationary_evidence.get("posture") == posture
            and posture in {"sitting", "lying"}
            and stationary_evidence.get("frame_id") == track.get("frame_id")
            and stationary_evidence.get("media_pts_ns")
            == current_media_pts_ns
            and track.get("world_bbox_stationary_supported") is True
        )
        upright_presence_evidence = provenance.get(
            "upright_presence_evidence"
        )
        upright_presence_hold = False
        if isinstance(upright_presence_evidence, Mapping):
            floor_candidate = track.get("world_floor_candidate")
            evidence_floor_candidate = upright_presence_evidence.get(
                "floor_candidate"
            )
            bbox = track.get("bbox")
            image_size = track.get("image_size")
            try:
                if (
                    not isinstance(floor_candidate, (list, tuple))
                    or len(floor_candidate) != 3
                    or not isinstance(evidence_floor_candidate, (list, tuple))
                    or len(evidence_floor_candidate) != 3
                ):
                    raise ValueError(
                        "upright presence floor candidates must be exact vectors"
                    )
                evidence_detector_confidence = float(
                    upright_presence_evidence.get("detector_confidence")
                )
                evidence_tracker_confidence = float(
                    upright_presence_evidence.get("tracker_confidence")
                )
                detector_confidence = float(track.get("confidence"))
                tracker_confidence = float(track.get("tracker_confidence"))
                candidate_point = tuple(
                    float(value) for value in floor_candidate[:3]
                )
                evidence_candidate_point = tuple(
                    float(value) for value in evidence_floor_candidate[:3]
                )
                output_distance_m = math.hypot(
                    candidate_point[0] - world_point[0],
                    candidate_point[2] - world_point[2],
                )
                evidence_output_distance_m = float(
                    upright_presence_evidence.get("output_distance_m")
                )
                evidence_output_distance_limit_m = float(
                    upright_presence_evidence.get(
                        "output_distance_limit_m"
                    )
                )
                bbox_width = float(bbox[2])
                bbox_height = float(bbox[3])
                image_height = float(image_size[1])
            except (
                TypeError,
                ValueError,
                IndexError,
                OverflowError,
            ):
                upright_presence_hold = False
            else:
                motion_mode = str(track.get("motion_mode") or "").strip().lower()
                upright_presence_hold = bool(
                    source == "anchor_hold"
                    and provenance_type == "bounded_output_hold"
                    and provenance_origin == "last_published_output"
                    and upright_presence_evidence.get("version") == 1
                    and upright_presence_evidence.get("kind")
                    == "pose_confirmed_nonseated_floor_near_output"
                    and upright_presence_evidence.get("frame_id")
                    == track.get("frame_id")
                    and upright_presence_evidence.get("media_pts_ns")
                    == current_media_pts_ns
                    and upright_presence_evidence.get("posture") == posture
                    and posture in {"standing", "unknown"}
                    and upright_presence_evidence.get("motion_mode")
                    == motion_mode
                    and motion_mode not in {"sit", "lie"}
                    and upright_presence_evidence.get("contact_basis")
                    == "pose:torso_motion"
                    and track.get("world_contact_basis")
                    == "pose:torso_motion"
                    and track.get("pose_present") is True
                    and track.get("lower_body_occluded") is False
                    and track.get("world_floor_admitted") is True
                    and track.get("world_floor_contact_plausible") is True
                    and track.get("world_observation_range_admitted") is True
                    and track.get("world_support_state") == "floor"
                    and all(
                        math.isfinite(value)
                        for value in (
                            detector_confidence,
                            tracker_confidence,
                            evidence_detector_confidence,
                            evidence_tracker_confidence,
                            *candidate_point,
                            *evidence_candidate_point,
                            output_distance_m,
                            evidence_output_distance_m,
                            evidence_output_distance_limit_m,
                            bbox_width,
                            bbox_height,
                            image_height,
                        )
                    )
                    and detector_confidence
                    >= cls.UPRIGHT_PRESENCE_MIN_DETECTOR_CONFIDENCE
                    and detector_confidence <= 1.0
                    and tracker_confidence
                    >= cls.UPRIGHT_PRESENCE_MIN_TRACKER_CONFIDENCE
                    and tracker_confidence <= 1.0
                    and 0.0 <= evidence_detector_confidence <= 1.0
                    and 0.0 <= evidence_tracker_confidence <= 1.0
                    and math.isclose(
                        evidence_detector_confidence,
                        detector_confidence,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        evidence_tracker_confidence,
                        tracker_confidence,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and bbox_width > 0.0
                    and bbox_height > 0.0
                    and image_height > 0.0
                    and bbox_height / image_height
                    >= cls.UPRIGHT_PRESENCE_MIN_BBOX_HEIGHT_FRACTION
                    and bbox_width / bbox_height
                    <= cls.UPRIGHT_PRESENCE_MAX_BBOX_ASPECT
                    and all(
                        math.isclose(
                            left,
                            right,
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                        for left, right in zip(
                            candidate_point,
                            evidence_candidate_point,
                        )
                    )
                    and math.isclose(
                        candidate_point[1],
                        world_point[1],
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    and output_distance_m
                    <= cls.UPRIGHT_PRESENCE_MAX_OUTPUT_DISTANCE_M
                    and math.isclose(
                        evidence_output_distance_m,
                        output_distance_m,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        evidence_output_distance_limit_m,
                        cls.UPRIGHT_PRESENCE_MAX_OUTPUT_DISTANCE_M,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
        current_presence_evidence = provenance.get(
            "current_presence_evidence"
        )
        current_presence_hold = bool(
            source == "anchor_hold"
            and provenance_type == "bounded_output_hold"
            and provenance_origin == "last_published_output"
            and cls._current_presence_evidence_is_canonical(
                track,
                current_presence_evidence,
                expected_origin=expected_origin,
                current_media_pts_ns=current_media_pts_ns,
            )
        )
        recent_projective_bridge = bool(
            source == "cv_prediction"
            and provenance_type == "bounded_cv_process"
            and provenance_origin == "recent_projective_process"
        )
        projective_bridge_root_media_pts_ns = (
            cls._active_projective_bridge_root(
                expected_origin,
                current_media_pts_ns=current_media_pts_ns,
            )
            if provenance_type
            in {"bounded_cv_process", "bounded_output_hold"}
            else None
        )
        projective_episode_continuation = bool(
            projective_bridge_root_media_pts_ns is not None
        )
        if recent_projective_bridge and not projective_episode_continuation:
            # The producer label is descriptive only. It can request this lane,
            # but only a live service-owned image root can authorize it.
            return False
        metric_horizon_s = (
            cls.STATIONARY_ANCHOR_HOLD_HORIZON_S
            if stationary_anchor_hold
            else (
                cls.UPRIGHT_PRESENCE_HOLD_HORIZON_S
                if upright_presence_hold
                else (
                    cls.CURRENT_PRESENCE_HOLD_HORIZON_S
                    if current_presence_hold
                    else cls.BOUNDED_PROCESS_HORIZON_S
                )
            )
        )
        max_metric_age_ns = int(
            round(
                (
                    metric_horizon_s
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        origin_horizon_s = (
            (
                cls.PROJECTIVE_PROCESS_BRIDGE_HORIZON_S
                + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
            )
            if projective_episode_continuation
            else (
                (
                    cls.UPRIGHT_PRESENCE_HOLD_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                if upright_presence_hold
                else (
                    (
                        cls.CURRENT_PRESENCE_HOLD_HORIZON_S
                        + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                    )
                    if current_presence_hold
                    else (
                        metric_horizon_s
                        + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                    )
                )
            )
        )
        matched_origin = cls._matched_committed_output_origin(
            expected_origin,
            origin_point=origin_point,
            origin_media_pts_ns=origin_media_pts_ns,
            origin_trail_segment_id=origin_trail_segment_id,
            current_media_pts_ns=current_media_pts_ns,
            maximum_age_s=origin_horizon_s,
        )
        if matched_origin is None:
            return False
        if provenance_type == "bounded_output_hold" and not (
            origin_trail_segment_id
            == int(expected_origin.trail_segment_id)
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(
                    origin_point,
                    expected_origin.position,
                )
            )
        ):
            # A queue-delayed proof may name an older service-committed hold,
            # but a gain-zero hold can only retain the latest public
            # coordinate in the same visible segment. ``matched_origin``
            # above proves the historical point was genuinely committed;
            # this equality prevents historical motion from being replayed.
            return False
        if not (
            matched_origin.metric_position is not None
            and matched_origin.metric_media_pts_ns is not None
            and matched_origin.metric_trail_segment_id is not None
            and matched_origin.metric_media_pts_ns == int(expected_metric_pts)
            and matched_origin.metric_trail_segment_id
            == int(expected_metric_segment)
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(
                    matched_origin.metric_position,
                    expected_metric_position,
                )
            )
        ):
            # A metric reanchor invalidates queued proofs formed before it.
            return False
        if (
            origin_media_pts_ns <= 0
            or current_media_pts_ns <= origin_media_pts_ns
            or current_media_pts_ns != track_media_pts_ns
            or metric_origin_media_pts_ns <= 0
            or metric_origin_media_pts_ns != int(expected_metric_pts)
            or metric_origin_media_pts_ns > origin_media_pts_ns
            or (
                projective_episode_continuation
                and current_media_pts_ns
                - int(projective_bridge_root_media_pts_ns or 0)
                > int(
                    round(
                        (
                            cls.PROJECTIVE_PROCESS_BRIDGE_HORIZON_S
                            + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                        )
                        * 1_000_000_000.0
                    )
                )
            )
            or (
                not projective_episode_continuation
                and not upright_presence_hold
                and not current_presence_hold
                and current_media_pts_ns - metric_origin_media_pts_ns
                > max_metric_age_ns
            )
            or metric_origin_trail_segment_id != int(expected_metric_segment)
            or origin_trail_segment_id != track_trail_segment_id
            or lifecycle_generation <= 0
            or lifecycle_generation != track_lifecycle_generation
            or transition.get("world_frame") != track.get("world_frame")
            or transition.get("world_frame_revision")
            != track.get("world_frame_revision")
            or transition.get("world_transform_sha256")
            != track.get("world_transform_sha256")
        ):
            return False
        if not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(
                metric_origin_point,
                expected_metric_position,
            )
        ):
            return False
        if not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(base_point, origin_point)
        ):
            return False
        if not all(
            math.isclose(point[1], world_point[1], rel_tol=0.0, abs_tol=1e-9)
            for point in (
                posterior_point,
                process_point,
                origin_point,
                metric_origin_point,
                base_point,
            )
        ):
            return False

        media_dt_s = (
            current_media_pts_ns - origin_media_pts_ns
        ) / 1_000_000_000.0
        expected_gate_dt_s = min(media_dt_s, cls.IMAGE_MOTION_RESET_AFTER_S)
        if (
            not math.isfinite(gate_dt_s)
            or gate_dt_s <= 0.0
            or not math.isclose(
                gate_dt_s,
                expected_gate_dt_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            return False

        transition_kind = str(transition.get("kind") or "")
        if provenance_type == "bounded_cv_process":
            if transition_kind != "bounded_process_step" or not math.isclose(
                position_gain,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        else:
            if transition_kind != "output_hold" or not math.isclose(
                position_gain,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        expected_posterior = (
            base_point[0] + position_gain * (process_point[0] - base_point[0]),
            base_point[1],
            base_point[2] + position_gain * (process_point[2] - base_point[2]),
        )
        if not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(world_point, posterior_point)
        ):
            return False
        if not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(world_point, expected_posterior)
        ):
            return False
        if (
            math.hypot(
                world_point[0] - origin_point[0],
                world_point[2] - origin_point[2],
            )
            > max_speed_mps * gate_dt_s + 1e-9
        ):
            return False
        return bool(
            cls._latest_output_step_is_physical(
                expected_origin,
                world_point=world_point,
                current_media_pts_ns=current_media_pts_ns,
                max_speed_mps=max_speed_mps,
            )
            and cls._latest_visible_output_step_is_physical(
                expected_origin,
                world_point=world_point,
                current_media_pts_ns=current_media_pts_ns,
                max_speed_mps=max_speed_mps,
            )
        )

    def _world_observation(
        self,
        track: Mapping[str, Any],
        *,
        calibration_revision: Any = None,
        expected_image_motion_origin: _CanonicalTrackOutput | None = None,
    ) -> WorldPositionObservation | None:
        if track.get("world_valid") is not True:
            return None
        world_source = str(track.get("world_source") or "").strip().lower()
        proof_bearing_continuation_sources = {
            "anchor_hold",
            "cv_prediction",
            "image_motion_prediction",
        }
        if (
            track.get("world_measurement_accepted") is False
            and world_source not in proof_bearing_continuation_sources
        ):
            # An ordinary metric source cannot become canonical after its own
            # producer explicitly rejected the measurement.  The three
            # continuation lanes below are the sole exception because this
            # service independently verifies their exact origin, transition,
            # and physical bounds before admitting them.
            return None
        if (
            str(track.get("world_quality") or "").strip().lower()
            == "held"
            and world_source not in proof_bearing_continuation_sources
        ):
            # ``held`` is process state, not a metric observation quality.
            # Fail closed if an ordinary source is relabelled held without a
            # proof-bearing continuation contract.
            return None
        if world_source == "relative_motion_prediction":
            # Retired producer lane: quarantine stale payloads rather than
            # allowing an old relative-body estimate to move canonical world.
            return None
        if world_source in {"cv_prediction", "anchor_hold"}:
            bounded_process_is_canonical = (
                self._bounded_process_observation_is_canonical(
                    track,
                    expected_origin=expected_image_motion_origin,
                )
            )
            if not bounded_process_is_canonical:
                return None
        if world_source == "image_motion_prediction":
            image_motion_is_canonical = (
                self._image_motion_observation_is_canonical(
                    track,
                    expected_origin=expected_image_motion_origin,
                )
            )
            if not image_motion_is_canonical:
                return None
        frame = str(track.get("world_frame") or "")
        if frame != "backend_world_m":
            return None
        world_frame_revision = self._required_revision(
            track.get("world_frame_revision")
        )
        world_transform_sha256 = self._sha256_text(
            track.get("world_transform_sha256")
        )
        if world_frame_revision is None or world_transform_sha256 is None:
            # A world-valid row without a complete registration identity is
            # display/debug data only.  The canonical backend must fail closed
            # instead of joining it to another registration.
            return None
        if track.get("calibration_revision") is not None:
            calibration_revision = self._required_revision(calibration_revision)
            if calibration_revision is None:
                return None
        elif calibration_revision is not None:
            calibration_revision = self._required_revision(calibration_revision)
        raw = track.get("world")
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            return None
        try:
            position = tuple(float(value) for value in raw[:3])
        except Exception:
            return None
        if not all(math.isfinite(value) for value in position):
            return None
        quantity_raw = track.get("world_quantity")
        if quantity_raw is not None and str(quantity_raw) != "ground_footprint":
            # The current canonical contract localizes the person's ground
            # footprint.  A body root or other quantity must use a future
            # explicit field/contract rather than masquerading as this point.
            return None
        quality_raw = str(track.get("world_quality") or "estimated").strip().lower()
        quality = quality_raw if quality_raw in {"good", "estimated", "held"} else "estimated"
        covariance = self._world_covariance(track.get("world_covariance"))
        if track.get("world_covariance") is not None and covariance is None:
            # A resolver-owned observation must never be relabelled with a
            # convenient legacy variance when its uncertainty contract is
            # malformed.  Older producers that do not publish covariance keep
            # the quality-bucket compatibility path below.
            return None
        if covariance is None:
            variance = {"good": 0.04, "estimated": 0.25, "held": 0.64}[quality]
            covariance = (
                variance,
                0.0,
                0.0,
                0.0,
                variance,
                0.0,
                0.0,
                0.0,
                variance,
            )
        confidence = {"good": 0.90, "estimated": 0.60, "held": 0.35}[quality]
        resolver_confidence = self._confidence(
            track.get("world_resolver_confidence")
        )
        if resolver_confidence is not None:
            confidence = resolver_confidence
        measured_confidence = self._confidence(track.get("tracker_confidence"))
        if measured_confidence is not None:
            confidence = min(confidence, measured_confidence)
        posture_raw = str(
            track.get("world_posture") or track.get("posture") or "unknown"
        ).strip().lower()
        posture = (
            posture_raw
            if posture_raw in {"standing", "sitting", "lying", "unknown"}
            else "unknown"
        )
        support_raw = str(
            track.get("world_support_state") or "unknown"
        ).strip().lower()
        support_state = (
            support_raw
            if support_raw in {"floor", "seat", "couch", "unknown"}
            else "unknown"
        )
        return WorldPositionObservation(
            position=Vector3(x=position[0], y=position[1], z=position[2]),
            covariance=Matrix3(values=covariance),
            frame="backend_world_m",
            world_frame="backend_world_m",
            units="meters",
            world_frame_revision=world_frame_revision,
            world_transform_sha256=world_transform_sha256,
            calibration_revision=calibration_revision,
            quantity="ground_footprint",
            support_state=support_state,  # type: ignore[arg-type]
            posture=posture,  # type: ignore[arg-type]
            source=str(track.get("world_source") or "unspecified"),
            quality=quality,  # type: ignore[arg-type]
            confidence=confidence,
            reason=(
                self._bounded_text(
                    track.get("world_quality_reason"),
                    maximum=200,
                )
            ),
        )

    @staticmethod
    def _world_covariance(value: Any) -> tuple[float, ...] | None:
        """Validate one row-major 3x3 covariance without importing NumPy.

        The canonical boundary accepts the resolver's anisotropic covariance
        only when it is finite, symmetric, and positive semidefinite.  The
        positive diagonal requirement keeps downstream precision weighting
        well defined while still allowing zero off-axis correlation.
        """

        if not isinstance(value, (list, tuple)) or len(value) != 9:
            return None
        try:
            matrix = tuple(float(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(item) for item in matrix):
            return None

        scale = max(1.0, *(abs(item) for item in matrix))
        tolerance = 1e-8 * scale
        if any(matrix[index] <= 0.0 for index in (0, 4, 8)):
            return None
        if (
            abs(matrix[1] - matrix[3]) > tolerance
            or abs(matrix[2] - matrix[6]) > tolerance
            or abs(matrix[5] - matrix[7]) > tolerance
        ):
            return None

        # Every principal minor of a symmetric 3x3 PSD matrix is
        # non-negative.  Evaluate them explicitly to keep this boundary small
        # and dependency-free.
        minors = (
            matrix[0] * matrix[4] - matrix[1] * matrix[3],
            matrix[0] * matrix[8] - matrix[2] * matrix[6],
            matrix[4] * matrix[8] - matrix[5] * matrix[7],
        )
        if any(minor < -tolerance for minor in minors):
            return None
        determinant = (
            matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
            - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
            + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
        )
        if determinant < -tolerance:
            return None
        return matrix

    def _world_diagnostics(
        self,
        track: Mapping[str, Any],
        *,
        world_observation: WorldPositionObservation | None,
        expected_continuity_origin: _CanonicalTrackOutput | None = None,
    ) -> WorldObservationDiagnostics | None:
        if (
            world_observation is not None
            and _WORLD_DIAGNOSTIC_TRACK_FIELDS.isdisjoint(track)
        ):
            return None
        resolver_fields = self._resolver_diagnostics_fields(track)
        return WorldObservationDiagnostics(
            estimator_evaluated=self._optional_bool(
                track.get("world_estimator_evaluated")
            ),
            first_divergence_reason=self._world_first_divergence_reason(
                track,
                world_observation=world_observation,
                expected_continuity_origin=expected_continuity_origin,
            ),
            floor_candidate_m=self._diagnostic_vector3(
                track.get("world_floor_candidate")
            ),
            depth_candidate_m=self._diagnostic_vector3(
                track.get("world_depth_candidate")
            ),
            prefilter_measurement_m=self._diagnostic_vector3(
                track.get("world_prefilter_measurement")
            ),
            filter_prediction_m=self._diagnostic_vector3(
                track.get("world_filter_prediction")
            ),
            floor_range_m=self._finite_float(track.get("world_floor_range_m")),
            floor_range_limit_m=self._finite_float(
                track.get("world_floor_range_limit_m")
            ),
            floor_incidence_sin=self._finite_float(
                track.get("world_floor_incidence_sin")
            ),
            floor_admitted=self._optional_bool(
                track.get("world_floor_admitted")
            ),
            floor_rejection_reason=self._bounded_text(
                track.get("world_floor_rejection_reason"),
                maximum=200,
            ),
            measurement_accepted=self._optional_bool(
                track.get("world_measurement_accepted")
            ),
            measurement_rejection_reason=self._bounded_text(
                track.get("world_rejection_reason"),
                maximum=200,
            ),
            innovation_m=self._finite_float(track.get("world_innovation_m")),
            innovation_limit_m=self._finite_float(
                track.get("world_innovation_limit_m")
            ),
            fusion_policy_id=self._bounded_text(
                track.get("world_fusion_policy_id"),
                maximum=200,
            ),
            floor_weight_scale=self._finite_float(
                track.get("world_floor_weight_scale")
            ),
            depth_weight_scale=self._finite_float(
                track.get("world_depth_weight_scale")
            ),
            floor_weight_effective=self._finite_float(
                track.get("world_floor_weight_effective")
            ),
            depth_weight_effective=self._finite_float(
                track.get("world_depth_weight_effective")
            ),
            depth_status=self._bounded_text(
                track.get("depth_status"),
                maximum=120,
            ),
            depth_anchor_m=self._finite_float(track.get("depth_anchor_m")),
            depth_registered_m=self._finite_float(
                track.get("depth_registered_m")
            ),
            depth_used_m=self._finite_float(track.get("depth_used_m")),
            depth_registration_status=self._bounded_text(
                track.get("depth_registration_status"),
                maximum=120,
            ),
            depth_registration_id=self._bounded_text(
                track.get("depth_registration_id"),
                maximum=200,
            ),
            **resolver_fields,
        )

    def _resolver_diagnostics_fields(
        self,
        track: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalize the bounded exact-cohort resolver summary.

        The richer dashboard payload stays track-local.  The canonical world
        contract retains only the decision and compact candidate diagnostics,
        and only when the record identifies this exact track observation.
        Malformed or stale optional diagnostics are omitted; they never alter
        admission of the already-validated canonical world point.
        """

        source = track.get("world_resolver")
        if not isinstance(source, Mapping):
            return {}
        if (
            source.get("contract") != "noesis.world_resolver_diagnostics"
            or source.get("version") != 1
        ):
            return {}

        exact_integer_fields = (
            ("source_id", "source_id"),
            ("tracker_id", "tracker_id"),
            ("frame_id", "frame_id"),
            ("observed_at_us", "observed_at_us"),
        )
        for diagnostic_key, track_key in exact_integer_fields:
            diagnostic_value = source.get(diagnostic_key)
            track_value = track.get(track_key)
            if (
                isinstance(diagnostic_value, bool)
                or isinstance(track_value, bool)
                or not isinstance(diagnostic_value, int)
            ):
                return {}
            try:
                if int(diagnostic_value) != int(track_value):
                    return {}
            except (TypeError, ValueError, OverflowError):
                return {}
        if str(source.get("camera_id") or "") != str(track.get("camera_id") or ""):
            return {}
        if str(source.get("world_frame") or "") != str(track.get("world_frame") or ""):
            return {}
        if str(source.get("world_frame_revision") or "") != str(
            track.get("world_frame_revision") or ""
        ):
            return {}

        supported_kinds = {
            "floor_ray",
            "registered_depth",
            "pose_scale",
            "gravity_reconstruction",
        }
        selected_kind_raw = source.get("selected_kind")
        selected_kind = (
            str(selected_kind_raw)
            if selected_kind_raw is not None
            and str(selected_kind_raw) in supported_kinds
            else None
        )
        selected_id = self._bounded_text(source.get("selected_id"), maximum=120)
        contributor_source = source.get("contributor_ids")
        if not isinstance(contributor_source, (list, tuple)) or len(contributor_source) > 4:
            return {}
        contributor_ids = tuple(
            item
            for item in (
                self._bounded_text(value, maximum=120)
                for value in contributor_source
            )
            if item is not None
        )
        if len(contributor_ids) != len(contributor_source):
            return {}

        raw_candidates = source.get("candidates")
        if not isinstance(raw_candidates, (list, tuple)) or len(raw_candidates) > 4:
            return {}
        candidates: list[WorldMeasurementCandidateDiagnostic] = []
        candidate_ids: set[str] = set()
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                return {}
            candidate_id = self._bounded_text(raw.get("id"), maximum=120)
            kind = str(raw.get("kind") or "")
            score = self._finite_float(raw.get("score"))
            pcf_score = self._finite_float(raw.get("pcf_score"))
            if (
                candidate_id is None
                or candidate_id in candidate_ids
                or kind not in supported_kinds
                or score is None
                or not 0.0 <= score <= 1.0
                or pcf_score is None
                or not 0.0 <= pcf_score <= 1.0
            ):
                return {}
            selected = raw.get("selected")
            compatible = raw.get("compatible_with_selected")
            alternate = raw.get("retained_as_alternate")
            if not all(isinstance(value, bool) for value in (selected, compatible, alternate)):
                return {}
            innovation = self._finite_float(raw.get("innovation_m"))
            agreement = self._finite_float(raw.get("agreement_mahalanobis_sq"))
            if innovation is not None and innovation < 0.0:
                return {}
            if agreement is not None and agreement < 0.0:
                return {}
            try:
                candidates.append(
                    WorldMeasurementCandidateDiagnostic(
                        candidate_id=candidate_id,
                        kind=kind,  # type: ignore[arg-type]
                        score=score,
                        pcf_score=pcf_score,
                        innovation_m=innovation,
                        agreement_mahalanobis_sq=agreement,
                        compatible_with_selected=compatible,
                        selected=selected,
                        retained_as_alternate=alternate,
                        rejection_reason=self._bounded_text(
                            raw.get("rejection_reason"), maximum=200
                        ),
                    )
                )
            except ValueError:
                return {}
            candidate_ids.add(candidate_id)

        if selected_id is not None and selected_id not in candidate_ids:
            return {}
        if any(item not in candidate_ids for item in contributor_ids):
            return {}
        alternate_id = self._bounded_text(source.get("alternate_id"), maximum=120)
        if alternate_id is not None and alternate_id not in candidate_ids:
            return {}
        fused = source.get("fused")
        if not isinstance(fused, bool):
            return {}
        confidence = self._finite_float(source.get("confidence"))
        disagreement = source.get("disagreement")
        disagreement_m = None
        if disagreement is not None:
            if not isinstance(disagreement, Mapping):
                return {}
            disagreement_m = self._finite_float(disagreement.get("distance_m"))
            if disagreement_m is not None and disagreement_m < 0.0:
                return {}
        agreement = self._finite_float(source.get("agreement_mahalanobis_sq"))
        pcf_score = self._finite_float(source.get("pcf_score"))
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            return {}
        if agreement is not None and agreement < 0.0:
            return {}
        if pcf_score is not None and not 0.0 <= pcf_score <= 1.0:
            return {}
        return {
            "resolver_selected_kind": selected_kind,
            "resolver_selected_candidate_id": selected_id,
            "resolver_contributor_ids": contributor_ids,
            "resolver_alternate_candidate_id": alternate_id,
            "resolver_fused": fused,
            "resolver_confidence": confidence,
            "resolver_disagreement_m": disagreement_m,
            "resolver_agreement_mahalanobis_sq": agreement,
            "resolver_pcf_score": pcf_score,
            "resolver_reason": self._bounded_text(source.get("reason"), maximum=240),
            "resolver_candidate_diagnostics": tuple(candidates),
        }

    def _world_first_divergence_reason(
        self,
        track: Mapping[str, Any],
        *,
        world_observation: WorldPositionObservation | None,
        expected_continuity_origin: _CanonicalTrackOutput | None = None,
    ) -> str | None:
        if world_observation is not None:
            return None
        if track.get("world_valid") is True:
            if str(track.get("world_frame") or "") != "backend_world_m":
                return "world_frame_invalid"
            source = str(track.get("world_source") or "").strip().lower()
            if source == "relative_motion_prediction":
                return "display_continuity_not_authoritative"
            if (
                source in {"cv_prediction", "anchor_hold"}
                and not self._bounded_process_observation_is_canonical(
                    track,
                    expected_origin=expected_continuity_origin,
                )
            ):
                return "bounded_process_continuity_provenance_invalid"
            if (
                source == "image_motion_prediction"
                and not self._image_motion_observation_is_canonical(
                    track,
                    expected_origin=expected_continuity_origin,
                )
            ):
                return "image_motion_continuity_provenance_invalid"
            if self._required_revision(track.get("world_frame_revision")) is None:
                return "world_frame_revision_missing"
            if self._sha256_text(track.get("world_transform_sha256")) is None:
                return "world_transform_sha256_missing"
            raw = track.get("world")
            if not isinstance(raw, (list, tuple)) or len(raw) < 3:
                return "world_position_missing"
            if self._diagnostic_vector3(raw) is None:
                return "world_position_invalid"
        for key in (
            "world_rejection_reason",
            "world_floor_rejection_reason",
            "world_quality_reason",
        ):
            reason = self._bounded_text(track.get(key), maximum=240)
            if reason is not None:
                return reason
        registration_status = self._bounded_text(
            track.get("depth_registration_status"),
            maximum=120,
        )
        if registration_status not in {None, "ok"}:
            return self._bounded_text(
                f"depth_registration_{registration_status}",
                maximum=240,
            )
        depth_status = self._bounded_text(
            track.get("depth_status"),
            maximum=120,
        )
        if depth_status not in {None, "ok"}:
            return self._bounded_text(
                f"depth_{depth_status}",
                maximum=240,
            )
        return "world_observation_unavailable"

    @staticmethod
    def _diagnostic_vector3(value: Any) -> Vector3 | None:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            result = tuple(float(item) for item in value[:3])
        except Exception:
            return None
        if not all(math.isfinite(item) for item in result):
            return None
        return Vector3(x=result[0], y=result[1], z=result[2])

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            result = float(value)
        except Exception:
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _bbox_geometry(
        value: Any,
    ) -> tuple[float, float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            result = tuple(float(item) for item in value[:4])
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            len(result) != 4
            or not all(math.isfinite(item) for item in result)
            or result[2] <= 0.0
            or result[3] <= 0.0
        ):
            return None
        return result

    @staticmethod
    def _optional_bool(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _zone_source(value: Any) -> ZoneSource | None:
        if value is None:
            return None
        if isinstance(value, str) and value in {
            "nvdsanalytics_roi",
            "camera_default",
        }:
            return value  # type: ignore[return-value]
        raise ValueError("zone_source must be an exact supported provenance value")

    @staticmethod
    def _exact_zone(value: Any) -> str | None:
        if value is None:
            return None
        if not is_exact_zone_label(value):
            raise ValueError(
                "zone labels must be nonempty, unpadded, and at most 160 characters"
            )
        return value

    @staticmethod
    def _bounded_text(value: Any, *, maximum: int) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result[:maximum] if result else None

    @staticmethod
    def _required_revision(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        result = value.strip()
        return result if 1 <= len(result) <= 200 else None

    @staticmethod
    def _sha256_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        result = value.strip().lower()
        if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
            return None
        return result

    def _subject(self, observation: ObservationEnvelope, track: Mapping[str, Any]) -> SubjectRef:
        stable_id = self._positive_int(track.get("stable_id"))
        kind_raw = str(track.get("identity_kind") or track.get("identity_state") or "provisional").lower()
        resident_uuid = str(track.get("resident_uuid") or "").strip()
        display_name = str(track.get("display_name") or "").strip() or None
        if kind_raw == "resident" and resident_uuid:
            return SubjectRef(
                subject_id=f"resident:{resident_uuid}",
                kind=IdentityKind.RESIDENT,
                resident_uuid=resident_uuid,
                display_name=display_name,
                stable_id=stable_id,
            )
        if kind_raw == "visitor" and stable_id is not None:
            generation = self._nonnegative_int(track.get("visitor_generation")) or 0
            return SubjectRef(
                subject_id=f"visitor:{self.producer.run_id}:{stable_id}:g{generation}",
                kind=IdentityKind.VISITOR,
                generation=generation,
                stable_id=stable_id,
            )
        tracklet = observation.payload.tracklet
        if kind_raw == "unknown":
            return SubjectRef(
                subject_id=(
                    f"unknown:{tracklet.run_id}:{tracklet.camera_id}:"
                    f"{tracklet.tracker_id}"
                ),
                kind=IdentityKind.UNKNOWN,
                stable_id=stable_id,
            )
        return SubjectRef(
            subject_id=(
                f"tracklet:{tracklet.run_id}:{tracklet.camera_id}:"
                f"{tracklet.tracker_id}"
            ),
            kind=IdentityKind.PROVISIONAL,
            stable_id=stable_id,
        )

    @staticmethod
    def _confidence(value: Any) -> float | None:
        try:
            result = float(value)
        except Exception:
            return None
        return result if math.isfinite(result) and 0.0 <= result <= 1.0 else None

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            result = int(value)
        except Exception:
            return None
        return result if result > 0 else None

    @staticmethod
    def _nonnegative_int(value: Any) -> int | None:
        try:
            result = int(value)
        except Exception:
            return None
        return result if result >= 0 else None
