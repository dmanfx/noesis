"""Private score-only shadow evidence capture for identity calibration."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Sequence, Tuple

from noesis_core.contracts.identity_calibration import (
    CalibrationCandidateScore,
    CalibrationEvidenceSource,
    IdentityEvidenceChainCheckpoint,
    ShadowIdentityEvidenceRecord,
)
from noesis_core.private_paths import (
    PrivatePathError,
    atomic_write_private_file,
    prepare_private_writable_file,
    read_private_file,
    validate_private_file,
)
from noesis_core.strict_json import strict_json_loads

from .coordinator import FrameBatchResult, PrimitiveFrameObservation
from .models import TrackletObservation
from .scoring import OpenSetScorer


class IdentityEvidenceError(RuntimeError):
    pass


DEFAULT_MAX_RECORDS = 100_000
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_AGE_S = 30.0 * 24.0 * 60.0 * 60.0
DEFAULT_PRUNE_INTERVAL_S = 300.0
MAX_RECORD_BYTES = 1024 * 1024


@dataclass(frozen=True)
class IdentityEvidenceRecorderHealth:
    enabled: bool
    contract: str
    contract_version: int
    session_id: str
    source: str
    runtime: str
    recorded_event_count: int
    last_observed_at_us: int | None
    retained_bytes: int
    max_records: int
    max_bytes: int
    max_age_s: float
    pruned_event_count: int


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def evidence_checkpoint_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.with_name(f"{source.name}.chain.json")


def _checkpoint_digest(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("checkpoint_sha256", None)
    return hashlib.sha256(
        b"noesis-identity-shadow-evidence-chain-v1\0" + _canonical_bytes(body)
    ).hexdigest()


def load_evidence_checkpoint(path: str | Path) -> IdentityEvidenceChainCheckpoint:
    checkpoint_path = evidence_checkpoint_path(path)
    try:
        raw = read_private_file(
            checkpoint_path,
            label="identity evidence chain checkpoint",
            max_bytes=64 * 1024,
        )
        checkpoint = IdentityEvidenceChainCheckpoint.model_validate(
            strict_json_loads(raw, label="identity evidence chain checkpoint")
        )
    except PrivatePathError as exc:
        raise IdentityEvidenceError(str(exc)) from exc
    except Exception as exc:
        raise IdentityEvidenceError(
            f"identity evidence chain checkpoint is invalid: {exc}"
        ) from exc
    canonical = _canonical_bytes(checkpoint.model_dump(mode="json")) + b"\n"
    if raw != canonical:
        raise IdentityEvidenceError(
            "identity evidence chain checkpoint is not canonical JSON"
        )
    validate_evidence_checkpoint(checkpoint)
    return checkpoint


def validate_evidence_checkpoint(
    checkpoint: IdentityEvidenceChainCheckpoint,
) -> None:
    if checkpoint.checkpoint_sha256 != _checkpoint_digest(
        checkpoint.model_dump(mode="json")
    ):
        raise IdentityEvidenceError(
            "identity evidence chain checkpoint digest mismatch"
        )


def _gallery_count(evidence: Sequence[str]) -> int:
    for token in evidence:
        if str(token).startswith("gallery_exemplars="):
            try:
                return max(0, int(str(token).split("=", 1)[1]))
            except (TypeError, ValueError):
                return 0
    return 0


class IdentityEvidenceRecorder:
    """Append immutable, owner-only JSONL without serializing embeddings."""

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str,
        source: str,
        runtime: str,
        max_records: int = DEFAULT_MAX_RECORDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        prune_interval_s: float = DEFAULT_PRUNE_INTERVAL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._max_records = int(max_records)
        self._max_bytes = int(max_bytes)
        self._max_age_s = float(max_age_s)
        self._prune_interval_s = float(prune_interval_s)
        if self._max_records < 1:
            raise IdentityEvidenceError(
                "identity evidence max_records must be positive"
            )
        if self._max_bytes < MAX_RECORD_BYTES:
            raise IdentityEvidenceError(
                f"identity evidence max_bytes must be at least {MAX_RECORD_BYTES}"
            )
        if self._max_age_s <= 0.0 or self._prune_interval_s <= 0.0:
            raise IdentityEvidenceError(
                "identity evidence max_age_s and prune_interval_s must be positive"
            )
        self._clock = clock
        self.session_id = str(session_id or "").strip()
        if not self.session_id:
            raise IdentityEvidenceError("identity evidence session_id is required")
        try:
            self.source = CalibrationEvidenceSource(str(source).strip().lower())
        except ValueError as exc:
            raise IdentityEvidenceError(
                "identity evidence source must be shadow or replay"
            ) from exc
        self.runtime = str(runtime or "").strip().lower()
        if self.runtime not in {"ds8", "ds9", "replay", "test"}:
            raise IdentityEvidenceError(
                "identity evidence runtime must be ds8, ds9, replay, or test"
            )
        try:
            self.path = prepare_private_writable_file(
                Path(path).expanduser(),
                label="identity evidence JSONL",
            )
        except PrivatePathError as exc:
            raise IdentityEvidenceError(str(exc)) from exc
        self._lock = threading.RLock()
        self._recorded_event_count = 0
        self._last_observed_at_us: int | None = None
        self._retained_bytes = 0
        self._pruned_event_count = 0
        self._seen_event_ids: set[str] = set()
        now_us = max(1, int(float(self._clock()) * 1_000_000))
        self._last_prune_check_us = now_us
        checkpoint_path = evidence_checkpoint_path(self.path)
        if checkpoint_path.is_symlink():
            raise IdentityEvidenceError(
                "identity evidence chain checkpoint must not be a symlink"
            )
        if checkpoint_path.exists():
            self._checkpoint = load_evidence_checkpoint(self.path)
        elif self.path.stat().st_size == 0:
            self._checkpoint = self._new_checkpoint(
                next_sequence=0,
                first_sequence=None,
                previous_event_id=None,
                tail_event_id=None,
                retained_count=0,
                retained_bytes=0,
                updated_at_us=now_us,
            )
            self._write_checkpoint(self._checkpoint)
        else:
            raise IdentityEvidenceError(
                "identity evidence chain checkpoint is missing for a non-empty file"
            )
        self._inspect_existing(now_us=now_us)

    @staticmethod
    def _new_checkpoint(
        *,
        next_sequence: int,
        first_sequence: int | None,
        previous_event_id: str | None,
        tail_event_id: str | None,
        retained_count: int,
        retained_bytes: int,
        updated_at_us: int,
    ) -> IdentityEvidenceChainCheckpoint:
        body: dict[str, object] = {
            "contract": "noesis.identity.shadow_score_evidence_chain",
            "contract_version": 1,
            "next_sequence": int(next_sequence),
            "first_sequence": first_sequence,
            "previous_event_id": previous_event_id,
            "tail_event_id": tail_event_id,
            "retained_count": int(retained_count),
            "retained_bytes": int(retained_bytes),
            "updated_at_us": max(1, int(updated_at_us)),
        }
        body["checkpoint_sha256"] = _checkpoint_digest(body)
        return IdentityEvidenceChainCheckpoint.model_validate(body)

    def _write_checkpoint(self, checkpoint: IdentityEvidenceChainCheckpoint) -> None:
        try:
            atomic_write_private_file(
                evidence_checkpoint_path(self.path),
                _canonical_bytes(checkpoint.model_dump(mode="json")) + b"\n",
                label="identity evidence chain checkpoint",
            )
        except PrivatePathError as exc:
            raise IdentityEvidenceError(str(exc)) from exc

    def _inspect_existing(self, *, now_us: int) -> None:
        retained, dropped, next_sequence, chain_tail = self._scan_retained(
            (), now_us=now_us
        )
        if dropped:
            self._rewrite_retained(retained)
            self._pruned_event_count += dropped
            self._checkpoint = self._checkpoint_for_retained(
                retained,
                next_sequence=next_sequence,
                chain_tail=chain_tail,
                updated_at_us=now_us,
            )
            self._write_checkpoint(self._checkpoint)
        self._set_retained_state(retained)

    @staticmethod
    def _validate_line(
        line: bytes,
        *,
        line_number: int,
    ) -> ShadowIdentityEvidenceRecord:
        if len(line) > MAX_RECORD_BYTES:
            raise IdentityEvidenceError(
                f"identity evidence record exceeds the byte bound at line {line_number}"
            )
        if not line.endswith(b"\n"):
            raise IdentityEvidenceError(
                f"incomplete identity evidence record at line {line_number}"
            )
        if not line.strip():
            raise IdentityEvidenceError(
                f"blank identity evidence line at {line_number}"
            )
        try:
            row = ShadowIdentityEvidenceRecord.model_validate(
                strict_json_loads(
                    line,
                    label=f"identity evidence line {line_number}",
                )
            )
        except Exception as exc:
            raise IdentityEvidenceError(
                f"invalid identity evidence line {line_number}: {exc}"
            ) from exc
        if line != _canonical_bytes(row.model_dump(mode="json")) + b"\n":
            raise IdentityEvidenceError(
                f"identity evidence line {line_number} is not canonical JSON"
            )
        body = row.model_dump(mode="json")
        body.pop("event_id", None)
        expected_event_id = hashlib.sha256(
            b"noesis-identity-shadow-evidence-v2\0" + _canonical_bytes(body)
        ).hexdigest()
        if row.event_id != expected_event_id:
            raise IdentityEvidenceError(
                f"identity evidence event digest mismatch at line {line_number}"
            )
        return row

    def _scan_retained(
        self,
        extra_lines: Sequence[bytes],
        *,
        now_us: int,
    ) -> tuple[
        Deque[tuple[ShadowIdentityEvidenceRecord, bytes]],
        int,
        int,
        str | None,
    ]:
        try:
            validated = validate_private_file(
                self.path,
                label="identity evidence JSONL",
            )
        except PrivatePathError as exc:
            raise IdentityEvidenceError(str(exc)) from exc
        size = validated.stat().st_size
        if size > self._max_bytes * 2:
            raise IdentityEvidenceError(
                "identity evidence file exceeds twice its configured byte bound; "
                "refusing an unbounded startup scan"
            )
        retained: Deque[tuple[ShadowIdentityEvidenceRecord, bytes]] = deque()
        retained_bytes = 0
        dropped = 0
        seen: set[str] = set()
        last_observed_at_us: int | None = None
        line_number = 0
        expected_sequence = (
            int(self._checkpoint.first_sequence)
            if self._checkpoint.retained_count > 0
            and self._checkpoint.first_sequence is not None
            else self._checkpoint.next_sequence
        )
        expected_previous = self._checkpoint.previous_event_id
        chain_tail = expected_previous
        existing_count = 0
        existing_bytes = 0

        def consume(line: bytes, *, existing: bool) -> None:
            nonlocal retained_bytes, dropped, last_observed_at_us, line_number
            nonlocal expected_sequence, expected_previous, chain_tail
            nonlocal existing_count, existing_bytes
            line_number += 1
            row = self._validate_line(line, line_number=line_number)
            if row.event_id in seen:
                raise IdentityEvidenceError(
                    f"duplicate identity evidence event {row.event_id}"
                )
            if (
                last_observed_at_us is not None
                and row.observed_at_us < last_observed_at_us
            ):
                raise IdentityEvidenceError(
                    f"identity evidence timestamps move backwards at line {line_number}"
                )
            if row.sequence != expected_sequence:
                raise IdentityEvidenceError(
                    f"identity evidence sequence gap at line {line_number}"
                )
            if row.previous_event_id != expected_previous:
                raise IdentityEvidenceError(
                    f"identity evidence hash-chain mismatch at line {line_number}"
                )
            seen.add(row.event_id)
            last_observed_at_us = row.observed_at_us
            expected_sequence += 1
            expected_previous = row.event_id
            chain_tail = row.event_id
            if existing:
                existing_count += 1
                existing_bytes += len(line)
            cutoff_us = int(now_us - self._max_age_s * 1_000_000)
            if row.observed_at_us < cutoff_us:
                dropped += 1
                return
            retained.append((row, line))
            retained_bytes += len(line)
            while len(retained) > self._max_records or retained_bytes > self._max_bytes:
                _old_row, old_line = retained.popleft()
                retained_bytes -= len(old_line)
                dropped += 1

        with self.path.open("rb") as stream:
            while True:
                line = stream.readline(MAX_RECORD_BYTES + 1)
                if not line:
                    break
                consume(line, existing=True)
                if line_number > self._max_records * 2:
                    raise IdentityEvidenceError(
                        "identity evidence file exceeds twice its configured record bound"
                    )
        if (
            existing_count != self._checkpoint.retained_count
            or existing_bytes != self._checkpoint.retained_bytes
            or expected_sequence != self._checkpoint.next_sequence
            or (
                self._checkpoint.retained_count > 0
                and chain_tail != self._checkpoint.tail_event_id
            )
        ):
            raise IdentityEvidenceError(
                "identity evidence file disagrees with its chain checkpoint"
            )
        for line in extra_lines:
            consume(bytes(line), existing=False)
        return retained, dropped, expected_sequence, chain_tail

    def _checkpoint_for_retained(
        self,
        retained: Sequence[tuple[ShadowIdentityEvidenceRecord, bytes]],
        *,
        next_sequence: int,
        chain_tail: str | None,
        updated_at_us: int,
    ) -> IdentityEvidenceChainCheckpoint:
        rows = tuple(retained)
        if rows:
            first = rows[0][0]
            return self._new_checkpoint(
                next_sequence=next_sequence,
                first_sequence=first.sequence,
                previous_event_id=first.previous_event_id,
                tail_event_id=rows[-1][0].event_id,
                retained_count=len(rows),
                retained_bytes=sum(len(line) for _row, line in rows),
                updated_at_us=updated_at_us,
            )
        return self._new_checkpoint(
            next_sequence=next_sequence,
            first_sequence=None,
            previous_event_id=chain_tail,
            tail_event_id=None,
            retained_count=0,
            retained_bytes=0,
            updated_at_us=updated_at_us,
        )

    def _rewrite_retained(
        self,
        retained: Sequence[tuple[ShadowIdentityEvidenceRecord, bytes]],
    ) -> None:
        payload = b"".join(line for _row, line in retained)
        try:
            self.path = atomic_write_private_file(
                self.path,
                payload,
                label="identity evidence JSONL",
            )
        except PrivatePathError as exc:
            raise IdentityEvidenceError(str(exc)) from exc

    def _set_retained_state(
        self,
        retained: Sequence[tuple[ShadowIdentityEvidenceRecord, bytes]],
    ) -> None:
        rows = tuple(retained)
        self._recorded_event_count = len(rows)
        self._retained_bytes = sum(len(line) for _row, line in rows)
        self._last_observed_at_us = rows[-1][0].observed_at_us if rows else None
        self._seen_event_ids = {row.event_id for row, _line in rows}

    def _append_payload(self, payload: bytes) -> None:
        try:
            validated = validate_private_file(
                self.path,
                label="identity evidence JSONL",
            )
        except PrivatePathError as exc:
            raise IdentityEvidenceError(str(exc)) from exc
        expected = validated.stat()
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            raise IdentityEvidenceError(
                "identity evidence file cannot be opened securely"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
                raise IdentityEvidenceError(
                    "identity evidence file changed while opening"
                )
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size != self._retained_bytes
            ):
                raise IdentityEvidenceError(
                    "identity evidence file changed outside the recorder"
                )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise IdentityEvidenceError(
                        "identity evidence append made no progress"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def health(self) -> IdentityEvidenceRecorderHealth:
        with self._lock:
            return IdentityEvidenceRecorderHealth(
                enabled=True,
                contract="noesis.identity.shadow_score_evidence",
                contract_version=2,
                session_id=self.session_id,
                source=self.source.value,
                runtime=self.runtime,
                recorded_event_count=self._recorded_event_count,
                last_observed_at_us=self._last_observed_at_us,
                retained_bytes=self._retained_bytes,
                max_records=self._max_records,
                max_bytes=self._max_bytes,
                max_age_s=self._max_age_s,
                pruned_event_count=self._pruned_event_count,
            )

    def append_frame(
        self,
        *,
        run_id: str,
        model_sha256: str,
        model_semantic_profile_sha256: str,
        model_layer: str,
        embedding_dim: int,
        observed_at_us: int,
        observations: Sequence[PrimitiveFrameObservation],
        candidate_rows: Sequence[TrackletObservation],
        batch: FrameBatchResult,
        scorer: OpenSetScorer,
        runtime_mode: str,
    ) -> Tuple[ShadowIdentityEvidenceRecord, ...]:
        observations_by_tracklet = {
            row.to_runtime().key.tracklet_id: row for row in observations
        }
        candidates_by_tracklet = {row.tracklet_id: row for row in candidate_rows}
        overlays_by_tracklet = {row.key.tracklet_id: row for row in batch.overlays}
        expected = set(observations_by_tracklet)
        if (
            set(candidates_by_tracklet) != expected
            or set(overlays_by_tracklet) != expected
        ):
            raise IdentityEvidenceError(
                "identity evidence frame surfaces disagree on tracklet membership"
            )
        record_bodies = []
        for tracklet_id in sorted(expected):
            observation = observations_by_tracklet[tracklet_id]
            candidate_row = candidates_by_tracklet[tracklet_id]
            overlay = overlays_by_tracklet[tracklet_id]
            score_row = scorer.score_tracklet(candidate_row)
            decision = overlay.resolver_decision
            candidates = tuple(
                CalibrationCandidateScore(
                    subject_id=row.identity_id,
                    identity_kind=row.identity_kind.value,
                    raw_similarity=row.raw_similarity,
                    hard_allowed=row.hard_allowed,
                    hard_constraint_reason=(
                        None
                        if row.hard_allowed
                        else str(row.hard_constraint_reason or "hard_constraint")
                    ),
                    gallery_exemplar_count=_gallery_count(row.evidence),
                )
                for row in sorted(
                    candidate_row.candidates, key=lambda item: item.identity_id
                )
            )
            final_outcome = (
                "unknown"
                if decision.is_unknown or decision.identity_kind is None
                else decision.identity_kind.value
            )
            body = {
                "contract": "noesis.identity.shadow_score_evidence",
                "contract_version": 2,
                "observed_at_us": max(1, int(observed_at_us)),
                "source": self.source.value,
                "session_id": self.session_id,
                "runtime": self.runtime,
                "runtime_mode": str(runtime_mode),
                "run_id": str(run_id),
                "camera_id": observation.camera_id,
                "tracker_id": observation.tracker_id,
                "frame_id": int(observation.frame_id),
                "observation_id": observation.observation_id,
                "model_sha256": str(model_sha256),
                "model_semantic_profile_sha256": str(model_semantic_profile_sha256),
                "model_layer": str(model_layer),
                "embedding_dim": int(embedding_dim),
                "quality": float(observation.quality),
                "candidates": [row.model_dump(mode="json") for row in candidates],
                "pre_prior_winner_id": score_row.pre_prior_winner_id,
                "final_winner_id": decision.identity_id,
                "final_outcome": final_outcome,
                "calibrated_confidence": float(decision.calibrated_confidence),
                "reject_reason": decision.reason if decision.is_unknown else None,
                "prior_changed_winner": (
                    "bounded_resident_prior_changed_winner" in decision.evidence
                ),
            }
            record_bodies.append(body)
        if not record_bodies:
            return ()
        with self._lock:
            if load_evidence_checkpoint(self.path) != self._checkpoint:
                raise IdentityEvidenceError(
                    "identity evidence checkpoint changed outside the recorder"
                )
            records = []
            sequence = self._checkpoint.next_sequence
            previous_event_id = (
                self._checkpoint.tail_event_id
                if self._checkpoint.retained_count > 0
                else self._checkpoint.previous_event_id
            )
            for raw_body in record_bodies:
                body = {
                    **raw_body,
                    "sequence": sequence,
                    "previous_event_id": previous_event_id,
                }
                event_id = hashlib.sha256(
                    b"noesis-identity-shadow-evidence-v2\0" + _canonical_bytes(body)
                ).hexdigest()
                row = ShadowIdentityEvidenceRecord.model_validate(
                    {"event_id": event_id, **body}
                )
                records.append(row)
                sequence += 1
                previous_event_id = event_id
            lines = tuple(
                _canonical_bytes(row.model_dump(mode="json")) + b"\n" for row in records
            )
            payload = b"".join(lines)
            if any(len(line) > MAX_RECORD_BYTES for line in lines):
                raise IdentityEvidenceError(
                    "identity evidence frame contains a record above the byte bound"
                )
            if len(lines) > self._max_records or len(payload) > self._max_bytes:
                raise IdentityEvidenceError(
                    "identity evidence frame exceeds the configured retention capacity"
                )
            event_ids = [row.event_id for row in records]
            if len(event_ids) != len(set(event_ids)):
                raise IdentityEvidenceError(
                    "identity evidence frame produced duplicate event IDs"
                )
            duplicate = sorted(set(event_ids) & self._seen_event_ids)
            if duplicate:
                raise IdentityEvidenceError(
                    f"identity evidence event was already recorded: {duplicate[0]}"
                )
            frame_observed_at_us = max(row.observed_at_us for row in records)
            if (
                self._last_observed_at_us is not None
                and min(row.observed_at_us for row in records)
                < self._last_observed_at_us
            ):
                raise IdentityEvidenceError(
                    "identity evidence timestamps moved backwards"
                )
            needs_retention = (
                self._recorded_event_count + len(records) > self._max_records
                or self._retained_bytes + len(payload) > self._max_bytes
                or frame_observed_at_us - self._last_prune_check_us
                >= int(self._prune_interval_s * 1_000_000)
            )
            if needs_retention:
                retained, dropped, next_sequence, chain_tail = self._scan_retained(
                    lines,
                    now_us=frame_observed_at_us,
                )
                self._rewrite_retained(retained)
                self._checkpoint = self._checkpoint_for_retained(
                    retained,
                    next_sequence=next_sequence,
                    chain_tail=chain_tail,
                    updated_at_us=frame_observed_at_us,
                )
                self._write_checkpoint(self._checkpoint)
                self._set_retained_state(retained)
                self._pruned_event_count += dropped
                self._last_prune_check_us = frame_observed_at_us
            else:
                self._append_payload(payload)
                first_sequence = self._checkpoint.first_sequence
                checkpoint_previous = self._checkpoint.previous_event_id
                if self._checkpoint.retained_count == 0:
                    first_sequence = records[0].sequence
                    checkpoint_previous = records[0].previous_event_id
                self._checkpoint = self._new_checkpoint(
                    next_sequence=records[-1].sequence + 1,
                    first_sequence=first_sequence,
                    previous_event_id=checkpoint_previous,
                    tail_event_id=records[-1].event_id,
                    retained_count=self._recorded_event_count + len(records),
                    retained_bytes=self._retained_bytes + len(payload),
                    updated_at_us=frame_observed_at_us,
                )
                self._write_checkpoint(self._checkpoint)
                self._recorded_event_count += len(records)
                self._retained_bytes += len(payload)
                self._last_observed_at_us = frame_observed_at_us
                self._seen_event_ids.update(event_ids)
        return tuple(records)


__all__ = [
    "DEFAULT_MAX_AGE_S",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_RECORDS",
    "DEFAULT_PRUNE_INTERVAL_S",
    "IdentityEvidenceError",
    "IdentityEvidenceRecorder",
    "IdentityEvidenceRecorderHealth",
    "evidence_checkpoint_path",
    "load_evidence_checkpoint",
    "validate_evidence_checkpoint",
]
