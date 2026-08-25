from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import Field

from noesis_core.contracts import CONTRACT_MODELS_BY_NAME
from noesis_core.contracts.base import ArtifactFingerprint, ContractModel, TimestampUs
from noesis_core.private_paths import (
    PrivatePathError,
    atomic_write_private_file,
    ensure_private_directory,
    read_private_file,
    validate_private_file,
)


class ReplayValidationError(ValueError):
    pass


MAX_REPLAY_RECORDS = 100_000
MAX_REPLAY_BYTES = 256 * 1024 * 1024
MAX_REPLAY_RECORD_BYTES = 2 * 1024 * 1024


class ReplayHeader(ContractModel):
    format: Literal["noesis.replay"]
    format_version: Literal[1]
    replay_id: str = Field(min_length=1, max_length=200)
    created_at_us: TimestampUs
    source: Literal["ds8", "ds9", "fixture", "synthetic"]
    purpose: str = Field(min_length=1, max_length=240)
    contains_biometrics: bool
    artifacts: tuple[ArtifactFingerprint, ...] = ()


@dataclass(frozen=True)
class ReplayContractRecord:
    sequence: int
    payload: Mapping[str, Any]
    previous_sha256: str
    record_sha256: str


@dataclass(frozen=True)
class ReplayArchive:
    header: ReplayHeader
    records: tuple[ReplayContractRecord, ...]
    final_sha256: str


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(f"replay payload is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def validate_contract_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    contract = raw.get("contract")
    if not isinstance(contract, str) or not contract:
        raise ReplayValidationError("replay contract payload requires an explicit contract name")
    model = CONTRACT_MODELS_BY_NAME.get(contract)
    if model is None:
        raise ReplayValidationError(f"unsupported replay contract: {contract}")
    try:
        validated = model.model_validate(dict(raw))
    except Exception as exc:
        raise ReplayValidationError(f"invalid {contract} replay payload: {exc}") from exc
    return validated.model_dump(mode="json")


def _project_validated_payload(
    raw: Any,
    validated: Any,
    *,
    path: str = "payload",
) -> Any:
    """Return current normalized values without inventing absent stored keys.

    Public contract models may gain optional fields with defaults while a
    journal or replay remains integrity-bound to the exact bytes written by an
    earlier binary.  Model validation must still reject malformed or unknown
    data, but comparing a full current ``model_dump`` to those historical bytes
    would mistake an additive default for corruption.  Projecting the fully
    validated value onto the stored JSON shape preserves both properties.
    """

    if isinstance(raw, Mapping):
        if not isinstance(validated, Mapping):
            raise ReplayValidationError(f"validated {path} changed JSON shape")
        projected: dict[str, Any] = {}
        for key, raw_value in raw.items():
            if key not in validated:
                raise ReplayValidationError(
                    f"validated {path} omitted stored key {key!r}"
                )
            projected[str(key)] = _project_validated_payload(
                raw_value,
                validated[key],
                path=f"{path}.{key}",
            )
        return projected
    if isinstance(raw, list):
        if not isinstance(validated, list) or len(raw) != len(validated):
            raise ReplayValidationError(f"validated {path} changed list shape")
        return [
            _project_validated_payload(
                raw_value,
                validated[index],
                path=f"{path}[{index}]",
            )
            for index, raw_value in enumerate(raw)
        ]
    return validated


def validate_stored_contract_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an integrity-bound historical payload without rewriting it.

    New writes use :func:`validate_contract_payload` and therefore contain the
    complete current normalized schema.  Reads use this compatibility path so
    additive optional defaults do not invalidate an otherwise canonical hash
    chain.  Existing stored values are still normalized and compared exactly,
    so whitespace coercion, type coercion, unknown keys, and malformed values
    remain rejected.
    """

    validated = validate_contract_payload(raw)
    projected = _project_validated_payload(dict(raw), validated)
    if not isinstance(projected, dict):  # pragma: no cover - root is guarded
        raise ReplayValidationError("validated replay payload is not an object")
    return projected


def _header_wrapper(header: ReplayHeader) -> dict[str, Any]:
    return {"record_type": "header", "header": header.model_dump(mode="json")}


def write_replay(
    path: str | Path,
    *,
    header: ReplayHeader,
    payloads: Iterable[Mapping[str, Any]],
) -> ReplayArchive:
    """Atomically write one deterministic, integrity-chained contract replay."""

    destination = Path(path).expanduser()
    try:
        ensure_private_directory(destination.parent, label="Noesis replay parent")
        if destination.is_symlink():
            raise PrivatePathError("Noesis replay must not be a symlink")
        if destination.exists():
            validate_private_file(destination, label="Noesis replay")
    except PrivatePathError as exc:
        raise ReplayValidationError(str(exc)) from exc

    header_wrapper = _header_wrapper(header)
    header_line = _canonical_bytes(header_wrapper) + b"\n"
    if len(header_line) > MAX_REPLAY_RECORD_BYTES:
        raise ReplayValidationError("replay header exceeds the record byte bound")
    if len(header_line) > MAX_REPLAY_BYTES:
        raise ReplayValidationError("replay header exceeds the archive byte bound")
    encoded_lines = [header_line]
    encoded_bytes = len(header_line)
    previous_sha256 = _sha256(header_wrapper)
    records: list[ReplayContractRecord] = []
    for sequence, raw_payload in enumerate(payloads):
        if sequence >= MAX_REPLAY_RECORDS:
            raise ReplayValidationError(
                f"replay exceeds the {MAX_REPLAY_RECORDS}-record bound"
            )
        payload = validate_contract_payload(raw_payload)
        core = {
            "record_type": "contract",
            "sequence": sequence,
            "payload": payload,
            "previous_sha256": previous_sha256,
        }
        record_sha256 = _sha256(core)
        wrapper = {**core, "record_sha256": record_sha256}
        line = _canonical_bytes(wrapper) + b"\n"
        if len(line) > MAX_REPLAY_RECORD_BYTES:
            raise ReplayValidationError(
                f"replay record {sequence} exceeds the record byte bound"
            )
        if encoded_bytes + len(line) > MAX_REPLAY_BYTES:
            raise ReplayValidationError(
                f"replay exceeds the {MAX_REPLAY_BYTES}-byte bound"
            )
        encoded_lines.append(line)
        encoded_bytes += len(line)
        records.append(
            ReplayContractRecord(
                sequence=sequence,
                payload=payload,
                previous_sha256=previous_sha256,
                record_sha256=record_sha256,
            )
        )
        previous_sha256 = record_sha256

    body = b"".join(encoded_lines)
    try:
        atomic_write_private_file(
            destination,
            body,
            label="Noesis replay",
        )
    except PrivatePathError as exc:
        raise ReplayValidationError(str(exc)) from exc

    return ReplayArchive(header=header, records=tuple(records), final_sha256=previous_sha256)


def read_replay(path: str | Path) -> ReplayArchive:
    try:
        raw = read_private_file(
            Path(path).expanduser(),
            label="Noesis replay",
            max_bytes=MAX_REPLAY_BYTES,
        )
    except PrivatePathError as exc:
        raise ReplayValidationError(f"replay cannot be read: {exc}") from exc
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise ReplayValidationError("replay is empty")
    if len(lines) > MAX_REPLAY_RECORDS + 1:
        raise ReplayValidationError("replay exceeds the record bound")
    for line_number, line in enumerate(lines, 1):
        if len(line) > MAX_REPLAY_RECORD_BYTES:
            raise ReplayValidationError(
                f"replay record exceeds the byte bound at line {line_number}"
            )
        if not line.endswith(b"\n"):
            raise ReplayValidationError(
                f"replay has an incomplete final record at line {line_number}"
            )

    try:
        header_wrapper = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ReplayValidationError(f"invalid replay header JSON: {exc}") from exc
    if (
        not isinstance(header_wrapper, dict)
        or set(header_wrapper) != {"record_type", "header"}
        or header_wrapper.get("record_type") != "header"
    ):
        raise ReplayValidationError("replay first record must be a header")
    if lines[0][:-1] != _canonical_bytes(header_wrapper):
        raise ReplayValidationError("replay header is not canonical JSON")
    try:
        header = ReplayHeader.model_validate(header_wrapper.get("header"))
    except Exception as exc:
        raise ReplayValidationError(f"invalid replay header: {exc}") from exc

    previous_sha256 = _sha256(_header_wrapper(header))
    records: list[ReplayContractRecord] = []
    for expected_sequence, line in enumerate(lines[1:]):
        try:
            wrapper = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayValidationError(f"invalid replay record JSON at sequence {expected_sequence}: {exc}") from exc
        if (
            not isinstance(wrapper, dict)
            or set(wrapper)
            != {
                "record_type",
                "sequence",
                "payload",
                "previous_sha256",
                "record_sha256",
            }
            or wrapper.get("record_type") != "contract"
        ):
            raise ReplayValidationError(f"invalid replay record type at sequence {expected_sequence}")
        if line[:-1] != _canonical_bytes(wrapper):
            raise ReplayValidationError(
                f"replay record is not canonical JSON at sequence {expected_sequence}"
            )
        if wrapper.get("sequence") != expected_sequence:
            raise ReplayValidationError(
                f"non-contiguous replay sequence: expected={expected_sequence} received={wrapper.get('sequence')}"
            )
        if wrapper.get("previous_sha256") != previous_sha256:
            raise ReplayValidationError(f"replay hash-chain mismatch at sequence {expected_sequence}")
        supplied_sha256 = wrapper.get("record_sha256")
        core = {key: value for key, value in wrapper.items() if key != "record_sha256"}
        expected_sha256 = _sha256(core)
        if supplied_sha256 != expected_sha256:
            raise ReplayValidationError(f"replay record checksum mismatch at sequence {expected_sequence}")
        payload = wrapper.get("payload")
        if not isinstance(payload, Mapping):
            raise ReplayValidationError(f"replay payload is not an object at sequence {expected_sequence}")
        validated = validate_stored_contract_payload(payload)
        if validated != payload:
            raise ReplayValidationError(f"replay payload is not canonical at sequence {expected_sequence}")
        records.append(
            ReplayContractRecord(
                sequence=expected_sequence,
                payload=validated,
                previous_sha256=previous_sha256,
                record_sha256=expected_sha256,
            )
        )
        previous_sha256 = expected_sha256
    return ReplayArchive(header=header, records=tuple(records), final_sha256=previous_sha256)
