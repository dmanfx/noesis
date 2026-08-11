from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import noesis_core.replay as replay_module
from noesis_core.contracts.health import CapabilityHealth, CapabilityState, CapabilityStatus
from noesis_core.replay import ReplayHeader, ReplayValidationError, read_replay, write_replay


def _header() -> ReplayHeader:
    return ReplayHeader(
        format="noesis.replay",
        format_version=1,
        replay_id="fixture-1",
        created_at_us=1_000_000,
        source="fixture",
        purpose="contract parity test",
        contains_biometrics=False,
    )


def _health_payload() -> dict[str, object]:
    health = CapabilityHealth(
        contract="noesis.capability.health",
        contract_version=1,
        instance_id="appliance",
        run_id="run-1",
        generated_at_us=1_000_010,
        capabilities=(
            CapabilityState(
                capability="tracking",
                status=CapabilityStatus.HEALTHY,
                checked_at_us=1_000_010,
                last_success_at_us=1_000_009,
                evidence={"sequence": 4},
            ),
        ),
    )
    return health.model_dump(mode="json")


def test_replay_is_deterministic_validated_and_owner_only(tmp_path: Path) -> None:
    first = tmp_path / "first.noesis-replay.ndjson"
    second = tmp_path / "second.noesis-replay.ndjson"
    archive = write_replay(first, header=_header(), payloads=[_health_payload()])
    write_replay(second, header=_header(), payloads=[_health_payload()])

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_mode & 0o777 == 0o600
    loaded = read_replay(first)
    assert loaded == archive
    assert loaded.records[0].payload["contract"] == "noesis.capability.health"


def test_replay_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "tampered.ndjson"
    write_replay(path, header=_header(), payloads=[_health_payload()])
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["payload"]["run_id"] = "tampered"
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ReplayValidationError, match="checksum mismatch"):
        read_replay(path)


def test_replay_rejects_missing_or_unsupported_contract(tmp_path: Path) -> None:
    with pytest.raises(ReplayValidationError, match="explicit contract"):
        write_replay(tmp_path / "missing.ndjson", header=_header(), payloads=[{"contract_version": 1}])
    with pytest.raises(ReplayValidationError, match="unsupported replay contract"):
        write_replay(
            tmp_path / "future.ndjson",
            header=_header(),
            payloads=[{"contract": "noesis.future", "contract_version": 1}],
        )


def test_replay_rejects_incomplete_linked_and_insecure_state_without_chmod(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.ndjson"
    write_replay(valid, header=_header(), payloads=[_health_payload()])
    valid.write_bytes(valid.read_bytes().rstrip(b"\n"))
    with pytest.raises(ReplayValidationError, match="incomplete final record"):
        read_replay(valid)

    insecure = tmp_path / "insecure.ndjson"
    insecure.write_bytes(valid.read_bytes() + b"\n")
    insecure.chmod(0o644)
    with pytest.raises(ReplayValidationError, match="mode must be 0600"):
        read_replay(insecure)
    with pytest.raises(ReplayValidationError, match="mode must be 0600"):
        write_replay(insecure, header=_header(), payloads=[_health_payload()])
    assert insecure.stat().st_mode & 0o777 == 0o644

    linked = tmp_path / "linked.ndjson"
    linked.symlink_to(valid)
    with pytest.raises(ReplayValidationError, match="symlink"):
        read_replay(linked)
    with pytest.raises(ReplayValidationError, match="symlink"):
        write_replay(linked, header=_header(), payloads=[_health_payload()])
    assert linked.is_symlink()

    hardlinked = tmp_path / "hardlinked.ndjson"
    os.link(valid, hardlinked)
    with pytest.raises(ReplayValidationError, match="exactly one hard link"):
        read_replay(hardlinked)
    with pytest.raises(ReplayValidationError, match="exactly one hard link"):
        write_replay(hardlinked, header=_header(), payloads=[_health_payload()])


def test_replay_enforces_byte_bound_before_consuming_unbounded_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumed = 0

    def payloads():
        nonlocal consumed
        for _index in range(10_000):
            consumed += 1
            yield _health_payload()

    monkeypatch.setattr(replay_module, "MAX_REPLAY_BYTES", 1024)
    destination = tmp_path / "bounded.ndjson"
    with pytest.raises(ReplayValidationError, match="archive byte bound|byte bound"):
        write_replay(destination, header=_header(), payloads=payloads())
    assert consumed < 10_000
    assert not destination.exists()


def test_replay_rejects_noncanonical_and_extra_wrapper_keys(tmp_path: Path) -> None:
    path = tmp_path / "canonical.ndjson"
    write_replay(path, header=_header(), payloads=[_health_payload()])
    lines = path.read_text(encoding="utf-8").splitlines()

    header = json.loads(lines[0])
    lines[0] = json.dumps(header, indent=2).replace("\n", " ")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ReplayValidationError, match="header is not canonical"):
        read_replay(path)

    write_replay(path, header=_header(), payloads=[_health_payload()])
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["unexpected"] = True
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ReplayValidationError, match="invalid replay record type"):
        read_replay(path)
