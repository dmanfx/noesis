from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "runtime_ownership_registry.py"
SPEC = importlib.util.spec_from_file_location(
    "runtime_ownership_registry_test_module", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
registry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = registry
SPEC.loader.exec_module(registry)


MATRIX_ID = "noesis-ds8-ds9-runtime-ownership"
MATRIX_A = "a" * 64
MATRIX_B = "b" * 64
CHECKOUT = "c" * 64


def _runtime_root(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _append(
    root: Path,
    *,
    matrix_sha256: str = MATRIX_A,
    supersedes: str | None = None,
    event_type: str = "promote",
):
    def build(snapshot):
        return registry.build_event(
            snapshot,
            event_type=event_type,
            matrix_id=MATRIX_ID,
            matrix_sha256=matrix_sha256,
            capability_id="artifacts.canonical_graph",
            evidence_type="asset_realization",
            subject="ds9",
            selector={"profiles": ["canonical"]} if event_type == "promote" else None,
            runtime_binding=None,
            checkout_sha256=CHECKOUT,
            artifact_binding={"profiles": ["canonical"]}
            if event_type == "promote"
            else None,
            evidence_sha256s_digest=None,
            supersedes_event_digest=supersedes,
            recorded_at_utc=(
                "2026-07-11T12:00:00Z"
                if not snapshot.events
                else f"2026-07-11T12:00:0{len(snapshot.events)}Z"
            ),
        )

    return registry.append_registry_event(root, build)


def test_registry_append_is_canonical_locked_durable_and_session_external(
    tmp_path: Path,
) -> None:
    root = _runtime_root(tmp_path)
    session = root / "evidence" / "session-a" / "launcher"
    session.mkdir(parents=True, mode=0o700)
    session.chmod(0o700)
    session_file = session / "SHA256SUMS"
    session_file.write_text("fixture\n", encoding="utf-8")
    session_file.chmod(0o600)
    before = session_file.read_bytes()
    lock_was_held = False

    def build(snapshot):
        nonlocal lock_was_held
        lock_path = (
            root / registry.REGISTRY_DIRECTORY / registry.REGISTRY_LOCK_FILENAME
        )
        descriptor = os.open(lock_path, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_was_held = True
        finally:
            os.close(descriptor)
        return registry.build_event(
            snapshot,
            event_type="promote",
            matrix_id=MATRIX_ID,
            matrix_sha256=MATRIX_A,
            capability_id="artifacts.canonical_graph",
            evidence_type="asset_realization",
            subject="ds9",
            selector={"profiles": ["canonical"]},
            runtime_binding=None,
            checkout_sha256=CHECKOUT,
            artifact_binding={"profiles": ["canonical"]},
            evidence_sha256s_digest=None,
            supersedes_event_digest=None,
            recorded_at_utc="2026-07-11T12:00:00Z",
        )

    event = registry.append_registry_event(root, build)
    snapshot = registry.read_registry(root)

    assert lock_was_held is True
    assert session_file.read_bytes() == before
    assert len(snapshot.events) == 1
    assert snapshot.tail_event_digest == event["event_digest"]
    assert snapshot.active_for_matrix(MATRIX_ID, MATRIX_A)
    registry_path = root / registry.REGISTRY_DIRECTORY / registry.REGISTRY_FILENAME
    head_path = root / registry.REGISTRY_DIRECTORY / registry.REGISTRY_HEAD_FILENAME
    assert registry_path.stat().st_mode & 0o777 == 0o600
    assert head_path.stat().st_mode & 0o777 == 0o600
    line = registry_path.read_bytes().splitlines()[0]
    assert line == registry.canonical_json_bytes(json.loads(line))


def test_registry_requires_explicit_linear_supersession_and_revoke(
    tmp_path: Path,
) -> None:
    root = _runtime_root(tmp_path)
    first = _append(root)

    with pytest.raises(registry.RegistryError, match="supersession"):
        _append(root, supersedes=None)

    second = _append(root, supersedes=first["event_digest"])
    revoked = _append(
        root,
        supersedes=second["event_digest"],
        event_type="revoke",
    )
    snapshot = registry.read_registry(root)
    key = (
        "artifacts.canonical_graph",
        "asset_realization",
        "ds9",
    )
    assert snapshot.key_heads[key]["event_digest"] == revoked["event_digest"]
    assert key not in snapshot.active


def test_new_matrix_epoch_supersedes_old_promotion_without_rewriting_history(
    tmp_path: Path,
) -> None:
    root = _runtime_root(tmp_path)
    first = _append(root, matrix_sha256=MATRIX_A)
    _append(
        root,
        matrix_sha256=MATRIX_B,
        supersedes=first["event_digest"],
    )
    snapshot = registry.read_registry(root)

    assert snapshot.active_for_matrix(MATRIX_ID, MATRIX_A) == {}
    assert len(snapshot.active_for_matrix(MATRIX_ID, MATRIX_B)) == 1
    assert len(snapshot.events) == 2


@pytest.mark.parametrize("mutation", ("digest", "splice", "duplicate_key", "rollback"))
def test_registry_rejects_digest_splice_duplicate_and_rollback(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _runtime_root(tmp_path)
    first = _append(root)
    directory = root / registry.REGISTRY_DIRECTORY
    registry_path = directory / registry.REGISTRY_FILENAME
    head_path = directory / registry.REGISTRY_HEAD_FILENAME
    first_registry = registry_path.read_bytes()
    _first_head = head_path.read_bytes()
    second = _append(root, supersedes=first["event_digest"])

    if mutation == "rollback":
        registry_path.write_bytes(first_registry)
    else:
        rows = registry_path.read_bytes().splitlines()
        event = json.loads(rows[-1])
        if mutation == "digest":
            event["event_digest"] = "f" * 64
            rows[-1] = registry.canonical_json_bytes(event)
        elif mutation == "splice":
            event["previous_event_digest"] = "e" * 64
            event["event_digest"] = registry._event_digest(event)
            rows[-1] = registry.canonical_json_bytes(event)
        else:
            text = rows[-1].decode("utf-8")
            rows[-1] = text.replace(
                '"schema_version":1',
                '"schema_version":1,"schema_version":1',
                1,
            ).encode()
        registry_path.write_bytes(b"\n".join(rows) + b"\n")
    registry_path.chmod(0o600)

    with pytest.raises(registry.RegistryError):
        registry.read_registry(root)
    assert second["sequence"] == 2


def test_registry_rejects_symlinked_registry_file(tmp_path: Path) -> None:
    root = _runtime_root(tmp_path)
    _append(root)
    directory = root / registry.REGISTRY_DIRECTORY
    registry_path = directory / registry.REGISTRY_FILENAME
    target = directory / "attacker.jsonl"
    registry_path.rename(target)
    registry_path.symlink_to(target)

    with pytest.raises(registry.RegistryError):
        registry.read_registry(root)


def test_registry_rejects_extra_entries_and_symlinked_root_component(
    tmp_path: Path,
) -> None:
    root = _runtime_root(tmp_path)
    _append(root)
    extra = root / registry.REGISTRY_DIRECTORY / "latest.json"
    extra.write_text("{}\n", encoding="utf-8")
    extra.chmod(0o600)
    with pytest.raises(registry.RegistryError, match="entry set"):
        registry.read_registry(root)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_root = linked_parent / "runtime"
    linked_root.mkdir(mode=0o700)
    with pytest.raises(registry.RegistryError, match="symlink component"):
        registry.read_registry(linked_root)
