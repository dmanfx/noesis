from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pytest

from noesis.server.internal_auth import (
    InternalAuthConfigurationError,
    load_or_create_internal_token,
)
from noesis_core.journal import ContractJournal, ContractJournalError
import noesis_core.private_paths as private_paths
from noesis_core.private_paths import (
    PrivatePathError,
    atomic_create_private_file,
    atomic_write_private_file,
    read_private_file,
    require_fresh_private_file_bundle,
)
from noesis_core.scene_store import SceneReleaseStore, SceneReleaseStoreError
from reid.identity_v2.store import IdentityStore, IdentityStoreError


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


@pytest.mark.parametrize(
    ("factory", "error_type"),
    (
        (lambda path: ContractJournal(path), ContractJournalError),
        (lambda path: SceneReleaseStore(path), SceneReleaseStoreError),
        (lambda path: IdentityStore(path), IdentityStoreError),
    ),
)
def test_state_stores_reject_shared_parent_without_mutating_it(
    tmp_path: Path,
    factory,
    error_type: type[Exception],
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    before = _mode(shared)

    with pytest.raises(error_type, match="directory mode must be 0700"):
        factory(shared / "state.sqlite3")

    assert _mode(shared) == before == 0o755
    assert not (shared / "state.sqlite3").exists()


def test_contract_journal_rejects_system_temp_parent_without_mutating_it() -> None:
    system_temp = Path(tempfile.gettempdir())
    before = _mode(system_temp)
    target = system_temp / f"noesis-private-path-test-{os.getpid()}.sqlite3"
    target.unlink(missing_ok=True)

    with pytest.raises(
        ContractJournalError,
        match="owned by the service user|directory mode must be 0700",
    ):
        ContractJournal(target)

    assert _mode(system_temp) == before
    assert not target.exists()


@pytest.mark.parametrize(
    ("factory", "error_type"),
    (
        (lambda path: ContractJournal(path), ContractJournalError),
        (lambda path: SceneReleaseStore(path), SceneReleaseStoreError),
        (lambda path: IdentityStore(path), IdentityStoreError),
    ),
)
def test_state_stores_reject_insecure_existing_database_without_chmod(
    tmp_path: Path,
    factory,
    error_type: type[Exception],
) -> None:
    database = tmp_path / "state.sqlite3"
    database.touch(mode=0o644)

    with pytest.raises(error_type, match="mode must be 0600"):
        factory(database)

    assert _mode(database) == 0o644


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_contract_journal_rejects_linked_state_files(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "target.sqlite3"
    target.touch(mode=0o600)
    linked = tmp_path / "linked.sqlite3"
    if kind == "symlink":
        linked.symlink_to(target)
        expected = "symlink"
    else:
        os.link(target, linked)
        expected = "exactly one hard link"

    with pytest.raises(ContractJournalError, match=expected):
        ContractJournal(linked)


def test_existing_secure_token_load_does_not_rewrite_shared_parent(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    token_file = shared / "gateway-token"
    token_file.write_text("t" * 64 + "\n", encoding="utf-8")
    token_file.chmod(0o600)
    before = _mode(shared)

    assert load_or_create_internal_token(token_file) == "t" * 64
    assert _mode(shared) == before == 0o755
    assert _mode(token_file) == 0o600


def test_token_creation_rejects_shared_parent_without_mutating_it(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    before = _mode(shared)

    with pytest.raises(
        InternalAuthConfigurationError,
        match="directory mode must be 0700",
    ):
        load_or_create_internal_token(shared / "gateway-token")

    assert _mode(shared) == before == 0o755
    assert not (shared / "gateway-token").exists()


def test_normal_nested_state_creation_uses_private_leaf_and_file(tmp_path: Path) -> None:
    state_dir = tmp_path / "application" / "journal"
    database = state_dir / "events.sqlite3"

    journal = ContractJournal(database)

    assert journal.path == database.resolve()
    assert _mode(state_dir) == 0o700
    assert _mode(database) == 0o600


def test_private_state_rejects_intermediate_symlink_components(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(PrivatePathError, match="symlink components"):
        atomic_write_private_file(
            linked_parent / "state.json",
            b"{}\n",
            label="test private state",
        )

    assert not (real_parent / "state.json").exists()


def test_atomic_write_private_file_replaces_existing_private_state(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    destination = parent / "state.json"
    destination.write_bytes(b"old\n")
    destination.chmod(0o600)

    created = atomic_write_private_file(
        destination,
        b"new\n",
        label="mutable private state",
    )

    assert created == destination
    assert destination.read_bytes() == b"new\n"
    assert _mode(destination) == 0o600
    assert destination.stat().st_nlink == 1
    assert not list(parent.glob(".state.json.tmp-*"))


def test_atomic_write_private_file_does_not_overwrite_last_boundary_appearance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    destination = parent / "state.json"
    real_link = private_paths.os.link

    def publish_race_winner_then_link(*args, **kwargs):
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.write(descriptor, b"race winner\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(private_paths.os, "link", publish_race_winner_then_link)
    with pytest.raises(PrivatePathError, match="appeared while it was being written"):
        atomic_write_private_file(
            destination,
            b"loser\n",
            label="mutable private state",
        )

    assert destination.read_bytes() == b"race winner\n"
    assert not list(parent.glob(".state.json.tmp-*"))


def test_atomic_write_private_file_fails_if_parent_mode_changes_at_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    destination = parent / "state.json"
    destination.write_bytes(b"old\n")
    destination.chmod(0o600)
    real_replace = private_paths.os.replace

    def chmod_parent_then_replace(*args, **kwargs):
        parent.chmod(0o777)
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(private_paths.os, "replace", chmod_parent_then_replace)
    with pytest.raises(PrivatePathError, match="parent changed or became unsafe"):
        atomic_write_private_file(
            destination,
            b"new\n",
            label="mutable private state",
        )

    assert _mode(parent) == 0o777
    assert destination.read_bytes() == b"new\n"
    assert not list(parent.glob(".state.json.tmp-*"))


def test_atomic_write_private_file_fails_on_parent_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "private"
    displaced = tmp_path / "private-displaced"
    parent.mkdir(mode=0o700)
    destination = parent / "state.json"
    destination.write_bytes(b"old\n")
    destination.chmod(0o600)
    real_replace = private_paths.os.replace

    def replace_parent_path_then_publish(*args, **kwargs):
        parent.rename(displaced)
        parent.mkdir(mode=0o700)
        substitute = parent / destination.name
        substitute.write_bytes(b"substitute\n")
        substitute.chmod(0o600)
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(
        private_paths.os,
        "replace",
        replace_parent_path_then_publish,
    )
    with pytest.raises(PrivatePathError, match="parent changed or became unsafe"):
        atomic_write_private_file(
            destination,
            b"new\n",
            label="mutable private state",
        )

    assert destination.read_bytes() == b"substitute\n"
    assert (displaced / destination.name).read_bytes() == b"new\n"
    assert not list(displaced.glob(".state.json.tmp-*"))


def test_atomic_create_private_file_publishes_exactly_once(tmp_path: Path) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    destination = parent / "report.json"

    created = atomic_create_private_file(
        destination,
        b'{"ok":true}\n',
        label="immutable test evidence",
        max_bytes=1024,
    )

    assert created == destination
    assert destination.read_bytes() == b'{"ok":true}\n'
    assert _mode(destination) == 0o600
    assert destination.stat().st_nlink == 1
    assert not list(parent.glob(".report.json.tmp-*"))

    with pytest.raises(PrivatePathError, match="already exists"):
        atomic_create_private_file(
            destination,
            b'{"ok":false}\n',
            label="immutable test evidence",
            max_bytes=1024,
        )
    assert destination.read_bytes() == b'{"ok":true}\n'


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_atomic_create_private_file_never_replaces_linked_destination(
    tmp_path: Path,
    kind: str,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    backing = parent / "backing.json"
    backing.write_bytes(b"original\n")
    backing.chmod(0o600)
    destination = parent / "report.json"
    if kind == "symlink":
        destination.symlink_to(backing.name)
    else:
        os.link(backing, destination)

    expected_error = "symlink" if kind == "symlink" else "already exists"
    with pytest.raises(PrivatePathError, match=expected_error):
        atomic_create_private_file(
            destination,
            b"replacement\n",
            label="immutable linked evidence",
            max_bytes=1024,
        )

    assert backing.read_bytes() == b"original\n"
    if kind == "symlink":
        assert destination.is_symlink()
    else:
        assert destination.stat().st_ino == backing.stat().st_ino


def test_atomic_create_private_file_loses_create_race_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    destination = parent / "report.json"
    real_link = private_paths.os.link

    def race_link(
        source: str,
        target: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            os.write(descriptor, b"race-winner\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(private_paths.os, "link", race_link)
    with pytest.raises(PrivatePathError, match="appeared while it was being written"):
        atomic_create_private_file(
            destination,
            b"loser\n",
            label="immutable raced evidence",
            max_bytes=1024,
        )

    assert destination.read_bytes() == b"race-winner\n"
    assert not list(parent.glob(".report.json.tmp-*"))


def test_atomic_create_private_file_refuses_incomplete_temp_residue(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    residue = parent / ".report.json.tmp-crashed"
    residue.write_bytes(b"partial\n")
    residue.chmod(0o600)

    with pytest.raises(PrivatePathError, match="incomplete immutable publication"):
        atomic_create_private_file(
            parent / "report.json",
            b"new\n",
            label="immutable residue evidence",
            max_bytes=1024,
        )

    assert residue.read_bytes() == b"partial\n"
    assert not (parent / "report.json").exists()


def test_atomic_create_private_file_rejects_residue_injected_after_initial_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    destination = parent / "report.json"
    injected = parent / ".report.json.tmp-injected"
    real_link = private_paths.os.link

    def inject_residue_then_link(*args, **kwargs):
        descriptor = os.open(
            injected,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.write(descriptor, b"foreign residue\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(private_paths.os, "link", inject_residue_then_link)
    with pytest.raises(PrivatePathError, match="concurrent immutable publication residue"):
        atomic_create_private_file(
            destination,
            b"new\n",
            label="immutable residue-race evidence",
            max_bytes=1024,
        )

    assert destination.read_bytes() == b"new\n"
    assert injected.read_bytes() == b"foreign residue\n"
    residues = list(parent.glob(".report.json.tmp-*"))
    assert len(residues) == 2
    staged = next(path for path in residues if path != injected)
    assert staged.stat().st_ino == destination.stat().st_ino
    assert destination.stat().st_nlink == 2


def test_fresh_private_bundle_rejects_partial_session_without_repair(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    source = parent / "source.json"
    report = parent / "report.json"
    source.write_bytes(b"sealed source\n")
    source.chmod(0o600)

    with pytest.raises(PrivatePathError, match="start a new evidence session"):
        require_fresh_private_file_bundle(
            (source, report),
            label="behavior evidence bundle",
        )

    assert source.read_bytes() == b"sealed source\n"
    assert not report.exists()


def test_fresh_private_bundle_requires_one_shared_private_directory(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir(mode=0o700)
    right.mkdir(mode=0o700)

    with pytest.raises(PrivatePathError, match="share one private directory"):
        require_fresh_private_file_bundle(
            (left / "source.json", right / "report.json"),
            label="behavior evidence bundle",
        )


@pytest.mark.parametrize("boundary", ("link", "unlink"))
def test_atomic_create_private_file_fails_if_parent_mode_changes_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    destination = parent / "report.json"
    real_operation = getattr(private_paths.os, boundary)

    def chmod_parent_then_continue(*args, **kwargs):
        parent.chmod(0o777)
        return real_operation(*args, **kwargs)

    monkeypatch.setattr(private_paths.os, boundary, chmod_parent_then_continue)
    with pytest.raises(PrivatePathError, match="parent changed or became unsafe"):
        atomic_create_private_file(
            destination,
            b"new\n",
            label="immutable parent-race evidence",
            max_bytes=1024,
        )

    assert _mode(parent) == 0o777
    assert destination.read_bytes() == b"new\n"
    residue = list(parent.glob(".report.json.tmp-*"))
    if boundary == "link":
        assert len(residue) == 1
        assert residue[0].stat().st_ino == destination.stat().st_ino
        assert destination.stat().st_nlink == 2
    else:
        assert residue == []
        assert destination.stat().st_nlink == 1


def test_atomic_create_private_file_rejects_same_size_final_name_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    destination = parent / "report.json"
    real_stat = private_paths.os.stat
    named_stat_count = 0

    def replace_before_final_stat(path, *args, **kwargs):
        nonlocal named_stat_count
        if path == destination.name and kwargs.get("dir_fd") is not None:
            named_stat_count += 1
            if named_stat_count == 3:
                destination.unlink()
                destination.write_bytes(b"bad\n")
                destination.chmod(0o600)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(private_paths.os, "stat", replace_before_final_stat)
    with pytest.raises(PrivatePathError, match="changed during final validation"):
        atomic_create_private_file(
            destination,
            b"new\n",
            label="immutable final-race evidence",
            max_bytes=1024,
        )

    assert named_stat_count == 3
    assert destination.read_bytes() == b"bad\n"
    assert _mode(destination) == 0o600
    assert destination.stat().st_nlink == 1


def test_read_private_file_rejects_mode_change_while_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "private.json"
    path.write_bytes(b'{"ok":true}\n')
    path.chmod(0o600)
    real_open = private_paths.os.open

    def chmod_then_open(target, flags, *args, **kwargs):
        if Path(target) == path:
            path.chmod(0o644)
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(private_paths.os, "open", chmod_then_open)
    with pytest.raises(PrivatePathError, match="changed or became unsafe while open"):
        read_private_file(path, label="private read race", max_bytes=1024)
    assert _mode(path) == 0o644


def test_read_private_file_rejects_mode_change_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "private.json"
    path.write_bytes(b'{"ok":true}\n')
    path.chmod(0o600)
    real_read = private_paths.os.read
    changed = False

    def chmod_then_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        if not changed:
            changed = True
            path.chmod(0o644)
        return real_read(descriptor, size)

    monkeypatch.setattr(private_paths.os, "read", chmod_then_read)
    with pytest.raises(PrivatePathError, match="changed or became unsafe while open"):
        read_private_file(path, label="private read race", max_bytes=1024)
    assert changed is True
    assert _mode(path) == 0o644


@pytest.mark.parametrize(
    ("factory", "error_type"),
    (
        (lambda path: ContractJournal(path), ContractJournalError),
        (lambda path: SceneReleaseStore(path), SceneReleaseStoreError),
        (lambda path: IdentityStore(path), IdentityStoreError),
    ),
)
def test_state_stores_reject_intermediate_symlink_components(
    tmp_path: Path,
    factory,
    error_type: type[Exception],
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(error_type, match="symlink components"):
        factory(linked_parent / "state.sqlite3")

    assert not (real_parent / "state.sqlite3").exists()
