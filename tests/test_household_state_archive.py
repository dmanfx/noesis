from __future__ import annotations

import stat
from pathlib import Path

from reid.household_state import archive_legacy_identity_state, default_household_backup_dir


def test_archive_legacy_identity_state_moves_files(tmp_path) -> None:
    legacy_a = tmp_path / "reid_gallery.npz"
    legacy_b = tmp_path / "reid_aliases.json"
    legacy_a.write_bytes(b"fake-npz")
    legacy_b.write_text('{"aliases": {}}')

    backup_dir = tmp_path / "backup"
    archived = archive_legacy_identity_state(
        str(backup_dir),
        [str(legacy_a), str(legacy_b), str(tmp_path / "missing.json")],
    )

    assert len(archived) == 2
    assert not legacy_a.exists()
    assert not legacy_b.exists()
    assert (backup_dir / "reid_gallery.npz").exists()
    assert (backup_dir / "reid_aliases.json").exists()
    assert stat.S_IMODE(backup_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((backup_dir / "reid_gallery.npz").stat().st_mode) == 0o600
    assert stat.S_IMODE((backup_dir / "reid_aliases.json").stat().st_mode) == 0o600


def test_default_household_backup_dir_is_timestamped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path_a = default_household_backup_dir()
    path_b = default_household_backup_dir()
    assert path_a != path_b
    assert "backups" in path_a
