from __future__ import annotations

import json
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "identity_v2_migrate.py"
FINGERPRINT = "cli-test-profile"
DIMENSION = 4


def _legacy(root: Path) -> None:
    root.mkdir()
    resident = {
        "uuid": str(uuid.uuid4()),
        "stable_id": 1,
        "display_name": "Alice",
        "created_ts": 1.0,
        "embedding_count": 1,
    }
    (root / "residents.json").write_text(
        json.dumps({"version": 1, "residents": [resident]}),
        encoding="utf-8",
    )
    (root / "identity_manifest.json").write_text(
        json.dumps({"model_fingerprint": FINGERPRINT, "embedding_dim": DIMENSION}),
        encoding="utf-8",
    )
    np.savez_compressed(
        root / "resident_gallery.npz",
        sids=np.asarray([1], dtype=np.int64),
        emb_1=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        ts_1=np.asarray([1.0], dtype=np.float64),
    )


def _base(command: str, root: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        command,
        "--source-root",
        str(root),
        "--model-fingerprint",
        FINGERPRINT,
        "--embedding-dim",
        str(DIMENSION),
        "--assert-source-model-fingerprint",
        FINGERPRINT,
        "--assert-source-embedding-dim",
        str(DIMENSION),
    ]


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=SCRIPT.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_requires_explicit_provenance_and_dry_run_never_writes(tmp_path) -> None:
    root = tmp_path / "legacy"
    _legacy(root)
    missing_assertions = _run(
        [
            sys.executable,
            str(SCRIPT),
            "dry-run",
            "--source-root",
            str(root),
            "--model-fingerprint",
            FINGERPRINT,
            "--embedding-dim",
            str(DIMENSION),
        ]
    )
    assert missing_assertions.returncode == 2
    assert "--assert-source-model-fingerprint" in missing_assertions.stderr

    before = {path.name: path.read_bytes() for path in root.iterdir()}
    result = _run(_base("dry-run", root))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "dry-run"
    assert payload["apply_disposition"] == "full"
    assert payload["full_migration_safe"] is True
    assert "safe_to_apply" not in payload
    assert len(payload["source_digest"]) == 64
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert not (root / "identity.sqlite3").exists()

    archive_root = tmp_path / "dry-run-archives"
    archived_dry_run = _run(
        _base("dry-run", root) + ["--archive-root", str(archive_root)]
    )
    assert archived_dry_run.returncode == 0, archived_dry_run.stderr
    archived_payload = json.loads(archived_dry_run.stdout)["pre_apply_archive"]
    assert len(archived_payload["receipt"]) == 64
    assert archived_payload["source_digest"] == payload["source_digest"]
    assert Path(archived_payload["archive_path"]).is_dir()
    assert not (root / "identity.sqlite3").exists()


def test_cli_apply_requires_fresh_digest_and_is_idempotent(tmp_path) -> None:
    root = tmp_path / "legacy"
    _legacy(root)
    digest = json.loads(_run(_base("dry-run", root)).stdout)["source_digest"]
    database = tmp_path / "state" / "identity.sqlite3"
    archive_root = tmp_path / "archives"
    archived = _run(
        _base("archive", root)
        + [
            "--archive-root",
            str(archive_root),
            "--confirm-source-digest",
            digest,
        ]
    )
    assert archived.returncode == 0, archived.stderr
    archive_receipt = json.loads(archived.stdout)["receipt"]

    wrong = _run(
        _base("apply", root)
        + [
            "--database",
            str(database),
            "--archive-root",
            str(archive_root),
            "--archive-receipt",
            archive_receipt,
            "--confirm-source-digest",
            "0" * 64,
        ]
    )
    assert wrong.returncode == 1
    assert "does not match" in wrong.stderr
    assert not database.exists()

    command = _base("apply", root) + [
        "--database",
        str(database),
        "--archive-root",
        str(archive_root),
        "--archive-receipt",
        archive_receipt,
        "--confirm-source-digest",
        digest,
    ]
    first = _run(command)
    assert first.returncode == 0, first.stderr
    first_payload = json.loads(first.stdout)
    assert first_payload["idempotent"] is False
    assert first_payload["apply_disposition"] == "full"
    assert first_payload["full_migration_completed"] is True
    assert first_payload["pre_apply_archive_receipt"] == archive_receipt
    assert first_payload["residents_created"] == 1
    assert first_payload["anchors_created"] == 1
    second = _run(command)
    assert second.returncode == 0, second.stderr
    second_payload = json.loads(second.stdout)
    assert second_payload["idempotent"] is True
    assert second_payload["health"]["resident_count"] == 1


def test_cli_archive_is_verified_private_idempotent_and_precedes_apply(
    tmp_path,
) -> None:
    root = tmp_path / "legacy"
    _legacy(root)
    digest = json.loads(_run(_base("dry-run", root)).stdout)["source_digest"]
    archive_root = tmp_path / "archives"
    archive_command = _base("archive", root) + [
        "--archive-root",
        str(archive_root),
        "--confirm-source-digest",
        digest,
    ]

    first = _run(archive_command)
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    assert payload["idempotent"] is False
    assert payload["archive_phase"] == "pre-apply"
    assert len(payload["receipt"]) == 64
    destination = Path(payload["archive_path"])
    assert destination.is_dir()
    assert stat.S_IMODE(archive_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    archived_files = set(payload["files_archived"])
    assert {
        "residents.json",
        "resident_gallery.npz",
        "identity_manifest.json",
    } <= archived_files
    for name in archived_files | {"archive_manifest.json"}:
        assert stat.S_IMODE((destination / name).stat().st_mode) == 0o600
    assert (root / "residents.json").exists()

    second = _run(archive_command)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["idempotent"] is True


def test_cli_partial_report_requires_explicit_acceptance_and_names_every_exclusion(
    tmp_path,
) -> None:
    root = tmp_path / "legacy"
    _legacy(root)
    residents = json.loads((root / "residents.json").read_text(encoding="utf-8"))
    first_uuid = residents["residents"][0]["uuid"]
    duplicate_uuid = str(uuid.uuid4())
    unique_uuid = str(uuid.uuid4())
    residents["residents"].extend(
        [
            {
                "uuid": duplicate_uuid,
                "stable_id": 2,
                "display_name": " ALICE ",
                "embedding_count": 1,
            },
            {
                "uuid": unique_uuid,
                "stable_id": 3,
                "display_name": "Bob",
                "embedding_count": 1,
            },
        ]
    )
    (root / "residents.json").write_text(json.dumps(residents), encoding="utf-8")
    np.savez_compressed(
        root / "resident_gallery.npz",
        sids=np.asarray([1, 2, 3], dtype=np.int64),
        emb_1=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        ts_1=np.asarray([1.0], dtype=np.float64),
        emb_2=np.asarray([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32),
        ts_2=np.asarray([1.0], dtype=np.float64),
        emb_3=np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        ts_3=np.asarray([1.0], dtype=np.float64),
    )
    (root / "visitor_pool.json").write_text(
        json.dumps({"free_visitor_sids": [1000]}), encoding="utf-8"
    )
    (root / "sid_pool.json").write_text(
        json.dumps({"free_sids": [1001]}), encoding="utf-8"
    )

    dry_run = _run(_base("dry-run", root))
    assert dry_run.returncode == 3
    report = json.loads(dry_run.stdout)
    assert report["apply_disposition"] == "partial"
    assert report["full_migration_safe"] is False
    assert report["requires_accept_partial"] is True
    assert set(report["excluded_resident_uuids"]) == {first_uuid, duplicate_uuid}
    assert "duplicate_normalized_name" in report["error_codes"]
    assert "multiple_visitor_pool_sources" in report["error_codes"]
    assert report["visitor_state_imported"] is False

    review_path = tmp_path / "private" / "migration-review.json"
    review_run = _run(
        _base("dry-run", root) + ["--report-output", str(review_path)]
    )
    assert review_run.returncode == 3
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review["contract"] == "noesis.identity.legacy_migration_review"
    assert review["contract_version"] == 1
    assert review["apply_disposition"] == "partial"
    assert review["duplicate_normalized_name_groups"][0]["normalized_name"] == "alice"
    assert review["duplicate_normalized_name_groups"][0]["residents"]
    assert "anchor_vectors" not in json.dumps(review)
    assert stat.S_IMODE(review_path.stat().st_mode) == 0o600

    digest = report["source_digest"]
    archive_root = tmp_path / "archives"
    archived = _run(
        _base("archive", root)
        + [
            "--archive-root",
            str(archive_root),
            "--confirm-source-digest",
            digest,
        ]
    )
    assert archived.returncode == 0, archived.stderr
    receipt = json.loads(archived.stdout)["receipt"]
    database = tmp_path / "identity.sqlite3"
    apply_command = _base("apply", root) + [
        "--database",
        str(database),
        "--archive-root",
        str(archive_root),
        "--archive-receipt",
        receipt,
        "--confirm-source-digest",
        digest,
    ]
    refused = _run(apply_command)
    assert refused.returncode == 1
    assert "--accept-partial" in refused.stderr
    assert not database.exists()

    accepted = _run(apply_command + ["--accept-partial"])
    assert accepted.returncode == 0, accepted.stderr
    applied = json.loads(accepted.stdout)
    assert applied["apply_disposition"] == "partial"
    assert applied["partial_migration_accepted"] is True
    assert applied["full_migration_completed"] is False
    assert set(applied["blocked_resident_uuids"]) == {first_uuid, duplicate_uuid}
    assert applied["residents_created"] == 1
    assert applied["health"]["resident_count"] == 1
