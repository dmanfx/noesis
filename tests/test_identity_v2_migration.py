from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from reid.identity_v2 import (
    IdentityStore,
    LegacyMigrationBlocked,
    MigrationConflict,
    apply_legacy_identity,
    archive_legacy_identity,
    inspect_legacy_identity,
)

EXPECTED_FINGERPRINT = "swin-tiny:sha256:expected"
EXPECTED_DIMENSION = 4


def _archive_arguments(report, tmp_path) -> dict:
    archive_root = tmp_path / "pre_apply_archives"
    archived = archive_legacy_identity(report, archive_root)
    return {
        "archive_root": archive_root,
        "archive_receipt": archived.receipt,
    }


def _row(
    name: str,
    sid: int,
    *,
    resident_uuid: str | None = None,
    count: int = 0,
) -> dict:
    return {
        "uuid": resident_uuid or str(uuid.uuid4()),
        "stable_id": sid,
        "display_name": name,
        "created_ts": 10.0,
        "embedding_count": count,
    }


def _write_legacy(
    root: Path,
    residents: list[dict],
    *,
    galleries: dict[int, list[list[float]]] | None = None,
    fingerprint: str | None = EXPECTED_FINGERPRINT,
    manifest_dimension: int | None = EXPECTED_DIMENSION,
    visitor_pool: dict | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "residents.json").write_text(
        json.dumps({"version": 1, "residents": residents}, sort_keys=True),
        encoding="utf-8",
    )
    if galleries is not None:
        arrays: dict[str, np.ndarray] = {
            "sids": np.asarray(sorted(galleries), dtype=np.int64),
        }
        for sid, vectors in galleries.items():
            if vectors:
                arrays[f"emb_{sid}"] = np.asarray(vectors, dtype=np.float32)
            else:
                arrays[f"emb_{sid}"] = np.empty(
                    (0, EXPECTED_DIMENSION), dtype=np.float32
                )
            arrays[f"ts_{sid}"] = np.arange(len(vectors), dtype=np.float64)
        np.savez_compressed(root / "resident_gallery.npz", **arrays)
    if fingerprint is not None or manifest_dimension is not None:
        payload = {}
        if fingerprint is not None:
            payload["model_fingerprint"] = fingerprint
        if manifest_dimension is not None:
            payload["embedding_dim"] = manifest_dimension
        (root / "identity_manifest.json").write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    if visitor_pool is not None:
        (root / "visitor_pool.json").write_text(
            json.dumps(visitor_pool, sort_keys=True),
            encoding="utf-8",
        )


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.name: (path.stat().st_mode, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def _issue_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_legacy_migration_rejects_duplicate_resident_inventory(tmp_path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "residents.json").write_text(
        '{"version":1,"residents":[],"residents":['
        '{"uuid":"hidden","stable_id":1,"display_name":"Hidden"}]}',
        encoding="utf-8",
    )

    report = inspect_legacy_identity(
        root,
        expected_model_fingerprint=EXPECTED_FINGERPRINT,
        expected_embedding_dim=EXPECTED_DIMENSION,
    )

    assert "residents_invalid" in _issue_codes(report)
    assert report.residents == ()
    assert all("Hidden" not in issue.detail for issue in report.issues)


def test_dry_run_is_read_only_and_reports_every_unsafe_legacy_condition(
    tmp_path,
) -> None:
    root = tmp_path / "legacy"
    alice_a = _row("Alice Smith", 1, count=99)
    alice_b = _row("  ALICE   smith ", 2, count=0)
    bob = _row("Bob", 3, count=1)
    cara = _row("Cara", 4, count=0)
    _write_legacy(
        root,
        [alice_a, alice_b, bob, cara],
        galleries={
            1: [[1.0, 0.0, 0.0, 0.0]],
            2: [],
            3: [[0.0, 1.0, 0.0]],
            1000: [[0.0, 0.0, 1.0, 0.0]],
        },
        fingerprint="different-model",
        visitor_pool={
            "free_visitor_sids": [1000, 1000, 1001],
            "visitor_last_seen": {"1001": 123.0},
            "visitor_generations": {"1001": 2},
        },
    )
    before = _snapshot(root)

    report = inspect_legacy_identity(
        root,
        expected_model_fingerprint=EXPECTED_FINGERPRINT,
        expected_embedding_dim=EXPECTED_DIMENSION,
    )

    assert _snapshot(root) == before
    assert not (root / "identity.sqlite3").exists()
    codes = _issue_codes(report)
    assert {
        "duplicate_normalized_name",
        "empty_resident_gallery",
        "embedding_dimension_mismatch",
        "model_fingerprint_mismatch",
        "stale_embedding_count",
        "visitor_slot_collision",
    }.issubset(codes)
    assert report.visitor_collisions == (1000, 1001)
    assert report.blocked_resident_count == 2
    assert report.migratable_resident_count == 2
    assert all(
        "duplicate_normalized_name" in resident.blocked_reasons
        for resident in report.residents
        if resident.compatibility_sid in (1, 2)
    )
    assert report.to_dict()["blocked_resident_count"] == 2


def test_clean_apply_is_atomic_restart_safe_and_idempotent(tmp_path) -> None:
    root = tmp_path / "legacy"
    alice = _row("Alice", 1, count=2)
    bob = _row("Bob", 2, count=1)
    _write_legacy(
        root,
        [alice, bob],
        galleries={
            1: [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
            2: [[0.0, 1.0, 0.0, 0.0]],
        },
    )
    report = inspect_legacy_identity(
        root,
        expected_model_fingerprint=EXPECTED_FINGERPRINT,
        expected_embedding_dim=EXPECTED_DIMENSION,
    )
    assert not report.fatal
    assert report.blocked_resident_count == 0
    assert all(row.gallery_compatible for row in report.residents)
    archive = _archive_arguments(report, tmp_path)

    db_path = tmp_path / "state" / "identity.sqlite3"
    with IdentityStore(db_path) as store:
        applied = apply_legacy_identity(report, store, now=100.0, **archive)
        assert not applied.idempotent
        assert applied.residents_created == 2
        assert applied.residents_existing == 0
        assert applied.anchors_created == 3
        assert applied.blocked_resident_uuids == ()
        assert applied.anchorless_resident_uuids == ()
        health = store.health(now=100.0)
        assert health.resident_count == 2
        assert health.enrollment_anchor_count == 3
        alice_gallery = store.load_resident_gallery(
            alice["uuid"],
            model_fingerprint=EXPECTED_FINGERPRINT,
            embedding_dim=EXPECTED_DIMENSION,
        )
        vectors = sorted((row.vector for row in alice_gallery), reverse=True)
        assert vectors[0] == pytest.approx((1.0, 0.0, 0.0, 0.0))
        assert vectors[1] == pytest.approx((0.9, 0.1, 0.0, 0.0))

    with IdentityStore(db_path) as reopened:
        second_report = inspect_legacy_identity(
            root,
            expected_model_fingerprint=EXPECTED_FINGERPRINT,
            expected_embedding_dim=EXPECTED_DIMENSION,
        )
        assert second_report.source_digest == report.source_digest
        second = apply_legacy_identity(second_report, reopened, now=200.0, **archive)
        assert second.idempotent
        assert second.residents_created == 2
        assert second.anchors_created == 3
        assert reopened.health().enrollment_anchor_count == 3


def test_duplicate_records_remain_blocked_and_are_never_auto_merged(tmp_path) -> None:
    root = tmp_path / "legacy"
    first = _row("Family Member", 1, count=1)
    duplicate = _row(" family   member ", 2, count=1)
    unique = _row("Unique Resident", 3, count=1)
    _write_legacy(
        root,
        [first, duplicate, unique],
        galleries={
            1: [[1.0, 0.0, 0.0, 0.0]],
            2: [[0.9, 0.1, 0.0, 0.0]],
            3: [[0.0, 1.0, 0.0, 0.0]],
        },
    )
    report = inspect_legacy_identity(
        root,
        expected_model_fingerprint=EXPECTED_FINGERPRINT,
        expected_embedding_dim=EXPECTED_DIMENSION,
    )
    assert report.blocked_resident_count == 2
    archive = _archive_arguments(report, tmp_path)

    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        with pytest.raises(LegacyMigrationBlocked, match="accept_partial=True"):
            apply_legacy_identity(report, store, **archive)
        applied = apply_legacy_identity(report, store, accept_partial=True, **archive)
        assert applied.residents_created == 1
        assert applied.anchors_created == 1
        assert set(applied.blocked_resident_uuids) == {first["uuid"], duplicate["uuid"]}
        residents = store.list_residents()
        assert [(row.display_name, row.compatibility_sid) for row in residents] == [
            ("Unique Resident", 3)
        ]
        assert store.health().resident_count == 1


@pytest.mark.parametrize(
    ("fingerprint", "vectors", "expected_issue"),
    [
        ("wrong-fingerprint", [[1.0, 0.0, 0.0, 0.0]], "model_fingerprint_mismatch"),
        (EXPECTED_FINGERPRINT, [[1.0, 0.0, 0.0]], "embedding_dimension_mismatch"),
    ],
)
def test_unverified_biometrics_are_not_imported_as_anchors(
    tmp_path,
    fingerprint,
    vectors,
    expected_issue,
) -> None:
    root = tmp_path / "legacy"
    resident = _row("Alice", 1, count=1)
    _write_legacy(root, [resident], galleries={1: vectors}, fingerprint=fingerprint)
    report = inspect_legacy_identity(
        root,
        expected_model_fingerprint=EXPECTED_FINGERPRINT,
        expected_embedding_dim=EXPECTED_DIMENSION,
    )
    assert expected_issue in _issue_codes(report)
    assert not report.residents[0].blocked
    assert not report.residents[0].gallery_compatible
    assert report.residents[0].anchor_vectors == ()
    archive = _archive_arguments(report, tmp_path)

    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        applied = apply_legacy_identity(report, store, accept_partial=True, **archive)
        assert applied.residents_created == 1
        assert applied.anchors_created == 0
        assert applied.anchorless_resident_uuids == (resident["uuid"],)
        health = store.health()
        assert health.resident_count == 1
        assert health.residents_without_anchors == 1


def test_missing_fingerprint_requires_explicit_operator_assertion(tmp_path) -> None:
    root = tmp_path / "legacy"
    resident = _row("Alice", 1, count=1)
    _write_legacy(
        root,
        [resident],
        galleries={1: [[1.0, 0.0, 0.0, 0.0]]},
        fingerprint=None,
        manifest_dimension=None,
    )
    unknown = inspect_legacy_identity(
        root,
        expected_model_fingerprint=EXPECTED_FINGERPRINT,
        expected_embedding_dim=EXPECTED_DIMENSION,
    )
    assert "model_fingerprint_unknown" in _issue_codes(unknown)
    assert not unknown.residents[0].gallery_compatible

    asserted = inspect_legacy_identity(
        root,
        expected_model_fingerprint=EXPECTED_FINGERPRINT,
        expected_embedding_dim=EXPECTED_DIMENSION,
        source_model_fingerprint=EXPECTED_FINGERPRINT,
        source_embedding_dim=EXPECTED_DIMENSION,
    )
    assert "model_fingerprint_unknown" not in _issue_codes(asserted)
    assert asserted.residents[0].gallery_compatible


def test_changed_source_cannot_reuse_recorded_migration_key(tmp_path) -> None:
    root = tmp_path / "legacy"
    resident = _row("Alice", 1, count=1)
    _write_legacy(
        root,
        [resident],
        galleries={1: [[1.0, 0.0, 0.0, 0.0]]},
    )
    first = inspect_legacy_identity(
        root,
        expected_model_fingerprint=EXPECTED_FINGERPRINT,
        expected_embedding_dim=EXPECTED_DIMENSION,
    )
    first_archive = _archive_arguments(first, tmp_path)
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        apply_legacy_identity(first, store, **first_archive)
        residents_payload = json.loads(
            (root / "residents.json").read_text(encoding="utf-8")
        )
        residents_payload["residents"][0]["embedding_count"] = 999
        (root / "residents.json").write_text(
            json.dumps(residents_payload, sort_keys=True),
            encoding="utf-8",
        )
        changed = inspect_legacy_identity(
            root,
            expected_model_fingerprint=EXPECTED_FINGERPRINT,
            expected_embedding_dim=EXPECTED_DIMENSION,
        )
        assert changed.migration_key == first.migration_key
        assert changed.source_digest != first.source_digest
        changed_archive = _archive_arguments(changed, tmp_path)
        with pytest.raises(MigrationConflict, match="source changed"):
            apply_legacy_identity(changed, store, **changed_archive)
        assert store.health().resident_count == 1


def test_fatal_inspection_report_cannot_be_applied(tmp_path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    before = _snapshot(root)
    report = inspect_legacy_identity(
        root,
        expected_model_fingerprint=EXPECTED_FINGERPRINT,
        expected_embedding_dim=EXPECTED_DIMENSION,
    )
    assert report.fatal
    assert _snapshot(root) == before
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        with pytest.raises(LegacyMigrationBlocked, match="residents_missing"):
            apply_legacy_identity(
                report,
                store,
                archive_root=tmp_path / "missing-archive",
                archive_receipt="0" * 64,
            )
        assert store.health().resident_count == 0
