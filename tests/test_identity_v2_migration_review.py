from __future__ import annotations

import json
from pathlib import Path

import pytest

from reid.identity_v2 import MigrationReviewError, load_migration_review


def _report() -> dict:
    return {
        "contract": "noesis.identity.legacy_migration_review",
        "contract_version": 1,
        "source_root": "/private/legacy/path",
        "source_digest": "a" * 64,
        "migration_key": "identity-v2:test",
        "apply_disposition": "partial",
        "full_migration_safe": False,
        "requires_accept_partial": True,
        "blocked_resident_count": 2,
        "migratable_resident_count": 2,
        "residents_without_importable_anchors": ["resident-3"],
        "duplicate_normalized_name_groups": [
            {
                "normalized_name": "alice",
                "residents": [
                    {
                        "resident_uuid": "resident-1",
                        "display_name": "Alice",
                        "compatibility_sid": 1,
                        "anchor_count": 0,
                    },
                    {
                        "resident_uuid": "resident-2",
                        "display_name": "ALICE",
                        "compatibility_sid": 2,
                        "anchor_count": 0,
                    },
                ],
            }
        ],
        "residents": [
            {
                "resident_uuid": "resident-1",
                "display_name": "Alice",
                "normalized_name": "alice",
                "compatibility_sid": 1,
                "anchor_count": 0,
                "recorded_embedding_count": 0,
                "actual_embedding_count": 0,
                "blocked_reasons": ["duplicate_normalized_name"],
                "gallery_compatible": False,
            }
        ],
        "issues": [
            {
                "code": "duplicate_normalized_name",
                "severity": "error",
                "subject": "alice",
                "detail": "ambiguous records are not auto-merged",
            }
        ],
        "operator_action": "Resolve every duplicate explicitly; no apply is available.",
    }


def test_migration_review_is_biometric_free_path_free_and_never_applies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "review.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    path.chmod(0o600)
    review = load_migration_review(path)
    assert review["apply_disposition"] == "partial"
    assert review["apply_available_from_api"] is False
    assert review["duplicate_normalized_name_groups"][0]["normalized_name"] == "alice"
    assert review["residents_without_importable_anchors"] == ["resident-3"]
    assert "source_root" not in review
    assert "/private/legacy/path" not in json.dumps(review)


@pytest.mark.parametrize(
    "field", ["embedding", "anchor_vectors", "vectors", "feature_vector"]
)
def test_migration_review_rejects_raw_biometric_fields(
    tmp_path: Path, field: str
) -> None:
    payload = _report()
    payload["residents"][0][field] = [[1.0, 0.0]]
    path = tmp_path / f"review-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(MigrationReviewError, match="forbidden biometric field"):
        load_migration_review(path)


def test_migration_review_rejects_duplicate_key_hidden_biometrics(
    tmp_path: Path,
) -> None:
    canonical = json.dumps(_report(), sort_keys=True, separators=(",", ":"))
    marker = '"residents":'
    offset = canonical.index(marker)
    ambiguous = (
        canonical[:offset]
        + '"residents":[{"embedding":[0.1,0.2]}],'
        + canonical[offset:]
    )
    path = tmp_path / "duplicate-review.json"
    path.write_text(ambiguous, encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(MigrationReviewError, match="duplicate JSON object key"):
        load_migration_review(path)


def test_migration_review_recursively_allowlists_and_redacts_private_paths(
    tmp_path: Path,
) -> None:
    payload = _report()
    payload["residents"][0]["private_path"] = "/home/private/gallery.npz"
    payload["duplicate_normalized_name_groups"][0]["residents"][0][
        "source_root"
    ] = "/srv/private"
    payload["issues"][0]["detail"] = "failed at /home/private/residents.json"
    payload["issues"][0]["archive_path"] = "/mnt/private/archive"
    path = tmp_path / "nested-private-review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)

    review = load_migration_review(path)
    serialized = json.dumps(review, sort_keys=True)

    assert "/home/private" not in serialized
    assert "/srv/private" not in serialized
    assert "/mnt/private" not in serialized
    assert "private_path" not in serialized
    assert "source_root" not in serialized
    assert "archive_path" not in serialized
    assert review["issues"][0]["detail"] == "[private path redacted]"
