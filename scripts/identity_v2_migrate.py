#!/usr/bin/env python3
"""Inspect, apply, and archive a legacy household identity snapshot safely."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reid.identity_v2 import (  # noqa: E402
    IdentityStore,
    apply_legacy_identity,
    archive_legacy_identity,
    inspect_legacy_identity,
    verify_legacy_archive,
)
from noesis_core.private_paths import atomic_write_private_file  # noqa: E402


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-root", required=True, help="Legacy household state directory"
    )
    parser.add_argument(
        "--model-fingerprint",
        required=True,
        help="Exact fingerprint expected by the identity-v2 runtime",
    )
    parser.add_argument(
        "--embedding-dim",
        required=True,
        type=int,
        help="Embedding dimension expected by the identity-v2 runtime",
    )
    parser.add_argument(
        "--assert-source-model-fingerprint",
        required=True,
        help="Operator assertion of the model that produced the legacy gallery",
    )
    parser.add_argument(
        "--assert-source-embedding-dim",
        required=True,
        type=int,
        help="Operator assertion of the legacy gallery embedding dimension",
    )


def _inspect(args: argparse.Namespace):
    return inspect_legacy_identity(
        args.source_root,
        expected_model_fingerprint=args.model_fingerprint,
        expected_embedding_dim=args.embedding_dim,
        source_model_fingerprint=args.assert_source_model_fingerprint,
        source_embedding_dim=args.assert_source_embedding_dim,
    )


def _require_confirmed_digest(args: argparse.Namespace, actual: str) -> None:
    confirmed = str(args.confirm_source_digest or "").strip().lower()
    if confirmed != actual.lower():
        raise RuntimeError(
            "--confirm-source-digest does not match the freshly inspected legacy snapshot"
        )


def _emit(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _write_private_json(path: str, payload) -> str:
    destination = Path(path).expanduser()
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return str(
        atomic_write_private_file(
            destination,
            body,
            label="identity migration review",
        )
    )


def _duplicate_name_groups(report) -> list[dict]:
    groups = {}
    for resident in report.residents:
        if "duplicate_normalized_name" not in resident.blocked_reasons:
            continue
        groups.setdefault(resident.normalized_name, []).append(
            {
                "resident_uuid": resident.resident_uuid,
                "display_name": resident.display_name,
                "compatibility_sid": resident.compatibility_sid,
                "anchor_count": len(resident.anchor_vectors),
            }
        )
    return [
        {"normalized_name": name, "residents": sorted(rows, key=lambda row: row["resident_uuid"])}
        for name, rows in sorted(groups.items())
    ]


def _dry_run(args: argparse.Namespace) -> int:
    report = _inspect(args)
    payload = report.to_dict()
    payload["contract"] = "noesis.identity.legacy_migration_review"
    payload["contract_version"] = 1
    payload["duplicate_normalized_name_groups"] = _duplicate_name_groups(report)
    payload["command"] = "dry-run"
    payload["operator_action"] = {
        "blocked": "Resolve fatal source errors before migration.",
        "partial": "Review every exclusion, create the pre-apply archive, then pass --accept-partial explicitly.",
        "full": "Create the pre-apply archive before applying this exact digest.",
    }[report.apply_disposition]
    if args.archive_root:
        archived = archive_legacy_identity(report, args.archive_root)
        payload["pre_apply_archive"] = {
            "archive_path": archived.archive_path,
            "receipt": archived.receipt,
            "source_digest": archived.source_digest,
            "idempotent": archived.idempotent,
            "files_archived": list(archived.files_archived),
        }
    if args.report_output:
        payload["report_output"] = _write_private_json(args.report_output, payload)
    _emit(payload)
    if report.apply_disposition == "blocked":
        return 2
    if report.apply_disposition == "partial":
        return 3
    return 0


def _apply(args: argparse.Namespace) -> int:
    report = _inspect(args)
    _require_confirmed_digest(args, report.source_digest)
    if report.apply_disposition == "blocked":
        raise RuntimeError("migration report is blocked by fatal source errors")
    if report.apply_disposition == "partial" and not args.accept_partial:
        raise RuntimeError(
            "migration is partial; review exclusions and pass --accept-partial explicitly"
        )
    archive = verify_legacy_archive(
        report,
        args.archive_root,
        expected_receipt=args.archive_receipt,
    )
    with IdentityStore(args.database) as store:
        result = apply_legacy_identity(
            report,
            store,
            archive_root=args.archive_root,
            archive_receipt=args.archive_receipt,
            accept_partial=args.accept_partial,
        )
        health = store.health()
    _emit(
        {
            "command": "apply",
            "database": str(Path(args.database).expanduser().resolve()),
            "source_digest": report.source_digest,
            "migration_key": report.migration_key,
            "apply_disposition": result.disposition,
            "full_migration_completed": result.disposition == "full",
            "partial_migration_accepted": result.disposition == "partial",
            "pre_apply_archive_receipt": archive.receipt,
            "pre_apply_archive_path": archive.archive_path,
            "idempotent": result.idempotent,
            "residents_created": result.residents_created,
            "residents_existing": result.residents_existing,
            "anchors_created": result.anchors_created,
            "blocked_resident_uuids": list(result.blocked_resident_uuids),
            "anchorless_resident_uuids": list(result.anchorless_resident_uuids),
            "source_error_codes": list(result.source_error_codes),
            "visitor_state_imported": result.visitor_state_imported,
            "health": {
                "schema_version": health.schema_version,
                "resident_count": health.resident_count,
                "enrollment_anchor_count": health.enrollment_anchor_count,
                "residents_without_anchors": health.residents_without_anchors,
            },
            "issues": report.to_dict()["issues"],
        }
    )
    return 0


def _archive(args: argparse.Namespace) -> int:
    report = _inspect(args)
    _require_confirmed_digest(args, report.source_digest)
    result = archive_legacy_identity(
        report,
        args.archive_root,
        remove_source=False,
    )
    _emit(
        {
            "command": "archive",
            "archive_path": result.archive_path,
            "source_digest": result.source_digest,
            "files_archived": list(result.files_archived),
            "sources_removed": list(result.sources_removed),
            "idempotent": result.idempotent,
            "receipt": result.receipt,
            "archive_phase": "pre-apply",
        }
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser(
        "dry-run", help="Inspect only; never creates a database"
    )
    _add_source_arguments(dry_run)
    dry_run.add_argument(
        "--archive-root",
        default=None,
        help="Optionally produce the required verified pre-apply archive",
    )
    dry_run.add_argument(
        "--report-output",
        default=None,
        help="Write an owner-only biometric-free review for the authenticated UI",
    )
    dry_run.set_defaults(func=_dry_run)

    apply = subparsers.add_parser(
        "apply", help="Atomically apply one confirmed snapshot"
    )
    _add_source_arguments(apply)
    apply.add_argument(
        "--database", required=True, help="Identity-v2 SQLite database path"
    )
    apply.add_argument(
        "--archive-root",
        required=True,
        help="Parent containing the verified pre-apply archive",
    )
    apply.add_argument(
        "--archive-receipt",
        required=True,
        help="SHA-256 receipt emitted by the pre-apply archive",
    )
    apply.add_argument(
        "--accept-partial",
        action="store_true",
        help="Explicitly accept every exclusion listed by a partial report",
    )
    apply.add_argument(
        "--confirm-source-digest",
        required=True,
        help="Exact digest printed by a fresh dry-run",
    )
    apply.set_defaults(func=_apply)

    archive = subparsers.add_parser(
        "archive",
        help="Copy and verify a snapshot before any migration is applied",
    )
    _add_source_arguments(archive)
    archive.add_argument(
        "--archive-root", required=True, help="Private archive parent directory"
    )
    archive.add_argument(
        "--confirm-source-digest",
        required=True,
        help="Exact digest printed by a fresh dry-run",
    )
    archive.set_defaults(func=_archive)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"identity-v2 migration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
