#!/usr/bin/env python3
"""Score persisted canonical track positions against a promoted authored OBJ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.validation.authored_scene import (  # noqa: E402
    GeometryThresholds,
    build_journal_oracle_report,
)
from noesis_core.private_paths import atomic_write_private_file  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_room_group_map(
    path: str | Path | None,
    *,
    authored_scene_path: str | Path | None = None,
) -> Mapping[str, Sequence[str]] | None:
    if path is None or not str(path).strip():
        return None
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("room group map must be a JSON object")
    declared_scene_sha256 = str(payload.get("authored_scene_sha256") or "").strip().lower()
    if declared_scene_sha256:
        if authored_scene_path is None:
            raise ValueError(
                "room group map declares authored_scene_sha256 but no authored scene was supplied"
            )
        scene_path = Path(authored_scene_path).expanduser().resolve()
        actual_scene_sha256 = _sha256_file(scene_path)
        if declared_scene_sha256 != actual_scene_sha256:
            raise ValueError(
                "room group map authored_scene_sha256 does not match the supplied OBJ "
                f"(declared {declared_scene_sha256}, actual {actual_scene_sha256})"
            )
    candidate: Any = payload.get("rooms") if isinstance(payload.get("rooms"), Mapping) else payload
    if not isinstance(candidate, Mapping):
        raise ValueError("room group map must contain a rooms object")
    normalized: dict[str, Sequence[str]] = {}
    for label, groups in candidate.items():
        if not isinstance(groups, Sequence) or isinstance(
            groups, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"room group map entry {label!r} must be an array of OBJ group names"
            )
        normalized[str(label)] = groups
    return normalized


def _failure_payload(error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "summary": {
            "status": "fail",
            "level": "invalid",
            "failure_count": 1,
            "warning_count": 0,
            "blocked_count": 0,
            "pass_count": 0,
            "skipped_count": 0,
            "check_count": 1,
        },
        "checks": [
            {
                "id": "ORACLE.input",
                "domain": "infrastructure",
                "name": "authored_scene_oracle_input",
                "status": "fail",
                "level": "invalid",
                "failure_type": "infrastructure_failure",
                "severity": "error",
                "detail": str(error),
                "suggested_next_diagnostic": (
                    "Verify the read-only journal, OBJ, similarity JSON, and optional room map paths."
                ),
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the declared backend_world_m-to-menon_scene similarity exactly once, "
            "then score journaled person positions against authored ground/room floor surfaces."
        )
    )
    parser.add_argument("--journal", required=True, help="Persisted canonical world SQLite journal.")
    parser.add_argument("--obj", required=True, help="Promoted authored OBJ in menon_scene coordinates.")
    parser.add_argument(
        "--similarity",
        required=True,
        help=(
            "JSON containing world_to_scene_col_major, align.scene_similarity, "
            "or pose_correction.world_to_menon_scene_col_major."
        ),
    )
    parser.add_argument(
        "--room-group-map",
        default="",
        help=(
            "Optional reviewed JSON map from household room labels to authored room_* OBJ groups. "
            "Without it, expected-room containment is explicitly blocked."
        ),
    )
    parser.add_argument("--camera", action="append", default=[], help="Camera ID to include; repeatable.")
    parser.add_argument(
        "--journal-limit",
        type=int,
        default=0,
        help="Score only the newest N person observations; zero scores the full journal.",
    )
    parser.add_argument("--support-good-m", type=float, default=0.12)
    parser.add_argument("--support-fail-m", type=float, default=0.25)
    parser.add_argument("--outside-warning-m", type=float, default=0.15)
    parser.add_argument("--horizontal-tolerance-deg", type=float, default=15.0)
    parser.add_argument("--max-divergences", type=int, default=100)
    parser.add_argument("--run-id", default="", help="Stable report run ID.")
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional JSON report path. Its parent must be an owner-only 0700 directory; "
            "the report is atomically written as 0600."
        ),
    )
    args = parser.parse_args()

    run_id = str(args.run_id or f"authored_scene_{int(time.time())}")
    try:
        report = build_journal_oracle_report(
            journal_path=args.journal,
            obj_path=args.obj,
            similarity_path=args.similarity,
            run_id=run_id,
            cameras=args.camera,
            journal_limit=max(0, int(args.journal_limit)),
            max_divergences=max(0, int(args.max_divergences)),
            thresholds=GeometryThresholds(
                support_good_m=float(args.support_good_m),
                support_fail_m=float(args.support_fail_m),
                outside_warning_m=float(args.outside_warning_m),
            ),
            horizontal_tolerance_deg=float(args.horizontal_tolerance_deg),
            room_group_map=_load_room_group_map(
                args.room_group_map,
                authored_scene_path=args.obj,
            ),
        )
    except Exception as exc:
        report = _failure_payload(exc)

    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        previous_umask = os.umask(0o077)
        try:
            atomic_write_private_file(
                args.output,
                encoded.encode("utf-8"),
                label="authored-scene geometry oracle report",
            )
        finally:
            os.umask(previous_umask)
    else:
        sys.stdout.write(encoded)

    status = str(report.get("summary", {}).get("status") or "fail")
    return 1 if status in ("fail", "blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
