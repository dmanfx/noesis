#!/usr/bin/env python3
"""Join the authored-scene oracle and Menon production trace into one gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.validation.artifacts import (  # noqa: E402
    ensure_private_artifact_tree,
    private_artifact_writer,
)
from noesis.validation.authored_scene import GeometryThresholds  # noqa: E402
from noesis.validation.track_geometry_acceptance import (  # noqa: E402
    ACCEPTANCE_CONTRACT,
    ACCEPTANCE_CONTRACT_VERSION,
    build_track_geometry_acceptance,
)
from noesis_core.private_paths import (  # noqa: E402
    PrivatePathError,
    atomic_write_private_file,
)


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _failure_payload(run_id: str, error: BaseException) -> dict[str, Any]:
    detail = str(error)
    return {
        "schema_version": 1,
        "contract": ACCEPTANCE_CONTRACT,
        "contract_version": ACCEPTANCE_CONTRACT_VERSION,
        "run_id": run_id,
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
                "id": "ACCEPTANCE.input",
                "domain": "infrastructure",
                "name": "track_geometry_acceptance_input",
                "status": "fail",
                "level": "invalid",
                "failure_type": "infrastructure_failure",
                "severity": "error",
                "detail": detail,
                "suggested_next_diagnostic": (
                    "Verify the journal, bound room map, authored OBJ, similarity, "
                    "and captured Menon production-geometry trace."
                ),
            }
        ],
        "gates": {
            name: {
                "status": "blocked",
                "check_count": 0,
                "missing_check_ids": ["ACCEPTANCE.input"],
                "checks": [],
            }
            for name in ("producer_coverage", "physical_placement", "menon_renderer")
        },
        "artifacts": {},
    }


@private_artifact_writer
def run_track_geometry_acceptance(
    *,
    journal_path: str | Path,
    obj_path: str | Path,
    similarity_path: str | Path,
    room_group_map_path: str | Path,
    menon_trace_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    cameras: Sequence[str] = (),
    journal_limit: int = 0,
    max_divergences: int = 100,
    thresholds: GeometryThresholds = GeometryThresholds(),
    horizontal_tolerance_deg: float = 15.0,
) -> tuple[Path, dict[str, Any]]:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run id must contain only letters, digits, periods, underscores, or hyphens"
        )
    run_dir_path = Path(output_dir).expanduser() / run_id
    existed = run_dir_path.exists()
    run_dir = ensure_private_artifact_tree(
        run_dir_path,
        label="track geometry acceptance artifacts",
    )
    if existed and any(run_dir.iterdir()):
        raise PrivatePathError(
            "track geometry acceptance run directory already contains artifacts"
        )

    try:
        acceptance, authored_report, menon_report = build_track_geometry_acceptance(
            journal_path=journal_path,
            obj_path=obj_path,
            similarity_path=similarity_path,
            room_group_map_path=room_group_map_path,
            menon_trace_path=menon_trace_path,
            run_id=run_id,
            cameras=cameras,
            journal_limit=max(0, int(journal_limit)),
            max_divergences=max(0, int(max_divergences)),
            thresholds=thresholds,
            horizontal_tolerance_deg=float(horizontal_tolerance_deg),
        )
    except Exception as exc:
        acceptance = _failure_payload(run_id, exc)
        authored_report = None
        menon_report = None

    if authored_report is not None and menon_report is not None:
        authored_bytes = _json_bytes(authored_report)
        menon_bytes = _json_bytes(menon_report)
        trace_bytes = Path(menon_trace_path).expanduser().resolve().read_bytes()
        menon_dir = ensure_private_artifact_tree(
            run_dir / "menon",
            label="track geometry Menon evidence",
        )
        atomic_write_private_file(
            run_dir / "authored_scene_report.json",
            authored_bytes,
            label="authored scene component report",
        )
        atomic_write_private_file(
            run_dir / "menon_trace_report.json",
            menon_bytes,
            label="Menon trace component report",
        )
        atomic_write_private_file(
            menon_dir / "trace.json",
            trace_bytes,
            label="captured Menon trace evidence",
        )
        acceptance["components"]["authored_scene"]["sha256"] = _sha256_bytes(
            authored_bytes
        )
        acceptance["components"]["menon_trace"]["sha256"] = _sha256_bytes(
            menon_bytes
        )
        acceptance["artifacts"]["menon_trace_sha256"] = _sha256_bytes(trace_bytes)

    result_path = atomic_write_private_file(
        run_dir / "track_geometry_acceptance.json",
        _json_bytes(acceptance),
        label="track geometry acceptance result",
    )
    ensure_private_artifact_tree(
        run_dir,
        label="track geometry acceptance artifacts",
    )
    return result_path, acceptance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join a canonical world journal and captured Menon production-geometry "
            "trace into one fail-closed offline acceptance result."
        )
    )
    parser.add_argument("--journal", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--similarity", required=True)
    parser.add_argument("--room-group-map", required=True)
    parser.add_argument("--menon-trace", required=True)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--journal-limit", type=int, default=0)
    parser.add_argument("--max-divergences", type=int, default=100)
    parser.add_argument("--support-good-m", type=float, default=0.12)
    parser.add_argument("--support-fail-m", type=float, default=0.25)
    parser.add_argument("--outside-warning-m", type=float, default=0.15)
    parser.add_argument("--horizontal-tolerance-deg", type=float, default=15.0)
    parser.add_argument("--output-dir", default="diagnostics/validation")
    parser.add_argument("--run-id", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_id = str(args.run_id or f"track_geometry_{int(time.time())}")
    try:
        result_path, payload = run_track_geometry_acceptance(
            journal_path=args.journal,
            obj_path=args.obj,
            similarity_path=args.similarity,
            room_group_map_path=args.room_group_map,
            menon_trace_path=args.menon_trace,
            output_dir=args.output_dir,
            run_id=run_id,
            cameras=args.camera,
            journal_limit=args.journal_limit,
            max_divergences=args.max_divergences,
            thresholds=GeometryThresholds(
                support_good_m=args.support_good_m,
                support_fail_m=args.support_fail_m,
                outside_warning_m=args.outside_warning_m,
            ),
            horizontal_tolerance_deg=args.horizontal_tolerance_deg,
        )
    except Exception as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2))
        return 1
    status = str(payload.get("summary", {}).get("status") or "fail")
    print(
        json.dumps(
            {
                "status": status,
                "result": str(result_path),
                "gates": {
                    name: gate.get("status")
                    for name, gate in payload.get("gates", {}).items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if status in {"fail", "blocked"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
