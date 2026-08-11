#!/usr/bin/env python3
"""Validate score-only evidence and emit a held-out identity-v2 calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reid.identity_v2.calibration import (  # noqa: E402
    IdentityCalibrationError,
    build_calibration_dataset,
    calibrate_dataset_files,
    dataset_review,
    load_and_validate_evidence,
    load_calibration_dataset,
    sha256_file,
    write_private_json,
)


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate_evidence(args: argparse.Namespace) -> int:
    _, summary = load_and_validate_evidence(args.evidence)
    payload = dict(summary.__dict__)
    payload["evidence_chain"] = summary.evidence_chain.model_dump(mode="json")
    _emit({"valid": True, **payload})
    return 0


def _build_dataset(args: argparse.Namespace) -> int:
    dataset = build_calibration_dataset(args.evidence, args.labels)
    destination = write_private_json(args.output, dataset.model_dump(mode="json"))
    _emit(
        {
            "dataset": str(destination),
            "dataset_sha256": sha256_file(destination),
            "review": dataset_review(dataset),
        }
    )
    return 0


def _validate_dataset(args: argparse.Namespace) -> int:
    dataset = load_calibration_dataset(args.dataset)
    _emit(
        {
            "valid": True,
            "dataset_sha256": sha256_file(args.dataset),
            "review": dataset_review(dataset),
        }
    )
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    artifact = calibrate_dataset_files(
        args.benchmark_dataset,
        args.household_dataset,
        generated_at_us=args.generated_at_us,
        generator_revision=args.generator_revision,
        max_benchmark_holdout_far_upper_confidence_bound=(
            args.max_benchmark_far_upper_bound
        ),
        max_benchmark_holdout_frr=args.max_benchmark_frr,
        max_benchmark_holdout_misidentification_upper_confidence_bound=(
            args.max_benchmark_misidentification_upper_bound
        ),
    )
    destination = write_private_json(args.output, artifact.model_dump(mode="json"))
    _emit(
        {
            "artifact": str(destination),
            "artifact_sha256": sha256_file(destination),
            "model_sha256": artifact.model_sha256,
            "benchmark_dataset": artifact.benchmark_dataset.model_dump(mode="json"),
            "household_dataset": artifact.household_dataset.model_dump(mode="json"),
            "benchmark_policy": artifact.benchmark_policy.model_dump(mode="json"),
            "policy": artifact.policy.model_dump(mode="json"),
            "benchmark_training_metrics": artifact.benchmark_training_metrics.model_dump(
                mode="json"
            ),
            "benchmark_holdout_metrics": artifact.benchmark_holdout_metrics.model_dump(
                mode="json"
            ),
            "household_training_metrics": artifact.household_training_metrics.model_dump(
                mode="json"
            ),
            "household_holdout_metrics": artifact.household_holdout_metrics.model_dump(
                mode="json"
            ),
            "acceptance": artifact.acceptance.model_dump(mode="json"),
        }
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    evidence = commands.add_parser("validate-evidence")
    evidence.add_argument("--evidence", required=True)
    evidence.set_defaults(func=_validate_evidence)

    dataset = commands.add_parser("build-dataset")
    dataset.add_argument("--evidence", required=True)
    dataset.add_argument("--labels", required=True)
    dataset.add_argument("--output", required=True)
    dataset.set_defaults(func=_build_dataset)

    validate_dataset = commands.add_parser("validate-dataset")
    validate_dataset.add_argument("--dataset", required=True)
    validate_dataset.set_defaults(func=_validate_dataset)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--benchmark-dataset", required=True)
    calibrate.add_argument("--household-dataset", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--generated-at-us", type=int, required=True)
    calibrate.add_argument("--generator-revision", required=True)
    calibrate.add_argument("--max-benchmark-far-upper-bound", type=float, default=0.01)
    calibrate.add_argument("--max-benchmark-frr", type=float, default=0.35)
    calibrate.add_argument(
        "--max-benchmark-misidentification-upper-bound",
        type=float,
        default=0.01,
    )
    calibrate.set_defaults(func=_calibrate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (IdentityCalibrationError, OSError, ValueError) as exc:
        print(f"identity-v2 calibration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
