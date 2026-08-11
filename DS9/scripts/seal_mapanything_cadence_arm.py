#!/usr/bin/env python3
"""Seal one named MapAnything cadence arm from exact live-session artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DS9.scripts import evaluate_mapanything_cadence_ab as cadence  # noqa: E402


def _parse_control(value: str) -> tuple[str, str]:
    key, separator, digest = value.partition("=")
    name = key.strip()
    if not separator or not name:
        raise argparse.ArgumentTypeError("expected CONTROL_NAME=SHA256")
    try:
        return name, cadence._require_sha256(  # noqa: SLF001
            digest.strip(),
            f"control {name}",
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _control_mapping(rows: list[tuple[str, str]]) -> dict[str, str]:
    controls: dict[str, str] = {}
    for key, digest in rows:
        if key in controls:
            raise ValueError(f"duplicate controlled input: {key}")
        controls[key] = digest
    if set(controls) != set(cadence.REQUIRED_CONTROL_FINGERPRINTS):
        missing = sorted(cadence.REQUIRED_CONTROL_FINGERPRINTS - set(controls))
        extra = sorted(set(controls) - cadence.REQUIRED_CONTROL_FINGERPRINTS)
        raise ValueError(
            f"controlled input set mismatch; missing={missing} extra={extra}"
        )
    return controls


def _require_regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser().absolute()
    if expanded.is_symlink():
        raise ValueError(f"{label} must be a non-symlink regular file")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a non-symlink regular file")
    return resolved


def _require_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser().absolute()
    if expanded.is_symlink():
        raise ValueError(f"{label} must be a non-symlink directory")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a non-symlink directory")
    return resolved


def _load_source_session(
    *,
    report_path: Path,
    source_path: Path,
) -> tuple[dict[str, Any], list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    if report_path.name != cadence.live_gate.CANONICAL_REPORT_FILENAME:
        raise ValueError("floorplan report filename is not canonical")
    if source_path.name != cadence.live_gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME:
        raise ValueError("floorplan source filename is not canonical")
    source = cadence._strict_json_load(source_path)  # noqa: SLF001
    cadence.live_gate.load_and_validate_sealed_authority(
        report_path,
        source_path,
        session_id=cadence._require_text(source.get("session_id"), "session_id"),  # noqa: SLF001
        runtime_lane=cadence._require_text(  # noqa: SLF001
            source.get("runtime_lane"),
            "runtime_lane",
        ),
        runtime_instance_id=cadence._require_text(  # noqa: SLF001
            source.get("runtime_instance_id"),
            "runtime_instance_id",
        ),
        runtime_run_id=cadence._require_text(  # noqa: SLF001
            source.get("runtime_run_id"),
            "runtime_run_id",
        ),
    )
    messages = source.get("messages")
    if not isinstance(messages, list):
        raise ValueError("floorplan source transcript has no messages")
    captures: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for message in messages:
        if (
            isinstance(message, Mapping)
            and message.get("type") == "validated_exact_floorplan_capture"
        ):
            event = message.get("capture_event")
            result = message.get("result")
            if not isinstance(event, Mapping) or not isinstance(result, Mapping):
                raise ValueError("floorplan exact capture evidence is malformed")
            captures.append((event, result))
    camera_ids = source.get("camera_ids")
    if (
        not isinstance(camera_ids, list)
        or not camera_ids
        or [event.get("camera_id") for event, _result in captures] != camera_ids
    ):
        raise ValueError("floorplan exact captures do not cover camera order")
    return source, captures


def _copy_private_file(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"sealer destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)
    if cadence._sha256_file(source) != cadence._sha256_file(destination):  # noqa: SLF001
        raise ValueError(f"copied file digest changed: {source}")


def _copy_snapshot(
    *,
    source_store: Path,
    destination_store: Path,
    storage_ref: str,
    camera_id: str,
) -> None:
    source = cadence._resolve_storage_ref(  # noqa: SLF001
        source_store,
        storage_ref,
        "snapshot storage_ref",
    )
    source_loaded = cadence._load_snapshot(  # noqa: SLF001
        source,
        expected_camera=camera_id,
    )

    def same_identity(
        left: cadence.LoadedSnapshot,
        right: cadence.LoadedSnapshot,
    ) -> bool:
        return bool(
            left.identity.camera_id == right.identity.camera_id
            and left.identity.timestamp_us == right.identity.timestamp_us
            and left.identity.sequence == right.identity.sequence
            and left.identity.write_id == right.identity.write_id
            and left.identity.manifest_sha256 == right.identity.manifest_sha256
            and left.identity.content_sha256 == right.identity.content_sha256
            and left.identity.component_sha256s
            == right.identity.component_sha256s
        )

    destination = destination_store / Path(storage_ref)
    if destination.exists():
        existing = cadence._load_snapshot(  # noqa: SLF001
            destination,
            expected_camera=camera_id,
        )
        if not same_identity(existing, source_loaded):
            raise ValueError(f"duplicate storage_ref has different identity: {storage_ref}")
        return
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"snapshot contains a symlink: {candidate}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    copied = cadence._load_snapshot(  # noqa: SLF001
        destination,
        expected_camera=camera_id,
    )
    if not same_identity(copied, source_loaded):
        raise ValueError(f"copied snapshot identity changed: {storage_ref}")


def _named_snapshot_refs(
    analysis: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    fused = analysis.get("fused_snapshot")
    raw = analysis.get("raw_snapshot_identities")
    if not isinstance(fused, Mapping) or not isinstance(raw, list):
        raise ValueError("capture analysis omitted named snapshot identities")
    fused_ref = cadence._require_text(  # noqa: SLF001
        fused.get("storage_ref"),
        "fused storage_ref",
    )
    raw_refs = tuple(
        cadence._require_text(row.get("storage_ref"), "raw storage_ref")  # noqa: SLF001
        for row in raw
        if isinstance(row, Mapping)
    )
    if len(raw_refs) != len(raw) or len(set(raw_refs)) != len(raw_refs):
        raise ValueError("capture analysis raw storage references are invalid")
    return fused_ref, raw_refs


def seal_arm(
    *,
    output_root: Path,
    interval_frames: int,
    depth_root: Path,
    report_path: Path,
    source_path: Path,
    infer_config_path: Path,
    controlled_input_sha256s: Mapping[str, str],
) -> Path:
    if interval_frames not in cadence.EXPECTED_INTERVALS:
        raise ValueError("interval_frames must be 89 or 59")
    controls = _control_mapping(list(controlled_input_sha256s.items()))
    source_store = _require_directory(depth_root, "depth_root")
    report = _require_regular_file(report_path, "floorplan report")
    source = _require_regular_file(source_path, "floorplan source")
    infer_config = _require_regular_file(infer_config_path, "infer config")
    cadence._load_nvinfer_semantics(  # noqa: SLF001
        infer_config,
        expected_interval=interval_frames,
    )
    _source_document, captures = _load_source_session(
        report_path=report,
        source_path=source,
    )
    source_analyses: list[
        tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]
    ] = []
    for event, result in captures:
        analysis, _cohort = cadence._analyze_capture(  # noqa: SLF001
            event=event,
            result=result,
            store_root=source_store,
            interval_frames=interval_frames,
            source_fps=cadence.EXPECTED_SOURCE_FPS,
        )
        source_analyses.append((event, result, analysis))

    output = output_root.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to replace existing arm root: {output}")
    if not output.parent.is_dir():
        raise ValueError("arm root parent must already exist")
    source_store_resolved = source_store.resolve()
    output_resolved = output.resolve(strict=False)
    try:
        output_resolved.relative_to(source_store_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("arm root cannot be inside the source depth store")
    staging = output.parent / f".{output.name}.partial-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise ValueError(f"staging root already exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        report_destination = staging / cadence.live_gate.CANONICAL_REPORT_FILENAME
        source_destination = (
            staging / cadence.live_gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME
        )
        config_destination = (
            staging / "config_infer_secondary_mapanything.ini"
        )
        _copy_private_file(report, report_destination)
        _copy_private_file(source, source_destination)
        _copy_private_file(infer_config, config_destination)

        destination_store = staging / "depth"
        destination_store.mkdir(mode=0o700)
        copied_refs: set[str] = set()
        for _event, _result, analysis in source_analyses:
            camera_id = cadence._require_text(  # noqa: SLF001
                analysis.get("camera_id"),
                "analysis camera_id",
            )
            fused_ref, raw_refs = _named_snapshot_refs(analysis)
            for storage_ref in (*raw_refs, fused_ref):
                if storage_ref in copied_refs:
                    continue
                _copy_snapshot(
                    source_store=source_store,
                    destination_store=destination_store,
                    storage_ref=storage_ref,
                    camera_id=camera_id,
                )
                copied_refs.add(storage_ref)

        for event, result, expected in source_analyses:
            copied, _cohort = cadence._analyze_capture(  # noqa: SLF001
                event=event,
                result=result,
                store_root=destination_store,
                interval_frames=interval_frames,
                source_fps=cadence.EXPECTED_SOURCE_FPS,
            )
            if copied != expected:
                raise ValueError("copied capture replay differs from source replay")

        receipt: dict[str, Any] = {
            "contract": cadence.ARM_RECEIPT_CONTRACT,
            "contract_version": cadence.ARM_RECEIPT_VERSION,
            "arm_id": f"interval-{interval_frames}",
            "interval_frames": interval_frames,
            "source_fps": cadence.EXPECTED_SOURCE_FPS,
            "burst_seconds": cadence.EXPECTED_BURST_SECONDS,
            "min_observations": cadence.EXPECTED_MIN_OBSERVATIONS,
            "depth_agreement_m": cadence.EXPECTED_AGREEMENT_M,
            "infer_config": {
                "path": config_destination.relative_to(staging).as_posix(),
                "sha256": cadence._sha256_file(config_destination),  # noqa: SLF001
            },
            "live_gate": {
                "report_path": report_destination.relative_to(staging).as_posix(),
                "report_sha256": cadence._sha256_file(report_destination),  # noqa: SLF001
                "source_path": source_destination.relative_to(staging).as_posix(),
                "source_sha256": cadence._sha256_file(source_destination),  # noqa: SLF001
            },
            "snapshot_store": "depth",
            "controlled_input_sha256s": dict(sorted(controls.items())),
        }
        receipt["receipt_sha256"] = cadence.canonical_json_sha256(receipt)
        receipt_path = staging / cadence.ARM_RECEIPT_FILENAME
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(receipt_path, 0o600)
        cadence.load_arm(staging, expected_interval=interval_frames)
        staging.rename(output)
    except BaseException as exc:
        failed = staging / "FAILED"
        try:
            failed.write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            os.chmod(failed, 0o600)
        except OSError:
            pass
        raise
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy only the exact live-gate capture snapshots and named receipt "
            "artifacts into one immutable cadence arm root."
        )
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--interval-frames",
        required=True,
        type=int,
        choices=cadence.EXPECTED_INTERVALS,
    )
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument("--floorplan-report", required=True, type=Path)
    parser.add_argument("--floorplan-source", required=True, type=Path)
    parser.add_argument("--infer-config", required=True, type=Path)
    parser.add_argument(
        "--control",
        required=True,
        action="append",
        type=_parse_control,
        metavar="NAME=SHA256",
        help=(
            "Repeat exactly for engine, mapanything_input_contract, "
            "calibration_bundle, dewarper_bundle, and source_corpus."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    controls = _control_mapping(args.control)
    output = seal_arm(
        output_root=args.output_root,
        interval_frames=args.interval_frames,
        depth_root=args.depth_root,
        report_path=args.floorplan_report,
        source_path=args.floorplan_source,
        infer_config_path=args.infer_config,
        controlled_input_sha256s=controls,
    )
    receipt = cadence._strict_json_load(output / cadence.ARM_RECEIPT_FILENAME)  # noqa: SLF001
    print(
        f"[OK] sealed {output} "
        f"receipt_sha256={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
