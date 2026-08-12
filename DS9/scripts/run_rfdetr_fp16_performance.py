#!/usr/bin/env python3
"""Run the reviewed RF-DETR B3 performance protocol against FP16 engines.

The original FP32 benchmark script is treated as an immutable protocol
template.  This launcher validates its exact digest, creates run-local source
and model-matrix snapshots for the unpromoted ``fp16_tf32`` engine receipts,
and then invokes that snapshot.  The deployed RF-DETR matrix is never changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


RELEASE = "1.8.3"
PROFILE = "fp16_tf32"
PRECISION = "fp16"
TF32_ENABLED = True
BATCH_SIZE = 3
MODEL_MATRIX_SCHEMA = "noesis.ds9.rfdetr-model-matrix.v1"
ENGINE_RECEIPT_SCHEMA = "noesis.ds9.rfdetr-runtime-engine-provenance.v1"
PERFORMANCE_REPORT_SCHEMA = "noesis.ds9.rfdetr-performance-benchmark.v1"
PERFORMANCE_MATRIX_SCHEMA = "noesis.ds9.rfdetr-performance-matrix.v1"
LAUNCH_MANIFEST_SCHEMA = "noesis.ds9.rfdetr-performance-launch.v1"
BASELINE_SCRIPT_SHA256 = (
    "f400fe9ed667e80133650f076b26ce68cde0f17739f7cd5fb56d22dfb259e26d"
)
BASELINE_MEDIA_MANIFEST_SHA256 = (
    "8be04f483aea3a14b1a398dd0269db63ca6da0fe12d1fdd573c5a7607ce8b52c"
)
IMAGE_ID = (
    "sha256:88d80ad35f12ec3a574cf2555a8242d33ac4110abdcc5f88a"
    "6cbdee40dfcf872"
)
IMAGE_REF = "noesis-ds9-dev:9.1-20260812"
TRT_VERSION = "10.16.0.72"
IMAGE_TRT_VERSION = "10.16.0.72"
CUDA_VERSION = "13.2.0.046"
EXPECTED_MODEL_IDS = (
    "detect_nano",
    "detect_small",
    "detect_medium",
    "detect_large",
    "seg_nano",
    "seg_small",
    "seg_medium",
    "seg_large",
    "seg_xlarge",
    "seg_2xlarge",
    "keypoint_preview",
)
PROTOCOL_ARGUMENTS = (
    "--rounds",
    "3",
    "--duration",
    "10",
    "--warmup-ms",
    "1000",
    "--graph-duration",
    "5",
    "--profile-duration",
    "3",
)


class PreparationError(RuntimeError):
    """Raised when the immutable benchmark inputs are incomplete or drifted."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PreparationError(f"{label} must be a JSON object: {path}")
    return payload


def require_absolute(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.is_absolute():
            raise PreparationError(f"path must be absolute: {path}")


def enabled_rows(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    if matrix.get("schema") != MODEL_MATRIX_SCHEMA:
        raise PreparationError("RF-DETR model-matrix schema drifted")
    raw_rows = matrix.get("models")
    if not isinstance(raw_rows, list):
        raise PreparationError("RF-DETR model matrix lacks models")
    rows = [
        dict(row)
        for row in raw_rows
        if isinstance(row, Mapping) and row.get("enabled") is True
    ]
    ids = tuple(str(row.get("id")) for row in rows)
    if ids != EXPECTED_MODEL_IDS:
        raise PreparationError(
            f"enabled RF-DETR model order drifted: {ids}"
        )
    return rows


def validate_receipt(
    *,
    artifact_root: Path,
    row: Mapping[str, Any],
) -> dict[str, str]:
    model_id = str(row["id"])
    receipt_path = (
        artifact_root
        / "models"
        / "provenance"
        / "rfdetr"
        / RELEASE
        / "runtime"
        / PROFILE
        / f"{model_id}.engine.json"
    )
    receipt = load_mapping(receipt_path, f"{model_id} engine receipt")
    receipt_digest = receipt.get("receipt_sha256")
    digest_payload = dict(receipt)
    digest_payload.pop("receipt_sha256", None)
    if (
        not isinstance(receipt_digest, str)
        or receipt_digest != json_digest(digest_payload)
    ):
        raise PreparationError(f"{model_id} receipt digest mismatch")

    build = receipt.get("build_contract")
    platform = receipt.get("platform")
    engine = receipt.get("engine")
    release = receipt.get("release")
    runtime_contract = receipt.get("runtime_input_contract")
    runtime = row.get("runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (
            build,
            platform,
            engine,
            release,
            runtime_contract,
            runtime,
        )
    ):
        raise PreparationError(f"{model_id} receipt contract is incomplete")
    assert isinstance(build, Mapping)
    assert isinstance(platform, Mapping)
    assert isinstance(engine, Mapping)
    assert isinstance(release, Mapping)
    assert isinstance(runtime_contract, Mapping)
    assert isinstance(runtime, Mapping)

    if (
        receipt.get("schema") != ENGINE_RECEIPT_SCHEMA
        or receipt.get("artifact_role") != "runtime_input_engine"
        or receipt.get("runtime_selected") is not False
        or receipt.get("promotion_status") != "unpromoted"
        or receipt.get("model_id") != model_id
        or receipt.get("family") != row.get("family")
        or receipt.get("variant") != row.get("variant")
        or release.get("version") != RELEASE
        or release.get("batch_size") != BATCH_SIZE
    ):
        raise PreparationError(f"{model_id} receipt identity drifted")
    if (
        build.get("profile") != PROFILE
        or build.get("precision") != PRECISION
        or build.get("fp16_enabled") is not True
        or build.get("tf32_enabled") is not TF32_ENABLED
        or build.get("trtexec_precision_args") != ["--fp16"]
        or (build.get("batch") or {}).get("size") != BATCH_SIZE
        or (build.get("batch") or {}).get("mode") != "static"
    ):
        raise PreparationError(f"{model_id} is not the FP16+TF32 B3 build")
    if (
        platform.get("image_ref") != IMAGE_REF
        or platform.get("image_id") != IMAGE_ID
        or platform.get("tensorrt_version") != TRT_VERSION
        or platform.get("image_tensorrt_version") != IMAGE_TRT_VERSION
        or platform.get("cuda_version") != CUDA_VERSION
    ):
        raise PreparationError(f"{model_id} platform contract drifted")
    if (
        runtime_contract.get("input_contract")
        != runtime.get("input_contract")
        or runtime_contract.get("adapter_revision")
        != runtime.get("adapter_revision")
    ):
        raise PreparationError(f"{model_id} runtime input contract drifted")

    engine_relative = engine.get("path")
    engine_sha256 = engine.get("sha256")
    engine_size = engine.get("size_bytes")
    if (
        not isinstance(engine_relative, str)
        or not isinstance(engine_sha256, str)
        or not isinstance(engine_size, int)
    ):
        raise PreparationError(f"{model_id} engine identity is incomplete")
    engine_path = (artifact_root / engine_relative).resolve()
    expected_parent = (
        artifact_root
        / "models"
        / "engines"
        / "rfdetr"
        / RELEASE
        / "runtime"
        / PROFILE
    ).resolve()
    if engine_path.parent != expected_parent:
        raise PreparationError(f"{model_id} engine escaped its profile root")
    onnx_filename = str(runtime.get("onnx_filename", ""))
    expected_name = (
        f"{onnx_filename[:-len('.onnx')]}_{PROFILE}.engine"
        if onnx_filename.endswith(".onnx")
        else ""
    )
    if engine_path.name != expected_name:
        raise PreparationError(f"{model_id} engine filename drifted")
    if (
        not engine_path.is_file()
        or engine_path.is_symlink()
        or engine_path.stat().st_size != engine_size
        or sha256(engine_path) != engine_sha256
    ):
        raise PreparationError(f"{model_id} engine bytes mismatch")
    return {
        "engine_profile": PROFILE,
        "engine_filename": engine_path.name,
        "engine_sha256": engine_sha256,
        "engine_receipt_sha256": sha256(receipt_path),
    }


def adapted_script(baseline_script: Path) -> str:
    if sha256(baseline_script) != BASELINE_SCRIPT_SHA256:
        raise PreparationError("immutable FP32 benchmark script digest drifted")
    source = baseline_script.read_text(encoding="utf-8")
    replacements = (
        ("fp32_no_tf32", PROFILE, 3),
        ('"precision": "fp32"', f'"precision": "{PRECISION}"', 1),
        ('"tf32_enabled": False', '"tf32_enabled": True', 1),
    )
    for original, replacement, expected_count in replacements:
        observed = source.count(original)
        if observed != expected_count:
            raise PreparationError(
                f"benchmark template occurrence drift for {original!r}: "
                f"expected {expected_count}, observed {observed}"
            )
        source = source.replace(original, replacement)
    return source


def prepare_payloads(
    *,
    repo_root: Path,
    artifact_root: Path,
    media_run: Path,
    baseline_script: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    matrix_path = repo_root / "DS9" / "config" / "rfdetr_1_8_3_models.json"
    matrix = load_mapping(matrix_path, "RF-DETR model matrix")
    rows = enabled_rows(matrix)
    engine_bindings = {
        str(row["id"]): validate_receipt(
            artifact_root=artifact_root,
            row=row,
        )
        for row in rows
    }
    snapshot = json.loads(json.dumps(matrix))
    for row in snapshot["models"]:
        model_id = str(row.get("id"))
        if model_id in engine_bindings:
            row["runtime"].update(engine_bindings[model_id])

    media_manifest = media_run / "run.json"
    if (
        not media_manifest.is_file()
        or sha256(media_manifest) != BASELINE_MEDIA_MANIFEST_SHA256
    ):
        raise PreparationError(
            "benchmark media manifest differs from the FP32 protocol"
        )
    script_source = adapted_script(baseline_script)
    manifest = {
        "schema": LAUNCH_MANIFEST_SCHEMA,
        "release": RELEASE,
        "runtime_engine_profile": PROFILE,
        "precision": PRECISION,
        "tf32_enabled": TF32_ENABLED,
        "batch_size": BATCH_SIZE,
        "models": list(EXPECTED_MODEL_IDS),
        "protocol_arguments": list(PROTOCOL_ARGUMENTS),
        "sources": {
            "deployed_matrix": str(matrix_path),
            "deployed_matrix_sha256": sha256(matrix_path),
            "baseline_script": str(baseline_script),
            "baseline_script_sha256": BASELINE_SCRIPT_SHA256,
            "media_run": str(media_run),
            "media_manifest_sha256": BASELINE_MEDIA_MANIFEST_SHA256,
        },
        "engine_bindings": engine_bindings,
    }
    return snapshot, script_source, manifest


def write_new_or_matching(path: Path, content: str, mode: int) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise PreparationError(f"refusing to overwrite drifted snapshot: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_snapshots(
    *,
    run_dir: Path,
    matrix: Mapping[str, Any],
    script_source: str,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    run_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
    run_dir.chmod(0o700)
    snapshot_root = run_dir / "source_snapshot"
    matrix_path = (
        snapshot_root / "DS9" / "config" / "rfdetr_1_8_3_models.json"
    )
    script_path = snapshot_root / "run_benchmark.py"
    manifest_path = run_dir / "launcher_manifest.json"
    write_new_or_matching(
        matrix_path,
        json.dumps(matrix, indent=2, sort_keys=True, allow_nan=False) + "\n",
        0o600,
    )
    write_new_or_matching(script_path, script_source, 0o700)
    write_new_or_matching(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        0o600,
    )
    return snapshot_root, script_path


def flatten_mapping(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, prefix=name))
        elif isinstance(value, list):
            flattened[name] = json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        else:
            flattened[name] = value
    return flattened


def write_joinable_outputs(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    report_path = run_dir / "performance_report.json"
    report = load_mapping(report_path, "FP16 performance report")
    if (
        report.get("schema") != PERFORMANCE_REPORT_SCHEMA
        or report.get("status") != "passed"
        or report.get("precision") != PRECISION
        or report.get("tf32_enabled") is not TF32_ENABLED
        or report.get("batch_size") != BATCH_SIZE
    ):
        raise PreparationError("FP16 performance report contract drifted")
    raw_models = report.get("models")
    if not isinstance(raw_models, list):
        raise PreparationError("FP16 performance report lacks model rows")
    bindings = manifest["engine_bindings"]
    rows: list[dict[str, Any]] = []
    for model in raw_models:
        if not isinstance(model, Mapping):
            raise PreparationError("FP16 performance row is not an object")
        model_id = str(model.get("model_id"))
        binding = bindings.get(model_id)
        if not isinstance(binding, Mapping):
            raise PreparationError(f"unexpected performance model: {model_id}")
        rows.append(
            {
                "framework": "rfdetr",
                "release": RELEASE,
                "runtime_engine_profile": PROFILE,
                "precision": PRECISION,
                "tf32_enabled": TF32_ENABLED,
                "engine_sha256": binding["engine_sha256"],
                "engine_receipt_sha256": binding["engine_receipt_sha256"],
                **flatten_mapping(model),
            }
        )
    if tuple(str(row["model_id"]) for row in rows) != EXPECTED_MODEL_IDS:
        raise PreparationError("FP16 performance report model order drifted")

    output = {
        "schema": PERFORMANCE_MATRIX_SCHEMA,
        "status": "passed",
        "join_key": ["framework", "model_id", "runtime_engine_profile"],
        "source_report": str(report_path),
        "source_report_sha256": sha256(report_path),
        "rows": rows,
    }
    json_path = run_dir / "performance_matrix.json"
    csv_path = run_dir / "performance_matrix.csv"
    write_new_or_matching(
        json_path,
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        0o600,
    )
    fieldnames = sorted({key for row in rows for key in row})
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    write_new_or_matching(csv_path, csv_buffer.getvalue(), 0o600)
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--docker-root", type=Path, required=True)
    parser.add_argument("--media-run", type=Path, required=True)
    parser.add_argument("--baseline-script", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help="validate readiness and print the immutable run plan without writes",
    )
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="materialize run-local snapshots without starting TensorRT",
    )
    args = parser.parse_args()
    os.umask(0o077)
    require_absolute(
        (
            args.repo_root,
            args.artifact_root,
            args.docker_root,
            args.media_run,
            args.baseline_script,
            args.run_dir,
        )
    )
    try:
        matrix, script_source, manifest = prepare_payloads(
            repo_root=args.repo_root,
            artifact_root=args.artifact_root,
            media_run=args.media_run,
            baseline_script=args.baseline_script,
        )
        if args.plan:
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "run_dir": str(args.run_dir),
                        **manifest,
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            return 0
        snapshot_root, script_path = materialize_snapshots(
            run_dir=args.run_dir,
            matrix=matrix,
            script_source=script_source,
            manifest=manifest,
        )
        if args.prepare_only:
            print(f"[READY] snapshots={snapshot_root}")
            return 0
        command = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(snapshot_root),
            "--artifact-root",
            str(args.artifact_root),
            "--docker-root",
            str(args.docker_root),
            "--media-run",
            str(args.media_run),
            "--run-dir",
            str(args.run_dir),
            *PROTOCOL_ARGUMENTS,
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
        json_path, csv_path = write_joinable_outputs(
            run_dir=args.run_dir,
            manifest=manifest,
        )
        print(f"[PASS] matrix_json={json_path}")
        print(f"[PASS] matrix_csv={csv_path}")
        return 0
    except PreparationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
