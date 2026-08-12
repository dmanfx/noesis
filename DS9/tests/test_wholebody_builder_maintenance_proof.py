from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import stat
from pathlib import Path

import pytest

from DS9.scripts import validate_asset_manifest as validator


ENGINE = "wholebody49_s_masks"
RUN_ID = "20260711T120000000000Z"
ONNX_SHA256 = "1" * 64
OUTPUT_SHA256 = "2" * 64
EXECUTABLE_SHA256 = "3" * 64
SOURCE_CONTRACTS_SHA256 = "4" * 64
VARIANT_FIXTURES = {
    "wholebody49_s_masks": {
        "artifact_id": "engine.wholebody49_s_masks",
        "variant": "s_masks",
        "onnx_name": "deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx",
        "output_size": 33_429_356,
        "onnx_size": 42_539_309,
    },
    "wholebody49_x_boxes": {
        "artifact_id": "engine.wholebody49_x_boxes",
        "variant": "x_boxes",
        "onnx_name": "deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx",
        "output_size": 113_269_708,
        "onnx_size": 202_881_311,
    },
}


def _copy_stat(path: str, *, sha256: str, size: int, mode: str) -> dict[str, object]:
    return {
        "path": path,
        "device": 1,
        "inode": 2,
        "mode": mode,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "nlink": 1,
        "size_bytes": size,
        "mtime_ns": 3,
        "ctime_ns": 4,
        "sha256": sha256,
    }


def _copy_record(
    source: str,
    destination: str,
    *,
    sha256: str,
    size: int,
    source_mode: str,
) -> dict[str, object]:
    return {
        "method": "exclusive_nofollow_stream_copy",
        "source": _copy_stat(source, sha256=sha256, size=size, mode=source_mode),
        "destination": _copy_stat(destination, sha256=sha256, size=size, mode="0600"),
        "durability": "file_and_destination_directory_fsynced",
    }


def _command(
    run_directory: Path,
    label: str,
    proof: str,
    command: list[str],
) -> dict[str, object]:
    return {
        "label": label,
        "command": command,
        "proof": proof,
        "returncode": 0,
        "duration_seconds": 1.0,
        "log": str(
            Path("/workspace/DS9/models/engine_maintenance")
            / run_directory.name
            / "logs"
            / f"{label}.log"
        ),
        "timed_out": False,
        "status": "passed",
    }


def _write_private(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    path.chmod(0o600)


def _fixture(root: Path, engine: str = ENGINE) -> dict[str, object]:
    variant_fixture = VARIANT_FIXTURES[engine]
    variant = str(variant_fixture["variant"])
    output_size = int(variant_fixture["output_size"])
    onnx_size = int(variant_fixture["onnx_size"])
    contracts = json.loads(
        (validator.REPO_ROOT / "DS9/config/engine_source_contracts.json").read_text(
            encoding="utf-8"
        )
    )["contracts"]
    expected_build = copy.deepcopy(contracts[engine]["maintenance_build"])
    onnx_contract = copy.deepcopy(contracts[engine]["onnx"])
    source_contract = {
        "raw_sha256": ONNX_SHA256,
        "bundle_sha256": "5" * 64,
        "onnx": onnx_contract,
        "maintenance_build": expected_build,
    }
    source_document = {"schema_version": 1, "contracts": {engine: source_contract}}

    run_directory = root / f"{RUN_ID}-{engine}"
    manifest_path = run_directory / "manifest.json"
    _write_private(manifest_path, b"{}\n")
    compile_log = run_directory / "logs/compile-wholebody-builder.log"
    build_log = run_directory / "logs/build.log"
    _write_private(compile_log, b"")
    build_transcript = "\n".join(
        (
            "[NOESIS_TRT_BUILDER] contract=noesis.ds9.wholebody49_builder.v1",
            "[NOESIS_TRT_BUILDER] network_mode=explicit_batch_trt10_default",
            f"[NOESIS_TRT_BUILDER] variant={variant}",
            "[NOESIS_TRT_BUILDER] profile=images:3x3x640x640",
            "[NOESIS_TRT_BUILDER] workspace_bytes="
            f"{expected_build['memory_pool_limits_bytes']['workspace']}",
            "[NOESIS_TRT_BUILDER] tactic_dram_bytes=2147483648",
            "[NOESIS_TRT_BUILDER] builder_optimization_level="
            f"{expected_build['builder_optimization_level']}",
            "[NOESIS_TRT_BUILDER] logger_minimum_severity=info",
            "[NOESIS_TRT_BUILDER] logger_verbose_policy=ignored_before_copy",
            "[NOESIS_TRT_BUILDER] logger_captured_truncation=fatal",
            "[NOESIS_TRT_BUILDER] logger_error_state=sticky_fatal",
            f"[NOESIS_TRT_BUILDER] engine_bytes={output_size}",
            "[NOESIS_TRT_BUILDER] status=PASS",
            "",
        )
    ).encode()
    _write_private(build_log, build_transcript)

    builder_source_path = (
        "/workspace/DS9/csrc/wholebody49_engine_builder/wholebody49_engine_builder.cpp"
    )
    staged_onnx_path = "/workspace/DS9/models/onnx/" + str(variant_fixture["onnx_name"])
    snapshot_root = Path(f"/tmp/noesis-wholebody49-build-{RUN_ID}")
    builder_snapshot = snapshot_root / "wholebody49_engine_builder.cpp"
    onnx_snapshot = snapshot_root / Path(staged_onnx_path).name
    executable_path = Path(f"/tmp/noesis-wholebody49-engine-builder-{RUN_ID}")
    candidate_path = (
        f"/workspace/DS9/models/engines/.deimv2_wholebody49.engine.building-{RUN_ID}"
    )
    compile_command = [
        "/usr/bin/g++",
        "-std=c++17",
        "-O2",
        "-fstack-protector-strong",
        "-D_FORTIFY_SOURCE=3",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-Wno-deprecated-declarations",
        "-I/usr/local/cuda/include",
        str(builder_snapshot),
        "-o",
        str(executable_path),
        "-lnvonnxparser",
        "-lnvinfer",
        "-pthread",
    ]
    build_command = [
        str(executable_path),
        "--onnx",
        str(onnx_snapshot),
        "--output",
        candidate_path,
        "--variant",
        variant,
    ]
    commands = [
        _command(
            run_directory, "probe-trtexec", "trtexec_probe", ["trtexec", "--help"]
        ),
        _command(
            run_directory,
            "probe-wholebody-builder-toolchain",
            "generic",
            ["/usr/bin/g++", "--version"],
        ),
        _command(
            run_directory,
            "compile-wholebody-builder",
            "generic",
            compile_command,
        ),
        _command(
            run_directory,
            "build",
            f"wholebody_builder_{variant}",
            build_command,
        ),
        _command(
            run_directory,
            "load-candidate",
            "trtexec_load",
            ["trtexec", f"--loadEngine={candidate_path}", "--skipInference"],
        ),
        _command(
            run_directory,
            "load-installed",
            "trtexec_load",
            ["trtexec", "--loadEngine=/workspace/final.engine", "--skipInference"],
        ),
    ]
    builder_size = validator.WHOLEBODY_BUILDER_SOURCE.stat().st_size
    builder_sha256 = expected_build["builder_source_sha256"]
    inputs = {
        "source_contracts": {
            "path": "/workspace/DS9/config/engine_source_contracts.json",
            "sha256": SOURCE_CONTRACTS_SHA256,
        },
        "wholebody_builder_source": {
            "path": builder_source_path,
            "sha256": builder_sha256,
            "size_bytes": builder_size,
        },
        "staged_onnx": {
            "path": staged_onnx_path,
            "sha256": ONNX_SHA256,
            "size_bytes": onnx_size,
        },
    }
    maintenance = {
        "schema_version": 1,
        "contract": "noesis.ds9.engine_maintenance",
        "run_id": RUN_ID,
        "engine": engine,
        "status": "complete",
        "inputs_revalidated_at_utc": "2026-07-11T12:00:00Z",
        "installed": {"sha256": OUTPUT_SHA256, "size_bytes": output_size},
        "candidate": {
            "path": candidate_path,
            "sha256": OUTPUT_SHA256,
            "size_bytes": output_size,
        },
        "inputs": inputs,
        "metadata": {
            "build_contract": {**copy.deepcopy(expected_build), "onnx": onnx_contract},
            "host_transaction": {
                "transaction_id": RUN_ID,
                "transaction_sha256": "6" * 64,
                "transaction_manifest": str(
                    Path("/workspace/DS9/models/engine_finalize")
                    / f"{RUN_ID}-{engine}"
                    / "transaction.json"
                ),
            },
            "platform": {
                "image": "noesis-ds9-dev:test",
                "image_id": "sha256:image",
                "base_digest": "sha256:base",
                "tensorrt_version": "10.16.0.72",
                "cuda_version": "13.2.0.046",
                "driver_version": "595.71.05",
                "gpu_name": "fixture-gpu",
                "gpu_uuid": "GPU-fixture",
                "gpu_compute_capability": "12.0",
                "gpu_memory_mib": "12288",
            },
        },
        "commands": commands,
        "evidence": {
            "wholebody_input_snapshots": {
                "directory": str(snapshot_root),
                "builder_source": _copy_record(
                    builder_source_path,
                    str(builder_snapshot),
                    sha256=builder_sha256,
                    size=builder_size,
                    source_mode="0644",
                ),
                "onnx": _copy_record(
                    staged_onnx_path,
                    str(onnx_snapshot),
                    sha256=ONNX_SHA256,
                    size=onnx_size,
                    source_mode="0664",
                ),
            },
            "wholebody_builder_executable": {
                "path": str(executable_path),
                "size_bytes": 65_808,
                "sha256": EXECUTABLE_SHA256,
                "mode": "0700",
                "compile_command": compile_command,
            },
        },
    }
    _write_private(
        manifest_path,
        (json.dumps(maintenance, indent=2, sort_keys=True) + "\n").encode(),
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_relative = (
        Path("DS9/models/engine_maintenance") / run_directory.name / "manifest.json"
    ).as_posix()
    target = {
        "build_image": {
            "reference": "noesis-ds9-dev:test",
            "image_id": "sha256:image",
            "base_digest": "sha256:base",
            "tensorrt_version": "10.16.0.72",
            "cuda_version": "13.2.0.046",
        }
    }
    artifact = {"compatibility": {"precision": "fp16", "batch": 3}}
    provenance = {
        "maintenance": {
            "manifest": manifest_relative,
            "manifest_sha256": manifest_sha256,
            "output_size_bytes": output_size,
            "image": "noesis-ds9-dev:test",
            "image_id": "sha256:image",
            "base_digest": "sha256:base",
            "tensorrt_version": "10.16.0.72",
            "cuda_version": "13.2.0.046",
            "driver_version": "595.71.05",
            "gpu": {
                "name": "fixture-gpu",
                "uuid": "GPU-fixture",
                "compute_capability": "12.0",
                "memory_mib": 12288,
            },
            "precision": "fp16",
            "batch": 3,
            "tensor_contract": onnx_contract,
        }
    }
    cohort = root / "models/engine_finalize" / f"{RUN_ID}-{engine}"
    cohort.mkdir(parents=True, mode=0o700)
    cohort.chmod(0o700)
    guard_path = cohort / "gpu-memory.jsonl"
    wrapper_pid = os.getpid()
    wrapper_start = validator.gpu_sampler._proc_start_time_ticks(wrapper_pid)
    guard_mib = validator.gpu_sampler.reviewed_guard_mib(engine)
    sampled_at = "2026-07-11T12:00:00.000001Z"
    guard_rows = [
        {
            "kind": "header",
            "schema_version": 1,
            "contract": validator.gpu_sampler.CONTRACT,
            "device_index": 0,
            "expected_uuid": "GPU-fixture",
            "engine": engine,
            "transaction_id": RUN_ID,
            "prepared_transaction_sha256": "6" * 64,
            "artifact_root_id": hashlib.sha256(
                str(root.resolve()).encode("utf-8")
            ).hexdigest(),
            "container_id": "7" * 64,
            "guard_mib": guard_mib,
            "guard_bytes": guard_mib * validator.gpu_sampler.MIB,
            "interval_ms": validator.gpu_sampler.REVIEWED_SAMPLE_INTERVAL_MS,
            "max_gap_ms": validator.gpu_sampler.REVIEWED_MAX_GAP_MS,
            "parent_pid": wrapper_pid,
            "parent_start_time_ticks": wrapper_start,
            "sampler_pid": wrapper_pid,
            "sampler_start_time_ticks": wrapper_start,
            "started_at_utc": "2026-07-11T12:00:00.000000Z",
        },
        {
            "kind": "sample",
            "sequence": 1,
            "sampled_at_utc": sampled_at,
            "monotonic_ns": 1_000_000_000,
            "total_mib": 12288,
            "total_bytes": 12288 * validator.gpu_sampler.MIB,
            "reserved_mib": 0,
            "reserved_bytes": 0,
            "used_mib": 512,
            "used_bytes": 512 * validator.gpu_sampler.MIB,
        },
        {
            "kind": "footer",
            "state": "stopped",
            "sample_count": 1,
            "peak_mib": 512,
            "peak_reserved_mib": 0,
            "first_sample_at_utc": sampled_at,
            "last_sample_at_utc": sampled_at,
            "maximum_gap_ms": 0.0,
            "ended_at_utc": "2026-07-11T12:00:00.000002Z",
        },
    ]
    _write_private(
        guard_path,
        (
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in guard_rows
            )
        ).encode(),
    )
    summary_args = type(
        "GuardArgs",
        (),
        {
            "evidence": guard_path,
            "device_index": 0,
            "expected_uuid": "GPU-fixture",
            "engine": engine,
            "transaction_id": RUN_ID,
            "prepared_transaction_sha256": "6" * 64,
            "artifact_root_id": hashlib.sha256(
                str(root.resolve()).encode("utf-8")
            ).hexdigest(),
            "container_id": "7" * 64,
            "guard_mib": guard_mib,
            "interval_ms": validator.gpu_sampler.REVIEWED_SAMPLE_INTERVAL_MS,
            "max_gap_ms": validator.gpu_sampler.REVIEWED_MAX_GAP_MS,
            "parent_pid": wrapper_pid,
            "parent_start_time_ticks": wrapper_start,
            "allow_active": False,
        },
    )()
    provenance["maintenance"]["gpu_memory_guard"] = {
        "contract": validator.gpu_sampler.CONTRACT,
        "path": guard_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(guard_path.read_bytes()).hexdigest(),
        "engine": engine,
        "transaction_id": RUN_ID,
        "prepared_transaction_sha256": "6" * 64,
        "artifact_root_id": hashlib.sha256(
            str(root.resolve()).encode("utf-8")
        ).hexdigest(),
        "container_id": "7" * 64,
        "wrapper_pid": wrapper_pid,
        "wrapper_start_time_ticks": wrapper_start,
        "device_index": 0,
        "gpu_uuid": "GPU-fixture",
        "guard_mib": guard_mib,
        "sample_interval_ms": validator.gpu_sampler.REVIEWED_SAMPLE_INTERVAL_MS,
        "maximum_gap_limit_ms": validator.gpu_sampler.REVIEWED_MAX_GAP_MS,
        "summary": validator.gpu_sampler.summarize(summary_args),
    }
    return {
        "artifact_root": root,
        "source_document": source_document,
        "maintenance": maintenance,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "compile_log": compile_log,
        "build_log": build_log,
        "target": target,
        "artifact": artifact,
        "provenance": provenance,
        "artifact_id": variant_fixture["artifact_id"],
        "output_size": output_size,
    }


def _validate(
    fixture: dict[str, object],
    *,
    bind_manifest: bool = True,
    manifest_sha256: str | None = None,
    sync_manifest: bool = True,
) -> None:
    if bind_manifest and sync_manifest:
        manifest_path = fixture["manifest_path"]
        assert isinstance(manifest_path, Path)
        _write_private(
            manifest_path,
            (
                json.dumps(fixture["maintenance"], indent=2, sort_keys=True) + "\n"
            ).encode(),
        )
        synchronized_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        fixture["manifest_sha256"] = synchronized_sha256
        fixture["provenance"]["maintenance"]["manifest_sha256"] = synchronized_sha256
    bound_sha256 = (
        manifest_sha256
        if manifest_sha256 is not None
        else (str(fixture["manifest_sha256"]) if bind_manifest else None)
    )
    validator._validate_engine_maintenance_proof(
        artifact_id=fixture["artifact_id"],
        artifact=fixture["artifact"],
        provenance=fixture["provenance"],
        maintenance_payload=fixture["maintenance"],
        target=fixture["target"],
        source_contract_document=fixture["source_document"],
        current_source_contracts_sha256=SOURCE_CONTRACTS_SHA256,
        accepted_source_contract_hashes=frozenset({SOURCE_CONTRACTS_SHA256}),
        artifact_root=fixture["artifact_root"],
        maintenance_manifest_path=(fixture["manifest_path"] if bind_manifest else None),
        maintenance_manifest_sha256=bound_sha256,
        output_sha256=OUTPUT_SHA256,
        output_size_bytes=fixture["output_size"],
    )


@pytest.mark.parametrize("engine", tuple(VARIANT_FIXTURES))
def test_wholebody_builder_private_maintenance_proof_passes(
    tmp_path: Path, engine: str
) -> None:
    _validate(_fixture(tmp_path, engine))


def test_wholebody_builder_proof_rejects_missing_caller_manifest_binding(
    tmp_path: Path,
) -> None:
    signature = inspect.signature(validator._validate_engine_maintenance_proof)
    assert (
        signature.parameters["maintenance_manifest_path"].default
        is inspect.Parameter.empty
    )
    assert (
        signature.parameters["maintenance_manifest_sha256"].default
        is inspect.Parameter.empty
    )
    with pytest.raises(ValueError, match="caller-known maintenance manifest digest"):
        _validate(_fixture(tmp_path), bind_manifest=False)


def test_wholebody_builder_proof_rejects_caller_manifest_digest_mismatch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="digest mismatch"):
        _validate(_fixture(tmp_path), manifest_sha256="0" * 64)


def test_wholebody_builder_proof_rejects_caller_manifest_payload_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["maintenance"]["status"] = "running"
    with pytest.raises(ValueError, match="differs from caller-known manifest"):
        _validate(fixture, sync_manifest=False)


def test_wholebody_builder_proof_requires_gpu_guard_without_legacy_exemption(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["provenance"]["maintenance"].pop("gpu_memory_guard")
    with pytest.raises(ValueError, match="requires a sealed GPU-memory guard"):
        _validate(fixture)


def test_wholebody_builder_proof_rejects_explicit_null_gpu_guard(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["provenance"]["maintenance"]["gpu_memory_guard"] = None
    with pytest.raises(ValueError, match="must be a mapping"):
        _validate(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["maintenance"]["inputs"].pop(
                "wholebody_builder_source"
            ),
            "lack builder/ONNX authority",
        ),
        (
            lambda payload: payload["maintenance"]["inputs"][
                "wholebody_builder_source"
            ].__setitem__("sha256", "9" * 64),
            "builder-source input differs",
        ),
        (
            lambda payload: payload["maintenance"]["evidence"][
                "wholebody_input_snapshots"
            ]["onnx"]["destination"].__setitem__("sha256", "9" * 64),
            "private snapshot differs",
        ),
        (
            lambda payload: payload["maintenance"]["evidence"][
                "wholebody_builder_executable"
            ].__setitem__("mode", "0755"),
            "builder-executable evidence drifted",
        ),
        (
            lambda payload: payload["maintenance"]["commands"][2]["command"].append(
                "-fno-omit-frame-pointer"
            ),
            "compile/toolchain command drifted",
        ),
        (
            lambda payload: payload["maintenance"]["commands"][3].__setitem__(
                "proof", "wholebody_builder_x_boxes"
            ),
            "command did not pass its proof gate",
        ),
        (
            lambda payload: payload["maintenance"]["commands"][3][
                "command"
            ].__setitem__(-1, "x_boxes"),
            "variant-bound build command/proof drifted",
        ),
    ),
)
def test_wholebody_builder_private_proof_rejects_tamper(
    tmp_path: Path, mutation, message: str
) -> None:
    fixture = _fixture(tmp_path)
    mutation(fixture)
    with pytest.raises(ValueError, match=message):
        _validate(fixture)


def test_wholebody_builder_rejects_compile_log_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    compile_log = fixture["compile_log"]
    assert isinstance(compile_log, Path)
    _write_private(compile_log, b"unexpected compiler output\n")
    with pytest.raises(ValueError, match="compilation emitted output"):
        _validate(fixture)


def test_wholebody_builder_rejects_build_transcript_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    build_log = fixture["build_log"]
    assert isinstance(build_log, Path)
    raw = build_log.read_bytes().replace(
        b"[NOESIS_TRT_BUILDER] status=PASS",
        b"[NOESIS_TRT_BUILDER] status=FAIL reason=tampered",
    )
    _write_private(build_log, raw)
    with pytest.raises(ValueError, match="exact positive markers"):
        _validate(fixture)


@pytest.mark.parametrize(
    "replacement",
    (
        b"[NOESIS_TRT_BUILDER] workspace_bytes=4294967296",
        b"[NOESIS_TRT_BUILDER] workspace_bytes=2147483648",
    b"[NOESIS_TRT_BUILDER] workspace_bytes=6442450944\n"
    b"[NOESIS_TRT_BUILDER] workspace_bytes=6442450944",
    ),
)
def test_wholebody_builder_rejects_wrong_or_duplicate_workspace_marker(
    tmp_path: Path, replacement: bytes
) -> None:
    fixture = _fixture(tmp_path)
    build_log = fixture["build_log"]
    assert isinstance(build_log, Path)
    raw = build_log.read_bytes().replace(
        b"[NOESIS_TRT_BUILDER] workspace_bytes=6442450944",
        replacement,
    )
    _write_private(build_log, raw)
    with pytest.raises(ValueError, match="exact positive markers"):
        _validate(fixture)


@pytest.mark.parametrize(
    "replacement",
    (
        b"",
        b"[NOESIS_TRT_BUILDER] builder_optimization_level=3",
        b"[NOESIS_TRT_BUILDER] builder_optimization_level=0\n"
        b"[NOESIS_TRT_BUILDER] builder_optimization_level=0",
    ),
)
def test_wholebody_builder_rejects_missing_wrong_or_duplicate_optimization_marker(
    tmp_path: Path, replacement: bytes
) -> None:
    fixture = _fixture(tmp_path)
    build_log = fixture["build_log"]
    assert isinstance(build_log, Path)
    raw = build_log.read_bytes().replace(
        b"[NOESIS_TRT_BUILDER] builder_optimization_level=0",
        replacement,
    )
    _write_private(build_log, raw)
    with pytest.raises(ValueError, match="exact positive markers"):
        _validate(fixture)


@pytest.mark.parametrize(
    "marker",
    (
        b"logger_minimum_severity=info",
        b"logger_verbose_policy=ignored_before_copy",
        b"logger_captured_truncation=fatal",
        b"logger_error_state=sticky_fatal",
    ),
)
def test_wholebody_builder_rejects_missing_logger_policy_marker(
    tmp_path: Path, marker: bytes
) -> None:
    fixture = _fixture(tmp_path)
    build_log = fixture["build_log"]
    assert isinstance(build_log, Path)
    raw = build_log.read_bytes().replace(b"[NOESIS_TRT_BUILDER] " + marker + b"\n", b"")
    _write_private(build_log, raw)
    with pytest.raises(ValueError, match="exact positive markers"):
        _validate(fixture)


def test_wholebody_builder_rejects_current_source_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    source_contract = fixture["source_document"]["contracts"][ENGINE]
    source_contract["maintenance_build"]["builder_source_sha256"] = "9" * 64
    fixture["maintenance"]["metadata"]["build_contract"]["builder_source_sha256"] = (
        "9" * 64
    )
    fixture["maintenance"]["inputs"]["wholebody_builder_source"]["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="tracked Wholebody49 builder source"):
        _validate(fixture)


def test_wholebody_builder_rejects_non_power_of_two_tactic_pool_authority(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_contract = fixture["source_document"]["contracts"][ENGINE]
    source_contract["maintenance_build"]["memory_pool_limits_bytes"]["tactic_dram"] = (
        3221225472
    )
    fixture["maintenance"]["metadata"]["build_contract"]["memory_pool_limits_bytes"][
        "tactic_dram"
    ] = 3221225472
    with pytest.raises(ValueError, match="positive power of two"):
        _validate(fixture)


def test_wholebody_builder_rejects_valid_but_wrong_variant_workspace(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_contract = fixture["source_document"]["contracts"][ENGINE]
    source_contract["maintenance_build"]["memory_pool_limits_bytes"]["workspace"] = (
        4294967296
    )
    fixture["maintenance"]["metadata"]["build_contract"]["memory_pool_limits_bytes"][
        "workspace"
    ] = 4294967296
    with pytest.raises(ValueError, match="dedicated-builder authority drifted"):
        _validate(fixture)


def test_wholebody_builder_rejects_wrong_variant_optimization_level(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_contract = fixture["source_document"]["contracts"][ENGINE]
    source_contract["maintenance_build"]["builder_optimization_level"] = 3
    fixture["maintenance"]["metadata"]["build_contract"][
        "builder_optimization_level"
    ] = 3
    with pytest.raises(ValueError, match="dedicated-builder authority drifted"):
        _validate(fixture)


def test_wholebody_builder_rejects_boolean_optimization_level(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_contract = fixture["source_document"]["contracts"][ENGINE]
    source_contract["maintenance_build"]["builder_optimization_level"] = False
    fixture["maintenance"]["metadata"]["build_contract"][
        "builder_optimization_level"
    ] = False
    with pytest.raises(ValueError, match="integer from 0 to 5"):
        _validate(fixture)


def test_wholebody_builder_rejects_logger_policy_authority_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_contract = fixture["source_document"]["contracts"][ENGINE]
    source_contract["maintenance_build"]["logger_policy"]["verbose"] = "captured"
    fixture["maintenance"]["metadata"]["build_contract"]["logger_policy"]["verbose"] = (
        "captured"
    )
    with pytest.raises(ValueError, match="dedicated-builder authority drifted"):
        _validate(fixture)


def test_wholebody_builder_log_must_be_private_regular_file(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    compile_log = fixture["compile_log"]
    assert isinstance(compile_log, Path)
    compile_log.chmod(0o644)
    assert stat.S_IMODE(compile_log.stat().st_mode) == 0o644
    with pytest.raises(ValueError, match="owner-only single-link regular"):
        _validate(fixture)
