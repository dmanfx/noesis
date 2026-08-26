#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from engine_maintenance_common import (  # noqa: E402
    EngineMaintenanceRun,
    MAPANYTHING_QUALITY_COMMAND_LABEL,
    MAPANYTHING_QUALITY_OUTPUT_MAX_BYTES,
    MAPANYTHING_QUALITY_OUTPUT_NAME,
    copy_regular_file_exclusive,
    engine_maintenance_lock,
    ensure_private_directory,
    fsync_directory,
    fsync_file,
    load_source_contracts,
    mapanything_quality_gate_from_source_contracts,
    native_host_build_authority,
    native_host_maintenance_platform,
    new_run_id,
    require_absent_candidate_path,
    required_regular_file,
    validate_wholebody_memory_pool_limits,
    validate_maintenance_build_contract,
    validate_source_contract,
    sha256_file,
)


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
_MODEL_OVERRIDE = str(os.environ.get("NOESIS_MODEL_DIR", "") or "").strip()
_ARTIFACT_OVERRIDE = str(os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "") or "").strip()
DS9_MODEL_ROOT = Path(
    _MODEL_OVERRIDE
    or ((Path(_ARTIFACT_OVERRIDE).expanduser() / "models") if _ARTIFACT_OVERRIDE else DS9_ROOT / "models")
).expanduser()
DS9_ONNX = Path(os.environ.get("NOESIS_ONNX_DIR", DS9_MODEL_ROOT / "onnx")).expanduser()
DS9_ENGINES = Path(os.environ.get("NOESIS_ENGINE_DIR", DS9_MODEL_ROOT / "engines")).expanduser()
SOURCE_MODELS_ROOT = Path(
    os.environ.get("NOESIS_DS9_SOURCE_MODELS_ROOT", str(REPO_ROOT / "models"))
).expanduser()
RFDETR_TRT_PLUGIN = Path(os.environ.get("NOESIS_RFDETR_TRT_PLUGIN_LIB", str(DS9_ROOT / "plugins" / "libnvdsinfer_custom_impl_Yolo_seg.so")))
SOURCE_CONTRACTS = DS9_ROOT / "config" / "engine_source_contracts.json"
WHOLEBODY_BUILDER_SOURCE = (
    DS9_ROOT
    / "csrc"
    / "wholebody49_engine_builder"
    / "wholebody49_engine_builder.cpp"
)
WHOLEBODY_BUILDER_CONTRACT = "noesis.ds9.wholebody49_builder.v1"
WHOLEBODY_S_MASKS_WORKSPACE_BYTES = 6144 * 1024 * 1024
WHOLEBODY_X_BOXES_WORKSPACE_BYTES = 4096 * 1024 * 1024
WHOLEBODY_TACTIC_DRAM_BYTES = 2048 * 1024 * 1024
WHOLEBODY_S_MASKS_BUILDER_OPTIMIZATION_LEVEL = 0
WHOLEBODY_X_BOXES_BUILDER_OPTIMIZATION_LEVEL = 3
WHOLEBODY_LOGGER_POLICY = {
    "minimum_severity": "info",
    "verbose": "ignored_before_copy",
    "captured_message_truncation": "fatal",
    "error_state": "sticky_fatal",
}
WHOLEBODY_PROFILE_SHAPE = [3, 3, 640, 640]


def _require_explicit_model_root() -> None:
    if _MODEL_OVERRIDE or _ARTIFACT_OVERRIDE:
        return
    raise SystemExit(
        "NOESIS_MODEL_DIR or NOESIS_DS9_ARTIFACT_ROOT must explicitly select "
        "the external DS9 model artifact root"
    )


@dataclass(frozen=True)
class EngineSpec:
    name: str
    source_onnx: Path
    staged_onnx: Path
    engine: Path
    trtexec_args: tuple[str, ...]
    precision_arg: str | None = "--fp16"
    builder: str = "trtexec"
    builder_variant: str | None = None


def _print_run(cmd: Sequence[str | Path]) -> None:
    print("[RUN]", " ".join(str(part) for part in cmd))


def _uses_wholebody_builder(spec: EngineSpec) -> bool:
    return spec.builder == WHOLEBODY_BUILDER_CONTRACT


def _wholebody_workspace_bytes(variant: str | None) -> int:
    values = {
        "s_masks": WHOLEBODY_S_MASKS_WORKSPACE_BYTES,
        "x_boxes": WHOLEBODY_X_BOXES_WORKSPACE_BYTES,
    }
    try:
        return values[variant]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Wholebody49 has an invalid dedicated-builder variant"
        ) from exc


def _wholebody_builder_optimization_level(variant: str | None) -> int:
    values = {
        "s_masks": WHOLEBODY_S_MASKS_BUILDER_OPTIMIZATION_LEVEL,
        "x_boxes": WHOLEBODY_X_BOXES_BUILDER_OPTIMIZATION_LEVEL,
    }
    try:
        return values[variant]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Wholebody49 has an invalid dedicated-builder variant"
        ) from exc


def _wholebody_compile_command(
    compiler: str, source: Path, output: Path
) -> list[str | Path]:
    return [
        compiler,
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
        source,
        "-o",
        output,
        "-lnvonnxparser",
        "-lnvinfer",
        "-pthread",
    ]


def _wholebody_build_command(
    executable: Path, spec: EngineSpec, candidate: Path, *, onnx: Path
) -> list[str | Path]:
    if spec.builder_variant not in {"s_masks", "x_boxes"}:
        raise RuntimeError(
            f"{spec.name} has an invalid Wholebody49 builder variant"
        )
    return [
        executable,
        "--onnx",
        onnx,
        "--output",
        candidate,
        "--variant",
        spec.builder_variant,
    ]


def _build_contract(
    spec: EngineSpec,
    *,
    onnx_metadata: object,
    plugin_args: Sequence[str],
) -> dict[str, object]:
    if _uses_wholebody_builder(spec):
        if plugin_args:
            raise RuntimeError("Wholebody49 dedicated builder forbids plugin arguments")
        memory_pool_limits = validate_wholebody_memory_pool_limits(
            {
                "workspace": _wholebody_workspace_bytes(spec.builder_variant),
                "tactic_dram": WHOLEBODY_TACTIC_DRAM_BYTES,
            }
        )
        return {
            "builder": WHOLEBODY_BUILDER_CONTRACT,
            "builder_source_sha256": sha256_file(WHOLEBODY_BUILDER_SOURCE),
            "precision": "fp16",
            "network_mode": "explicit_batch_trt10_default",
            "builder_optimization_level": (
                _wholebody_builder_optimization_level(spec.builder_variant)
            ),
            "logger_policy": dict(WHOLEBODY_LOGGER_POLICY),
            "profile": {
                "input": "images",
                "min": WHOLEBODY_PROFILE_SHAPE,
                "opt": WHOLEBODY_PROFILE_SHAPE,
                "max": WHOLEBODY_PROFILE_SHAPE,
            },
            "memory_pool_limits_bytes": memory_pool_limits,
            "variant": spec.builder_variant,
            "plugin_args": [],
            "onnx": onnx_metadata,
        }
    precision: dict[str, object] = (
        {"precision": "fp32"}
        if spec.precision_arg is None
        else {"precision_arg": spec.precision_arg}
    )
    return {
        **precision,
        "trtexec_args": list(spec.trtexec_args),
        "plugin_args": list(plugin_args),
        "onnx": onnx_metadata,
    }


def _stage_onnx(src: Path, dst: Path, *, dry_run: bool = False) -> None:
    if not src.exists() or src.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing ONNX source: {src}")
    if src.resolve() == dst.resolve():
        print(f"[STAGE] {src} already staged")
        return
    source_digest = sha256_file(src)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and sha256_file(dst) == source_digest:
        print(f"[STAGE] {dst} already matches {src}")
        return
    print(f"[STAGE] {src} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        temporary = dst.with_name(f".{dst.name}.staging-{os.getpid()}")
        try:
            shutil.copy2(src, temporary)
            fsync_file(temporary)
            if sha256_file(temporary) != source_digest:
                raise RuntimeError(f"staged ONNX failed SHA-256 verification: {temporary}")
            os.replace(temporary, dst)
            fsync_directory(dst.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _build(
    spec: EngineSpec,
    *,
    trtexec: str,
    dry_run: bool = False,
    validate_load: bool = False,
    evidence_root: Path | None = None,
    build_timeout_seconds: int = 1800,
    load_timeout_seconds: int = 120,
    inspect_onnx_contract: bool = True,
) -> None:
    source_contract: dict[str, object] | None = None
    quality_authority: dict[str, object] | None = None
    quality_fixture: Path | None = None
    _stage_onnx(spec.source_onnx, spec.staged_onnx, dry_run=dry_run)
    if inspect_onnx_contract:
        validate_source_contract(spec.name, spec.source_onnx, SOURCE_CONTRACTS)
        source_contract = validate_source_contract(
            spec.name, spec.staged_onnx, SOURCE_CONTRACTS
        )
        onnx_metadata = source_contract.get("onnx") or source_contract.get(
            "tensor_contract"
        )
        if spec.name == "mapanything":
            quality_authority = mapanything_quality_gate_from_source_contracts(
                SOURCE_CONTRACTS
            )
            relative_fixture = Path(
                str(quality_authority["fixture"]["artifact_relative_path"])
            )
            quality_fixture = DS9_MODEL_ROOT / relative_fixture.relative_to("models")
    else:
        onnx_metadata = {"inspection": "disabled_by_test"}
    plugin_args: list[str] = []
    if spec.name.startswith("rfdetr"):
        if not RFDETR_TRT_PLUGIN.exists() or RFDETR_TRT_PLUGIN.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing RF-DETR TensorRT plugin library: {RFDETR_TRT_PLUGIN}")
        plugin_args.append(f"--dynamicPlugins={RFDETR_TRT_PLUGIN}")

    build_contract = _build_contract(
        spec,
        onnx_metadata=onnx_metadata,
        plugin_args=plugin_args,
    )
    if inspect_onnx_contract:
        validate_maintenance_build_contract(
            spec.name, build_contract, SOURCE_CONTRACTS
        )

    # Kept for CLI compatibility; guarded real builds always validate both the
    # candidate and the final installed path.
    _ = validate_load
    if dry_run:
        temporary_engine = spec.engine.with_name(f".{spec.engine.name}.building-PLAN")
        _print_run([trtexec, "--help"])
        if _uses_wholebody_builder(spec):
            compiler = shutil.which("g++")
            if not compiler:
                raise RuntimeError("g++ not found for the DS9 Wholebody49 builder")
            builder_executable = Path(
                "/tmp/noesis-wholebody49-engine-builder-PLAN"
            )
            snapshot_root = Path("/tmp/noesis-wholebody49-build-PLAN")
            builder_source_snapshot = snapshot_root / "wholebody49_engine_builder.cpp"
            onnx_snapshot = snapshot_root / spec.staged_onnx.name
            print(
                "[PLAN] securely snapshot reviewed builder/ONNX bytes into "
                f"{snapshot_root} before compilation or parsing"
            )
            _print_run([compiler, "--version"])
            _print_run(
                _wholebody_compile_command(
                    compiler, builder_source_snapshot, builder_executable
                )
            )
            _print_run(
                _wholebody_build_command(
                    builder_executable,
                    spec,
                    temporary_engine,
                    onnx=onnx_snapshot,
                )
            )
        else:
            precision_args = [spec.precision_arg] if spec.precision_arg else []
            _print_run(
                [
                    trtexec,
                    f"--onnx={spec.staged_onnx}",
                    *precision_args,
                    *plugin_args,
                    *spec.trtexec_args,
                    f"--saveEngine={temporary_engine}",
                    "--skipInference",
                ]
            )
        _print_run([trtexec, f"--loadEngine={temporary_engine}", "--skipInference"])
        if spec.name == "mapanything":
            if quality_authority is None or quality_fixture is None:
                raise RuntimeError("MapAnything quality authority was not resolved")
            quality_output = (
                evidence_root or DS9_MODEL_ROOT / "engine_maintenance"
            ) / "PLAN-mapanything" / MAPANYTHING_QUALITY_OUTPUT_NAME
            _print_run(
                [
                    trtexec,
                    f"--loadEngine={temporary_engine}",
                    f"--loadInputs=images:{quality_fixture}",
                    *quality_authority["trtexec_args"],
                    "--dumpOutput",
                    f"--exportOutput={quality_output}",
                ]
            )
            print(
                "[PLAN] require the pinned MapAnything functional-quality "
                "receipt before installation"
            )
        print(f"[PLAN] preserve prior bytes and atomically install {temporary_engine} -> {spec.engine}")
        _print_run([trtexec, f"--loadEngine={spec.engine}", "--skipInference"])
        print("[PLAN] require positive candidate/final load markers and reject fail-closed markers")
        return

    spec.engine.parent.mkdir(parents=True, exist_ok=True)
    evidence = evidence_root or (DS9_MODEL_ROOT / "engine_maintenance")
    inputs = {
        "source_onnx": spec.source_onnx,
        "staged_onnx": spec.staged_onnx,
        "source_contracts": SOURCE_CONTRACTS,
        "rebuild_implementation": Path(__file__).resolve(),
        "maintenance_implementation": SCRIPT_DIR / "engine_maintenance_common.py",
    }
    if _uses_wholebody_builder(spec):
        inputs["wholebody_builder_source"] = WHOLEBODY_BUILDER_SOURCE
    if plugin_args:
        inputs["tensorrt_plugin"] = RFDETR_TRT_PLUGIN
    if spec.name == "mapanything":
        if quality_authority is None or quality_fixture is None:
            raise RuntimeError("MapAnything quality authority was not resolved")
        inputs["functional_quality_fixture"] = quality_fixture
    run_id = new_run_id()
    temporary_engine = spec.engine.with_name(
        f".{spec.engine.name}.building-{run_id}"
    )
    require_absent_candidate_path(temporary_engine)
    platform = native_host_maintenance_platform()
    run = EngineMaintenanceRun(
        name=spec.name,
        target=spec.engine,
        evidence_root=evidence,
        inputs=inputs,
        repo_root=REPO_ROOT,
        run_id=run_id,
        metadata={
            "platform": platform,
            "build_contract": build_contract,
            **(
                {"quality_gate_authority": quality_authority}
                if quality_authority is not None
                else {}
            ),
        },
    )
    try:
        run.run_command(
            "probe-trtexec",
            [trtexec, "--help"],
            proof="trtexec_probe",
            timeout_seconds=30,
        )
        if _uses_wholebody_builder(spec):
            compiler = shutil.which("g++")
            if not compiler:
                raise RuntimeError("g++ not found for the DS9 Wholebody49 builder")
            snapshot_root = ensure_private_directory(
                Path(f"/tmp/noesis-wholebody49-build-{run_id}")
            )
            builder_source_snapshot = (
                snapshot_root / "wholebody49_engine_builder.cpp"
            )
            onnx_snapshot = snapshot_root / spec.staged_onnx.name
            source_snapshot_record = copy_regular_file_exclusive(
                WHOLEBODY_BUILDER_SOURCE,
                builder_source_snapshot,
                expected_sha256=str(build_contract["builder_source_sha256"]),
            )
            onnx_snapshot_record = copy_regular_file_exclusive(
                spec.staged_onnx,
                onnx_snapshot,
                expected_sha256=(
                    str(source_contract["sha256"])
                    if source_contract is not None
                    else sha256_file(spec.staged_onnx)
                ),
            )
            run.record_evidence(
                "wholebody_input_snapshots",
                {
                    "directory": str(snapshot_root),
                    "builder_source": source_snapshot_record,
                    "onnx": onnx_snapshot_record,
                },
            )
            builder_executable = Path(
                f"/tmp/noesis-wholebody49-engine-builder-{run_id}"
            )
            compile_command = _wholebody_compile_command(
                compiler, builder_source_snapshot, builder_executable
            )
            run.run_command(
                "probe-wholebody-builder-toolchain",
                [compiler, "--version"],
                proof="generic",
                timeout_seconds=30,
            )
            run.run_command(
                "compile-wholebody-builder",
                compile_command,
                proof="generic",
                timeout_seconds=120,
            )
            required_regular_file(
                builder_executable, "compiled Wholebody49 builder"
            )
            os.chmod(builder_executable, 0o700)
            run.record_evidence(
                "wholebody_builder_executable",
                {
                    "path": str(builder_executable),
                    "size_bytes": builder_executable.stat().st_size,
                    "sha256": sha256_file(builder_executable),
                    "mode": f"{stat.S_IMODE(builder_executable.stat().st_mode):04o}",
                    "compile_command": [str(value) for value in compile_command],
                },
            )
            build_command = _wholebody_build_command(
                builder_executable,
                spec,
                temporary_engine,
                onnx=onnx_snapshot,
            )
            build_proof = f"wholebody_builder_{spec.builder_variant}"
        else:
            precision_args = [spec.precision_arg] if spec.precision_arg else []
            build_command = [
                trtexec,
                f"--onnx={spec.staged_onnx}",
                *precision_args,
                *plugin_args,
                *spec.trtexec_args,
                f"--saveEngine={temporary_engine}",
                "--skipInference",
            ]
            build_proof = "trtexec_build"
        run.run_command(
            "build",
            build_command,
            proof=build_proof,
            timeout_seconds=build_timeout_seconds,
        )
        candidate_record = run.record_candidate(temporary_engine)
        run.record_native_host_build(
            native_host_build_authority(
                source_sha256=(
                    str(source_contract["sha256"])
                    if source_contract is not None
                    else sha256_file(spec.staged_onnx)
                ),
                output_sha256=str(candidate_record["sha256"]),
                command=build_command,
            )
        )
        run.run_command(
            "load-candidate",
            [trtexec, f"--loadEngine={temporary_engine}", "--skipInference"],
            proof="trtexec_load",
            timeout_seconds=load_timeout_seconds,
        )
        if spec.name == "mapanything":
            if quality_authority is None or quality_fixture is None:
                raise RuntimeError("MapAnything quality authority was not resolved")
            quality_output = run.run_dir / MAPANYTHING_QUALITY_OUTPUT_NAME
            require_absent_candidate_path(quality_output)
            quality_command = [
                trtexec,
                f"--loadEngine={temporary_engine}",
                f"--loadInputs=images:{quality_fixture}",
                *quality_authority["trtexec_args"],
                "--dumpOutput",
                f"--exportOutput={quality_output}",
            ]
            quality_command_record = run.run_command(
                MAPANYTHING_QUALITY_COMMAND_LABEL,
                quality_command,
                proof="trtexec_inference",
                timeout_seconds=load_timeout_seconds,
                max_output_bytes=MAPANYTHING_QUALITY_OUTPUT_MAX_BYTES,
            )
            required_regular_file(
                quality_output, "MapAnything functional output evidence"
            )
            output_info = quality_output.stat()
            if output_info.st_uid != os.getuid() or output_info.st_nlink != 1:
                raise RuntimeError(
                    "MapAnything functional output evidence is not owner-owned/single-link"
                )
            os.chmod(quality_output, 0o600)
            fsync_file(quality_output)
            fsync_directory(quality_output.parent)
            run.record_mapanything_functional_quality(
                candidate=temporary_engine,
                fixture_path=quality_fixture,
                output_path=quality_output,
                authority=quality_authority,
                command_record=quality_command_record,
            )
        run.revalidate_inputs()
        run.install_candidate(temporary_engine)
        print(f"[INSTALL] {spec.engine}")
        try:
            run.run_command(
                "load-installed",
                [trtexec, f"--loadEngine={spec.engine}", "--skipInference"],
                proof="trtexec_load",
                timeout_seconds=load_timeout_seconds,
            )
        except BaseException as final_load_error:
            run.rollback_after_install_failure(final_load_error)
            raise
        run.complete()
        print(f"[EVIDENCE] {run.manifest_path}")
    except BaseException as exc:
        run.fail(exc)
        raise


def _specs(include_mapanything: bool) -> list[EngineSpec]:
    specs = [
        EngineSpec(
            "yolo11_seg",
            DS9_ONNX / "yolo11s-seg_cust_fused.onnx",
            DS9_ONNX / "yolo11s-seg_cust_fused.onnx",
            DS9_ENGINES / "yolo11s-seg_cust_fused.engine",
            (
                "--minShapes=images:3x3x640x640",
                "--optShapes=images:3x3x640x640",
                "--maxShapes=images:3x3x640x640",
            ),
        ),
        EngineSpec(
            "yolo11",
            DS9_ONNX / "yolo11m.onnx",
            DS9_ONNX / "yolo11m.onnx",
            DS9_ENGINES / "yolo11m_b3_fp16.engine",
            (
                "--minShapes=input:3x3x640x640",
                "--optShapes=input:3x3x640x640",
                "--maxShapes=input:3x3x640x640",
            ),
        ),
        EngineSpec(
            "reid_swin",
            DS9_ONNX / "reid_swin_tiny_market1501_aicity156_featuredim256.onnx",
            DS9_ONNX / "reid_swin_tiny_market1501_aicity156_featuredim256.onnx",
            DS9_ENGINES / "reid_swin_tiny_aicity156_dyn_b16_fp16.engine",
            (
                "--minShapes=input:1x3x256x128",
                "--optShapes=input:16x3x256x128",
                "--maxShapes=input:16x3x256x128",
            ),
        ),
        EngineSpec(
            "yolo26_pose_n",
            DS9_ONNX / "yolo26n-pose_b3.onnx",
            DS9_ONNX / "yolo26n-pose_b3.onnx",
            DS9_ENGINES / "yolo26n-pose_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "depth_anything_v2_tracking",
            DS9_ONNX / "depth_anything_v2_metric_hypersim_vits_294x518_b3.onnx",
            DS9_ONNX / "depth_anything_v2_metric_hypersim_vits_294x518_b3.onnx",
            DS9_ENGINES / "depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "bodypose3dnet",
            DS9_ONNX / "bodypose3dnet_accuracy.onnx",
            DS9_ONNX / "bodypose3dnet_accuracy.onnx",
            DS9_ENGINES / "bodypose3dnet_accuracy_b1_fp16.engine",
            (
                "--minShapes=input0:1x3x256x192,k_inv:1x3x3,t_form_inv:1x3x3,scale_normalized_mean_limb_lengths:1x36,mean_limb_lengths:1x36",
                "--optShapes=input0:1x3x256x192,k_inv:1x3x3,t_form_inv:1x3x3,scale_normalized_mean_limb_lengths:1x36,mean_limb_lengths:1x36",
                "--maxShapes=input0:1x3x256x192,k_inv:1x3x3,t_form_inv:1x3x3,scale_normalized_mean_limb_lengths:1x36,mean_limb_lengths:1x36",
                "--memPoolSize=workspace:4096",
            ),
        ),
        EngineSpec(
            "yolo26_seg_n",
            DS9_ONNX / "yolo26n-seg_fused.onnx",
            DS9_ONNX / "yolo26n-seg_fused.onnx",
            DS9_ENGINES / "yolo26n-seg_fused_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_seg_s",
            DS9_ONNX / "yolo26s-seg_fused.onnx",
            DS9_ONNX / "yolo26s-seg_fused.onnx",
            DS9_ENGINES / "yolo26s-seg_fused_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_seg_m",
            DS9_ONNX / "yolo26m-seg_fused.onnx",
            DS9_ONNX / "yolo26m-seg_fused.onnx",
            DS9_ENGINES / "yolo26m-seg_fused_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_n",
            DS9_ONNX / "yolo26n.onnx",
            DS9_ONNX / "yolo26n.onnx",
            DS9_ENGINES / "yolo26n_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_s",
            DS9_ONNX / "yolo26s.onnx",
            DS9_ONNX / "yolo26s.onnx",
            DS9_ENGINES / "yolo26s_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_m",
            DS9_ONNX / "yolo26m.onnx",
            DS9_ONNX / "yolo26m.onnx",
            DS9_ENGINES / "yolo26m_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_l",
            DS9_ONNX / "yolo26l.onnx",
            DS9_ONNX / "yolo26l.onnx",
            DS9_ENGINES / "yolo26l_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_x",
            DS9_ONNX / "yolo26x.onnx",
            DS9_ONNX / "yolo26x.onnx",
            DS9_ENGINES / "yolo26x_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "wholebody49_s_masks",
            DS9_ONNX / "deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx",
            DS9_ONNX / "deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx",
            DS9_ENGINES / "deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine",
            (),
            builder=WHOLEBODY_BUILDER_CONTRACT,
            builder_variant="s_masks",
        ),
        EngineSpec(
            "wholebody49_x_boxes",
            DS9_ONNX / "deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx",
            DS9_ONNX / "deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx",
            DS9_ENGINES / "deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine",
            (),
            builder=WHOLEBODY_BUILDER_CONTRACT,
            builder_variant="x_boxes",
        ),
        EngineSpec(
            "rfdetr_n",
            DS9_ONNX / "rfdetr_n_384.onnx",
            DS9_ONNX / "rfdetr_n_384.onnx",
            DS9_ENGINES / "rfdetr_n_384_b3_fp16.engine",
            ("--memPoolSize=workspace:4096",),
        ),
        EngineSpec(
            "rfdetr_s",
            DS9_ONNX / "rfdetr_s_512.onnx",
            DS9_ONNX / "rfdetr_s_512.onnx",
            DS9_ENGINES / "rfdetr_s_512_b3_fp16.engine",
            ("--memPoolSize=workspace:4096",),
        ),
        EngineSpec(
            "rfdetr_m",
            DS9_ONNX / "rfdetr_m_576.onnx",
            DS9_ONNX / "rfdetr_m_576.onnx",
            DS9_ENGINES / "rfdetr_m_576_b3_fp16.engine",
            ("--memPoolSize=workspace:4096",),
        ),
        EngineSpec(
            "rfdetr_seg_n",
            SOURCE_MODELS_ROOT / "onnx" / "rfdetr_seg_n_312.onnx",
            DS9_ONNX / "rfdetr_seg_n_312.onnx",
            DS9_ENGINES / "rfdetr_seg_n_312_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "rfdetr_seg_s",
            SOURCE_MODELS_ROOT / "onnx" / "rfdetr_seg_s_384.onnx",
            DS9_ONNX / "rfdetr_seg_s_384.onnx",
            DS9_ENGINES / "rfdetr_seg_s_384_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "rfdetr_seg_m",
            SOURCE_MODELS_ROOT / "onnx" / "rfdetr_seg_m_432.onnx",
            DS9_ONNX / "rfdetr_seg_m_432.onnx",
            DS9_ENGINES / "rfdetr_seg_m_432_b3_fp16.engine",
            (),
        ),
    ]
    if include_mapanything:
        specs.append(
            EngineSpec(
                "mapanything",
                DS9_ONNX / "mapanything_images_294x518_b3.onnx",
                DS9_ONNX / "mapanything_images_294x518_b3.onnx",
                DS9_ENGINES / "mapanything_images_294x518_b3_fp32.plan",
                (
                    "--minShapes=images:3x3x294x518",
                    "--optShapes=images:3x3x294x518",
                    "--maxShapes=images:3x3x294x518",
                    "--builderOptimizationLevel=0",
                    "--maxAuxStreams=0",
                    "--memPoolSize=workspace:1024",
                ),
                None,
            )
        )
    return specs


def _preflight_selected_spec(spec: EngineSpec) -> None:
    """Validate every selected source/build contract before any engine build."""

    source_contract = validate_source_contract(
        spec.name, spec.source_onnx, SOURCE_CONTRACTS
    )
    plugin_args: list[str] = []
    if spec.name.startswith("rfdetr"):
        if not RFDETR_TRT_PLUGIN.is_file() or RFDETR_TRT_PLUGIN.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Missing RF-DETR TensorRT plugin library: {RFDETR_TRT_PLUGIN}"
            )
        plugin_args.append(f"--dynamicPlugins={RFDETR_TRT_PLUGIN}")
    validate_maintenance_build_contract(
        spec.name,
        _build_contract(
            spec,
            onnx_metadata=source_contract.get("onnx")
            or source_contract.get("tensor_contract"),
            plugin_args=plugin_args,
        ),
        SOURCE_CONTRACTS,
    )
    if spec.name == "mapanything":
        mapanything_quality_gate_from_source_contracts(SOURCE_CONTRACTS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild Noesis TensorRT engines for DeepStream 9.1 / TensorRT 10.16")
    parser.add_argument("--only", default="", help="Comma-separated engine names to build.")
    parser.add_argument("--include-mapanything", action="store_true", help="Also build MapAnything from DS9/models/onnx/mapanything_images_294x518_b3.onnx.")
    parser.add_argument(
        "--validate-load",
        action="store_true",
        help="Compatibility flag; guarded real builds always validate candidate and installed paths.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=DS9_MODEL_ROOT / "engine_maintenance",
        help="Durable manifest/log/prior-engine evidence root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise SystemExit("trtexec not found on PATH")

    selected = {item.strip() for item in str(args.only or "").split(",") if item.strip()}
    if not selected:
        raise SystemExit(
            "--only is required; unscoped DS9 engine rebuilds are forbidden"
        )
    _require_explicit_model_root()
    specs = _specs(include_mapanything=bool(args.include_mapanything))
    reviewed = set(load_source_contracts(SOURCE_CONTRACTS))
    known = {spec.name for spec in specs if spec.name in reviewed}
    unknown = sorted(selected - known)
    if unknown:
        raise SystemExit(
            f"unknown DS9 engine selection(s): {', '.join(unknown)}; "
            f"valid names: {', '.join(sorted(known))}"
        )
    selected_specs = [spec for spec in specs if spec.name in selected]
    for spec in selected_specs:
        _preflight_selected_spec(spec)
    lock_path = DS9_ENGINES / ".noesis-ds9-engine-maintenance.lock"
    with engine_maintenance_lock(lock_path, dry_run=bool(args.dry_run)):
        for spec in selected_specs:
            _build(
                spec,
                trtexec=trtexec,
                dry_run=bool(args.dry_run),
                validate_load=bool(args.validate_load),
                evidence_root=args.evidence_root,
            )

    print("[OK] DS9 engine rebuild complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
