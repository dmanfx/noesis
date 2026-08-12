#!/usr/bin/env python3
"""Build the DS9 tracker-internal TAO ReID engine through the NvMOT API."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Sequence

DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (str(SCRIPT_DIR), str(DS9_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from engine_maintenance_common import (  # noqa: E402
    EngineMaintenanceError,
    EngineMaintenanceRun,
    copy_regular_file_exclusive,
    engine_maintenance_lock,
    maintenance_provenance_from_environment,
    new_run_id,
    require_absent_candidate_path,
    sha256_file,
    validate_maintenance_build_contract,
    validate_prepared_transaction_authority,
    validate_source_contract,
)

from noesis.v3dt_assets import (  # noqa: E402
    BODYPOSE_ENGINE,
    BODYPOSE_SOURCE,
    TRACKER_REID_ENGINE,
    TRACKER_REID_SOURCE,
    V3DTAssetBundle,
    derive_nvmot_tracker_engine_path,
    materialize_v3dt_tracker_build_config,
    validate_v3dt_assets,
)


TRACKER_CONFIG = DS9_ROOT / "config" / "v3dt" / "nvtracker_v3dt.yaml"
PIPELINE_CONFIG = DS9_ROOT / "config" / "infer_v3dt.yaml"
TRACKER_SOURCE = TRACKER_REID_SOURCE
TRACKER_ENGINE = TRACKER_REID_ENGINE
HELPER_SOURCE = (
    DS9_ROOT / "csrc" / "v3dt_engine_builder" / "v3dt_tracker_engine_builder.cpp"
)
SOURCE_CONTRACTS = DS9_ROOT / "config" / "engine_source_contracts.json"
TRACKER_BATCH_SIZE = 32
TRACKER_GPU_ID = 0
TRACKER_NETWORK_MODE = 1


def _print_run(command: Sequence[str | Path]) -> None:
    print("[RUN]", " ".join(str(value) for value in command))


def _required_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")


def _tracker_library(deepstream_home: Path) -> Path:
    return deepstream_home / "lib" / "libnvds_nvmultiobjecttracker.so"


def _temporary_tracker_config(
    bundle: V3DTAssetBundle,
    destination: Path,
    *,
    output_root: Path,
    staged_source: Path,
    generated_engine: Path,
) -> None:
    materialize_v3dt_tracker_build_config(
        bundle,
        destination,
        output_root=output_root,
        tracker_reid_source=staged_source,
        tracker_reid_generated_engine=generated_engine,
        gpu_id=TRACKER_GPU_ID,
        bodypose_engine=BODYPOSE_ENGINE,
    )


def _sdk_source_inventory(
    root: Path, *, expected_names: set[str]
) -> dict[str, object]:
    root = root.expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise EngineMaintenanceError(f"NvMOT SDK source workspace is unsafe: {root}")
    root_info = root.stat()
    if root_info.st_uid != os.getuid() or stat.S_IMODE(root_info.st_mode) != 0o700:
        raise EngineMaintenanceError(
            f"NvMOT SDK source workspace must be owner-private: {root}"
        )
    rows: list[dict[str, object]] = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        info = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_size <= 0
        ):
            raise EngineMaintenanceError(
                "NvMOT SDK source workspace contains a symlink, nonregular, "
                f"foreign-owned, hardlinked, or empty entry: {path}"
            )
        rows.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "size_bytes": info.st_size,
                "sha256": sha256_file(path),
                "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                "uid": info.st_uid,
                "gid": info.st_gid,
                "nlink": info.st_nlink,
            }
        )
    observed_names = {str(row["name"]) for row in rows}
    if observed_names != expected_names or len(rows) != len(expected_names):
        raise EngineMaintenanceError(
            "NvMOT SDK source workspace inventory differs from the exact contract: "
            f"expected={sorted(expected_names)} observed={sorted(observed_names)}"
        )
    return {"root": str(root.resolve()), "entries": rows}


def build_tracker_engine(
    *,
    dry_run: bool,
    validate_load: bool,
    evidence_root: Path | None = None,
    transaction_manifest: Path | None = None,
    expected_transaction_sha256: str | None = None,
) -> None:
    prepared_authority: dict[str, object] | None = None
    if not dry_run:
        if transaction_manifest is None or not expected_transaction_sha256:
            raise RuntimeError(
                "real DS9 tracker builds require a prepared host transaction"
            )
        prepared = validate_prepared_transaction_authority(
            transaction_manifest=transaction_manifest,
            expected_sha256=expected_transaction_sha256,
            engine_name="v3dt_tracker_reid",
            engine_target=TRACKER_ENGINE,
        )
        prepared_authority = {
            "transaction_id": prepared["transaction_id"],
            "transaction_sha256": expected_transaction_sha256,
            "transaction_manifest": str(transaction_manifest.expanduser().absolute()),
        }
    deepstream_home = Path(
        os.environ.get(
            "NOESIS_DEEPSTREAM_HOME", "/opt/nvidia/deepstream/deepstream-9.1"
        )
    ).expanduser()
    tracker_library = _tracker_library(deepstream_home)
    for path, label in (
        (PIPELINE_CONFIG, "DS9 V3DT pipeline config"),
        (TRACKER_CONFIG, "DS9 V3DT tracker config"),
        (TRACKER_SOURCE, "DS9 tracker ReID source"),
        (BODYPOSE_SOURCE, "DS9 BodyPose3DNet source"),
        (HELPER_SOURCE, "DS9 NvMOT build helper source"),
    ):
        _required_file(path, label)
    bundle = validate_v3dt_assets(PIPELINE_CONFIG, require_engines=False)
    tracker_source_contract = validate_source_contract(
        "v3dt_tracker_reid", TRACKER_SOURCE, SOURCE_CONTRACTS
    )
    validate_source_contract("bodypose3dnet", BODYPOSE_SOURCE, SOURCE_CONTRACTS)
    build_contract = {
        "builder": "NvMOT",
        "precision": "fp16",
        "streams": 3,
        "width": 1920,
        "height": 1080,
        "gpu_id": TRACKER_GPU_ID,
        "tensor_contract": {
            **dict(tracker_source_contract["tensor_contract"])
        },
    }
    validate_maintenance_build_contract(
        "v3dt_tracker_reid", build_contract, SOURCE_CONTRACTS
    )
    if not dry_run:
        _required_file(BODYPOSE_ENGINE, "DS9 BodyPose3DNet engine (build it first)")
        _required_file(tracker_library, "DeepStream 9 tracker library")

    cxx = shutil.which(os.environ.get("CXX", "c++"))
    trtexec_name = os.environ.get("TRTEXEC", "trtexec")
    trtexec = shutil.which(trtexec_name)
    if not cxx:
        raise FileNotFoundError("C++ compiler not found on PATH")
    if not dry_run and not trtexec:
        raise FileNotFoundError("trtexec not found on PATH")

    # Kept for CLI compatibility. Real builds always validate the candidate
    # and independently validate the installed target.
    _ = validate_load

    plan_root = Path("/tmp/noesis-ds9-v3dt-engine-plan")
    if dry_run:
        work_root = plan_root
        helper = work_root / "v3dt_tracker_engine_builder"
        generated_config = work_root / "nvtracker_v3dt_build.yaml"
        sdk_source_root = work_root / "sdk-source"
        staged_source = sdk_source_root / TRACKER_SOURCE.name
        generated_engine = derive_nvmot_tracker_engine_path(
            staged_source,
            batch_size=TRACKER_BATCH_SIZE,
            gpu_id=TRACKER_GPU_ID,
            network_mode=TRACKER_NETWORK_MODE,
        )
        temporary_engine = TRACKER_ENGINE.with_name(
            f".{TRACKER_ENGINE.name}.building-<run-id>"
        )
        _print_run([trtexec or trtexec_name, "--help"])
        _print_run(
            [
                cxx,
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{deepstream_home / 'sources' / 'includes'}",
                "-I/usr/local/cuda/include",
                str(HELPER_SOURCE),
                "-ldl",
                "-o",
                str(helper),
            ]
        )
        print(f"[PLAN] exclusively stage locked ETLT {TRACKER_SOURCE} -> {staged_source}")
        print(
            f"[PLAN] inventory {sdk_source_root}; require only {staged_source.name}"
        )
        print(
            f"[PLAN] materialize {generated_config} with NvMOT output {generated_engine}"
        )
        _print_run(
            [
                helper,
                "--tracker-config",
                generated_config,
                "--tracker-lib",
                tracker_library,
                "--streams",
                "3",
                "--width",
                "1920",
                "--height",
                "1080",
                "--gpu-id",
                str(TRACKER_GPU_ID),
            ]
        )
        print(
            f"[PLAN] inventory {sdk_source_root}; require only the staged ETLT and {generated_engine.name}"
        )
        print(
            f"[PLAN] exclusively adopt {generated_engine} -> transaction candidate {temporary_engine}"
        )
        _print_run(
            [
                trtexec or trtexec_name,
                f"--loadEngine={temporary_engine}",
                "--skipInference",
            ]
        )
        print(
            f"[PLAN] preserve prior bytes and atomically install {temporary_engine} -> {TRACKER_ENGINE}"
        )
        _print_run(
            [
                trtexec or trtexec_name,
                f"--loadEngine={TRACKER_ENGINE}",
                "--skipInference",
            ]
        )
        print("[PLAN] require positive candidate/final load markers and durable manifest evidence")
        return

    TRACKER_ENGINE.parent.mkdir(parents=True, exist_ok=True)
    lock_path = TRACKER_ENGINE.parent / ".noesis-ds9-engine-maintenance.lock"
    with engine_maintenance_lock(lock_path, dry_run=False):
        with tempfile.TemporaryDirectory(prefix="noesis-ds9-v3dt-engine-") as raw_work:
            work_root = Path(raw_work)
            helper = work_root / "v3dt_tracker_engine_builder"
            generated_config = work_root / "nvtracker_v3dt_build.yaml"
            sdk_source_root = work_root / "sdk-source"
            sdk_source_root.mkdir(mode=0o700)
            os.chmod(sdk_source_root, 0o700)
            staged_source = sdk_source_root / TRACKER_SOURCE.name
            staged_source_copy = copy_regular_file_exclusive(
                TRACKER_SOURCE,
                staged_source,
                expected_sha256=str(tracker_source_contract["sha256"]),
            )
            generated_engine = derive_nvmot_tracker_engine_path(
                staged_source,
                batch_size=TRACKER_BATCH_SIZE,
                gpu_id=TRACKER_GPU_ID,
                network_mode=TRACKER_NETWORK_MODE,
            )
            require_absent_candidate_path(generated_engine)
            pre_build_inventory = _sdk_source_inventory(
                sdk_source_root,
                expected_names={staged_source.name},
            )
            run_id = new_run_id()
            temporary_engine = TRACKER_ENGINE.with_name(
                f".{TRACKER_ENGINE.name}.building-{run_id}"
            )
            require_absent_candidate_path(temporary_engine)
            _temporary_tracker_config(
                bundle,
                generated_config,
                output_root=work_root,
                staged_source=staged_source,
                generated_engine=generated_engine,
            )
            run = EngineMaintenanceRun(
                name="v3dt_tracker_reid",
                target=TRACKER_ENGINE,
                evidence_root=evidence_root
                or (TRACKER_ENGINE.parent.parent / "engine_maintenance"),
                inputs={
                    "pipeline_config": PIPELINE_CONFIG,
                    "tracker_config": TRACKER_CONFIG,
                    "generated_tracker_config": generated_config,
                    "tracker_reid_source": TRACKER_SOURCE,
                    "staged_tracker_reid_source": staged_source,
                    "bodypose_source": BODYPOSE_SOURCE,
                    "bodypose_engine": BODYPOSE_ENGINE,
                    "builder_source": HELPER_SOURCE,
                    "tracker_library": tracker_library.resolve(),
                    "source_contracts": SOURCE_CONTRACTS,
                    "tracker_build_implementation": Path(__file__).resolve(),
                    "maintenance_implementation": SCRIPT_DIR
                    / "engine_maintenance_common.py",
                },
                repo_root=REPO_ROOT,
                run_id=run_id,
                metadata={
                    "platform": maintenance_provenance_from_environment(),
                    "build_contract": build_contract,
                    "host_transaction": prepared_authority,
                    "staged_tracker_reid_source_copy": staged_source_copy,
                },
            )
            try:
                run.record_evidence("sdk_source_pre_build", pre_build_inventory)
                run.run_command(
                    "probe-trtexec",
                    [trtexec or trtexec_name, "--help"],
                    proof="trtexec_probe",
                    timeout_seconds=30,
                )
                run.run_command(
                    "compile-helper",
                    [
                        cxx,
                        "-std=c++17",
                        "-O2",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        f"-I{deepstream_home / 'sources' / 'includes'}",
                        "-I/usr/local/cuda/include",
                        str(HELPER_SOURCE),
                        "-ldl",
                        "-o",
                        str(helper),
                    ],
                    proof="generic",
                    timeout_seconds=120,
                )
                run.run_command(
                    "build-tracker-engine",
                    [
                        helper,
                        "--tracker-config",
                        generated_config,
                        "--tracker-lib",
                        tracker_library,
                        "--streams",
                        "3",
                        "--width",
                        "1920",
                        "--height",
                        "1080",
                        "--gpu-id",
                        str(TRACKER_GPU_ID),
                    ],
                    proof="generic",
                    timeout_seconds=900,
                )
                post_build_inventory = _sdk_source_inventory(
                    sdk_source_root,
                    expected_names={staged_source.name, generated_engine.name},
                )
                run.record_evidence("sdk_source_post_build", post_build_inventory)
                run.adopt_derived_candidate(
                    generated_engine,
                    temporary_engine,
                    workspace_root=sdk_source_root,
                )
                run.record_candidate(temporary_engine)
                run.run_command(
                    "load-candidate",
                    [
                        trtexec or trtexec_name,
                        f"--loadEngine={temporary_engine}",
                        "--skipInference",
                    ],
                    proof="trtexec_load",
                    timeout_seconds=120,
                )
                run.revalidate_inputs()
                run.install_candidate(temporary_engine)
                try:
                    run.run_command(
                        "load-installed",
                        [
                            trtexec or trtexec_name,
                            f"--loadEngine={TRACKER_ENGINE}",
                            "--skipInference",
                        ],
                        proof="trtexec_load",
                        timeout_seconds=120,
                    )
                except BaseException as final_load_error:
                    run.rollback_after_install_failure(final_load_error)
                    raise
                run.complete()
            except BaseException as exc:
                run.fail(exc)
                raise
            print(f"[EVIDENCE] {run.manifest_path}")
    print(f"[OK] installed DS9 tracker ReID engine: {TRACKER_ENGINE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-load",
        action="store_true",
        help="Compatibility flag; guarded real builds always validate candidate and installed paths.",
    )
    parser.add_argument("--transaction-manifest", type=Path)
    parser.add_argument("--expected-transaction-sha256")
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=TRACKER_ENGINE.parent.parent / "engine_maintenance",
    )
    args = parser.parse_args()
    build_tracker_engine(
        dry_run=bool(args.dry_run),
        validate_load=bool(args.validate_load),
        evidence_root=args.evidence_root,
        transaction_manifest=args.transaction_manifest,
        expected_transaction_sha256=args.expected_transaction_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
