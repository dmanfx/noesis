#!/usr/bin/env python3
"""Prepare and inspect isolated DS9 MapAnything quality candidates.

This tool never edits the active runtime config, canonical engine, asset
manifest, or engine-source authority. GPU work is represented as exact command
vectors unless an operator explicitly runs those commands.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DS9_ROOT = SCRIPT_DIR.parent
REPO_ROOT = DS9_ROOT.parent
for import_root in (DS9_ROOT, SCRIPT_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine_maintenance_common import (  # noqa: E402
    DS9_TRTEXEC_BANNER,
    EngineMaintenanceError,
    engine_maintenance_lock,
    input_bundle_record,
    onnx_contract,
    reject_symlink_ancestors,
    sha256_file,
    validate_trtexec_probe,
)
from noesis.mapanything_profiles import (  # noqa: E402
    MapAnythingProfile,
    MapAnythingProfileError,
    get_mapanything_profile,
)
from noesis_core.strict_json import StrictJSONError, strict_json_loads  # noqa: E402


SOURCE_CONTRACT = "noesis.ds9.mapanything_candidate_source.v1"
FIXTURE_CONTRACT = "noesis.ds9.mapanything_raw_fixture.v1"
PLAN_CONTRACT = "noesis.ds9.mapanything_hr0_plan.v1"
OUTPUT_SUMMARY_CONTRACT = "noesis.ds9.mapanything_candidate_output_summary.v1"
GUARDED_RUN_CONTRACT = "noesis.ds9.mapanything_hr0_guarded_run.v1"
MAPANYTHING_HF_MODEL_ID = "facebook/map-anything-apache"
MAPANYTHING_HF_REVISION = "4cf3561e403dcec91b41629f0ce7793e3f04d15c"
MAPANYTHING_SOURCE_VERSION = "1.1.3"
MAPANYTHING_SOURCE_COMMIT = "9d1db2dd728bd8a10e74b15d2eb646e1bf933791"
MIN_UNICEPTION_VERSION = "0.1.7"
MAX_TRTEXEC_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_COMMAND_LOG_BYTES = 256 * 1024 * 1024
GIB = 1024**3
ARTIFACT_GROWTH_BYTES = 16 * GIB
CACHE_GROWTH_BYTES = 8 * GIB
TEMP_GROWTH_BYTES = 24 * GIB
RESIDUAL_HEADROOM_BYTES = 8 * GIB
GPU_DEVICE_INDEX = 0
GUARDED_PHASE_TIMEOUT_S = {
    "export_onnx": 3_600,
    "prepare_functional_fixture": 300,
    "prepare_scene_fixture": 300,
    "inspect_source": 300,
    "build_candidate": 7_200,
    "deserialize_candidate": 600,
    "functional_inference": 900,
    "summarize_functional_output": 300,
    "benchmark_compute_only": 900,
    "benchmark_with_transfers": 900,
    "inspect_engine": 600,
}
PROTECTED_RUNTIME_AUTHORITIES = (
    DS9_ROOT / "config" / "infer.yaml",
    DS9_ROOT / "pipelines" / "config_infer_secondary_mapanything.ini",
    DS9_ROOT / "asset_manifest.yaml",
    DS9_ROOT / "config" / "engine_source_contracts.json",
)


def _workspace_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _logical_or_absolute(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _exclusive_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if stat.S_IMODE(path.stat().st_mode) != mode:
        raise RuntimeError(f"new file has unexpected permissions: {path}")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _command_plan(
    profile: MapAnythingProfile,
    *,
    mapanything_repo: Path,
    trtexec: str,
    artifact_root: Path | None = None,
    python_executable: str = "python3",
    export_device: str = "cuda",
) -> dict[str, Any]:
    onnx_path = profile.workspace_path("onnx", artifact_root=artifact_root)
    engine_path = profile.workspace_path("engine", artifact_root=artifact_root)
    fixture_path = profile.workspace_path(
        "functional_fixture",
        artifact_root=artifact_root,
    )
    evidence_root = (
        (artifact_root / "models" / "engine_maintenance")
        if artifact_root is not None
        else (DS9_ROOT / "build")
    ) / f"mapanything_{profile.name}_evidence"
    scene_fixture = evidence_root / "scene-b3.raw"
    source_contract = evidence_root / "source-contract.json"
    functional_output = evidence_root / "functional-output.json"
    functional_summary = evidence_root / "functional-summary.json"
    compute_times = evidence_root / "timings-compute-only.json"
    io_times = evidence_root / "timings-with-transfers.json"
    exporter = REPO_ROOT / "utils/onnx2trt/export_ma_onnx/export_to_onnx.py"
    profile_tool = Path(__file__).resolve()
    common_benchmark = [
        f"--loadEngine={engine_path}",
        "--warmUp=500",
        "--duration=10",
        "--avgRuns=10",
        "--percentile=50,90,95,99",
    ]
    commands = {
        "prepare_candidate_directories": [
            "install",
            "-d",
            "-m",
            "0700",
            str(engine_path.parent),
            str(evidence_root),
        ],
        "export_onnx": [
            python_executable,
            str(exporter),
            "--repo",
            str(mapanything_repo),
            "--outdir",
            str(onnx_path.parent),
            "--output-name",
            onnx_path.name,
            "--report-name",
            f"{onnx_path.stem}.export_report.txt",
            "--fail-if-output-exists",
            "--h",
            str(profile.input_height),
            "--w",
            str(profile.input_width),
            "--opset",
            str(profile.opset),
            "--hf-model-id",
            MAPANYTHING_HF_MODEL_ID,
            "--hf-revision",
            MAPANYTHING_HF_REVISION,
            "--device",
            export_device,
            "--skip-eager-smoke",
            "--no-fused-input",
            "--no-include-intrinsics",
            "--return-conf-mask",
            "--skip-ort",
            "--skip-simplify",
            "--skip-shape-inference",
        ],
        "prepare_functional_fixture": [
            python_executable,
            str(profile_tool),
            "prepare-fixture",
            "--profile",
            profile.name,
            "--image",
            "{FULL_RES_RGB}",
            "--output",
            str(fixture_path),
        ],
        "prepare_scene_fixture": [
            python_executable,
            str(profile_tool),
            "prepare-fixture",
            "--profile",
            profile.name,
            "--image",
            "{CAMERA_0_FULL_RES_RGB}",
            "--image",
            "{CAMERA_1_FULL_RES_RGB}",
            "--image",
            "{CAMERA_2_FULL_RES_RGB}",
            "--output",
            str(scene_fixture),
        ],
        "inspect_source": [
            python_executable,
            str(profile_tool),
            "inspect-source",
            "--profile",
            profile.name,
            "--onnx",
            str(onnx_path),
            "--output",
            str(source_contract),
        ],
        "build_candidate": [
            trtexec,
            f"--onnx={onnx_path}",
            *profile.trtexec_build_args,
            f"--saveEngine={engine_path}",
            "--skipInference",
        ],
        "deserialize_candidate": [
            trtexec,
            f"--loadEngine={engine_path}",
            "--skipInference",
        ],
        "functional_inference": [
            trtexec,
            f"--loadEngine={engine_path}",
            f"--loadInputs=images:{fixture_path}",
            "--iterations=1",
            "--duration=0",
            "--warmUp=0",
            "--avgRuns=1",
            "--dumpOutput",
            f"--exportOutput={functional_output}",
        ],
        "summarize_functional_output": [
            python_executable,
            str(profile_tool),
            "summarize-output",
            "--profile",
            profile.name,
            "--input",
            str(functional_output),
            "--output",
            str(functional_summary),
        ],
        "benchmark_compute_only": [
            trtexec,
            *common_benchmark,
            "--noDataTransfers",
            f"--exportTimes={compute_times}",
        ],
        "benchmark_with_transfers": [
            trtexec,
            *common_benchmark,
            f"--loadInputs=images:{scene_fixture}",
            f"--exportTimes={io_times}",
        ],
        "inspect_engine": [
            "polygraphy",
            "inspect",
            "model",
            str(engine_path),
            "--model-type=engine",
        ],
    }
    return {
        "contract": PLAN_CONTRACT,
        "profile": profile.as_dict(),
        "promotion_state": "isolated_candidate_only",
        "canonical_runtime_modified": False,
        "model_source": {
            "mapanything_version": MAPANYTHING_SOURCE_VERSION,
            "mapanything_commit": MAPANYTHING_SOURCE_COMMIT,
            "uniception_min_version": MIN_UNICEPTION_VERSION,
            "huggingface_model_id": MAPANYTHING_HF_MODEL_ID,
            "huggingface_revision": MAPANYTHING_HF_REVISION,
            "strict_checkpoint_load": True,
            "export_device": export_device,
        },
        "artifact_root": (
            _logical_or_absolute(artifact_root)
            if artifact_root is not None
            else "repo_local_DS9"
        ),
        "evidence_root": _logical_or_absolute(evidence_root),
        "commands": commands,
    }


class GuardedRunError(RuntimeError):
    """Raised before or during an isolated guarded HR candidate run."""


def _existing_ancestor(path: Path) -> Path:
    current = path.expanduser().resolve(strict=False)
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise GuardedRunError(f"no existing ancestor for {path}")
        current = parent
    return current


def _require_external_artifact_root(path: Path) -> Path:
    if not path.is_absolute():
        raise GuardedRunError("--artifact-root must be an absolute path")
    raw = path.expanduser()
    reject_symlink_ancestors(raw, "MapAnything artifact root")
    if raw.is_symlink():
        raise GuardedRunError("--artifact-root must not be a symlink")
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise GuardedRunError("--artifact-root must be an existing directory")
    info = resolved.stat()
    if info.st_uid != os.getuid() or not os.access(resolved, os.W_OK | os.X_OK):
        raise GuardedRunError(
            "--artifact-root must be caller-owned, writable, and searchable"
        )
    repo = REPO_ROOT.resolve()
    if resolved == repo or resolved.is_relative_to(repo):
        raise GuardedRunError(
            "guarded execution forbids a repo-local artifact root"
        )
    if info.st_dev == repo.stat().st_dev:
        raise GuardedRunError(
            "guarded execution requires an artifact root on a different "
            "filesystem from the checkout"
        )
    models_root = resolved / "models"
    if (
        models_root.is_symlink()
        or not models_root.is_dir()
        or models_root.stat().st_uid != os.getuid()
        or not os.access(models_root, os.W_OK | os.X_OK)
    ):
        raise GuardedRunError(
            "--artifact-root must contain a caller-owned writable models/ "
            "directory"
        )
    return resolved


def _capacity_evidence(
    *,
    artifact_root: Path,
    cache_root: Path,
    temp_root: Path,
) -> list[dict[str, object]]:
    requirements = (
        ("candidate_artifacts", artifact_root, ARTIFACT_GROWTH_BYTES),
        ("model_cache", cache_root, CACHE_GROWTH_BYTES),
        ("export_and_builder_temp", temp_root, TEMP_GROWTH_BYTES),
    )
    by_device: dict[int, dict[str, object]] = {}
    for label, path, required in requirements:
        ancestor = _existing_ancestor(path)
        device = ancestor.stat().st_dev
        row = by_device.setdefault(
            device,
            {
                "device": int(device),
                "sample_path": str(ancestor),
                "purposes": [],
                "growth_bytes": 0,
            },
        )
        row["purposes"].append(label)  # type: ignore[union-attr]
        row["growth_bytes"] = int(row["growth_bytes"]) + required
    evidence = []
    for device in sorted(by_device):
        row = by_device[device]
        free_bytes = shutil.disk_usage(str(row["sample_path"])).free
        required_bytes = int(row["growth_bytes"]) + RESIDUAL_HEADROOM_BYTES
        if free_bytes < required_bytes:
            raise GuardedRunError(
                "insufficient free space for guarded HR-0 work: "
                f"device={device} free={free_bytes} required={required_bytes} "
                f"purposes={row['purposes']}"
            )
        evidence.append(
            {
                **row,
                "residual_headroom_bytes": RESIDUAL_HEADROOM_BYTES,
                "required_free_bytes": required_bytes,
                "observed_free_bytes": free_bytes,
            }
        )
    return evidence


def _guarded_output_paths(
    profile: MapAnythingProfile,
    *,
    artifact_root: Path,
    evidence_root: Path,
    temp_root: Path,
) -> tuple[Path, ...]:
    onnx_path = profile.workspace_path("onnx", artifact_root=artifact_root)
    fixture_path = profile.workspace_path(
        "functional_fixture",
        artifact_root=artifact_root,
    )
    return (
        onnx_path.parent,
        profile.workspace_path("engine", artifact_root=artifact_root),
        fixture_path.parent,
        evidence_root,
        temp_root,
    )


def _require_outputs_absent(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise GuardedRunError(
            "refusing existing HR-0 output path(s): "
            + ", ".join(str(path) for path in existing)
        )


def _protected_authority_snapshot() -> dict[str, str]:
    missing = [path for path in PROTECTED_RUNTIME_AUTHORITIES if not path.is_file()]
    if missing:
        raise GuardedRunError(
            "protected runtime authority is missing: "
            + ", ".join(str(path) for path in missing)
        )
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in PROTECTED_RUNTIME_AUTHORITIES
    }


def _require_protected_authorities_unchanged(expected: Mapping[str, str]) -> None:
    observed = _protected_authority_snapshot()
    if observed != dict(expected):
        raise GuardedRunError(
            "a deployment selector, canonical inference config, manifest, or "
            "source authority changed during the guarded candidate run"
        )


def _resolve_executable(
    value: str,
    label: str,
    *,
    preserve_invocation_path: bool = False,
) -> Path:
    raw = str(value or "").strip()
    candidate = Path(raw).expanduser()
    resolved_raw = (
        str(candidate.absolute())
        if candidate.is_absolute() or candidate.parent != Path(".")
        else shutil.which(raw)
    )
    if not resolved_raw:
        raise GuardedRunError(f"unable to resolve {label}: {raw!r}")
    invocation_path = Path(resolved_raw).absolute()
    resolved = invocation_path.resolve(strict=True)
    if not resolved.is_file() or not os.access(invocation_path, os.X_OK):
        raise GuardedRunError(f"{label} is not an executable regular file")
    return invocation_path if preserve_invocation_path else resolved


def _probe_trtexec_toolchain(trtexec: Path) -> dict[str, str]:
    """Require the exact DS9 TensorRT builder ABI before candidate work."""

    try:
        result = subprocess.run(
            [str(trtexec), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardedRunError(f"unable to probe trtexec toolchain: {exc}") from exc
    transcript = result.stdout or ""
    try:
        validate_trtexec_probe(result.returncode, transcript)
    except EngineMaintenanceError as exc:
        raise GuardedRunError(
            "MapAnything candidate builder does not match the DS9 TensorRT "
            f"toolchain ({DS9_TRTEXEC_BANNER} required): {exc}"
        ) from exc
    banner_line = next(
        (
            line.strip()
            for line in transcript.splitlines()
            if DS9_TRTEXEC_BANNER in line
        ),
        DS9_TRTEXEC_BANNER,
    )
    return {
        "expected_banner": DS9_TRTEXEC_BANNER,
        "observed_banner_line": banner_line,
        "probe_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
    }


def _probe_mapanything_source_environment(
    *,
    mapanything_repo: Path,
    python_executable: Path,
) -> dict[str, str]:
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(mapanything_repo), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(mapanything_repo),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardedRunError(
            f"unable to inspect MapAnything source revision: {exc}"
        ) from exc
    commit = (commit_result.stdout or "").strip()
    if commit_result.returncode != 0 or commit != MAPANYTHING_SOURCE_COMMIT:
        raise GuardedRunError(
            "MapAnything export requires official source commit "
            f"{MAPANYTHING_SOURCE_COMMIT}; found {commit or 'unavailable'}"
        )
    if status_result.returncode != 0:
        raise GuardedRunError(
            "unable to inspect MapAnything source worktree state: "
            + (status_result.stdout or "").strip()
        )
    if (status_result.stdout or "").strip():
        raise GuardedRunError(
            "MapAnything export source has tracked modifications; use a clean "
            f"checkout of {MAPANYTHING_SOURCE_COMMIT}"
        )

    probe = r"""
import importlib.metadata
import inspect
import json
import re
import sys
import tomllib
from pathlib import Path

repo = Path(sys.argv[1]).resolve(strict=True)
expected_source_version = sys.argv[2]
minimum_uniception = sys.argv[3]
with (repo / "pyproject.toml").open("rb") as handle:
    source_version = str(tomllib.load(handle)["project"]["version"])
if source_version != expected_source_version:
    raise RuntimeError(
        f"MapAnything source version {source_version} does not match "
        f"{expected_source_version}"
    )
sys.path.insert(0, str(repo))
from mapanything.models import MapAnything
from uniception.models.encoders.dinov2 import DINOv2Encoder

model_path = Path(inspect.getfile(MapAnything)).resolve()
if not model_path.is_relative_to(repo):
    raise RuntimeError(
        f"MapAnything imported from {model_path}, outside requested source {repo}"
    )
uniception_version = importlib.metadata.version("uniception")
python_prefix = Path(sys.prefix).resolve()
base_prefix = Path(sys.base_prefix).resolve()
if python_prefix == base_prefix:
    raise RuntimeError(
        "MapAnything export Python must be a task-isolated virtual environment"
    )
def version_tuple(value):
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise RuntimeError(f"invalid dependency version {value!r}")
    return tuple(int(item) for item in match.groups())
if version_tuple(uniception_version) < version_tuple(minimum_uniception):
    raise RuntimeError(
        f"UniCeption {uniception_version} is older than "
        f"{minimum_uniception}"
    )
model_parameters = set(inspect.signature(MapAnything.__init__).parameters)
required_model_parameters = {
    "use_register_tokens_from_encoder",
    "info_sharing_mlp_layer_str",
}
missing_model_parameters = sorted(required_model_parameters - model_parameters)
if missing_model_parameters:
    raise RuntimeError(
        "MapAnything source is missing checkpoint parameters: "
        + ", ".join(missing_model_parameters)
    )
encoder_parameters = set(inspect.signature(DINOv2Encoder.__init__).parameters)
required_encoder_parameters = {
    "norm_returned_features",
    "torch_hub_pretrained",
}
missing_encoder_parameters = sorted(
    required_encoder_parameters - encoder_parameters
)
if missing_encoder_parameters:
    raise RuntimeError(
        "UniCeption source is missing checkpoint parameters: "
        + ", ".join(missing_encoder_parameters)
    )
print(json.dumps({
    "mapanything_version": source_version,
    "mapanything_module": str(model_path),
    "uniception_version": uniception_version,
    "python": str(Path(sys.executable).absolute()),
    "python_prefix": str(python_prefix),
}, sort_keys=True))
"""
    try:
        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                probe,
                str(mapanything_repo),
                MAPANYTHING_SOURCE_VERSION,
                MIN_UNICEPTION_VERSION,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardedRunError(
            f"unable to probe MapAnything export environment: {exc}"
        ) from exc
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        raise GuardedRunError(
            "MapAnything export environment is incompatible: "
            + (output or f"probe exited {result.returncode}")
        )
    try:
        payload = strict_json_loads(
            output.encode("utf-8"),
            label="MapAnything export environment probe",
        )
    except StrictJSONError as exc:
        raise GuardedRunError(
            "MapAnything export environment probe returned invalid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise GuardedRunError(
            "MapAnything export environment probe returned a non-object"
        )
    return {
        "mapanything_version": str(payload["mapanything_version"]),
        "mapanything_commit": commit,
        "mapanything_module": str(payload["mapanything_module"]),
        "uniception_version": str(payload["uniception_version"]),
        "python": str(payload["python"]),
        "python_prefix": str(payload["python_prefix"]),
        "huggingface_model_id": MAPANYTHING_HF_MODEL_ID,
        "huggingface_revision": MAPANYTHING_HF_REVISION,
        "checkpoint_load": "strict",
    }


def _query_compute_owners(nvidia_smi: Path) -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            [
                str(nvidia_smi),
                f"--id={GPU_DEVICE_INDEX}",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardedRunError(f"unable to query GPU compute owners: {exc}") from exc
    if result.returncode != 0:
        raise GuardedRunError(
            "unable to query GPU compute owners: "
            + (result.stdout or "").strip()
        )
    owners: list[dict[str, object]] = []
    for raw_row in (result.stdout or "").splitlines():
        if not raw_row.strip() or raw_row.lower().startswith("no running"):
            continue
        values = [item.strip() for item in raw_row.split(",", 2)]
        if len(values) != 3 or not values[0].isascii() or not values[0].isdigit():
            raise GuardedRunError(
                f"unexpected GPU compute-owner row: {raw_row!r}"
            )
        pid = int(values[0])
        if pid <= 0:
            raise GuardedRunError(
                f"unexpected GPU compute-owner PID: {values[0]!r}"
            )
        owners.append(
            {
                "pid": pid,
                "process_name": values[1],
                "used_memory_mib": values[2],
            }
        )
    return owners


def _process_parent_pid(pid: int) -> int | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    closing = raw.rfind(")")
    if closing < 0:
        return None
    fields = raw[closing + 1 :].split()
    if len(fields) < 2 or not fields[1].isdigit():
        return None
    return int(fields[1])


def _is_process_descendant(pid: int, ancestor_pid: int) -> bool:
    current = int(pid)
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == ancestor_pid:
            return True
        seen.add(current)
        parent = _process_parent_pid(current)
        if parent is None:
            return False
        current = parent
    return current == ancestor_pid


def _query_conflicting_processes(
    *,
    allowed_root_pid: int | None = None,
) -> list[dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    self_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isascii() or not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid or (
            allowed_root_pid is not None
            and _is_process_descendant(pid, allowed_root_pid)
        ):
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        args = [
            item.decode("utf-8", errors="replace")
            for item in raw.split(b"\0")
            if item
        ]
        if not args:
            continue
        reason = _conflicting_process_reason(args)
        if reason:
            conflicts.append(
                {
                    "pid": pid,
                    "reason": reason,
                    "argv0": args[0],
                }
            )
    return sorted(conflicts, key=lambda row: int(row["pid"]))


def _conflicting_process_reason(args: Sequence[str]) -> str:
    """Classify actual build/export invocations, not paths mentioned by tools."""

    if not args:
        return ""
    executable = Path(args[0]).name
    if executable == "trtexec":
        return "trtexec"

    script = executable
    if executable.startswith("python"):
        # The guarded commands invoke Python files directly. Permit common
        # flag-only prefixes, but stop at -c/-m because their following value
        # is code or a module rather than a script path.
        script = ""
        for item in args[1:]:
            if item in {"-c", "-m"}:
                break
            if item == "--":
                continue
            if item.startswith("-"):
                continue
            script = Path(item).name
            break

    if script == "export_to_onnx.py":
        return "mapanything_export"
    if script == "mapanything_profile_tool.py" and "guarded-run" in args:
        return "mapanything_guarded_run"
    return ""


def _unexpected_compute_owners(
    owners: Sequence[Mapping[str, object]],
    *,
    allowed_root_pid: int | None = None,
) -> list[dict[str, object]]:
    unexpected = []
    for owner in owners:
        pid = int(owner["pid"])
        if allowed_root_pid is not None and _is_process_descendant(
            pid, allowed_root_pid
        ):
            continue
        unexpected.append(dict(owner))
    return unexpected


def _require_idle_gpu_and_processes(
    *,
    nvidia_smi: Path,
    allowed_root_pid: int | None = None,
) -> dict[str, object]:
    owners = _query_compute_owners(nvidia_smi)
    unexpected = _unexpected_compute_owners(
        owners,
        allowed_root_pid=allowed_root_pid,
    )
    if unexpected:
        raise GuardedRunError(
            f"GPU compute owner(s) are active: {unexpected}"
        )
    conflicts = _query_conflicting_processes(
        allowed_root_pid=allowed_root_pid
    )
    if conflicts:
        raise GuardedRunError(
            f"conflicting export/build process(es) are active: {conflicts}"
        )
    return {"compute_owners": [], "conflicting_processes": []}


def _probe_lock_available(lock_path: Path) -> dict[str, object]:
    if not lock_path.exists() and not lock_path.is_symlink():
        return {"path": str(lock_path), "state": "absent_will_create_on_execute"}
    if lock_path.is_symlink() or not lock_path.is_file():
        raise GuardedRunError("artifact transaction lock is not a regular file")
    flags = os.O_RDWR
    for flag_name in ("O_CLOEXEC", "O_NOFOLLOW"):
        flags |= getattr(os, flag_name, 0)
    descriptor = os.open(lock_path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise GuardedRunError(
                "artifact transaction lock must be caller-owned, single-link, "
                "and mode 0600"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GuardedRunError(
                "another DS9 artifact transaction owns the maintenance lock "
                f"{lock_path}"
            ) from exc
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
    return {"path": str(lock_path), "state": "available"}


def _coordination_lock_paths(artifact_root: Path) -> tuple[Path, ...]:
    transaction_lock = artifact_root / ".noesis-ds9-artifact-transaction.lock"
    discovered = {
        path
        for path in artifact_root.iterdir()
        if path.name.endswith(".lock") and (path.is_file() or path.is_symlink())
    }
    discovered.add(transaction_lock)
    return tuple(sorted(discovered, key=lambda path: path.name))


def _probe_coordination_locks(
    artifact_root: Path,
) -> list[dict[str, object]]:
    return [
        _probe_lock_available(path)
        for path in _coordination_lock_paths(artifact_root)
    ]


def _validate_fixture_sources(
    profile: MapAnythingProfile,
    *,
    functional_image: Path,
    scene_images: Sequence[Path],
) -> list[dict[str, object]]:
    import cv2

    paths = (functional_image, *scene_images)
    if len(scene_images) != profile.batch_size:
        raise GuardedRunError(
            f"guarded HR-0 run requires exactly {profile.batch_size} scene images"
        )
    records = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        if path.is_symlink() or not resolved.is_file():
            raise GuardedRunError(
                f"fixture source must be a non-symlink regular file: {path}"
            )
        image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise GuardedRunError(f"unable to decode fixture source: {path}")
        height, width = image.shape[:2]
        if height < profile.input_height or width < profile.input_width:
            raise GuardedRunError(
                f"fixture source {path} is too small for {profile.name}"
            )
        records.append(
            {
                "path": _logical_or_absolute(resolved),
                "sha256": sha256_file(resolved),
                "width": int(width),
                "height": int(height),
            }
        )
    return records


def _guarded_preflight(
    profile: MapAnythingProfile,
    *,
    artifact_root: Path,
    cache_root: Path,
    temp_root: Path,
    mapanything_repo: Path,
    trtexec: str,
    functional_image: Path,
    scene_images: Sequence[Path],
    check_live_owners: bool = True,
    python: str = sys.executable,
    export_device: str = "cuda",
) -> dict[str, Any]:
    if profile.status != "candidate":
        raise GuardedRunError("guarded execution accepts candidate profiles only")
    artifact_root = _require_external_artifact_root(artifact_root)
    cache_root_raw = cache_root.expanduser()
    temp_root_raw = temp_root.expanduser()
    for label, path in (
        ("cache", cache_root_raw),
        ("temporary", temp_root_raw),
    ):
        reject_symlink_ancestors(path, f"MapAnything {label} root")
    cache_root = cache_root_raw.resolve(strict=False)
    temp_root = temp_root_raw.resolve(strict=False)
    for label, path in (("cache", cache_root), ("temporary", temp_root)):
        if path == REPO_ROOT.resolve() or path.is_relative_to(REPO_ROOT.resolve()):
            raise GuardedRunError(
                f"{label} root must stay outside the checkout"
            )
    repo = mapanything_repo.expanduser().resolve(strict=True)
    if not repo.is_dir() or not (repo / "mapanything").is_dir():
        raise GuardedRunError(
            "--mapanything-repo must contain the MapAnything Python package"
        )
    python_path = _resolve_executable(
        python,
        "MapAnything export Python",
        preserve_invocation_path=True,
    )
    model_source = _probe_mapanything_source_environment(
        mapanything_repo=repo,
        python_executable=python_path,
    )
    trtexec_path = _resolve_executable(trtexec, "trtexec")
    trtexec_toolchain = _probe_trtexec_toolchain(trtexec_path)
    polygraphy_path = _resolve_executable("polygraphy", "polygraphy")
    nvidia_smi_path = _resolve_executable("nvidia-smi", "nvidia-smi")
    plan = _command_plan(
        profile,
        mapanything_repo=repo,
        trtexec=str(trtexec_path),
        artifact_root=artifact_root,
        python_executable=str(python_path),
        export_device=export_device,
    )
    evidence_root = Path(plan["evidence_root"])
    outputs = _guarded_output_paths(
        profile,
        artifact_root=artifact_root,
        evidence_root=evidence_root,
        temp_root=temp_root,
    )
    _require_outputs_absent(outputs)
    capacity = _capacity_evidence(
        artifact_root=artifact_root,
        cache_root=cache_root,
        temp_root=temp_root,
    )
    fixture_sources = _validate_fixture_sources(
        profile,
        functional_image=functional_image,
        scene_images=scene_images,
    )
    locks = _probe_coordination_locks(artifact_root)
    owner_state = (
        _require_idle_gpu_and_processes(nvidia_smi=nvidia_smi_path)
        if check_live_owners
        else {"compute_owners": "not_queried", "conflicting_processes": "not_queried"}
    )
    protected = _protected_authority_snapshot()
    return {
        "contract": GUARDED_RUN_CONTRACT,
        "mode": "dry_run",
        "ready": True,
        "profile": profile.as_dict(),
        "artifact_root": str(artifact_root),
        "cache_root": str(cache_root),
        "temp_root": str(temp_root),
        "evidence_root": str(evidence_root),
        "locks": locks,
        "capacity": capacity,
        "fixture_sources": fixture_sources,
        "owner_state": owner_state,
        "model_source": model_source,
        "export_device": export_device,
        "trtexec_toolchain": trtexec_toolchain,
        "executables": {
            "python": str(python_path),
            "trtexec": str(trtexec_path),
            "polygraphy": str(polygraphy_path),
            "nvidia_smi": str(nvidia_smi_path),
        },
        "protected_runtime_authorities": protected,
        "canonical_runtime_modified": False,
        "plan": plan,
    }


def _replace_path_argument(
    command: Sequence[str],
    old_path: Path,
    new_path: Path,
) -> list[str]:
    old = str(old_path)
    new = str(new_path)
    return [str(item).replace(old, new) for item in command]


def _materialize_guarded_commands(
    preflight: Mapping[str, Any],
    *,
    functional_image: Path,
    scene_images: Sequence[Path],
    run_token: str,
) -> tuple[dict[str, list[str]], dict[str, Path]]:
    profile = get_mapanything_profile(str(preflight["profile"]["name"]))
    artifact_root = Path(str(preflight["artifact_root"]))
    evidence_root = Path(str(preflight["evidence_root"]))
    final_onnx = profile.workspace_path("onnx", artifact_root=artifact_root)
    final_engine = profile.workspace_path("engine", artifact_root=artifact_root)
    final_fixture = profile.workspace_path(
        "functional_fixture",
        artifact_root=artifact_root,
    )
    staging_onnx_dir = final_onnx.parent.with_name(
        f".{final_onnx.parent.name}.partial-{run_token}"
    )
    staging_onnx = staging_onnx_dir / final_onnx.name
    staging_engine = final_engine.with_name(
        f".{final_engine.name}.partial-{run_token}"
    )
    staging_fixture = final_fixture.with_name(
        f".{final_fixture.name}.partial-{run_token}"
    )
    commands = {
        name: [str(item) for item in command]
        for name, command in preflight["plan"]["commands"].items()
        if name != "prepare_candidate_directories"
    }
    for name in tuple(commands):
        commands[name] = _replace_path_argument(
            commands[name], final_onnx, staging_onnx
        )
        commands[name] = _replace_path_argument(
            commands[name], final_engine, staging_engine
        )
        commands[name] = _replace_path_argument(
            commands[name], final_fixture, staging_fixture
        )
        if name in {
            "export_onnx",
            "prepare_functional_fixture",
            "prepare_scene_fixture",
            "inspect_source",
            "summarize_functional_output",
        }:
            commands[name][0] = str(preflight["executables"]["python"])
        elif Path(commands[name][0]).name == "trtexec":
            commands[name][0] = str(preflight["executables"]["trtexec"])
        elif Path(commands[name][0]).name == "polygraphy":
            commands[name][0] = str(preflight["executables"]["polygraphy"])
    export = commands["export_onnx"]
    export[export.index("--outdir") + 1] = str(staging_onnx_dir)
    functional = commands["prepare_functional_fixture"]
    functional[functional.index("--image") + 1] = str(
        functional_image.expanduser().resolve(strict=True)
    )
    scene = commands["prepare_scene_fixture"]
    image_indices = [
        index + 1 for index, item in enumerate(scene) if item == "--image"
    ]
    if len(image_indices) != profile.batch_size:
        raise GuardedRunError("scene fixture command shape drifted")
    for index, path in zip(image_indices, scene_images):
        scene[index] = str(path.expanduser().resolve(strict=True))
    inspect_command = commands["inspect_source"]
    inspect_command[inspect_command.index("--output") + 1] = str(
        evidence_root / "source-contract.preinstall.json"
    )
    return commands, {
        "final_onnx_dir": final_onnx.parent,
        "final_onnx": final_onnx,
        "staging_onnx_dir": staging_onnx_dir,
        "staging_onnx": staging_onnx,
        "final_engine": final_engine,
        "staging_engine": staging_engine,
        "final_fixture": final_fixture,
        "staging_fixture": staging_fixture,
        "evidence_root": evidence_root,
    }


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _run_guarded_command(
    *,
    label: str,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    nvidia_smi: Path,
) -> dict[str, object]:
    timeout_s = GUARDED_PHASE_TIMEOUT_S[label]
    _require_idle_gpu_and_processes(nvidia_smi=nvidia_smi)
    started = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout_s:
                    raise GuardedRunError(
                        f"{label} exceeded its {timeout_s}s timeout"
                    )
                if log_path.stat().st_size > MAX_COMMAND_LOG_BYTES:
                    raise GuardedRunError(
                        f"{label} log exceeded {MAX_COMMAND_LOG_BYTES} bytes"
                    )
                _require_idle_gpu_and_processes(
                    nvidia_smi=nvidia_smi,
                    allowed_root_pid=process.pid,
                )
                time.sleep(0.5)
        except BaseException:
            _terminate_process(process)
            raise
        returncode = int(process.wait())
    if returncode != 0:
        raise GuardedRunError(
            f"{label} failed with exit {returncode}; see {log_path}"
        )
    _require_idle_gpu_and_processes(nvidia_smi=nvidia_smi)
    return {
        "label": label,
        "command": list(command),
        "returncode": returncode,
        "duration_s": time.monotonic() - started,
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def _install_regular_file_no_replace(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise GuardedRunError(f"candidate output is not a nonempty regular file: {source}")
    info = source.stat()
    if info.st_uid != os.getuid() or info.st_nlink != 1:
        raise GuardedRunError(
            f"candidate output must be caller-owned and single-link: {source}"
        )
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() or target.is_symlink():
        raise GuardedRunError(f"refusing to replace candidate output: {target}")
    try:
        os.link(source, target, follow_symlinks=False)
    except FileExistsError as exc:
        raise GuardedRunError(
            f"candidate output appeared during installation: {target}"
        ) from exc
    os.chmod(target, 0o600)
    with target.open("rb") as handle:
        os.fsync(handle.fileno())
    source.unlink()


def _install_onnx_bundle_no_replace(source_dir: Path, target_dir: Path) -> None:
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise GuardedRunError("staged ONNX bundle directory is invalid")
    entries = sorted(source_dir.iterdir(), key=lambda path: path.name)
    if not entries:
        raise GuardedRunError("staged ONNX bundle is empty")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise GuardedRunError(
            "staged ONNX bundle must contain only non-symlink regular files"
        )
    target_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        for source in entries:
            _install_regular_file_no_replace(source, target_dir / source.name)
    except BaseException:
        for target in target_dir.iterdir():
            if target.is_file() and not target.is_symlink():
                target.unlink()
        target_dir.rmdir()
        raise
    source_dir.rmdir()


def _execute_guarded_run(
    preflight: Mapping[str, Any],
    *,
    functional_image: Path,
    scene_images: Sequence[Path],
) -> dict[str, Any]:
    run_token = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    commands, paths = _materialize_guarded_commands(
        preflight,
        functional_image=functional_image,
        scene_images=scene_images,
        run_token=run_token,
    )
    artifact_root = Path(str(preflight["artifact_root"]))
    cache_root = Path(str(preflight["cache_root"]))
    temp_root = Path(str(preflight["temp_root"]))
    evidence_root = paths["evidence_root"]
    nvidia_smi = Path(str(preflight["executables"]["nvidia_smi"]))
    protected = dict(preflight["protected_runtime_authorities"])
    outputs = _guarded_output_paths(
        get_mapanything_profile(str(preflight["profile"]["name"])),
        artifact_root=artifact_root,
        evidence_root=evidence_root,
        temp_root=temp_root,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HOME": str(cache_root / "huggingface"),
            "HUGGINGFACE_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "TORCH_HOME": str(cache_root / "torch"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "TMPDIR": str(temp_root),
        }
    )
    command_receipts: list[dict[str, object]] = []
    with ExitStack() as coordination_locks:
        for lock_path in _coordination_lock_paths(artifact_root):
            coordination_locks.enter_context(
                engine_maintenance_lock(lock_path, dry_run=False)
            )
        _require_outputs_absent(outputs)
        _require_outputs_absent(
            (
                paths["staging_onnx_dir"],
                paths["staging_engine"],
                paths["staging_fixture"],
                paths["staging_fixture"].with_suffix(
                    paths["staging_fixture"].suffix + ".receipt.json"
                ),
            )
        )
        _capacity_evidence(
            artifact_root=artifact_root,
            cache_root=cache_root,
            temp_root=temp_root,
        )
        _require_idle_gpu_and_processes(nvidia_smi=nvidia_smi)
        _require_protected_authorities_unchanged(protected)
        for path in (
            paths["staging_onnx_dir"].parent,
            paths["staging_engine"].parent,
            paths["staging_fixture"].parent,
            cache_root,
            temp_root,
            evidence_root,
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path in (cache_root, temp_root, evidence_root):
            os.chmod(path, 0o700)
        try:
            for label in GUARDED_PHASE_TIMEOUT_S:
                receipt = _run_guarded_command(
                    label=label,
                    command=commands[label],
                    cwd=REPO_ROOT,
                    environment=environment,
                    log_path=evidence_root / f"{label}.log",
                    nvidia_smi=nvidia_smi,
                )
                command_receipts.append(receipt)
            _require_protected_authorities_unchanged(protected)
            _install_onnx_bundle_no_replace(
                paths["staging_onnx_dir"], paths["final_onnx_dir"]
            )
            _install_regular_file_no_replace(
                paths["staging_engine"], paths["final_engine"]
            )
            _install_regular_file_no_replace(
                paths["staging_fixture"], paths["final_fixture"]
            )
            staging_receipt = paths["staging_fixture"].with_suffix(
                paths["staging_fixture"].suffix + ".receipt.json"
            )
            final_receipt = paths["final_fixture"].with_suffix(
                paths["final_fixture"].suffix + ".receipt.json"
            )
            _install_regular_file_no_replace(staging_receipt, final_receipt)
            profile = get_mapanything_profile(str(preflight["profile"]["name"]))
            source_contract = inspect_source(profile, paths["final_onnx"])
            _exclusive_bytes(
                evidence_root / "source-contract.json",
                _json_bytes(source_contract),
            )
            _require_protected_authorities_unchanged(protected)
            result = {
                "contract": GUARDED_RUN_CONTRACT,
                "mode": "execute",
                "status": "complete_unpromoted_candidate",
                "profile": profile.name,
                "artifact_root": str(artifact_root),
                "cache_root": str(cache_root),
                "temp_root": str(temp_root),
                "model_source": dict(preflight["model_source"]),
                "commands": command_receipts,
                "outputs": {
                    "onnx": input_bundle_record(paths["final_onnx"]),
                    "engine": {
                        "path": str(paths["final_engine"]),
                        "size_bytes": paths["final_engine"].stat().st_size,
                        "sha256": sha256_file(paths["final_engine"]),
                    },
                    "functional_fixture": {
                        "path": str(paths["final_fixture"]),
                        "size_bytes": paths["final_fixture"].stat().st_size,
                        "sha256": sha256_file(paths["final_fixture"]),
                    },
                    "evidence_root": str(evidence_root),
                },
                "protected_runtime_authorities": protected,
                "canonical_runtime_modified": False,
                "promotion_state": "not_promoted",
            }
            _exclusive_bytes(
                evidence_root / "guarded-run-receipt.json",
                _json_bytes(result),
            )
            return result
        except BaseException as exc:
            failure = {
                "contract": GUARDED_RUN_CONTRACT,
                "mode": "execute",
                "status": "failed_closed_unpromoted_candidate",
                "profile": str(preflight["profile"]["name"]),
                "error": f"{type(exc).__name__}: {exc}",
                "model_source": dict(preflight["model_source"]),
                "commands": command_receipts,
                "canonical_runtime_modified": False,
                "promotion_state": "not_promoted",
            }
            failure_path = evidence_root / "guarded-run-failure.json"
            if not failure_path.exists() and not failure_path.is_symlink():
                _exclusive_bytes(failure_path, _json_bytes(failure))
            raise


def _shape_matches_batch(value: object, batch_size: int) -> bool:
    return value == "batch" or value == batch_size


def inspect_source(
    profile: MapAnythingProfile,
    onnx_path: Path,
) -> dict[str, Any]:
    bundle = input_bundle_record(onnx_path)
    contract = onnx_contract(onnx_path)
    opsets = contract.get("opsets")
    if (
        not isinstance(opsets, list)
        or {"domain": "ai.onnx", "version": profile.opset} not in opsets
    ):
        raise MapAnythingProfileError(
            f"{profile.name} ONNX must use ai.onnx opset {profile.opset}"
        )
    inputs = contract.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise MapAnythingProfileError(
            f"{profile.name} ONNX must expose exactly one input"
        )
    input_row = inputs[0]
    expected_input = [
        "batch",
        3,
        profile.input_height,
        profile.input_width,
    ]
    actual_input = input_row.get("shape") if isinstance(input_row, Mapping) else None
    if (
        not isinstance(input_row, Mapping)
        or input_row.get("name") != "images"
        or input_row.get("dtype") != "FLOAT"
        or not isinstance(actual_input, list)
        or len(actual_input) != 4
        or not _shape_matches_batch(actual_input[0], profile.batch_size)
        or actual_input[1:] != expected_input[1:]
    ):
        raise MapAnythingProfileError(
            f"{profile.name} ONNX input differs from {expected_input}"
        )
    outputs = contract.get("outputs")
    if not isinstance(outputs, list) or [
        row.get("name") if isinstance(row, Mapping) else None for row in outputs
    ] != list(profile.output_layers):
        raise MapAnythingProfileError(
            f"{profile.name} ONNX outputs must be depth/conf/mask"
        )
    for row in outputs:
        shape = row.get("shape")
        if (
            row.get("dtype") != "FLOAT"
            or not isinstance(shape, list)
            or len(shape) != 4
            or not _shape_matches_batch(shape[0], profile.batch_size)
            or shape[1] != 1
            or shape[2] != profile.input_height
            or shape[3] != profile.input_width
        ):
            raise MapAnythingProfileError(
                f"{profile.name} ONNX output tensor contract is invalid"
            )
    external_count = contract.get("external_initializer_count")
    if (
        isinstance(external_count, bool)
        or not isinstance(external_count, int)
        or external_count <= 0
    ):
        raise MapAnythingProfileError(
            f"{profile.name} ONNX lacks its external initializer bundle"
        )
    return {
        "contract": SOURCE_CONTRACT,
        "profile": profile.name,
        "status": "structurally_valid_unpromoted_candidate",
        "source": bundle,
        "onnx": contract,
        "expected_runtime": {
            "batch_size": profile.batch_size,
            "input_height": profile.input_height,
            "input_width": profile.input_width,
            "output_layers": list(profile.output_layers),
            "output_bytes_per_frame": profile.output_bytes_per_frame,
        },
    }


def _preprocess_rgb(path: Path, profile: MapAnythingProfile) -> tuple[Any, dict[str, Any]]:
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"unable to decode one 3-channel RGB fixture source: {path}")
    source_height, source_width = image.shape[:2]
    if source_height < profile.input_height or source_width < profile.input_width:
        raise ValueError(
            f"fixture source {path} is {source_width}x{source_height}; "
            f"HR-0 requires a non-upscaled source at least "
            f"{profile.input_width}x{profile.input_height}"
        )
    scale = min(
        profile.input_width / source_width,
        profile.input_height / source_height,
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    pad_x = profile.input_width - resized_width
    pad_y = profile.input_height - resized_height
    left = pad_x // 2
    top = pad_y // 2
    canvas = np.zeros(
        (profile.input_height, profile.input_width, 3),
        dtype=np.uint8,
    )
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(
        np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)),
        dtype=np.dtype("<f4"),
    )
    return tensor, {
        "path": _logical_or_absolute(path),
        "sha256": sha256_file(path),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "pad_left": int(left),
        "pad_top": int(top),
    }


def prepare_fixture(
    profile: MapAnythingProfile,
    image_paths: Sequence[Path],
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    import numpy as np

    if len(image_paths) not in {1, profile.batch_size}:
        raise ValueError(
            f"{profile.name} fixture requires one image repeated across the "
            f"batch or exactly {profile.batch_size} images"
        )
    tensors = []
    sources = []
    for path in image_paths:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"fixture image is not a regular file: {path}")
        tensor, source = _preprocess_rgb(path, profile)
        tensors.append(tensor)
        sources.append(source)
    identical_batch_members = len(tensors) == 1
    if identical_batch_members:
        tensors = tensors * profile.batch_size
    batch = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.dtype("<f4"))
    expected_shape = (
        profile.batch_size,
        3,
        profile.input_height,
        profile.input_width,
    )
    if batch.shape != expected_shape or batch.nbytes != profile.input_batch_bytes:
        raise RuntimeError("prepared MapAnything fixture violated its profile contract")
    raw = batch.tobytes(order="C")
    _exclusive_bytes(output_path, raw)
    receipt = {
        "contract": FIXTURE_CONTRACT,
        "profile": profile.name,
        "tensor": {
            "name": "images",
            "dtype": "float32_le",
            "shape": list(expected_shape),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "identical_batch_members": identical_batch_members,
        "preprocess": {
            "color": "RGB",
            "range": [0.0, 1.0],
            "layout": "NCHW",
            "resize": "aspect_preserving_area",
            "padding": "symmetric_zero",
            "upscaling": "forbidden",
        },
        "sources": sources,
    }
    try:
        _exclusive_bytes(receipt_path, _json_bytes(receipt))
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return receipt


def summarize_output(
    profile: MapAnythingProfile,
    output_path: Path,
) -> dict[str, Any]:
    import numpy as np

    if output_path.is_symlink() or not output_path.is_file():
        raise FileNotFoundError(
            f"trtexec output is not a regular file: {output_path}"
        )
    size_bytes = output_path.stat().st_size
    if size_bytes <= 0 or size_bytes > MAX_TRTEXEC_OUTPUT_BYTES:
        raise ValueError(
            "trtexec output must be nonempty and no larger than "
            f"{MAX_TRTEXEC_OUTPUT_BYTES} bytes"
        )
    try:
        payload = strict_json_loads(
            output_path.read_bytes(),
            label="MapAnything candidate trtexec output",
        )
    except (OSError, StrictJSONError) as exc:
        raise ValueError("trtexec output is not strict JSON") from exc
    if not isinstance(payload, list) or len(payload) != len(profile.output_layers):
        raise ValueError("trtexec output must contain exactly depth/conf/mask")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in payload:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "dimensions",
            "values",
        }:
            raise ValueError("trtexec output tensor has an invalid record")
        name = str(row.get("name") or "")
        if name in rows:
            raise ValueError(f"duplicate trtexec output tensor {name!r}")
        rows[name] = row
    if set(rows) != set(profile.output_layers):
        raise ValueError("trtexec output tensor names differ from depth/conf/mask")

    expected_dimensions = (
        f"{profile.batch_size}x1x{profile.input_height}x{profile.input_width}"
    )
    values_per_batch = profile.pixels_per_frame
    expected_count = profile.batch_size * values_per_batch
    summaries: dict[str, Any] = {}
    for name in profile.output_layers:
        row = rows[name]
        if row.get("dimensions") != expected_dimensions:
            raise ValueError(
                f"{name} dimensions differ from {expected_dimensions}"
            )
        raw_values = row.get("values")
        if not isinstance(raw_values, list) or len(raw_values) != expected_count:
            raise ValueError(f"{name} value count differs from its tensor shape")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_values
        ):
            raise ValueError(f"{name} values must be JSON numbers")
        try:
            values = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} values are not numeric") from exc
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"{name} contains NaN or Inf")
        first = values[:values_per_batch]
        batch_delta = 0.0
        for batch_index in range(1, profile.batch_size):
            current = values[
                batch_index * values_per_batch :
                (batch_index + 1) * values_per_batch
            ]
            batch_delta = max(
                batch_delta,
                float(np.max(np.abs(first - current), initial=0.0)),
            )
        summary = {
            "shape": [
                profile.batch_size,
                1,
                profile.input_height,
                profile.input_width,
            ],
            "finite_fraction": 1.0,
            "positive_fraction": float(np.count_nonzero(values > 0.0) / values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "p01": float(np.percentile(values, 1)),
            "p50": float(np.percentile(values, 50)),
            "p99": float(np.percentile(values, 99)),
            "batch_max_abs_delta": batch_delta,
        }
        if name == "mask":
            summary["mask_coverage_at_0_5"] = float(
                np.count_nonzero(values >= 0.5) / values.size
            )
        summaries[name] = summary
    return {
        "contract": OUTPUT_SUMMARY_CONTRACT,
        "profile": profile.name,
        "source": {
            "path": _logical_or_absolute(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": size_bytes,
        },
        "outputs": summaries,
        "promotion_state": "measurement_only",
    }


def _write_or_print(payload: Mapping[str, Any], output: Path | None) -> None:
    encoded = _json_bytes(payload)
    if output is None:
        sys.stdout.buffer.write(encoded)
        return
    _exclusive_bytes(output, encoded)
    print(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and prepare isolated DS9 MapAnything profiles"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan", help="Print exact export/build/inference/benchmark commands"
    )
    plan.add_argument("--profile", default="hr0_378x672_b3_fp32")
    plan.add_argument("--mapanything-repo", type=Path, default=Path("src/mapanything"))
    plan.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "Physical DS9 artifact root containing models/. Defaults to "
            "NOESIS_DS9_ARTIFACT_ROOT, then repo-local DS9/."
        ),
    )
    plan.add_argument("--trtexec", default="trtexec")
    plan.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable containing the pinned MapAnything dependencies.",
    )
    plan.add_argument(
        "--export-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Exact device used for ONNX export; no fallback is permitted.",
    )
    plan.add_argument("--shell", action="store_true")

    fixture = subparsers.add_parser(
        "prepare-fixture",
        help="Create a full-resolution-derived float32 NCHW raw fixture",
    )
    fixture.add_argument("--profile", default="hr0_378x672_b3_fp32")
    fixture.add_argument("--image", action="append", required=True, type=Path)
    fixture.add_argument("--output", required=True, type=Path)
    fixture.add_argument("--receipt", type=Path)

    inspect = subparsers.add_parser(
        "inspect-source",
        help="Validate and fingerprint a candidate ONNX plus external data",
    )
    inspect.add_argument("--profile", default="hr0_378x672_b3_fp32")
    inspect.add_argument("--onnx", type=Path)
    inspect.add_argument("--artifact-root", type=Path)
    inspect.add_argument("--output", type=Path)

    summarize = subparsers.add_parser(
        "summarize-output",
        help="Validate and summarize one candidate trtexec output JSON",
    )
    summarize.add_argument("--profile", default="hr0_378x672_b3_fp32")
    summarize.add_argument("--input", required=True, type=Path)
    summarize.add_argument("--output", type=Path)

    guarded = subparsers.add_parser(
        "guarded-run",
        help=(
            "Fail-closed external-artifact preflight; execute only with both "
            "--execute and --confirm-exclusive-gpu-window"
        ),
    )
    guarded.add_argument("--profile", default="hr0_378x672_b3_fp32")
    guarded.add_argument("--artifact-root", type=Path)
    guarded.add_argument("--cache-root", type=Path)
    guarded.add_argument("--temp-root", type=Path)
    guarded.add_argument(
        "--mapanything-repo",
        type=Path,
        default=Path("src/mapanything"),
    )
    guarded.add_argument("--trtexec", default="trtexec")
    guarded.add_argument(
        "--python",
        default=sys.executable,
        help="Task-isolated Python executable used for export and inspection.",
    )
    guarded.add_argument(
        "--export-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Exact device used for ONNX export; no fallback is permitted.",
    )
    guarded.add_argument("--functional-image", required=True, type=Path)
    guarded.add_argument(
        "--scene-image",
        action="append",
        required=True,
        type=Path,
    )
    mode = guarded.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all read-only guards and print the exact plan (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute the isolated candidate transaction after all guards pass.",
    )
    guarded.add_argument(
        "--confirm-exclusive-gpu-window",
        action="store_true",
        help="Required with --execute; never implied by dry-run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = get_mapanything_profile(args.profile)
    if args.command == "plan":
        artifact_root_raw = args.artifact_root or os.environ.get(
            "NOESIS_DS9_ARTIFACT_ROOT"
        )
        artifact_root = (
            _workspace_path(artifact_root_raw)
            if artifact_root_raw
            else None
        )
        payload = _command_plan(
            profile,
            mapanything_repo=_workspace_path(args.mapanything_repo),
            trtexec=str(args.trtexec),
            artifact_root=artifact_root,
            python_executable=str(args.python),
            export_device=str(args.export_device),
        )
        if args.shell:
            for name, command in payload["commands"].items():
                print(f"# {name}")
                print(shlex.join(str(item) for item in command))
        else:
            _write_or_print(payload, None)
        return 0
    if args.command == "prepare-fixture":
        output = _workspace_path(args.output)
        receipt = (
            _workspace_path(args.receipt)
            if args.receipt is not None
            else output.with_suffix(output.suffix + ".receipt.json")
        )
        payload = prepare_fixture(
            profile,
            [_workspace_path(path) for path in args.image],
            output,
            receipt,
        )
        print(
            json.dumps(
                {
                    "output": _logical_or_absolute(output),
                    "receipt": _logical_or_absolute(receipt),
                    "sha256": payload["tensor"]["sha256"],
                    "size_bytes": payload["tensor"]["size_bytes"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "inspect-source":
        artifact_root_raw = args.artifact_root or os.environ.get(
            "NOESIS_DS9_ARTIFACT_ROOT"
        )
        artifact_root = (
            _workspace_path(artifact_root_raw)
            if artifact_root_raw
            else None
        )
        onnx_path = (
            _workspace_path(args.onnx)
            if args.onnx is not None
            else profile.workspace_path("onnx", artifact_root=artifact_root)
        )
        payload = inspect_source(profile, onnx_path)
        output = _workspace_path(args.output) if args.output is not None else None
        _write_or_print(payload, output)
        return 0
    if args.command == "summarize-output":
        payload = summarize_output(profile, _workspace_path(args.input))
        output = _workspace_path(args.output) if args.output is not None else None
        _write_or_print(payload, output)
        return 0
    if args.command == "guarded-run":
        artifact_root_raw = args.artifact_root or os.environ.get(
            "NOESIS_DS9_ARTIFACT_ROOT"
        )
        if artifact_root_raw is None:
            raise GuardedRunError(
                "guarded-run requires --artifact-root or "
                "NOESIS_DS9_ARTIFACT_ROOT"
            )
        artifact_root = Path(artifact_root_raw).expanduser()
        cache_root = (
            args.cache_root.expanduser()
            if args.cache_root is not None
            else artifact_root
            / "model-work"
            / "mapanything"
            / profile.name
            / "cache"
        )
        temp_root = (
            args.temp_root.expanduser()
            if args.temp_root is not None
            else artifact_root
            / "model-work"
            / "mapanything"
            / profile.name
            / "tmp"
        )
        preflight = _guarded_preflight(
            profile,
            artifact_root=artifact_root,
            cache_root=cache_root,
            temp_root=temp_root,
            mapanything_repo=_workspace_path(args.mapanything_repo),
            trtexec=str(args.trtexec),
            functional_image=_workspace_path(args.functional_image),
            scene_images=[
                _workspace_path(path) for path in args.scene_image
            ],
            python=str(args.python),
            export_device=str(args.export_device),
        )
        if not args.execute:
            _write_or_print(preflight, None)
            return 0
        if not args.confirm_exclusive_gpu_window:
            raise GuardedRunError(
                "--execute also requires --confirm-exclusive-gpu-window"
            )
        result = _execute_guarded_run(
            preflight,
            functional_image=_workspace_path(args.functional_image),
            scene_images=[
                _workspace_path(path) for path in args.scene_image
            ],
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "profile": result["profile"],
                    "artifact_root": result["artifact_root"],
                    "evidence_root": result["outputs"]["evidence_root"],
                    "promotion_state": result["promotion_state"],
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        GuardedRunError,
        EngineMaintenanceError,
        MapAnythingProfileError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
