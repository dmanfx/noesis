#!/usr/bin/env python3
"""Rebase an external DS9 engine realization across safe manifest changes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CURRENT_MANIFEST = REPO_ROOT / "DS9/asset_manifest.yaml"
REALIZATION_FILENAME = "asset_realization.json"
REALIZATION_CONTRACT = "noesis.ds9.asset_realization"
MAX_BYTES = 8 * 1024 * 1024
REBASABLE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "updated_at",
        "target",
        "runtime",
        "shared_app_data",
        "artifacts",
    }
)
REPO_OWNED_ARTIFACT_KINDS = frozenset(
    {"native_extension", "nvinfer_parser", "gstreamer_plugin"}
)


class RebaseError(RuntimeError):
    pass


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_regular(path: Path, label: str, *, private: bool = False) -> bytes:
    path = Path(os.path.abspath(path))
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RebaseError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RebaseError(f"{label} must be a regular non-symlink file")
    if info.st_uid != os.geteuid() or info.st_nlink != 1:
        raise RebaseError(f"{label} must be an owned single-link file")
    if private and stat.S_IMODE(info.st_mode) != 0o600:
        raise RebaseError(f"{label} must have mode 0600")
    if info.st_size <= 0 or info.st_size > MAX_BYTES:
        raise RebaseError(f"{label} is outside the accepted size bound")
    raw = path.read_bytes()
    after = path.stat()
    if len(raw) != info.st_size or (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
        raise RebaseError(f"{label} changed while it was read")
    return raw


def _mapping_from_yaml(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RebaseError(f"{label} is not YAML: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RebaseError(f"{label} root must be a mapping")
    return value


def _mapping_from_json(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RebaseError(f"{label} is not JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RebaseError(f"{label} root must be a mapping")
    return value


def _git_manifest(ref: str) -> bytes:
    if not ref or any(ch.isspace() for ch in ref):
        raise RebaseError("old manifest ref must be one non-whitespace Git revision")
    result = subprocess.run(
        ["git", "show", f"{ref}:DS9/asset_manifest.yaml"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RebaseError(f"cannot read old manifest from {ref}: {detail}")
    if len(result.stdout) > MAX_BYTES:
        raise RebaseError("old manifest is outside the accepted size bound")
    return result.stdout


def _artifact_map(
    manifest: Mapping[str, Any], *, label: str
) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("artifacts")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise RebaseError(f"{label} artifacts must be a sequence")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RebaseError(f"{label} artifact record must be a mapping")
        artifact_id = str(record.get("id") or "").strip()
        if not artifact_id or artifact_id in result:
            raise RebaseError(f"{label} artifact IDs must be non-empty and unique")
        result[artifact_id] = record
    return result


def _repo_relative_ds9_path(value: Any, *, label: str) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts or path.parts[0] != "DS9":
        raise RebaseError(f"{label} must be a DS9-owned repository-relative path")
    return path


def _validate_repo_owned_artifact_change(
    artifact_id: str,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> None:
    if old.get("kind") != new.get("kind") or new.get("kind") not in (
        REPO_OWNED_ARTIFACT_KINDS
    ):
        raise RebaseError(
            f"changed unrealized artifact is not a stable repo-owned native artifact: {artifact_id}"
        )
    old_static = {key: value for key, value in old.items() if key != "provenance"}
    new_static = {key: value for key, value in new.items() if key != "provenance"}
    if old_static != new_static:
        raise RebaseError(
            f"changed unrealized artifact altered more than provenance: {artifact_id}"
        )
    output = _repo_relative_ds9_path(
        new.get("output"), label=f"{artifact_id} output"
    )
    matches = sorted(REPO_ROOT.glob(output.as_posix()))
    if len(matches) != 1:
        raise RebaseError(
            f"{artifact_id} output must resolve to exactly one repository file"
        )
    output_raw = _read_regular(matches[0], f"{artifact_id} output")
    provenance = new.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("output_sha256") != (
        _sha256(output_raw)
    ):
        raise RebaseError(f"{artifact_id} output digest does not match new provenance")
    sources = new.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise RebaseError(f"{artifact_id} sources must be a non-empty sequence")
    for index, source in enumerate(sources):
        source_path = _repo_relative_ds9_path(
            source, label=f"{artifact_id} source[{index}]"
        )
        if not (REPO_ROOT / source_path).is_file():
            raise RebaseError(f"{artifact_id} source is missing: {source_path}")


def _validate_rebase_change(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    realized_artifact_ids: set[str],
) -> tuple[list[str], list[str]]:
    if old.get("manifest_id") != "noesis-ds9-artifacts":
        raise RebaseError("old manifest ID is not the DS9 artifact authority")
    if new.get("manifest_id") != "noesis-ds9-artifacts":
        raise RebaseError("new manifest ID is not the DS9 artifact authority")
    if new.get("schema_version") != 3:
        raise RebaseError("new manifest must use native schema version 3")
    target = new.get("target")
    runtime = new.get("runtime")
    if not isinstance(target, Mapping) or not isinstance(runtime, Mapping):
        raise RebaseError("new manifest lacks native target/runtime authority")
    if target.get("native_host") != {
        "operating_system": "ubuntu-24.04",
        "architecture": "x86_64",
        "driver_minimum": "595.58.03",
        "gstreamer": "1.24.2",
        "sdk_root": "/opt/nvidia/deepstream/deepstream-9.1",
        "cuda_root": "/usr/local/cuda-13.2",
        "native_root_env": "NOESIS_DS91_NATIVE_ROOT",
    }:
        raise RebaseError("new manifest native-host authority drifted")
    if any(key in target for key in ("build_image", "image", "docker")):
        raise RebaseError("new manifest target retains container authority")
    if runtime.get("backend") != "native_host" or runtime.get("supervisor") != (
        "DS9/scripts/run_canonical_runtime_host.py"
    ):
        raise RebaseError("new manifest runtime is not the native host supervisor")
    if any(key in runtime for key in ("image", "container", "docker")):
        raise RebaseError("new manifest runtime retains container authority")

    changed: list[str] = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            changed.append(key)
            if key not in REBASABLE_MANIFEST_KEYS:
                raise RebaseError(f"manifest change touches non-rebasable authority: {key}")

    shared_paths = new.get("shared_app_data")
    if not isinstance(shared_paths, Sequence) or isinstance(shared_paths, (str, bytes)):
        raise RebaseError("new manifest shared_app_data must be a sequence")
    for index, value in enumerate(shared_paths):
        raw = str(value or "").strip()
        path = Path(raw)
        if not raw or path.is_absolute() or ".." in path.parts:
            raise RebaseError(
                f"shared_app_data[{index}] must be a repository-relative path"
            )
        if not (REPO_ROOT / path).is_file():
            raise RebaseError(f"shared_app_data path is missing: {raw}")

    old_artifacts = _artifact_map(old, label="old manifest")
    new_artifacts = _artifact_map(new, label="new manifest")
    changed_artifact_ids = [
        artifact_id
        for artifact_id in sorted(set(old_artifacts) | set(new_artifacts))
        if old_artifacts.get(artifact_id) != new_artifacts.get(artifact_id)
    ]
    for artifact_id in changed_artifact_ids:
        if artifact_id in realized_artifact_ids:
            raise RebaseError(
                f"manifest change touches realized engine authority: {artifact_id}"
            )
        old_record = old_artifacts.get(artifact_id)
        new_record = new_artifacts.get(artifact_id)
        if old_record is None or new_record is None:
            raise RebaseError(
                f"manifest rebase cannot add or remove artifact records: {artifact_id}"
            )
        _validate_repo_owned_artifact_change(artifact_id, old_record, new_record)
    return changed, changed_artifact_ids


def _atomic_write(path: Path, raw: bytes, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise RebaseError(f"temporary output already exists: {temporary}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RebaseError("short write while creating atomic replacement")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def rebase(
    *,
    artifact_root: Path,
    old_manifest_ref: str,
    expected_realization_sha256: str,
    apply: bool,
) -> dict[str, Any]:
    root = Path(os.path.abspath(artifact_root))
    if root.is_symlink() or not root.is_dir():
        raise RebaseError("artifact root must be a real directory")
    info = root.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise RebaseError("artifact root must be owned and not group/world writable")

    old_raw = _git_manifest(old_manifest_ref)
    new_raw = _read_regular(CURRENT_MANIFEST, "current asset manifest")
    old = _mapping_from_yaml(old_raw, "old asset manifest")
    new = _mapping_from_yaml(new_raw, "current asset manifest")
    realization_path = root / REALIZATION_FILENAME
    realization_raw = _read_regular(
        realization_path, "asset realization", private=True
    )
    before_sha256 = _sha256(realization_raw)
    if before_sha256 != expected_realization_sha256:
        raise RebaseError(
            "asset realization digest differs from the explicit compare-and-swap pin"
        )
    realization = _mapping_from_json(realization_raw, "asset realization")
    if realization.get("schema_version") != 1 or realization.get("contract") != (
        REALIZATION_CONTRACT
    ):
        raise RebaseError("asset realization contract drifted")
    realized_artifacts = realization.get("artifacts")
    if not isinstance(realized_artifacts, Mapping):
        raise RebaseError("asset realization artifacts must be a mapping")
    changed, changed_artifact_ids = _validate_rebase_change(
        old,
        new,
        realized_artifact_ids={str(value) for value in realized_artifacts},
    )
    if realization.get("base_manifest") != {
        "path": "DS9/asset_manifest.yaml",
        "sha256": _sha256(old_raw),
    }:
        raise RebaseError("asset realization does not bind the selected old manifest")

    replacement = copy.deepcopy(dict(realization))
    replacement["base_manifest"] = {
        "path": "DS9/asset_manifest.yaml",
        "sha256": _sha256(new_raw),
    }
    replacement["updated_at_utc"] = _utc_now()
    replacement_raw = (
        json.dumps(replacement, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    after_sha256 = _sha256(replacement_raw)
    result = {
        "ok": True,
        "applied": apply,
        "changed_manifest_keys": changed,
        "changed_unrealized_artifact_ids": changed_artifact_ids,
        "old_manifest_sha256": _sha256(old_raw),
        "new_manifest_sha256": _sha256(new_raw),
        "realization_before_sha256": before_sha256,
        "realization_after_sha256": after_sha256,
        "realized_artifact_count": len(realized_artifacts),
    }
    if not apply:
        return result

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    evidence_root = root / "manifest_rebase_native" / run_id
    evidence_root.mkdir(parents=True, mode=0o700)
    evidence_root.chmod(0o700)
    _atomic_write(evidence_root / "asset_realization.before.json", realization_raw, 0o600)
    _atomic_write(evidence_root / "asset_realization.after.json", replacement_raw, 0o600)
    evidence = {
        "schema_version": 1,
        "contract": "noesis.ds9.native_manifest_rebase",
        "created_at_utc": _utc_now(),
        **result,
    }
    evidence_raw = (
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(evidence_root / "rebase_evidence.json", evidence_raw, 0o600)
    if _sha256(_read_regular(realization_path, "asset realization", private=True)) != (
        before_sha256
    ):
        raise RebaseError("asset realization changed before commit")
    _atomic_write(realization_path, replacement_raw, 0o600)
    if _sha256(_read_regular(realization_path, "asset realization", private=True)) != (
        after_sha256
    ):
        raise RebaseError("asset realization replacement did not persist exact bytes")
    result["evidence"] = str(evidence_root / "rebase_evidence.json")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "")),
    )
    parser.add_argument("--old-manifest-ref", default="HEAD")
    parser.add_argument("--expected-realization-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if not str(args.artifact_root):
            raise RebaseError("--artifact-root or NOESIS_DS9_ARTIFACT_ROOT is required")
        result = rebase(
            artifact_root=args.artifact_root,
            old_manifest_ref=args.old_manifest_ref,
            expected_realization_sha256=args.expected_realization_sha256,
            apply=args.apply,
        )
    except RebaseError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
