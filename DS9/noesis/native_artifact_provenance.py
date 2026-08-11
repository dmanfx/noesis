"""Fail-closed content attestation for DS9 native extension binaries.

Filesystem timestamps are not build provenance: a fresh immutable checkout makes
tracked sources newer than separately hydrated binaries even when both still
match the reviewed artifact manifest byte-for-byte.  This module verifies the
exact manifest, source bundle, ABI filename, and binary bytes before native code
is imported.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_EXTENSION_BYTES = 128 * 1024 * 1024


class DS9NativeArtifactProvenanceError(RuntimeError):
    """Raised before import when a DS9 native artifact is not exactly reviewed."""


@dataclass(frozen=True)
class _NativeContract:
    artifact_id: str
    output: str
    sources: tuple[str, ...]
    builder: str
    required_profiles: tuple[str, ...]


_NATIVE_CONTRACTS: Mapping[str, _NativeContract] = {
    "noesis_pose_meta_ext": _NativeContract(
        artifact_id="native.pose_meta",
        output="DS9/native_extensions/noesis_pose_meta_ext*.so",
        sources=("DS9/native/noesis_pose_meta_ext.cpp",),
        builder="DS9/scripts/build_noesis_pose_meta_ext.sh",
        required_profiles=("runtime_common", "full"),
    ),
    "noesis_depth_meta_ext": _NativeContract(
        artifact_id="native.depth_meta",
        output="DS9/native_extensions/noesis_depth_meta_ext*.so",
        sources=("DS9/native/noesis_depth_meta_ext.cpp",),
        builder="DS9/scripts/build_noesis_depth_meta_ext.sh",
        required_profiles=("runtime_common", "full"),
    ),
    "noesis_depth_tracking_tensor_ext": _NativeContract(
        artifact_id="native.depth_tracking_tensor",
        output="DS9/native_extensions/noesis_depth_tracking_tensor_ext*.so",
        sources=(
            "DS9/native/noesis_depth_tracking_tensor_ext.cpp",
            "DS9/native/noesis_depth_tracking_tensor_kernels.cu",
        ),
        builder="DS9/scripts/build_noesis_depth_tracking_tensor_ext.sh",
        required_profiles=("runtime_common", "full"),
    ),
    "noesis_reid_meta_ext": _NativeContract(
        artifact_id="native.reid_meta",
        output="DS9/native_extensions/noesis_reid_meta_ext*.so",
        sources=("DS9/native/noesis_reid_meta_ext.cpp",),
        builder="DS9/scripts/build_noesis_reid_meta_ext.sh",
        required_profiles=("runtime_common", "full"),
    ),
    "noesis_v3dt_meta_ext": _NativeContract(
        artifact_id="native.v3dt_meta",
        output="DS9/native_extensions/noesis_v3dt_meta_ext*.so",
        sources=("DS9/native/noesis_v3dt_meta_ext.cpp",),
        builder="DS9/scripts/build_noesis_v3dt_meta_ext.sh",
        required_profiles=("v3dt", "full"),
    ),
    "noesis_latency_ext": _NativeContract(
        artifact_id="native.latency",
        output="DS9/native_extensions/noesis_latency_ext*.so",
        sources=("DS9/native/noesis_latency_ext.cpp",),
        builder="DS9/scripts/build_noesis_latency_ext.sh",
        required_profiles=("runtime_common", "full"),
    ),
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    explicit: set[Any] = set()
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=False)
        try:
            duplicate = key in explicit
        except TypeError as exc:
            raise DS9NativeArtifactProvenanceError(
                "DS9 asset manifest contains a non-scalar mapping key"
            ) from exc
        if duplicate:
            raise DS9NativeArtifactProvenanceError(
                f"DS9 asset manifest contains duplicate key {key!r}"
            )
        explicit.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    uid: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _identity(info: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=int(info.st_dev),
        inode=int(info.st_ino),
        mode=int(info.st_mode),
        links=int(info.st_nlink),
        uid=int(info.st_uid),
        size=int(info.st_size),
        mtime_ns=int(info.st_mtime_ns),
        ctime_ns=int(info.st_ctime_ns),
    )


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_no_symlink_chain(path: Path, *, root: Path, label: str) -> Path:
    root = _absolute(root)
    candidate = _absolute(path)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise DS9NativeArtifactProvenanceError(
            f"{label} escapes the reviewed checkout: {candidate}"
        ) from exc
    current = root
    for part in (Path("."), *relative.parts):
        if part != Path("."):
            current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            raise DS9NativeArtifactProvenanceError(
                f"{label} is unavailable: {current}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise DS9NativeArtifactProvenanceError(
                f"{label} contains a symlink: {current}"
            )
    return candidate


def _read_stable_regular(
    path: Path,
    *,
    root: Path,
    label: str,
    maximum_bytes: int,
) -> tuple[bytes, _FileIdentity]:
    candidate = _require_no_symlink_chain(path, root=root, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise DS9NativeArtifactProvenanceError(
            f"{label} cannot be opened safely: {candidate}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        before_identity = _identity(before)
        if not stat.S_ISREG(before.st_mode):
            raise DS9NativeArtifactProvenanceError(
                f"{label} is not a regular file: {candidate}"
            )
        if before.st_uid != os.geteuid():
            raise DS9NativeArtifactProvenanceError(
                f"{label} is not owned by the runtime user: {candidate}"
            )
        if before.st_nlink != 1:
            raise DS9NativeArtifactProvenanceError(
                f"{label} must have exactly one link: {candidate}"
            )
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise DS9NativeArtifactProvenanceError(
                f"{label} size is outside the reviewed bound: {candidate}"
            )
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise DS9NativeArtifactProvenanceError(
                    f"{label} ended before its inspected size: {candidate}"
                )
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise DS9NativeArtifactProvenanceError(
                f"{label} grew while it was read: {candidate}"
            )
        after_identity = _identity(os.fstat(descriptor))
        if after_identity != before_identity:
            raise DS9NativeArtifactProvenanceError(
                f"{label} changed while it was read: {candidate}"
            )
    finally:
        os.close(descriptor)
    try:
        path_identity = _identity(candidate.lstat())
    except OSError as exc:
        raise DS9NativeArtifactProvenanceError(
            f"{label} path disappeared while it was read: {candidate}"
        ) from exc
    if path_identity != before_identity:
        raise DS9NativeArtifactProvenanceError(
            f"{label} path identity changed while it was read: {candidate}"
        )
    return b"".join(chunks), before_identity


def _assert_identity(path: Path, expected: _FileIdentity, *, label: str) -> None:
    try:
        observed = _identity(path.lstat())
    except OSError as exc:
        raise DS9NativeArtifactProvenanceError(
            f"{label} disappeared during attestation: {path}"
        ) from exc
    if observed != expected:
        raise DS9NativeArtifactProvenanceError(
            f"{label} changed during attestation: {path}"
        )


def _load_manifest(payload: bytes) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DS9NativeArtifactProvenanceError(
            "DS9 asset manifest must be UTF-8"
        ) from exc
    loader = _UniqueKeySafeLoader(text)
    try:
        manifest = loader.get_single_data()
    except DS9NativeArtifactProvenanceError:
        raise
    except yaml.YAMLError as exc:
        raise DS9NativeArtifactProvenanceError(
            "DS9 asset manifest is not valid strict YAML"
        ) from exc
    finally:
        loader.dispose()
    if not isinstance(manifest, Mapping):
        raise DS9NativeArtifactProvenanceError(
            "DS9 asset manifest root must be a mapping"
        )
    return manifest


def _artifact_records(manifest: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise DS9NativeArtifactProvenanceError(
            "DS9 asset manifest artifacts must be a list"
        )
    records: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise DS9NativeArtifactProvenanceError(
                f"DS9 asset manifest artifact {index} must be a mapping"
            )
        artifact_id = str(item.get("id") or "")
        if not artifact_id or artifact_id in records:
            raise DS9NativeArtifactProvenanceError(
                f"DS9 asset manifest has missing or duplicate artifact id {artifact_id!r}"
            )
        records[artifact_id] = item
    return records


def _require_manifest_authority(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema_version") != 2
        or manifest.get("manifest_id") != "noesis-ds9-artifacts"
        or manifest.get("schema") != "DS9/docs/asset_manifest.schema.json"
    ):
        raise DS9NativeArtifactProvenanceError(
            "DS9 asset manifest schema authority is not the reviewed v2 contract"
        )
    target = manifest.get("target")
    deepstream = target.get("deepstream") if isinstance(target, Mapping) else None
    if (
        not isinstance(target, Mapping)
        or target.get("platform") != "linux-x86_64-dgpu"
        or target.get("python") != "3.12"
        or target.get("cuda") != "13.1"
        or target.get("tensorrt") != "10.14.1.48"
        or not isinstance(deepstream, Mapping)
        or deepstream.get("major") != 9
        or deepstream.get("version") != "9.0"
        or deepstream.get("home") != "/opt/nvidia/deepstream/deepstream-9.0"
    ):
        raise DS9NativeArtifactProvenanceError(
            "DS9 asset manifest target authority is not the reviewed SDK contract"
        )
    policy = manifest.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("ds8_binary_reuse") != "forbidden"
        or policy.get("root_engine_reuse") != "forbidden"
        or policy.get("root_native_extension_reuse") != "forbidden"
        or policy.get("require_ds9_owned_outputs") is not True
        or policy.get("require_provenance_for_validated") is not True
        or set(policy.get("allowed_states") or ())
        != {"missing", "staged_unverified", "validated"}
        or "DS9/native_extensions" not in set(policy.get("output_roots") or ())
        or "DS9/native" not in set(policy.get("source_roots") or ())
    ):
        raise DS9NativeArtifactProvenanceError(
            "DS9 asset manifest policy authority is not the reviewed ownership contract"
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def attest_ds9_native_artifacts(
    *,
    ds9_root: Path,
    native_dir: Path,
    module_names: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Attest reviewed DS9 native sources and binaries before any import."""

    ds9_root = _absolute(ds9_root)
    repo_root = ds9_root.parent
    expected_native_dir = ds9_root / "native_extensions"
    if _absolute(native_dir) != expected_native_dir:
        raise DS9NativeArtifactProvenanceError(
            "DS9 native extensions must load directly from DS9/native_extensions"
        )
    _require_no_symlink_chain(
        expected_native_dir,
        root=repo_root,
        label="DS9 native-extension directory",
    )
    requested = tuple(_NATIVE_CONTRACTS if module_names is None else module_names)
    if len(requested) != len(set(requested)):
        raise DS9NativeArtifactProvenanceError(
            "DS9 native attestation request contains duplicate modules"
        )
    unknown = sorted(set(requested) - set(_NATIVE_CONTRACTS))
    if unknown:
        raise DS9NativeArtifactProvenanceError(
            f"DS9 native attestation request contains unknown modules: {unknown}"
        )

    manifest_path = ds9_root / "asset_manifest.yaml"
    manifest_payload, manifest_identity = _read_stable_regular(
        manifest_path,
        root=repo_root,
        label="DS9 asset manifest",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest = _load_manifest(manifest_payload)
    _require_manifest_authority(manifest)
    records = _artifact_records(manifest)
    extension_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or "")
    if not extension_suffix.startswith(".") or not extension_suffix.endswith(".so"):
        raise DS9NativeArtifactProvenanceError(
            f"active Python extension suffix is not reviewed for DS9: {extension_suffix!r}"
        )

    source_identities: list[tuple[Path, _FileIdentity, str]] = []
    output_identities: list[tuple[Path, _FileIdentity, str]] = []
    attested: dict[str, Path] = {}
    for module_name in requested:
        contract = _NATIVE_CONTRACTS[module_name]
        artifact = records.get(contract.artifact_id)
        if artifact is None:
            raise DS9NativeArtifactProvenanceError(
                f"DS9 asset manifest is missing {contract.artifact_id}"
            )
        if (
            artifact.get("kind") != "native_extension"
            or artifact.get("output") != contract.output
            or tuple(artifact.get("sources") or ()) != contract.sources
            or artifact.get("builder") != contract.builder
            or tuple(artifact.get("required_profiles") or ())
            != contract.required_profiles
            or artifact.get("state") not in {"staged_unverified", "validated"}
        ):
            raise DS9NativeArtifactProvenanceError(
                f"DS9 asset manifest contract drifted for {contract.artifact_id}"
            )
        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping):
            raise DS9NativeArtifactProvenanceError(
                f"DS9 asset manifest provenance is missing for {contract.artifact_id}"
            )
        required_provenance = {
            "source_sha256",
            "output_sha256",
            "built_at_utc",
            "build_host",
            "command",
        }
        if required_provenance - set(provenance) or any(
            not str(provenance.get(key) or "").strip()
            for key in required_provenance
        ):
            raise DS9NativeArtifactProvenanceError(
                f"DS9 asset manifest provenance is incomplete for {contract.artifact_id}"
            )
        expected_source = str(provenance.get("source_sha256") or "")
        expected_output = str(provenance.get("output_sha256") or "")
        if not _SHA256_RE.fullmatch(expected_source) or not _SHA256_RE.fullmatch(
            expected_output
        ):
            raise DS9NativeArtifactProvenanceError(
                f"DS9 asset manifest hashes are malformed for {contract.artifact_id}"
            )

        source_digest = hashlib.sha256()
        for label in sorted(contract.sources):
            relative = Path(label)
            if relative.is_absolute() or ".." in relative.parts:
                raise DS9NativeArtifactProvenanceError(
                    f"unsafe native source path for {contract.artifact_id}: {label}"
                )
            source_path = repo_root / relative
            source_payload, source_identity = _read_stable_regular(
                source_path,
                root=repo_root,
                label=f"{contract.artifact_id} source",
                maximum_bytes=_MAX_SOURCE_BYTES,
            )
            source_digest.update(label.encode("utf-8"))
            source_digest.update(b"\0")
            source_digest.update(source_payload)
            source_digest.update(b"\0")
            source_identities.append(
                (source_path, source_identity, f"{contract.artifact_id} source")
            )
        if source_digest.hexdigest() != expected_source:
            raise DS9NativeArtifactProvenanceError(
                f"source SHA-256 mismatch for {contract.artifact_id}"
            )

        matches = sorted(expected_native_dir.glob(f"{module_name}*.so"))
        expected_path = expected_native_dir / f"{module_name}{extension_suffix}"
        if matches != [expected_path]:
            raise DS9NativeArtifactProvenanceError(
                f"{contract.artifact_id} must have exactly one active-ABI output: "
                f"{expected_path.name}"
            )
        output_payload, output_identity = _read_stable_regular(
            expected_path,
            root=repo_root,
            label=f"{contract.artifact_id} output",
            maximum_bytes=_MAX_EXTENSION_BYTES,
        )
        if _sha256(output_payload) != expected_output:
            raise DS9NativeArtifactProvenanceError(
                f"output SHA-256 mismatch for {contract.artifact_id}"
            )
        output_identities.append(
            (expected_path, output_identity, f"{contract.artifact_id} output")
        )
        attested[module_name] = expected_path

    for path, identity, label in (*source_identities, *output_identities):
        _assert_identity(path, identity, label=label)
    final_manifest, final_manifest_identity = _read_stable_regular(
        manifest_path,
        root=repo_root,
        label="DS9 asset manifest",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    if final_manifest != manifest_payload or final_manifest_identity != manifest_identity:
        raise DS9NativeArtifactProvenanceError(
            "DS9 asset manifest changed during native attestation"
        )
    return attested


__all__ = [
    "DS9NativeArtifactProvenanceError",
    "attest_ds9_native_artifacts",
]
