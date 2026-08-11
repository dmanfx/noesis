#!/usr/bin/env python3
"""Validate the exact offline 595-to-580 rollback without mutating the host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import stat
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "DS9" / "config" / "driver_rollback_580.json"
CHECKSUM_LINE = re.compile(r"^(?P<sha>[0-9a-f]{64})  \./(?P<path>.+)$")
SUMMARY_LINE = re.compile(
    r"(?P<upgraded>\d+) upgraded, (?P<installed>\d+) newly installed, "
    r"(?P<removed>\d+) to remove"
)
RELATIONSHIP_FIELDS = (
    "Pre-Depends",
    "Depends",
    "Recommends",
    "Suggests",
    "Breaks",
    "Conflicts",
    "Replaces",
    "Provides",
    "Multi-Arch",
)
PROTECTED_PACKAGE_FRAGMENTS = (
    "cuda",
    "cudnn",
    "deepstream",
    "libnvinfer",
    "tensorrt",
)


class RehearsalError(RuntimeError):
    """A fail-closed rollback rehearsal validation error."""


@dataclass(frozen=True, order=True)
class PackageKey:
    package: str
    version: str
    architecture: str


@dataclass(frozen=True)
class PackageRecord:
    key: PackageKey
    sha256: str
    relationships_sha256: str
    path: Path | None = None
    relationships: tuple[tuple[str, str], ...] = ()


def sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_unlinked_path(path: Path) -> Path:
    if any(part == ".." for part in path.parts):
        raise RehearsalError(f"output path cannot contain parent traversal: {path}")
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise RehearsalError(
                f"output parent component is missing: {current}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise RehearsalError(f"output path contains a symlink: {current}")
        if not stat.S_ISDIR(info.st_mode):
            raise RehearsalError(
                f"output parent component is not a directory: {current}"
            )
    return absolute


def write_private_atomic_json(path: Path, result: Mapping[str, Any]) -> str:
    output = _absolute_unlinked_path(path)
    parent = output.parent
    parent_info = parent.stat()
    if parent_info.st_uid != os.getuid():
        raise RehearsalError("output parent is not owned by the rehearsal user")
    if stat.S_IMODE(parent_info.st_mode) & 0o077:
        raise RehearsalError("output parent is not owner-only")
    try:
        output_info = output.lstat()
    except FileNotFoundError:
        output_info = None
    if output_info is not None:
        if stat.S_ISLNK(output_info.st_mode):
            raise RehearsalError(f"output evidence path is a symlink: {output}")
        raise RehearsalError(f"refusing to replace existing evidence: {output}")

    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = parent / f".{output.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RehearsalError("short write while persisting rollback evidence")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, output, follow_symlinks=False)
        linked = True
        temporary.unlink()
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise RehearsalError(
            f"refusing to replace existing evidence: {output}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()

    if not linked:
        raise RehearsalError("rollback evidence was not atomically published")
    info = _require_regular_file(output, owner_uid=os.getuid())
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise RehearsalError("rollback evidence permissions are not exactly 0600")
    return sha256_file(output)


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        env=dict(env) if env is not None else None,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit {completed.returncode}"
        raise RehearsalError(f"command failed: {shlex.join(command)}: {detail}")
    return completed


def _require_regular_file(
    path: Path, *, owner_uid: int | None = None
) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RehearsalError(f"required file is missing: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RehearsalError(f"required path is not a regular file: {path}")
    if info.st_nlink != 1:
        raise RehearsalError(f"required file must have exactly one link: {path}")
    if owner_uid is not None and info.st_uid != owner_uid:
        raise RehearsalError(f"required file has unexpected owner: {path}")
    return info


def load_spec(path: Path) -> dict[str, Any]:
    _require_regular_file(path)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RehearsalError(
            f"cannot read rollback specification: {path}: {exc}"
        ) from exc
    if spec.get("schema_version") != 1:
        raise RehearsalError("unsupported rollback specification version")
    if tuple(spec.get("relationship_fields", ())) != RELATIONSHIP_FIELDS:
        raise RehearsalError("rollback relationship field contract drifted")
    packages = spec.get("packages")
    if not isinstance(packages, list) or len(packages) != 33:
        raise RehearsalError("rollback specification must contain exactly 33 packages")
    expected_counts = Counter(str(item["architecture"]) for item in packages)
    declared_counts = Counter(
        {str(key): int(value) for key, value in spec["architecture_counts"].items()}
    )
    if expected_counts != declared_counts:
        raise RehearsalError("rollback architecture count contract is inconsistent")
    keys = {
        PackageKey(
            str(item["package"]), str(item["version"]), str(item["architecture"])
        )
        for item in packages
    }
    if len(keys) != len(packages):
        raise RehearsalError("rollback package keys are not unique")
    removals = set(spec.get("explicit_removals", ()))
    if len(removals) != len(spec.get("explicit_removals", ())):
        raise RehearsalError("rollback explicit removals are not unique")
    non_595 = {name for name in removals if "595" not in name}
    documented_non_595 = set(spec.get("solver_required_non_595_removals", {}))
    if non_595 != documented_non_595:
        raise RehearsalError(
            "every non-595 removal must have exact solver-necessity evidence"
        )
    return spec


def expected_packages(spec: Mapping[str, Any]) -> dict[PackageKey, PackageRecord]:
    records: dict[PackageKey, PackageRecord] = {}
    for item in spec["packages"]:
        key = PackageKey(item["package"], item["version"], item["architecture"])
        records[key] = PackageRecord(
            key=key,
            sha256=item["sha256"],
            relationships_sha256=item["relationships_sha256"],
        )
    return records


def _safe_manifest_relative(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or not relative.parts:
        raise RehearsalError(f"unsafe checkpoint manifest path: {raw!r}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise RehearsalError(f"unsafe checkpoint manifest path: {raw!r}")
    return relative


def verify_checkpoint(
    checkpoint: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise RehearsalError(f"checkpoint is not a real directory: {checkpoint}")
    checkpoint_info = checkpoint.stat()
    if checkpoint_info.st_uid != os.getuid():
        raise RehearsalError("checkpoint is not owned by the rehearsal user")
    if stat.S_IMODE(checkpoint_info.st_mode) & 0o077:
        raise RehearsalError("checkpoint directory is not owner-only")

    manifest = checkpoint / "SHA256SUMS"
    manifest_info = _require_regular_file(manifest, owner_uid=os.getuid())
    if stat.S_IMODE(manifest_info.st_mode) & 0o077:
        raise RehearsalError("checkpoint checksum manifest is not owner-only")
    manifest_sha256 = sha256_file(manifest)
    if manifest_sha256 != expected_manifest_sha256:
        raise RehearsalError(
            "checkpoint SHA256SUMS authority mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )

    declared: dict[Path, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise RehearsalError(f"invalid SHA256SUMS line {line_number}")
        relative = _safe_manifest_relative(match.group("path"))
        if relative in declared:
            raise RehearsalError(f"duplicate SHA256SUMS path: {relative}")
        declared[relative] = match.group("sha")

    required = {
        Path("README.md"),
        Path("VALIDATION.md"),
        Path("host/state/package-cache-inventory.tsv"),
        Path("host/state/dpkg-status"),
    }
    missing_required = sorted(str(path) for path in required - declared.keys())
    if missing_required:
        raise RehearsalError(
            "checkpoint manifest omits required recovery files: "
            + ", ".join(missing_required)
        )

    total_bytes = 0
    for relative, expected_sha256 in declared.items():
        path = checkpoint / relative
        info = _require_regular_file(path, owner_uid=os.getuid())
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RehearsalError(
                f"checkpoint content hash mismatch for {relative}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        total_bytes += info.st_size

    return {
        "manifest_sha256": manifest_sha256,
        "verified_file_count": len(declared),
        "verified_bytes": total_bytes,
        "owner_only": True,
    }


def _normalized_control_value(control: Mapping[str, str], field: str) -> str:
    return " ".join(str(control.get(field, "")).split())


def relationship_sha256(control: Mapping[str, str], fields: Iterable[str]) -> str:
    payload = "".join(
        f"{field}: {_normalized_control_value(control, field)}\n" for field in fields
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_deb(path: Path, fields: Sequence[str]) -> PackageRecord:
    info = _require_regular_file(path, owner_uid=os.getuid())
    if info.st_mode & stat.S_IWOTH:
        raise RehearsalError(f"rollback package is world-writable: {path.name}")
    completed = _run(("dpkg-deb", "-f", str(path)))
    control = Parser().parsestr(completed.stdout)
    if not control.get("Multi-Arch"):
        # dpkg-deb reports the Debian default as `no` when queried directly,
        # even when the optional field is absent from the raw control stanza.
        control["Multi-Arch"] = "no"
    try:
        key = PackageKey(
            _normalized_control_value(control, "Package"),
            _normalized_control_value(control, "Version"),
            _normalized_control_value(control, "Architecture"),
        )
    except KeyError as exc:
        raise RehearsalError(f"invalid Debian control metadata: {path.name}") from exc
    if not all((key.package, key.version, key.architecture)):
        raise RehearsalError(f"incomplete Debian control identity: {path.name}")
    return PackageRecord(
        key=key,
        sha256=sha256_file(path),
        relationships_sha256=relationship_sha256(control, fields),
        path=path,
        relationships=tuple(
            (field, _normalized_control_value(control, field)) for field in fields
        ),
    )


def _inventory_records(path: Path) -> dict[PackageKey, str]:
    _require_regular_file(path, owner_uid=os.getuid())
    records: dict[PackageKey, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        columns = line.split("\t")
        if len(columns) != 5:
            raise RehearsalError(f"invalid package inventory line {line_number}")
        scope, package, version, architecture, sha256 = columns
        if scope != "rollback-580":
            continue
        key = PackageKey(package, version, architecture)
        if key in records:
            raise RehearsalError(f"duplicate checkpoint inventory package: {key}")
        records[key] = sha256
    return records


def verify_packages(
    checkpoint: Path, spec: Mapping[str, Any]
) -> tuple[dict[PackageKey, PackageRecord], dict[str, Any]]:
    expected = expected_packages(spec)
    package_dir = checkpoint / _safe_manifest_relative(spec["package_directory"])
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise RehearsalError("rollback package directory is missing or linked")
    archives = sorted(package_dir.glob("*.deb"))
    if len(archives) != len(expected):
        raise RehearsalError(
            f"rollback cache must contain {len(expected)} debs, found {len(archives)}"
        )

    actual: dict[PackageKey, PackageRecord] = {}
    for archive in archives:
        record = _read_deb(archive, tuple(spec["relationship_fields"]))
        if record.key in actual:
            raise RehearsalError(f"duplicate rollback archive identity: {record.key}")
        actual[record.key] = record
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise RehearsalError(
            f"rollback package-set drift: missing={missing}, extra={extra}"
        )
    for key, expected_record in expected.items():
        record = actual[key]
        if record.sha256 != expected_record.sha256:
            raise RehearsalError(f"rollback package hash mismatch: {key}")
        if record.relationships_sha256 != expected_record.relationships_sha256:
            raise RehearsalError(f"rollback package relationship drift: {key}")

    for removal, evidence in spec["solver_required_non_595_removals"].items():
        required_by = str(evidence["required_by"])
        relationship = str(evidence["relationship"])
        candidates = [
            record for key, record in actual.items() if key.package == required_by
        ]
        if len(candidates) != 1:
            raise RehearsalError(
                f"solver necessity source is not an exact package: {required_by}"
            )
        relationship_value = dict(candidates[0].relationships).get(relationship, "")
        terms = {term.strip().split()[0] for term in relationship_value.split(",")}
        if removal not in terms:
            raise RehearsalError(
                f"non-595 removal lacks declared {relationship} evidence: {removal}"
            )

    inventory_path = checkpoint / _safe_manifest_relative(spec["checkpoint_inventory"])
    inventory = _inventory_records(inventory_path)
    expected_inventory = {key: record.sha256 for key, record in expected.items()}
    if inventory != expected_inventory:
        raise RehearsalError(
            "checkpoint package inventory disagrees with rollback specification"
        )

    counts = Counter(record.key.architecture for record in actual.values())
    return actual, {
        "package_count": len(actual),
        "archive_bytes": sum(
            record.path.stat().st_size for record in actual.values() if record.path
        ),
        "architectures": dict(sorted(counts.items())),
        "archive_hashes_verified": True,
        "relationship_hashes_verified": True,
        "non_595_solver_removals_verified": sorted(
            spec["solver_required_non_595_removals"]
        ),
        "checkpoint_inventory_verified": True,
    }


def installed_packages() -> dict[tuple[str, str], tuple[str, str]]:
    completed = _run(
        (
            "dpkg-query",
            "-W",
            "-f=${binary:Package}\\t${Version}\\t${Architecture}\\t${db:Status-Abbrev}\\n",
        )
    )
    installed: dict[tuple[str, str], tuple[str, str]] = {}
    for line in completed.stdout.splitlines():
        columns = line.split("\t")
        if len(columns) != 4 or len(columns[3]) < 2 or columns[3][1] != "i":
            continue
        binary, version, architecture, status = columns
        package = binary.rsplit(":", 1)[0] if ":" in binary else binary
        installed[(package, architecture)] = (version, status)
    return installed


def _apt_display_name(package: str) -> str:
    return package.removesuffix(":amd64")


def _parse_installed_actions(output: str) -> set[PackageKey]:
    actions: set[PackageKey] = set()
    pattern = re.compile(
        r"^Inst (?P<package>\S+)(?: \[[^]]+\])? "
        r"\((?P<version>\S+).* \[(?P<architecture>amd64|i386|all)\]\)(?: \[\])?$"
    )
    for line in output.splitlines():
        if not line.startswith("Inst "):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise RehearsalError(f"cannot parse apt install action: {line}")
        package = match.group("package").removesuffix(":i386").removesuffix(":amd64")
        actions.add(
            PackageKey(package, match.group("version"), match.group("architecture"))
        )
    return actions


def simulate_rollback(
    packages: Mapping[PackageKey, PackageRecord],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    current = installed_packages()
    exact_already_installed = {
        key
        for key in packages
        if current.get((key.package, key.architecture), (None, None))[0] == key.version
    }
    archives = [str(packages[key].path) for key in sorted(packages)]
    explicit_removals = [f"{name}-" for name in spec["explicit_removals"]]
    command = [
        "apt-get",
        "--simulate",
        "--no-download",
        "--allow-downgrades",
        "--allow-change-held-packages",
        "install",
        *archives,
        *explicit_removals,
    ]
    environment = dict(os.environ)
    environment.update(
        {"LC_ALL": "C", "LANG": "C", "DEBIAN_FRONTEND": "noninteractive"}
    )
    completed = _run(command, env=environment)
    output = completed.stdout

    summary_match = SUMMARY_LINE.search(output)
    if summary_match is None:
        raise RehearsalError("apt simulation did not emit a transaction summary")
    summary = {
        "upgraded": int(summary_match.group("upgraded")),
        "newly_installed": int(summary_match.group("installed")),
        "removed": int(summary_match.group("removed")),
    }
    if summary != spec["simulation"]:
        raise RehearsalError(f"apt transaction summary drift: {summary}")

    install_actions = _parse_installed_actions(output)
    expected_installs = set(packages) - exact_already_installed
    if install_actions != expected_installs:
        missing = sorted(expected_installs - install_actions)
        extra = sorted(install_actions - expected_installs)
        raise RehearsalError(
            f"apt install action drift: missing={missing}, extra={extra}"
        )

    removal_actions = {
        line.split()[1] for line in output.splitlines() if line.startswith("Remv ")
    }
    expected_removals = {_apt_display_name(name) for name in spec["explicit_removals"]}
    if removal_actions != expected_removals:
        missing = sorted(expected_removals - removal_actions)
        extra = sorted(removal_actions - expected_removals)
        raise RehearsalError(
            f"apt removal action drift: missing={missing}, extra={extra}"
        )

    changed_names = {key.package.lower() for key in install_actions} | {
        name.lower() for name in removal_actions
    }
    protected_changes = sorted(
        name
        for name in changed_names
        if any(fragment in name for fragment in PROTECTED_PACKAGE_FRAGMENTS)
    )
    if protected_changes:
        raise RehearsalError(
            "rollback simulation would change protected SDK packages: "
            + ", ".join(protected_changes)
        )

    necessity_probes: dict[str, str] = {}
    for removal in spec.get("solver_required_non_595_removals", {}):
        probe_command = [arg for arg in command if arg != f"{removal}-"]
        probe = _run(probe_command, env=environment)
        probe_removals = {
            line.split()[1]
            for line in probe.stdout.splitlines()
            if line.startswith("Remv ")
        }
        probe_summary = SUMMARY_LINE.search(probe.stdout)
        if probe_summary is None or removal not in probe_removals:
            raise RehearsalError(
                f"documented solver-required removal was not independently proven: {removal}"
            )
        if probe_removals != expected_removals:
            raise RehearsalError(
                f"solver-necessity probe changed the removal boundary: {removal}"
            )
        necessity_probes[removal] = "removed_by_solver_without_explicit_argument"

    return {
        "solver": "apt-get --simulate --no-download",
        "return_code": completed.returncode,
        "summary": summary,
        "exact_local_deb_inputs": len(archives),
        "already_installed_exact": len(exact_already_installed),
        "install_actions_verified": len(install_actions),
        "removal_actions_verified": len(removal_actions),
        "cuda_tensorrt_deepstream_changes": [],
        "network_downloads_allowed": False,
        "solver_required_removal_probes": necessity_probes,
        "dpkg_status_sha256": sha256_file(Path("/var/lib/dpkg/status")),
        "apt_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def current_platform_state() -> dict[str, Any]:
    dpkg_audit = _run(("dpkg", "--audit")).stdout.strip()
    if dpkg_audit:
        raise RehearsalError("current dpkg state is not clean")
    kernel = _run(("uname", "-r")).stdout.strip()
    proc_version_path = Path("/proc/driver/nvidia/version")
    _require_regular_file(proc_version_path)
    proc_version = proc_version_path.read_text(encoding="utf-8")
    version_match = re.search(r"\b(\d{3}\.\d+\.\d+)\b", proc_version)
    if version_match is None:
        raise RehearsalError("cannot determine loaded NVIDIA module version")
    loaded_version = version_match.group(1)
    if loaded_version != "595.71.05":
        raise RehearsalError(
            f"rehearsal source driver is not 595.71.05: {loaded_version}"
        )

    boot_kernels = sorted(
        path.name.removeprefix("vmlinuz-")
        for path in Path("/boot").glob("vmlinuz-*")
        if path.is_file()
    )
    if not boot_kernels or kernel not in boot_kernels:
        raise RehearsalError(
            "running kernel is absent from the installed boot-kernel set"
        )
    on_disk_versions: dict[str, str] = {}
    for boot_kernel in boot_kernels:
        version = _run(
            ("modinfo", "-k", boot_kernel, "-F", "version", "nvidia")
        ).stdout.strip()
        if version != loaded_version:
            raise RehearsalError(
                f"on-disk NVIDIA module mismatch for {boot_kernel}: {version}"
            )
        on_disk_versions[boot_kernel] = version

    dkms_output = _run(("dkms", "status")).stdout
    for boot_kernel in boot_kernels:
        expected = f"nvidia/{loaded_version}, {boot_kernel}, x86_64: installed"
        if expected not in dkms_output.splitlines():
            raise RehearsalError(
                f"DKMS lacks the loaded NVIDIA version for {boot_kernel}"
            )

    foreign_architectures = _run(
        ("dpkg", "--print-foreign-architectures")
    ).stdout.split()
    if "i386" not in foreign_architectures:
        raise RehearsalError("i386 is not enabled for the cached rollback libraries")
    cuda_default = Path("/usr/local/cuda").resolve(strict=True)
    if cuda_default.name != "cuda-13.0":
        raise RehearsalError(f"host CUDA default drifted: {cuda_default.name}")

    return {
        "source_driver": loaded_version,
        "running_kernel": kernel,
        "boot_kernel_modules": on_disk_versions,
        "dkms_entries_verified": len(boot_kernels),
        "dpkg_audit_empty": True,
        "foreign_architectures": foreign_architectures,
        "cuda_default": cuda_default.name,
        "gpu_api_used": False,
    }


def recovery_command(spec: Mapping[str, Any]) -> str:
    removals = " ".join(f"{name}-" for name in spec["explicit_removals"])
    return (
        "sudo apt-get -y --no-download --allow-downgrades "
        '--allow-change-held-packages install "$ROLLBACK_CACHE"/*.deb '
        f"{removals}"
    )


def recovery_plan(spec: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "order": "1",
            "action": "announce an exclusive maintenance window; stop DS8/DS9, every GPU owner, and the display manager; verify persistent console or SSH recovery",
        },
        {
            "order": "2",
            "action": "rerun this validator against the unchanged checkpoint and require status validated_rehearsal",
        },
        {
            "order": "3",
            "action": recovery_command(spec).replace(" -y ", " --simulate "),
        },
        {"order": "4", "action": recovery_command(spec)},
        {
            "order": "5",
            "action": "require empty `dpkg --audit`; require 580.167.08 DKMS and `modinfo` for every `/boot/vmlinuz-*`; run `sudo update-initramfs -u -k all`",
        },
        {
            "order": "6",
            "action": "reboot once; do not launch a GPU workload while userspace and the loaded kernel module disagree",
        },
        {
            "order": "7",
            "action": "run the complete post-rollback DS8 acceptance contract before restoring any appliance or DS9 service",
        },
    ]


def acceptance_contract() -> list[str]:
    return [
        "595-to-580 package transaction and initramfs evidence are preserved; dpkg audit is empty",
        "nvidia-smi, loaded module, every boot-kernel module, and DKMS agree on 580.167.08 with no current-boot Xid, API mismatch, GPU loss, or RmInitAdapter failure",
        "host CUDA remains 13.0, Python TensorRT remains 10.13.3.9, and DeepStream remains 8.0",
        "every configured DS8 TensorRT engine deserializes from its exact checkpointed bytes",
        "authenticated baseline `scripts/ds8_runtime_30s_gate.py` passes advancing REST/WS/world, RTSP DESCRIBE, strict acknowledged EOS, exit 0, and no forced kill",
        "authenticated V3DT `scripts/ds8_runtime_30s_gate.py --runtime-profile v3dt` passes the same lifecycle plus tracker/metadata contract",
        "decoded WebRTC, identity telemetry, MapAnything/depth/floorplan quality, persistence, and bounded CPU/GPU/memory behavior match the accepted DS8 baseline",
        "desktop and RDP recover, followed by the required supervised DS8 soak",
    ]


def rehearse(checkpoint: Path, spec_path: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    checkpoint_result = verify_checkpoint(
        checkpoint, spec["checkpoint_manifest_sha256"]
    )
    packages, package_result = verify_packages(checkpoint, spec)
    platform_result = current_platform_state()
    simulation_result = simulate_rollback(packages, spec)
    return {
        "schema_version": 1,
        "status": "validated_rehearsal",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rollback_executed": False,
        "host_mutation_performed": False,
        "checkpoint": checkpoint_result,
        "packages": package_result,
        "source_platform": platform_result,
        "simulation": simulation_result,
        "ordered_recovery": recovery_plan(spec),
        "post_rollback_ds8_acceptance": acceptance_contract(),
        "maintenance_reboot_required_for_execution": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-mutating offline rehearsal for the exact DS9-capable-host DS8 rollback"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="owner-only pre-driver checkpoint directory",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help="versioned exact rollback specification",
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically create an owner-only JSON evidence file; never overwrites",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = rehearse(args.checkpoint, args.spec)
        if args.output is not None:
            output = _absolute_unlinked_path(args.output)
            result["evidence_output"] = str(output)
            output_sha256 = write_private_atomic_json(output, result)
    except RehearsalError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    if args.output is not None:
        result["evidence_output_sha256"] = output_sha256
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
