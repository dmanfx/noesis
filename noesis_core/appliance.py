from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from threading import RLock
import time
from typing import Any, Callable, Mapping, TypeVar

from pydantic import ValidationError

from noesis_core.contracts.appliance import (
    DeploymentHealth,
    DeploymentSelector,
    DS8RuntimeSelector,
    DS9RuntimeSelector,
    RuntimeDeploymentContext,
    StateBaseline,
    StateRelease,
    WebSocketDeploymentHealth,
)
from noesis_core.contracts.health import CapabilityStatus
from noesis_core.health import CapabilityMonitor


MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_GIT_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_SAFE_INTEGER = (1 << 53) - 1
PRIVATE_FILE_MODE = 0o600
PRIVATE_DIRECTORY_MODE = 0o700
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_BOOT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,63}$")
_T = TypeVar("_T")

NOESIS_RUNTIME_DIRECTORY_ROOTS = (
    "config",
    "pipelines",
    "gst-plugins",
    "native_extensions",
    "artifacts/native",
    "plugins",
    "external/ds_preprocess_shim",
    "oai2-fe/dist",
    "DS9/artifacts",
    "DS9/native_extensions",
    "DS9/plugins",
    "DS9/gst-plugins",
    "DS9/pipelines",
)
NOESIS_RUNTIME_EXTENSION_MODULES = (
    "noesis_depth_meta_ext",
    "noesis_depth_tracking_tensor_ext",
    "noesis_latency_ext",
    "noesis_pose_meta_ext",
    "noesis_reid_meta_ext",
    "noesis_v3dt_meta_ext",
)
NOESIS_RUNTIME_MODEL_FILES = (
    "models/coco_labels.txt",
    "models/mapanything_depth/1/model.plan",
    "models/engines/reid_swin_tiny_aicity156_dyn_b16_fp16.engine",
    "models/engines/yolo26n-pose_b3_fp16.engine",
    "models/engines/depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
    "models/tracker_reid/resnet50_market1501.etlt",
    "models/tracker_reid/resnet50_market1501.etlt_b32_gpu0_fp16.engine",
    "models/bodypose3dnet/bodypose3dnet_accuracy.onnx",
    "models/engines/bodypose3dnet_accuracy_b1_fp16.engine",
    "models/deimv2_wholebody49/classes.txt",
    *(f"models/engines/yolo11{size}_b3_fp16.engine" for size in ("s", "m", "l")),
    "models/engines/yolo11s-seg_cust_fused.engine",
    *(f"models/engines/yolo11{size}-seg_cust.engine" for size in ("m", "l")),
    *(
        f"models/engines/yolo26{size}_dynamic_b1-3_fp16.engine"
        for size in ("n", "s", "m", "l", "x")
    ),
    *(
        f"models/engines/yolo26{size}-seg_fused_b3_fp16.engine"
        for size in ("n", "s", "m")
    ),
    "models/engines/rfdetr_n_384_b3_fp16.engine",
    "models/engines/rfdetr_s_512_b3_fp16.engine",
    "models/engines/rfdetr_m_576_b3_fp16.engine",
    "models/engines/rfdetr_seg_n_312_b3_fp16.engine",
    "models/engines/rfdetr_seg_s_384_b3_fp16.engine",
    "models/engines/rfdetr_seg_m_432_b3_fp16.engine",
    "models/engines/deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine",
    "models/engines/deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine",
)
NOESIS_RUNTIME_DIRECT_FILES = (
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/Makefile",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "nvdsparseseg_Yolo.cpp",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "nvdsparseseg_Yolo11.cpp",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/common.cpp",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/common.h",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/efficientNMSPlugin/efficientNMSInference.cu",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/efficientNMSPlugin/efficientNMSInference.h",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/efficientNMSPlugin/efficientNMSParameters.h",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/efficientNMSPlugin/efficientNMSPlugin.cpp",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/efficientNMSPlugin/efficientNMSPlugin.h",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/roiAlignPlugin/roiAlignKernel.cu",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/roiAlignPlugin/roiAlignKernel.h",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/roiAlignPlugin/roiAlignPlugin.cpp",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "trt_plugins/roiAlignPlugin/roiAlignPlugin.h",
    "external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/"
    "libnvdsinfer_custom_impl_Yolo_seg.so",
)
MAX_RUNTIME_FILES = 100_000
MAX_RUNTIME_BYTES = 4 * 1024 * 1024 * 1024
APPLIANCE_STATE_SCHEMAS = {
    "analytics_roi": 1,
    "identity": 1,
    "scene": 1,
    "world": 3,
}
APPLIANCE_STATE_FILES = {
    "analytics_config": "payload/analytics/nvdsanalytics.yaml",
    "analytics_exclude": "payload/analytics/config_nvdsanalytics_exclude.ini",
    "identity_store": "payload/identity.db",
    "scene_store": "payload/scene/scene_releases.sqlite3",
    "world_store": "payload/world.db",
}
FORBIDDEN_NESTED_LEASE_ENV = frozenset(
    {
        "NOESIS_STATE_RELEASE_LEASE",
        "NOESIS_STATE_RELEASE_LEASE_FILE",
        "NOESIS_DEPLOYMENT_LEASE_FILE",
        "MENON_STATE_RELEASE_LEASE",
        "MENON_STATE_RELEASE_LEASE_FILE",
        "MENON_APPLIANCE_STATE_LEASE_FILE",
    }
)


class ApplianceConfigurationError(ValueError):
    """Permanent selector, state-release, or producer binding failure."""


class ApplianceNotReadyError(RuntimeError):
    """The selected producer exists but has not proved advancing readiness."""


def _fail(message: str) -> None:
    raise ApplianceConfigurationError(message)


def canonical_json(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, indent=2, ensure_ascii=False, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def runtime_inventory_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        (json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def runtime_snapshot_digest(descriptor: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                dict(descriptor), ensure_ascii=False, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def _parse_unique_json(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ApplianceConfigurationError(f"{label} is not UTF-8") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ApplianceConfigurationError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except ApplianceConfigurationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ApplianceConfigurationError(f"{label} is invalid JSON") from exc


def _normalized_real_path(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or os.path.normpath(os.fspath(candidate)) != os.fspath(
        candidate
    ):
        _fail(f"{label} must be a normalized absolute path")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ApplianceConfigurationError(f"{label} is unavailable") from exc
    if resolved != candidate:
        _fail(f"{label} must not contain symlinks")
    return candidate


def _require_private_directory(path: Path, *, label: str) -> None:
    path = _normalized_real_path(path, label=label)
    try:
        info = path.lstat()
    except OSError as exc:
        raise ApplianceConfigurationError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != PRIVATE_DIRECTORY_MODE
    ):
        _fail(f"{label} must be an owner-owned mode-0700 real directory")


def _read_private_file(
    path: str | Path,
    *,
    label: str,
    maximum_bytes: int = MAX_CONTRACT_BYTES,
) -> tuple[Path, bytes]:
    candidate = _normalized_real_path(path, label=label)
    _require_private_directory(candidate.parent, label=f"{label} parent")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != PRIVATE_FILE_MODE
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            _fail(
                f"{label} must be an owner-owned, single-link, mode-0600 bounded file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"{label} was truncated while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"{label} grew while being read")
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            _fail(f"{label} changed while being read")
        return candidate, b"".join(chunks)
    except ApplianceConfigurationError:
        raise
    except OSError as exc:
        raise ApplianceConfigurationError(f"{label} could not be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_canonical_contract(
    path: str | Path,
    *,
    label: str,
    validator: Callable[[Any], _T],
) -> tuple[Path, bytes, _T]:
    resolved, payload = _read_private_file(path, label=label)
    parsed = _parse_unique_json(payload, label=label)
    try:
        value = validator(parsed)
    except ValidationError as exc:
        raise ApplianceConfigurationError(f"{label} violates its closed contract") from exc
    if canonical_json(value) != payload:
        _fail(f"{label} must be canonical JSON")
    return resolved, payload, value


def _run_git(root: Path, *arguments: str, allowed_status: tuple[int, ...] = ()) -> bytes:
    try:
        result = subprocess.run(
            ("git", "-C", os.fspath(root), *arguments),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ApplianceConfigurationError("Git checkout inspection failed") from exc
    if result.returncode != 0 and result.returncode not in allowed_status:
        detail = (result.stderr or result.stdout)[:2000].decode("utf-8", errors="replace")
        raise ApplianceConfigurationError(
            "Git checkout inspection failed" + (f": {detail.strip()}" if detail else "")
        )
    if len(result.stdout) > MAX_GIT_OUTPUT_BYTES:
        _fail("Git checkout inspection exceeded its output bound")
    return result.stdout


def _git_oid(payload: bytes, *, label: str) -> str:
    try:
        value = payload.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ApplianceConfigurationError(f"{label} is not an ASCII Git OID") from exc
    if re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", value) is None:
        _fail(f"{label} is not an exact Git OID")
    return value


@dataclass(frozen=True)
class CheckoutIdentity:
    revision: str
    tree: str
    submodules: tuple[str, ...]
    runtime_inventory_sha256: str
    runtime_file_count: int
    runtime_byte_count: int
    snapshot_sha256: str


def _same_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _contained_path(root: Path, candidate: Path) -> bool:
    try:
        return candidate != root and candidate.is_relative_to(root)
    except AttributeError:  # pragma: no cover - Python >=3.10 is required.
        try:
            candidate.relative_to(root)
            return candidate != root
        except ValueError:
            return False


def _hash_runtime_file(
    path: Path,
    before: os.stat_result,
    *,
    expected_real_path: Path,
) -> tuple[str, int]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not _same_entry(opened, before)
            or opened.st_size != before.st_size
            or opened.st_mtime_ns != before.st_mtime_ns
            or opened.st_nlink != 1
            or not stat.S_ISREG(opened.st_mode)
        ):
            _fail("Noesis runtime file changed during snapshot")
        try:
            opened_real_path = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ApplianceConfigurationError(
                "Noesis runtime file identity could not be retained"
            ) from exc
        if opened_real_path != expected_real_path:
            _fail("Noesis runtime file target changed during snapshot")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError as exc:
            raise ApplianceConfigurationError(
                "Noesis runtime file disappeared during snapshot"
            ) from exc
        if (
            not _same_entry(after, opened)
            or not _same_entry(current, opened)
            or after.st_size != total
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            _fail("Noesis runtime file changed during snapshot")
        return digest.hexdigest(), total
    except ApplianceConfigurationError:
        raise
    except OSError as exc:
        raise ApplianceConfigurationError(
            f"Noesis runtime file could not be hashed safely: {path}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass
class _RuntimeInventoryBuilder:
    root: Path
    uid: int
    rows: list[dict[str, Any]]
    seen: set[str]
    byte_count: int = 0

    def add_row(self, row: dict[str, Any]) -> None:
        relative = str(row.get("path") or "")
        target = row.get("target")
        if (
            re.fullmatch(r"[\x20-\x7e]+", relative) is None
            or (
                target is not None
                and re.fullmatch(r"[\x20-\x7e]+", str(target)) is None
            )
        ):
            _fail(
                "Noesis runtime inventory paths and link targets must be printable ASCII"
            )
        if not relative or relative in self.seen:
            _fail("Noesis runtime inventory contains a duplicate path")
        self.seen.add(relative)
        self.rows.append(row)
        if len(self.rows) > MAX_RUNTIME_FILES:
            _fail("Noesis runtime inventory exceeds its file-count bound")

    def add_file(
        self,
        absolute: Path,
        relative: str,
        *,
        containment_root: Path | None = None,
    ) -> None:
        containment = containment_root or self.root
        try:
            info = absolute.lstat()
            resolved = absolute.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ApplianceConfigurationError(
                f"Noesis runtime file is missing: {relative}"
            ) from exc
        if (
            info.st_uid != self.uid
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            _fail(f"Noesis runtime file is invalid: {relative}")
        if resolved != containment and not _contained_path(containment, resolved):
            _fail(f"Noesis runtime file escapes its bound root: {relative}")
        digest, total = _hash_runtime_file(
            absolute,
            info,
            expected_real_path=resolved,
        )
        self.byte_count += total
        if self.byte_count > MAX_RUNTIME_BYTES:
            _fail("Noesis runtime inventory exceeds its byte bound")
        self.add_row(
            {
                "path": relative,
                "type": "file",
                "mode": stat.S_IMODE(info.st_mode),
                "bytes": total,
                "sha256": digest,
            }
        )


def _compute_noesis_runtime_inventory(
    root: Path,
) -> tuple[str, int, int]:
    builder = _RuntimeInventoryBuilder(root=root, uid=os.getuid(), rows=[], seen=set())
    allowed_roots = tuple(root / relative for relative in NOESIS_RUNTIME_DIRECTORY_ROOTS)

    def visit(absolute: Path, relative: str) -> None:
        try:
            info = absolute.lstat()
        except OSError as exc:
            raise ApplianceConfigurationError(
                f"Noesis runtime entry is missing: {relative}"
            ) from exc
        if info.st_uid != builder.uid:
            _fail(f"Noesis runtime entry has the wrong owner: {relative}")
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            try:
                target = os.readlink(absolute)
                resolved = absolute.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ApplianceConfigurationError(
                    f"Noesis runtime symlink is invalid: {relative}"
                ) from exc
            if not any(
                resolved == allowed or _contained_path(allowed, resolved)
                for allowed in allowed_roots
            ):
                _fail(f"Noesis runtime symlink escapes its inventory: {relative}")
            builder.add_row(
                {"path": relative, "type": "symlink", "mode": mode, "target": target}
            )
            return
        if stat.S_ISDIR(info.st_mode):
            builder.add_row({"path": relative, "type": "directory", "mode": mode})
            try:
                names = sorted(entry.name for entry in os.scandir(absolute))
            except OSError as exc:
                raise ApplianceConfigurationError(
                    f"Noesis runtime directory cannot be inventoried: {relative}"
                ) from exc
            for name in names:
                visit(absolute / name, f"{relative}/{name}")
            try:
                after = absolute.lstat()
                after_names = sorted(entry.name for entry in os.scandir(absolute))
            except OSError as exc:
                raise ApplianceConfigurationError(
                    f"Noesis runtime directory changed: {relative}"
                ) from exc
            if (
                not _same_entry(info, after)
                or info.st_mtime_ns != after.st_mtime_ns
                or names != after_names
            ):
                _fail(f"Noesis runtime directory changed: {relative}")
            return
        builder.add_file(absolute, relative)

    for relative in NOESIS_RUNTIME_DIRECTORY_ROOTS:
        visit(root / relative, relative)
    for relative in NOESIS_RUNTIME_DIRECT_FILES:
        builder.add_file(root / relative, relative)

    try:
        root_names = sorted(entry.name for entry in os.scandir(root))
    except OSError as exc:
        raise ApplianceConfigurationError("Noesis checkout root cannot be inventoried") from exc
    for module_name in NOESIS_RUNTIME_EXTENSION_MODULES:
        matcher = re.compile(
            rf"^{re.escape(module_name)}(?:\.[A-Za-z0-9_-]+)+\.so$"
        )
        matches = [name for name in root_names if matcher.fullmatch(name)]
        if len(matches) != 1:
            _fail(
                f"Noesis runtime requires exactly one {module_name} extension; "
                f"found {len(matches)}"
            )
        builder.add_file(root / matches[0], matches[0])

    models_path = root / "models"
    try:
        models_entry = models_path.lstat()
        models_root = models_path.resolve(strict=True)
        model_root_info = models_root.lstat()
    except (OSError, RuntimeError) as exc:
        raise ApplianceConfigurationError("Noesis models root is missing") from exc
    if (
        models_entry.st_uid != builder.uid
        or model_root_info.st_uid != builder.uid
        or not stat.S_ISDIR(model_root_info.st_mode)
    ):
        _fail("Noesis models root must be an owner-controlled directory")
    if stat.S_ISLNK(models_entry.st_mode):
        builder.add_row(
            {
                "path": "models",
                "type": "external-directory-symlink",
                "mode": stat.S_IMODE(models_entry.st_mode),
                "target": os.readlink(models_path),
            }
        )
    elif stat.S_ISDIR(models_entry.st_mode) and models_root == models_path:
        builder.add_row(
            {
                "path": "models",
                "type": "directory",
                "mode": stat.S_IMODE(models_entry.st_mode),
            }
        )
    else:
        _fail("Noesis models root must be a real directory or one explicit symlink")
    for relative in NOESIS_RUNTIME_MODEL_FILES:
        builder.add_file(
            root / relative,
            relative,
            containment_root=models_root,
        )
    try:
        models_after = models_path.lstat()
        models_after_root = models_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ApplianceConfigurationError("Noesis models root changed") from exc
    if (
        not _same_entry(models_entry, models_after)
        or models_entry.st_mtime_ns != models_after.st_mtime_ns
        or models_after_root != models_root
    ):
        _fail("Noesis models root changed during snapshot")

    builder.rows.sort(key=lambda row: str(row["path"]))
    inventory_sha256 = runtime_inventory_digest(builder.rows)
    file_count = sum(row.get("type") == "file" for row in builder.rows)
    return inventory_sha256, int(file_count), builder.byte_count


def compute_noesis_checkout_identity(root: str | Path) -> CheckoutIdentity:
    """Reproduce Menon's frozen ``noesis-runtime-v1`` identity byte-for-byte."""

    checkout = _normalized_real_path(root, label="Noesis checkout root")
    info = checkout.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
    ):
        _fail("Noesis checkout root must be an owner-owned real directory")
    revision = _git_oid(
        _run_git(checkout, "rev-parse", "--verify", "HEAD"),
        label="Noesis checkout revision",
    )
    tree = _git_oid(
        _run_git(checkout, "rev-parse", "--verify", "HEAD^{tree}"),
        label="Noesis checkout tree",
    )
    status = _run_git(
        checkout,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        _fail("Noesis deployment checkout is not clean, including untracked files")
    submodule_raw = _run_git(
        checkout,
        "submodule",
        "status",
        "--recursive",
    )
    submodule_text = submodule_raw.decode("utf-8", errors="strict")
    submodules = tuple(sorted(line for line in submodule_text.strip().split("\n") if line))
    if any(row.startswith(("-", "+", "U")) for row in submodules):
        _fail("Noesis checkout submodules differ from their committed revisions")
    runtime_inventory_sha256, runtime_file_count, runtime_byte_count = (
        _compute_noesis_runtime_inventory(checkout)
    )
    try:
        checkout_after = checkout.lstat()
    except OSError as exc:
        raise ApplianceConfigurationError(
            "Noesis checkout root changed during snapshot"
        ) from exc
    revision_after = _git_oid(
        _run_git(checkout, "rev-parse", "--verify", "HEAD"),
        label="Noesis checkout revision",
    )
    tree_after = _git_oid(
        _run_git(checkout, "rev-parse", "--verify", "HEAD^{tree}"),
        label="Noesis checkout tree",
    )
    status_after = _run_git(
        checkout,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    submodule_after_raw = _run_git(
        checkout,
        "submodule",
        "status",
        "--recursive",
    )
    submodules_after = tuple(
        sorted(
            line
            for line in submodule_after_raw.decode("utf-8", errors="strict")
            .strip()
            .split("\n")
            if line
        )
    )
    if (
        not _same_entry(info, checkout_after)
        or info.st_mtime_ns != checkout_after.st_mtime_ns
        or revision_after != revision
        or tree_after != tree
        or status_after
        or submodules_after != submodules
        or any(row.startswith(("-", "+", "U")) for row in submodules_after)
    ):
        _fail("Noesis checkout changed during runtime snapshot admission")
    identity = {
        "algorithm": "noesis-runtime-v1",
        "revision": revision,
        "tree": tree,
        "submodules": list(submodules),
        "runtime_inventory_sha256": runtime_inventory_sha256,
        "runtime_file_count": runtime_file_count,
        "runtime_byte_count": runtime_byte_count,
    }
    snapshot = runtime_snapshot_digest(identity)
    return CheckoutIdentity(
        revision=revision,
        tree=tree,
        submodules=submodules,
        runtime_inventory_sha256=runtime_inventory_sha256,
        runtime_file_count=runtime_file_count,
        runtime_byte_count=runtime_byte_count,
        snapshot_sha256=snapshot,
    )


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return candidate != root and candidate.is_relative_to(root)
    except AttributeError:  # pragma: no cover - Python >=3.10 is required in production.
        try:
            candidate.relative_to(root)
            return candidate != root
        except ValueError:
            return False


def _paths_overlap(left: str | Path, right: str | Path) -> bool:
    try:
        common = os.path.commonpath((os.fspath(left), os.fspath(right)))
    except ValueError:
        return False
    return common in {os.fspath(left), os.fspath(right)}


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError) as exc:
        raise ApplianceConfigurationError("kernel boot identity is unavailable") from exc
    if _BOOT_ID_RE.fullmatch(value) is None:
        _fail("kernel boot identity is not canonical")
    return value


@dataclass(frozen=True)
class ApplianceStateEvidence:
    release: StateRelease
    baseline: StateBaseline
    manifest_path: Path
    baseline_path: Path
    runtime_files: Mapping[str, Path]
    build_directory: Path
    manifest_identity: tuple[int, int]
    baseline_identity: tuple[int, int]
    release_root_identity: tuple[int, int]
    runtime_file_identities: Mapping[str, tuple[int, int]]
    runtime_parent_identities: Mapping[str, tuple[int, int]]
    build_directory_identity: tuple[int, int]


class DeploymentHealthBinding:
    """Stable runtime identity used by the REST and WebSocket health surfaces."""

    def __init__(
        self,
        *,
        deployment_id: str,
        selector_sha256: str,
        state_release_id: str,
        runtime_family: str,
        runtime_variant: str,
        software_revision: str,
        boot_id: str,
    ) -> None:
        self.deployment_id = deployment_id
        self.selector_sha256 = selector_sha256
        self.state_release_id = state_release_id
        self.runtime_family = runtime_family
        self.runtime_variant = runtime_variant
        self.software_revision = software_revision
        self.boot_id = boot_id
        self._producer: tuple[str, str] | None = None
        self._last_generated_at_us = 0
        self._lock = RLock()

    def bind_producer(self, *, instance_id: str, run_id: str) -> None:
        identity = (str(instance_id), str(run_id))
        if not all(value and value == value.strip() for value in identity):
            _fail("appliance producer identity is not canonical")
        with self._lock:
            if self._producer is not None and self._producer != identity:
                _fail("appliance producer identity cannot change during a runtime")
            self._producer = identity

    def _generated_at_us(self) -> int:
        now = time.time_ns() // 1_000
        with self._lock:
            value = max(int(now), self._last_generated_at_us + 1)
            if value < 1 or value > MAX_SAFE_INTEGER:
                raise ApplianceNotReadyError("deployment health clock is invalid")
            self._last_generated_at_us = value
            return value

    def deployment_health(self, monitor: CapabilityMonitor) -> DeploymentHealth:
        with self._lock:
            producer = self._producer
        if producer is None:
            raise ApplianceNotReadyError("deployment producer is not bound")
        generated_at_us = self._generated_at_us()
        snapshot = monitor.snapshot(generated_at_us=generated_at_us)
        if (snapshot.instance_id, snapshot.run_id) != producer:
            raise ApplianceNotReadyError(
                "capability and deployment producer identities differ"
            )
        by_name = {row.capability: row for row in snapshot.capabilities}
        for capability in ("tracking_observations", "global_world"):
            row = by_name.get(capability)
            if row is None or row.status != CapabilityStatus.HEALTHY:
                raise ApplianceNotReadyError(
                    f"required appliance capability is not healthy: {capability}"
                )
        return DeploymentHealth(
            contract="noesis.appliance.deployment_health",
            contract_version=1,
            deployment_id=self.deployment_id,
            selector_sha256=self.selector_sha256,
            state_release_id=self.state_release_id,
            runtime_family=self.runtime_family,  # type: ignore[arg-type]
            runtime_variant=self.runtime_variant,
            instance_id=producer[0],
            run_id=producer[1],
            boot_id=self.boot_id,
            software_revision=self.software_revision,
            generated_at_us=generated_at_us,
            ready=True,
        )

    def websocket_health(self, monitor: CapabilityMonitor) -> WebSocketDeploymentHealth:
        health = self.deployment_health(monitor)
        payload = health.model_dump(mode="json")
        payload["type"] = "health"
        payload["contract"] = "noesis.ws.health"
        payload["contract_version"] = 2
        return WebSocketDeploymentHealth.model_validate(payload)


class ApplianceBinding(DeploymentHealthBinding):
    """Exact selected deployment identity shared by DS8 and DS9 adapters.

    The Menon adapter owns the single full-lifetime shared state-release lease.
    This object deliberately has no lock path and acquires no nested lease; it
    validates the immutable selector, checkout, release, and inherited identity
    while the orchestrator-held lease is already in force.
    """

    def __init__(
        self,
        *,
        selector: DeploymentSelector,
        selector_path: Path,
        selector_sha256: str,
        state: ApplianceStateEvidence,
        boot_id: str,
        selector_file_identity: tuple[int, int],
    ) -> None:
        super().__init__(
            deployment_id=selector.deployment_id,
            selector_sha256=selector_sha256,
            state_release_id=state.release.release_id,
            runtime_family=selector.runtime.family,
            runtime_variant=selector.runtime.variant,
            software_revision=selector.noesis_checkout.software_revision,
            boot_id=boot_id,
        )
        self.selector = selector
        self.selector_path = selector_path
        self.state = state
        self.selector_file_identity = selector_file_identity


def _validate_state_release(selector: DeploymentSelector) -> ApplianceStateEvidence:
    manifest_path, manifest_payload, release = _read_canonical_contract(
        selector.state_release.manifest_path,
        label="state release manifest",
        validator=StateRelease.model_validate,
    )
    manifest_info = manifest_path.lstat()
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if (
        manifest_sha256 != selector.state_release.manifest_sha256
        or release.release_id != selector.state_release.release_id
    ):
        _fail("state release manifest does not match the selector binding")
    release_root = _normalized_real_path(release.root, label="state release root")
    _require_private_directory(release_root, label="state release root")
    if any(
        _paths_overlap(os.fspath(release_root), checkout.root)
        for checkout in (selector.menon_checkout, selector.noesis_checkout)
    ):
        _fail("state release root must be disjoint from both deployment checkouts")
    if not _inside(release_root, manifest_path):
        _fail("state release manifest must be stored beneath its release root")
    if release.schemas != APPLIANCE_STATE_SCHEMAS:
        _fail(
            "state release schemas must be exactly analytics_roi=1, identity=1, "
            "scene=1, world=3"
        )
    baseline_path, baseline_payload, baseline = _read_canonical_contract(
        release.baseline.inventory_path,
        label="state activation baseline",
        validator=StateBaseline.model_validate,
    )
    baseline_info = baseline_path.lstat()
    if not _inside(release_root, baseline_path) or baseline_path == manifest_path:
        _fail("state activation baseline path is outside its release")
    if (
        hashlib.sha256(baseline_payload).hexdigest()
        != release.baseline.inventory_sha256
        or baseline.release_id != release.release_id
        or len(baseline.files) != release.baseline.file_count
        or sum(row.bytes for row in baseline.files.values())
        != release.baseline.byte_count
    ):
        _fail("state activation baseline does not match its release binding")
    missing_state_files = sorted(set(APPLIANCE_STATE_FILES.values()) - set(baseline.files))
    if missing_state_files:
        _fail(
            "state activation baseline lacks required appliance payloads: "
            + ", ".join(missing_state_files)
        )
    if any(
        relative == "runtime/noesis-build"
        or relative.startswith("runtime/noesis-build/")
        for relative in baseline.files
    ):
        _fail("appliance runtime build directory must remain outside the state baseline")
    runtime_files: dict[str, Path] = {}
    runtime_file_identities: dict[str, tuple[int, int]] = {}
    runtime_parent_identities: dict[str, tuple[int, int]] = {}
    for role, relative in APPLIANCE_STATE_FILES.items():
        path = release_root / relative
        resolved = _normalized_real_path(path, label=f"appliance state {role}")
        _require_private_directory(resolved.parent, label=f"appliance state {role} parent")
        info = resolved.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != PRIVATE_FILE_MODE
            or not os.access(resolved, os.R_OK | os.W_OK)
        ):
            _fail(
                f"appliance state {role} must be an owner-owned writable mode-0600 file"
            )
        runtime_files[role] = resolved
        runtime_file_identities[role] = (info.st_dev, info.st_ino)
        parent_info = resolved.parent.lstat()
        runtime_parent_identities[role] = (parent_info.st_dev, parent_info.st_ino)
    build_directory = release_root / "runtime" / "noesis-build"
    _require_private_directory(
        build_directory,
        label="appliance runtime build directory",
    )
    release_root_info = release_root.lstat()
    build_directory_info = build_directory.lstat()
    return ApplianceStateEvidence(
        release=release,
        baseline=baseline,
        manifest_path=manifest_path,
        baseline_path=baseline_path,
        runtime_files=runtime_files,
        build_directory=build_directory,
        manifest_identity=(manifest_info.st_dev, manifest_info.st_ino),
        baseline_identity=(baseline_info.st_dev, baseline_info.st_ino),
        release_root_identity=(release_root_info.st_dev, release_root_info.st_ino),
        runtime_file_identities=runtime_file_identities,
        runtime_parent_identities=runtime_parent_identities,
        build_directory_identity=(
            build_directory_info.st_dev,
            build_directory_info.st_ino,
        ),
    )


def require_appliance_state_current(binding: ApplianceBinding) -> None:
    """Re-admit the selected state cohort immediately before runtime use."""

    state = binding.state
    selector_path, selector_payload, selector = _read_canonical_contract(
        binding.selector_path,
        label="deployment selector",
        validator=DeploymentSelector.model_validate,
    )
    selector_info = selector_path.lstat()
    if (
        selector_path != binding.selector_path
        or (selector_info.st_dev, selector_info.st_ino)
        != binding.selector_file_identity
        or hashlib.sha256(selector_payload).hexdigest() != binding.selector_sha256
        or selector != binding.selector
    ):
        _fail("deployment selector changed after appliance admission")
    manifest_path, manifest_payload, release = _read_canonical_contract(
        state.manifest_path,
        label="state release manifest",
        validator=StateRelease.model_validate,
    )
    manifest_info = manifest_path.lstat()
    if (
        manifest_path != state.manifest_path
        or (manifest_info.st_dev, manifest_info.st_ino) != state.manifest_identity
        or hashlib.sha256(manifest_payload).hexdigest()
        != binding.selector.state_release.manifest_sha256
        or release != state.release
    ):
        _fail("state release manifest changed after appliance admission")
    baseline_path, baseline_payload, baseline = _read_canonical_contract(
        state.baseline_path,
        label="state activation baseline",
        validator=StateBaseline.model_validate,
    )
    baseline_info = baseline_path.lstat()
    if (
        baseline_path != state.baseline_path
        or (baseline_info.st_dev, baseline_info.st_ino) != state.baseline_identity
        or hashlib.sha256(baseline_payload).hexdigest()
        != release.baseline.inventory_sha256
        or baseline != state.baseline
    ):
        _fail("state activation baseline changed after appliance admission")
    release_root = _normalized_real_path(release.root, label="state release root")
    _require_private_directory(release_root, label="state release root")
    release_info = release_root.lstat()
    if (release_info.st_dev, release_info.st_ino) != state.release_root_identity:
        _fail("state release root changed after appliance admission")
    for role, selected_path in state.runtime_files.items():
        expected_path = release_root / APPLIANCE_STATE_FILES[role]
        current = _normalized_real_path(expected_path, label=f"appliance state {role}")
        if current != selected_path:
            _fail(f"appliance state {role} path changed after admission")
        _require_private_directory(current.parent, label=f"appliance state {role} parent")
        parent_info = current.parent.lstat()
        if (parent_info.st_dev, parent_info.st_ino) != state.runtime_parent_identities[role]:
            _fail(f"appliance state {role} parent changed after admission")
        info = current.lstat()
        if (
            (info.st_dev, info.st_ino) != state.runtime_file_identities[role]
            or not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != PRIVATE_FILE_MODE
            or not os.access(current, os.R_OK | os.W_OK)
        ):
            _fail(f"appliance state {role} changed after admission")
    _require_private_directory(
        state.build_directory,
        label="appliance runtime build directory",
    )
    build_info = state.build_directory.lstat()
    if (build_info.st_dev, build_info.st_ino) != state.build_directory_identity:
        _fail("appliance runtime build directory changed after admission")


def _require_environment_binding(
    *,
    env: Mapping[str, str],
    selector: DeploymentSelector,
    selector_path: Path,
    selector_sha256: str,
    state: ApplianceStateEvidence,
) -> None:
    inherited_leases = sorted(FORBIDDEN_NESTED_LEASE_ENV & set(env))
    if inherited_leases:
        _fail(
            "Noesis must not receive or acquire the Menon-owned state lease: "
            + ", ".join(inherited_leases)
        )
    required = {
        "NOESIS_DEPLOYMENT_ID": selector.deployment_id,
        "NOESIS_DEPLOYMENT_SELECTOR_FILE": os.fspath(selector_path),
        "NOESIS_DEPLOYMENT_SELECTOR_SHA256": selector_sha256,
        "NOESIS_STATE_RELEASE_ID": state.release.release_id,
        "NOESIS_STATE_RELEASE_MANIFEST": os.fspath(state.manifest_path),
        "NOESIS_STATE_RELEASE_ROOT": state.release.root,
        "NOESIS_RUNTIME_FAMILY": selector.runtime.family,
        "NOESIS_ANALYTICS_CONFIG": os.fspath(state.runtime_files["analytics_config"]),
        "NOESIS_ANALYTICS_EXCLUDE_CONFIG": os.fspath(
            state.runtime_files["analytics_exclude"]
        ),
        "NOESIS_IDENTITY_V2_STORE": os.fspath(state.runtime_files["identity_store"]),
        "NOESIS_SCENE_STORE_PATH": os.fspath(state.runtime_files["scene_store"]),
        "NOESIS_WORLD_JOURNAL_PATH": os.fspath(state.runtime_files["world_store"]),
        "NOESIS_BUILD_DIR": os.fspath(state.build_directory),
    }
    for name, expected in required.items():
        if str(env.get(name, "")) != expected:
            _fail(f"inherited appliance binding {name} does not match the selector")


def load_appliance_binding(
    *,
    selector_file: str | Path,
    selector_sha256: str,
    repo_root: str | Path,
    env: Mapping[str, str],
    expected_family: str,
    expected_profile: str | None = None,
    expected_size: str | None = None,
    expected_tracking_mode: str | None = None,
    expected_lane: str | None = None,
    verify_checkout: bool = True,
) -> ApplianceBinding:
    if _SHA256_RE.fullmatch(str(selector_sha256)) is None:
        _fail("selector SHA-256 must contain exactly 64 lowercase hex characters")
    selector_path, selector_payload, selector = _read_canonical_contract(
        selector_file,
        label="deployment selector",
        validator=DeploymentSelector.model_validate,
    )
    observed_sha256 = hashlib.sha256(selector_payload).hexdigest()
    if observed_sha256 != selector_sha256:
        _fail("deployment selector digest does not match --selector-sha256")
    if selector_path.name != f"{selector_sha256}.json":
        _fail("deployment selector path is not content-addressed by its digest")
    if selector.runtime.family != expected_family:
        _fail(
            f"selected runtime family is {selector.runtime.family}, not {expected_family}"
        )
    checkout_root = _normalized_real_path(repo_root, label="runtime checkout root")
    selector_checkout_root = _normalized_real_path(
        selector.noesis_checkout.root, label="selected Noesis checkout root"
    )
    if checkout_root != selector_checkout_root:
        _fail("running Noesis checkout root does not match the selector")
    if verify_checkout:
        observed_checkout = compute_noesis_checkout_identity(checkout_root)
        if (
            observed_checkout.revision != selector.noesis_checkout.software_revision
            or observed_checkout.snapshot_sha256
            != selector.noesis_checkout.snapshot_sha256
        ):
            _fail("running Noesis checkout identity does not match the selector")
    if isinstance(selector.runtime, DS8RuntimeSelector):
        expected = (
            expected_profile,
            expected_size,
            expected_tracking_mode,
        )
        observed = (
            selector.runtime.pgie_profile,
            selector.runtime.model_size,
            selector.runtime.tracking_mode,
        )
        if expected != observed:
            _fail(
                "DS8 runtime arguments do not match the selected profile/size/tracking tuple"
            )
    elif isinstance(selector.runtime, DS9RuntimeSelector):
        if expected_lane is not None and expected_lane != selector.runtime.lane:
            _fail("DS9 supervisor lane does not match the deployment selector")
    state = _validate_state_release(selector)
    _require_environment_binding(
        env=env,
        selector=selector,
        selector_path=selector_path,
        selector_sha256=selector_sha256,
        state=state,
    )
    selector_info = selector_path.lstat()
    return ApplianceBinding(
        selector=selector,
        selector_path=selector_path,
        selector_sha256=selector_sha256,
        state=state,
        boot_id=_boot_id(),
        selector_file_identity=(selector_info.st_dev, selector_info.st_ino),
    )


def optional_appliance_binding(
    *,
    selector_file: str | Path | None,
    selector_sha256: str | None,
    **kwargs: Any,
) -> ApplianceBinding | None:
    file_text = str(selector_file or "").strip()
    digest_text = str(selector_sha256 or "").strip()
    if not file_text and not digest_text:
        return None
    if not file_text or not digest_text:
        _fail(
            "--deployment-selector and --selector-sha256 must be supplied together"
        )
    return load_appliance_binding(
        selector_file=file_text,
        selector_sha256=digest_text,
        **kwargs,
    )


def runtime_deployment_context(binding: DeploymentHealthBinding) -> RuntimeDeploymentContext:
    return RuntimeDeploymentContext(
        contract="noesis.appliance.runtime_context",
        contract_version=1,
        deployment_id=binding.deployment_id,
        selector_sha256=binding.selector_sha256,
        state_release_id=binding.state_release_id,
        runtime_family=binding.runtime_family,  # type: ignore[arg-type]
        runtime_variant=binding.runtime_variant,
        boot_id=binding.boot_id,
        software_revision=binding.software_revision,
    )


def runtime_context_environment(
    binding: DeploymentHealthBinding,
) -> dict[str, str]:
    context = runtime_deployment_context(binding)
    payload = json.dumps(
        context.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "NOESIS_APPLIANCE_RUNTIME_CONTEXT": payload,
        "NOESIS_APPLIANCE_RUNTIME_CONTEXT_SHA256": hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest(),
    }


def optional_runtime_context_binding(
    env: Mapping[str, str],
) -> DeploymentHealthBinding | None:
    raw = str(env.get("NOESIS_APPLIANCE_RUNTIME_CONTEXT", ""))
    expected_sha256 = str(
        env.get("NOESIS_APPLIANCE_RUNTIME_CONTEXT_SHA256", "")
    )
    if not raw and not expected_sha256:
        return None
    inherited_leases = sorted(FORBIDDEN_NESTED_LEASE_ENV & set(env))
    if inherited_leases:
        _fail(
            "Noesis must not receive or acquire the Menon-owned state lease: "
            + ", ".join(inherited_leases)
        )
    if not raw or _SHA256_RE.fullmatch(expected_sha256) is None:
        _fail("appliance runtime context and digest must be supplied together")
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != expected_sha256:
        _fail("appliance runtime context digest does not match")
    parsed = _parse_unique_json(raw.encode("utf-8"), label="appliance runtime context")
    try:
        context = RuntimeDeploymentContext.model_validate(parsed)
    except ValidationError as exc:
        raise ApplianceConfigurationError(
            "appliance runtime context violates its closed contract"
        ) from exc
    canonical = json.dumps(
        context.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if canonical != raw:
        _fail("appliance runtime context is not canonical JSON")
    return DeploymentHealthBinding(
        deployment_id=context.deployment_id,
        selector_sha256=context.selector_sha256,
        state_release_id=context.state_release_id,
        runtime_family=context.runtime_family,
        runtime_variant=context.runtime_variant,
        software_revision=context.software_revision,
        boot_id=context.boot_id,
    )


__all__ = [
    "ApplianceBinding",
    "ApplianceConfigurationError",
    "ApplianceNotReadyError",
    "ApplianceStateEvidence",
    "CheckoutIdentity",
    "DeploymentHealthBinding",
    "FORBIDDEN_NESTED_LEASE_ENV",
    "canonical_json",
    "compute_noesis_checkout_identity",
    "load_appliance_binding",
    "optional_appliance_binding",
    "optional_runtime_context_binding",
    "runtime_context_environment",
    "runtime_deployment_context",
    "runtime_inventory_digest",
    "runtime_snapshot_digest",
    "require_appliance_state_current",
]
