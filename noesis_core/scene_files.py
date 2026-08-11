from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


MAX_SCENE_MANIFEST_BYTES = 5 * 1024 * 1024
MAX_SCENE_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_AUTHORED_SCENE_BYTES = 64 * 1024 * 1024
MAX_SCENE_VALIDATION_BYTES = 5 * 1024 * 1024
MAX_SCENE_RELEASE_BYTES = 1024 * 1024 * 1024
MAX_SCENE_CAMERAS = 16
MAX_SCENE_CAMERA_ARTIFACTS = 128
MAX_SCENE_AUTHORED_DEPENDENCIES = 64
MAX_SCENE_RELEASE_FILES = 512
MAX_OBJ_LINES = 1_000_000
MAX_MTL_BYTES = 4 * 1024 * 1024
MAX_MTL_LINES = 100_000
MAX_SCENE_TEXT_LINE_BYTES = 64 * 1024

_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_DIRECTORY_FLAGS = _READ_FLAGS | getattr(os, "O_DIRECTORY", 0)
_CREATE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_RENAME_NOREPLACE = 1


class SceneFileError(RuntimeError):
    """Raised when immutable scene bytes cannot be handled safely."""


@dataclass(frozen=True)
class VerifiedSceneFile:
    data: bytes
    sha256: str
    size_bytes: int


def absolute_path_without_resolving(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        rendered = os.fspath(candidate)
    except TypeError as exc:
        raise SceneFileError("scene path is invalid") from exc
    if "\x00" in rendered:
        raise SceneFileError("scene path contains a NUL byte")
    return Path(os.path.abspath(rendered))


def normalized_scene_relative_path(value: str, *, label: str) -> str:
    raw = str(value)
    if not raw or "\\" in raw or "\x00" in raw:
        raise SceneFileError(f"{label} must be a normalized relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts:
        raise SceneFileError(f"{label} must be a normalized relative path")
    for part in path.parts:
        if (
            part in {"", ".", ".."}
            or part != part.strip()
            or len(part.encode("utf-8")) > 255
            or any(ord(char) < 32 or ord(char) == 127 for char in part)
        ):
            raise SceneFileError(f"{label} must be a normalized relative path")
    if raw != path.as_posix():
        raise SceneFileError(f"{label} must be a normalized relative path")
    return raw


def load_strict_json(payload: bytes, *, label: str) -> Any:
    try:
        text = bytes(payload).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SceneFileError(f"{label} is not valid UTF-8 JSON") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SceneFileError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise SceneFileError(f"{label} contains non-finite JSON value {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except SceneFileError:
        raise
    except json.JSONDecodeError as exc:
        raise SceneFileError(f"{label} is invalid JSON") from exc


def _open_directory(path: Path, *, label: str, create: bool = False) -> int:
    absolute = absolute_path_without_resolving(path)
    descriptor = os.open(os.sep, _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise SceneFileError(f"{label} directory is missing") from None
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise SceneFileError(f"{label} directory cannot be created safely") from exc
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except OSError as exc:
                    raise SceneFileError(f"{label} directory cannot be opened safely") from exc
            except OSError as exc:
                raise SceneFileError(
                    f"{label} directory contains a symlink or non-directory component"
                ) from exc
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise SceneFileError(f"{label} path component is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def ensure_scene_directory(path: str | Path, *, label: str) -> Path:
    absolute = absolute_path_without_resolving(path)
    descriptor = _open_directory(absolute, label=label, create=True)
    os.close(descriptor)
    return absolute


def _relative_parts(relative_path: str, *, label: str) -> tuple[str, ...]:
    normalized = normalized_scene_relative_path(relative_path, label=label)
    return PurePosixPath(normalized).parts


def _open_root_file(
    root: Path,
    parts: tuple[str, ...],
    *,
    label: str,
) -> tuple[int, int, str]:
    root_descriptor = _open_directory(root, label=f"{label} root")
    parent_descriptor = root_descriptor
    try:
        for part in parts[:-1]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
            except OSError as exc:
                raise SceneFileError(
                    f"{label} contains a symlink, missing, or non-directory path component"
                ) from exc
            if parent_descriptor != root_descriptor:
                os.close(parent_descriptor)
            parent_descriptor = child
        try:
            file_descriptor = os.open(parts[-1], _READ_FLAGS, dir_fd=parent_descriptor)
        except FileNotFoundError:
            raise SceneFileError(f"{label} is missing") from None
        except OSError as exc:
            raise SceneFileError(f"{label} cannot be opened without following links") from exc
        if parent_descriptor == root_descriptor:
            root_descriptor = -1
        return file_descriptor, parent_descriptor, parts[-1]
    except Exception:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
            if parent_descriptor == root_descriptor:
                root_descriptor = -1
        if root_descriptor >= 0 and root_descriptor != parent_descriptor:
            os.close(root_descriptor)
            root_descriptor = -1
        raise
    finally:
        if root_descriptor >= 0 and root_descriptor != parent_descriptor:
            os.close(root_descriptor)


def _path_identity(root: Path, parts: tuple[str, ...], *, label: str) -> tuple[int, int]:
    descriptor, parent, leaf = _open_root_file(root, parts, label=label)
    try:
        info = os.fstat(descriptor)
        try:
            path_info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except OSError as exc:
            raise SceneFileError(f"{label} changed while it was being inspected") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or not stat.S_ISREG(path_info.st_mode)
            or info.st_nlink != 1
            or path_info.st_nlink != 1
            or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino)
        ):
            raise SceneFileError(
                f"{label} must remain one single-link regular file"
            )
        return int(info.st_dev), int(info.st_ino)
    finally:
        os.close(descriptor)
        os.close(parent)


def _read_verified_root_file(
    root: Path,
    parts: tuple[str, ...],
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None,
    expected_size: int | None,
) -> VerifiedSceneFile:
    limit = int(max_bytes)
    if limit < 0:
        raise ValueError("max_bytes cannot be negative")
    descriptor, parent, leaf = _open_root_file(root, parts, label=label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SceneFileError(f"{label} must be a regular file")
        if before.st_nlink != 1:
            raise SceneFileError(f"{label} must have exactly one hard link")
        if before.st_size > limit:
            raise SceneFileError(f"{label} exceeds the {limit}-byte limit")
        if expected_size is not None and before.st_size != int(expected_size):
            raise SceneFileError(
                f"{label} size mismatch: expected={expected_size} actual={before.st_size}"
            )

        chunks: list[bytes] = []
        consumed = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, limit - consumed + 1))
            if not block:
                break
            consumed += len(block)
            if consumed > limit:
                raise SceneFileError(f"{label} exceeds the {limit}-byte limit")
            if expected_size is not None and consumed > int(expected_size):
                raise SceneFileError(
                    f"{label} size mismatch: expected={expected_size} actual>{expected_size}"
                )
            chunks.append(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise SceneFileError(f"{label} changed while it was being read")
        try:
            current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except OSError as exc:
            raise SceneFileError(f"{label} changed while it was being read") from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise SceneFileError(f"{label} changed while it was being read")
    finally:
        os.close(descriptor)
        os.close(parent)

    reopened_identity = _path_identity(root, parts, label=label)
    if reopened_identity != (int(before.st_dev), int(before.st_ino)):
        raise SceneFileError(f"{label} path changed while it was being read")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_size is not None and len(payload) != int(expected_size):
        raise SceneFileError(
            f"{label} size mismatch: expected={expected_size} actual={len(payload)}"
        )
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
        raise SceneFileError(f"{label} fingerprint mismatch")
    return VerifiedSceneFile(
        data=payload,
        sha256=actual_sha256,
        size_bytes=len(payload),
    )


def read_scene_root_file(
    root: str | Path,
    relative_path: str,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> VerifiedSceneFile:
    absolute_root = absolute_path_without_resolving(root)
    parts = _relative_parts(relative_path, label=label)
    return _read_verified_root_file(
        absolute_root,
        parts,
        label=label,
        max_bytes=max_bytes,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )


def read_scene_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> VerifiedSceneFile:
    absolute = absolute_path_without_resolving(path)
    if absolute.parent == absolute:
        raise SceneFileError(f"{label} must name a file")
    return _read_verified_root_file(
        absolute.parent,
        (absolute.name,),
        label=label,
        max_bytes=max_bytes,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )


def _write_all(descriptor: int, payload: bytes, *, label: str) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SceneFileError(f"{label} write made no progress")
        view = view[written:]


def materialize_scene_file(
    destination: str | Path,
    payload: bytes,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> Path:
    body = bytes(payload)
    actual_sha256 = hashlib.sha256(body).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise SceneFileError(f"{label} payload fingerprint mismatch")
    destination_path = absolute_path_without_resolving(destination)
    parent = ensure_scene_directory(destination_path.parent, label=f"{label} parent")
    leaf = destination_path.name
    if not leaf or leaf in {".", ".."}:
        raise SceneFileError(f"{label} destination is invalid")

    try:
        existing = read_scene_file(
            destination_path,
            label=label,
            max_bytes=len(body),
            expected_sha256=actual_sha256,
            expected_size=len(body),
        )
    except SceneFileError as exc:
        try:
            os.lstat(destination_path)
        except FileNotFoundError:
            pass
        else:
            raise SceneFileError(
                f"immutable scene asset path contains unsafe or different content: {destination_path}"
            ) from exc
    else:
        if existing.sha256 == actual_sha256:
            return destination_path

    parent_descriptor = _open_directory(parent, label=f"{label} parent")
    temporary_name = f".{leaf}.tmp-{secrets.token_hex(12)}"
    descriptor = -1
    published = False
    try:
        descriptor = os.open(temporary_name, _CREATE_FLAGS, 0o600, dir_fd=parent_descriptor)
        _write_all(descriptor, body, label=label)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            _rename_noreplace(parent_descriptor, temporary_name, leaf)
            published = True
        except FileExistsError:
            read_scene_file(
                destination_path,
                label=label,
                max_bytes=len(body),
                expected_sha256=actual_sha256,
                expected_size=len(body),
            )
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(parent_descriptor)
    read_scene_file(
        destination_path,
        label=label,
        max_bytes=len(body),
        expected_sha256=actual_sha256,
        expected_size=len(body),
    )
    return destination_path


def _rename_noreplace(parent_descriptor: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SceneFileError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source),
        parent_descriptor,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        raise SceneFileError("atomic immutable scene directory publication failed") from OSError(
            error_number,
            os.strerror(error_number),
            destination,
        )


def _remove_tree(root: Path, *, label: str) -> None:
    descriptor = _open_directory(root, label=label)

    def remove_children(directory_descriptor: int) -> None:
        for name in os.listdir(directory_descriptor):
            info = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_descriptor)
                try:
                    remove_children(child)
                finally:
                    os.close(child)
                os.rmdir(name, dir_fd=directory_descriptor)
            else:
                os.unlink(name, dir_fd=directory_descriptor)

    try:
        remove_children(descriptor)
    finally:
        os.close(descriptor)
    root.rmdir()


def _tree_inventory(root: Path, *, label: str) -> tuple[set[str], set[str]]:
    descriptor = _open_directory(root, label=label)
    files: set[str] = set()
    directories: set[str] = set()

    def walk(directory_descriptor: int, prefix: PurePosixPath) -> None:
        for name in os.listdir(directory_descriptor):
            info = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            relative = (prefix / name).as_posix()
            if stat.S_ISDIR(info.st_mode):
                directories.add(relative)
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_descriptor)
                try:
                    walk(child, PurePosixPath(relative))
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise SceneFileError(f"{label} file {relative} must have one hard link")
                files.add(relative)
            else:
                raise SceneFileError(f"{label} contains a symlink or non-regular entry: {relative}")

    try:
        walk(descriptor, PurePosixPath())
    finally:
        os.close(descriptor)
    return files, directories


def validate_scene_tree(
    directory: str | Path,
    files: Mapping[str, bytes],
    *,
    label: str,
) -> Path:
    root = absolute_path_without_resolving(directory)
    expected_files = {
        normalized_scene_relative_path(path, label=f"{label} file path"): bytes(payload)
        for path, payload in files.items()
    }
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    actual_files, actual_directories = _tree_inventory(root, label=label)
    if actual_files != set(expected_files) or actual_directories != expected_directories:
        raise SceneFileError(f"{label} file inventory does not match immutable content")
    for relative, payload in expected_files.items():
        read_scene_root_file(
            root,
            relative,
            label=f"{label} file {relative}",
            max_bytes=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
        )
    return root


def materialize_scene_tree(
    directory: str | Path,
    files: Mapping[str, bytes],
    *,
    label: str,
) -> Path:
    destination = absolute_path_without_resolving(directory)
    normalized_files = {
        normalized_scene_relative_path(path, label=f"{label} file path"): bytes(payload)
        for path, payload in files.items()
    }
    if not normalized_files:
        raise SceneFileError(f"{label} must contain at least one file")
    try:
        os.lstat(destination)
    except FileNotFoundError:
        pass
    else:
        return validate_scene_tree(destination, normalized_files, label=label)

    parent = ensure_scene_directory(destination.parent, label=f"{label} parent")
    parent_descriptor = _open_directory(parent, label=f"{label} parent")
    staging_name = f".{destination.name}.stage-{secrets.token_hex(12)}"
    staging = parent / staging_name
    try:
        os.mkdir(staging_name, 0o700, dir_fd=parent_descriptor)
        for relative, payload in sorted(normalized_files.items()):
            materialize_scene_file(
                staging / relative,
                payload,
                label=f"{label} staging file {relative}",
                expected_sha256=hashlib.sha256(payload).hexdigest(),
            )
        validate_scene_tree(staging, normalized_files, label=f"{label} staging")
        try:
            _rename_noreplace(parent_descriptor, staging_name, destination.name)
        except FileExistsError:
            validate_scene_tree(destination, normalized_files, label=label)
            _remove_tree(staging, label=f"{label} staging")
        os.fsync(parent_descriptor)
    except Exception:
        try:
            os.lstat(staging)
        except FileNotFoundError:
            pass
        else:
            _remove_tree(staging, label=f"{label} staging")
        raise
    finally:
        os.close(parent_descriptor)
    return validate_scene_tree(destination, normalized_files, label=label)


__all__ = [
    "MAX_AUTHORED_SCENE_BYTES",
    "MAX_MTL_BYTES",
    "MAX_MTL_LINES",
    "MAX_OBJ_LINES",
    "MAX_SCENE_ARTIFACT_BYTES",
    "MAX_SCENE_AUTHORED_DEPENDENCIES",
    "MAX_SCENE_CAMERAS",
    "MAX_SCENE_CAMERA_ARTIFACTS",
    "MAX_SCENE_MANIFEST_BYTES",
    "MAX_SCENE_RELEASE_BYTES",
    "MAX_SCENE_RELEASE_FILES",
    "MAX_SCENE_TEXT_LINE_BYTES",
    "MAX_SCENE_VALIDATION_BYTES",
    "SceneFileError",
    "VerifiedSceneFile",
    "absolute_path_without_resolving",
    "ensure_scene_directory",
    "materialize_scene_file",
    "materialize_scene_tree",
    "load_strict_json",
    "normalized_scene_relative_path",
    "read_scene_file",
    "read_scene_root_file",
    "validate_scene_tree",
]
