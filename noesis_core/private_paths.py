"""Owner-private path primitives for cooperating Noesis processes.

The checks below reject symlink, hard-link, permission, inode, namespace, and
in-call mutation races.  Unix owner permissions cannot defend against a
malicious process running under the same effective UID after a call returns;
service-user isolation remains an operational trust boundary.
"""

from __future__ import annotations

import fcntl
import os
import secrets
import stat
from pathlib import Path
from typing import Iterable


PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class PrivatePathError(RuntimeError):
    """Raised when private state would rely on an unsafe filesystem path."""


def _mode_text(mode: int) -> str:
    return f"{int(mode):04o}"


def _validate_open_private_directory(
    path: Path,
    descriptor: int,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> os.stat_result:
    """Revalidate one open private directory and its current path binding."""

    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
    except OSError as exc:
        raise PrivatePathError(f"{label} parent cannot be revalidated") from exc
    expected = (int(expected_identity[0]), int(expected_identity[1]))
    if (
        (int(opened.st_dev), int(opened.st_ino)) != expected
        or (int(named.st_dev), int(named.st_ino)) != expected
        or not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or opened.st_uid != os.geteuid()
        or named.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != PRIVATE_DIRECTORY_MODE
        or stat.S_IMODE(named.st_mode) != PRIVATE_DIRECTORY_MODE
    ):
        raise PrivatePathError(
            f"{label} parent changed or became unsafe during private publication"
        )
    return opened


def _validate_open_private_file(
    descriptor: int,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> os.stat_result:
    """Revalidate one open exact-mode, owner-private, single-link file."""

    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise PrivatePathError(f"{label} cannot be revalidated") from exc
    if (
        (int(opened.st_dev), int(opened.st_ino))
        != (int(expected_identity[0]), int(expected_identity[1]))
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != PRIVATE_FILE_MODE
        or opened.st_nlink != 1
    ):
        raise PrivatePathError(f"{label} changed or became unsafe while open")
    return opened


def _without_symlink_components(path: str | Path, *, label: str) -> Path:
    """Return an absolute lexical path after rejecting every existing symlink."""

    candidate = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    anchor = Path(candidate.anchor)
    cursor = anchor
    for component in candidate.parts[1:]:
        cursor = cursor / component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise PrivatePathError(f"{label} path cannot be inspected") from exc
        if stat.S_ISLNK(info.st_mode):
            raise PrivatePathError(f"{label} path must not contain symlink components")
    return candidate


def ensure_private_directory(
    directory: str | Path,
    *,
    label: str,
) -> Path:
    """Create one private leaf directory or validate it without changing it."""

    path = _without_symlink_components(directory, label=label)
    try:
        path.mkdir(parents=True, exist_ok=False, mode=PRIVATE_DIRECTORY_MODE)
    except FileExistsError:
        pass
    except OSError as exc:
        raise PrivatePathError(f"{label} directory cannot be created") from exc
    path = _without_symlink_components(path, label=label)
    try:
        info = path.lstat()
    except OSError as exc:
        raise PrivatePathError(f"{label} directory cannot be inspected") from exc
    if stat.S_ISLNK(info.st_mode):
        raise PrivatePathError(f"{label} directory must not be a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise PrivatePathError(f"{label} directory must be a directory")
    if info.st_uid != os.geteuid():
        raise PrivatePathError(f"{label} directory must be owned by the service user")
    mode = stat.S_IMODE(info.st_mode)
    if mode != PRIVATE_DIRECTORY_MODE:
        raise PrivatePathError(
            f"{label} directory mode must be {_mode_text(PRIVATE_DIRECTORY_MODE)}; "
            f"found {_mode_text(mode)}"
        )
    return path


def validate_private_file(
    path: str | Path,
    *,
    label: str,
    allowed_modes: Iterable[int] = (PRIVATE_FILE_MODE,),
) -> Path:
    """Validate an existing owner-only, single-link regular file."""

    candidate = _without_symlink_components(path, label=label)
    try:
        info = candidate.lstat()
    except FileNotFoundError as exc:
        raise PrivatePathError(f"{label} is missing") from exc
    except OSError as exc:
        raise PrivatePathError(f"{label} cannot be inspected") from exc
    if stat.S_ISLNK(info.st_mode):
        raise PrivatePathError(f"{label} must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise PrivatePathError(f"{label} must be a regular file")
    if info.st_uid != os.geteuid():
        raise PrivatePathError(f"{label} must be owned by the service user")
    accepted = tuple(sorted({int(mode) for mode in allowed_modes}))
    mode = stat.S_IMODE(info.st_mode)
    if mode not in accepted:
        rendered = " or ".join(_mode_text(item) for item in accepted)
        raise PrivatePathError(
            f"{label} mode must be {rendered}; found {_mode_text(mode)}"
        )
    if info.st_nlink != 1:
        raise PrivatePathError(f"{label} must have exactly one hard link")
    return candidate


def prepare_private_writable_file(
    path: str | Path,
    *,
    label: str,
) -> Path:
    """Create an empty 0600 state file or validate an existing one."""

    candidate = _without_symlink_components(path, label=label)
    if candidate.is_symlink():
        raise PrivatePathError(f"{label} must not be a symlink")
    ensure_private_directory(candidate.parent, label=f"{label} parent")
    if candidate.exists():
        return validate_private_file(candidate, label=label)
    if candidate.is_symlink():
        raise PrivatePathError(f"{label} must not be a symlink")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(candidate, flags, PRIVATE_FILE_MODE)
    except FileExistsError:
        return validate_private_file(candidate, label=label)
    except OSError as exc:
        raise PrivatePathError(f"{label} cannot be created securely") from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return validate_private_file(candidate, label=label)


def atomic_write_private_file(
    path: str | Path,
    payload: bytes,
    *,
    label: str,
) -> Path:
    """Atomically replace one private file without following or chmodding paths.

    The destination's parent must be an owner-only directory.  A pre-existing
    destination must already satisfy the private-file contract and must not
    change between validation and replacement.  This deliberately refuses
    unsafe legacy paths instead of repairing their permissions in place.

    Writers using this helper are serialized on the held parent directory.
    Creating a previously absent destination uses hard-link publication so a
    name that appears at the last boundary is never overwritten.  Replacing an
    existing destination relies on that shared lock plus an immediate inode
    compare; direct writers that bypass this helper violate the state contract.
    """

    candidate = _without_symlink_components(path, label=label)
    body = bytes(payload)
    parent = ensure_private_directory(candidate.parent, label=f"{label} parent")
    candidate = parent / candidate.name
    parent_before = parent.lstat()
    parent_identity = (int(parent_before.st_dev), int(parent_before.st_ino))
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    original: tuple[int, int] | None = None
    temporary_name = f".{candidate.name}.tmp-{secrets.token_hex(12)}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    temporary_present = False
    destination_linked = False
    try:
        fcntl.flock(parent_descriptor, fcntl.LOCK_EX)
        _validate_open_private_directory(
            parent,
            parent_descriptor,
            expected_identity=parent_identity,
            label=label,
        )
        try:
            initial = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            initial = None
        if initial is not None:
            if (
                not stat.S_ISREG(initial.st_mode)
                or initial.st_uid != os.geteuid()
                or stat.S_IMODE(initial.st_mode) != PRIVATE_FILE_MODE
                or initial.st_nlink != 1
            ):
                raise PrivatePathError(
                    f"{label} must be an owner-only, single-link regular file"
                )
            original = (int(initial.st_dev), int(initial.st_ino))

        descriptor = os.open(
            temporary_name,
            flags,
            PRIVATE_FILE_MODE,
            dir_fd=parent_descriptor,
        )
        temporary_present = True
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        staged_initial = os.fstat(descriptor)
        staged_identity = (int(staged_initial.st_dev), int(staged_initial.st_ino))
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PrivatePathError(f"{label} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        staged = _validate_open_private_file(
            descriptor,
            expected_identity=staged_identity,
            label=f"{label} staged file",
        )
        if staged.st_size != len(body):
            raise PrivatePathError(f"{label} staged size is inconsistent")
        _validate_open_private_directory(
            parent,
            parent_descriptor,
            expected_identity=parent_identity,
            label=label,
        )

        try:
            current = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            current = None
        if original is None:
            if current is not None:
                raise PrivatePathError(f"{label} appeared while it was being written")
        else:
            if (
                current is None
                or not stat.S_ISREG(current.st_mode)
                or current.st_uid != os.geteuid()
                or stat.S_IMODE(current.st_mode) != PRIVATE_FILE_MODE
                or current.st_nlink != 1
                or (int(current.st_dev), int(current.st_ino)) != original
            ):
                raise PrivatePathError(f"{label} changed while it was being written")

        if original is None:
            try:
                os.link(
                    temporary_name,
                    candidate.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise PrivatePathError(
                    f"{label} appeared while it was being written"
                ) from exc
            destination_linked = True
            _validate_open_private_directory(
                parent,
                parent_descriptor,
                expected_identity=parent_identity,
                label=label,
            )
            linked = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            staged = os.fstat(descriptor)
            if (
                (int(linked.st_dev), int(linked.st_ino)) != staged_identity
                or (int(staged.st_dev), int(staged.st_ino)) != staged_identity
                or linked.st_nlink != 2
                or staged.st_nlink != 2
            ):
                raise PrivatePathError(f"{label} create publication is inconsistent")
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            temporary_present = False
        else:
            os.replace(
                temporary_name,
                candidate.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary_present = False
        published = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        staged = _validate_open_private_file(
            descriptor,
            expected_identity=staged_identity,
            label=f"{label} published file",
        )
        if (
            (int(published.st_dev), int(published.st_ino)) != staged_identity
            or not stat.S_ISREG(published.st_mode)
            or published.st_uid != os.geteuid()
            or stat.S_IMODE(published.st_mode) != PRIVATE_FILE_MODE
            or published.st_nlink != 1
            or published.st_size != len(body)
            or staged.st_size != len(body)
        ):
            raise PrivatePathError(f"{label} publication is inconsistent")
        os.fsync(parent_descriptor)
        _validate_open_private_directory(
            parent,
            parent_descriptor,
            expected_identity=parent_identity,
            label=label,
        )
    except Exception:
        if temporary_present and not destination_linked:
            try:
                named = os.stat(
                    temporary_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                opened = os.fstat(descriptor)
                if (int(named.st_dev), int(named.st_ino)) == (
                    int(opened.st_dev),
                    int(opened.st_ino),
                ):
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    return candidate


def atomic_create_private_file(
    path: str | Path,
    payload: bytes,
    *,
    label: str,
    max_bytes: int,
) -> Path:
    """Publish one immutable private file without replacing an existing name.

    The payload is durably staged in the destination's owner-only directory,
    then linked into place.  A hard-link publication is the portable
    create-if-absent primitive used here: unlike ``os.replace()``, it cannot
    overwrite a destination that appears during the write.  The temporary link
    is removed only after the destination has been verified against the open
    descriptor.  A crash in between deliberately leaves a multi-link file or a
    temporary residue, both of which fail the private-file contract and require
    a fresh evidence session.
    """

    limit = int(max_bytes)
    if limit < 1:
        raise ValueError("max_bytes must be positive")
    body = bytes(payload)
    if not body:
        raise PrivatePathError(f"{label} payload must not be empty")
    if len(body) > limit:
        raise PrivatePathError(f"{label} exceeds the {limit}-byte limit")

    candidate = _without_symlink_components(path, label=label)
    if not candidate.name or candidate.name in {".", ".."}:
        raise PrivatePathError(f"{label} filename is invalid")
    parent = ensure_private_directory(candidate.parent, label=f"{label} parent")
    candidate = parent / candidate.name
    temporary_prefix = f".{candidate.name}.tmp-"

    parent_before = parent.lstat()
    parent_identity = (int(parent_before.st_dev), int(parent_before.st_ino))
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = -1
    temporary_name: str | None = None
    destination_linked = False
    try:
        _validate_open_private_directory(
            parent,
            parent_descriptor,
            expected_identity=parent_identity,
            label=label,
        )

        entries = os.listdir(parent_descriptor)
        if candidate.name in entries:
            raise PrivatePathError(
                f"refusing to replace immutable evidence: {label} already exists"
            )
        if any(name.startswith(temporary_prefix) for name in entries):
            raise PrivatePathError(
                f"{label} has incomplete immutable publication residue"
            )

        temporary_name = f"{temporary_prefix}{secrets.token_hex(12)}"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(
            temporary_name,
            flags,
            PRIVATE_FILE_MODE,
            dir_fd=parent_descriptor,
        )
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PrivatePathError(f"{label} write made no progress")
            view = view[written:]
        os.fsync(descriptor)

        _validate_open_private_directory(
            parent,
            parent_descriptor,
            expected_identity=parent_identity,
            label=label,
        )

        try:
            os.link(
                temporary_name,
                candidate.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise PrivatePathError(f"{label} appeared while it was being written") from exc
        destination_linked = True
        _validate_open_private_directory(
            parent,
            parent_descriptor,
            expected_identity=parent_identity,
            label=label,
        )
        matching_temporaries = sorted(
            name
            for name in os.listdir(parent_descriptor)
            if name.startswith(temporary_prefix)
        )
        if matching_temporaries != [temporary_name]:
            raise PrivatePathError(
                f"{label} has concurrent immutable publication residue"
            )

        staged = os.fstat(descriptor)
        published = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            (staged.st_dev, staged.st_ino) != (published.st_dev, published.st_ino)
            or not stat.S_ISREG(staged.st_mode)
            or staged.st_uid != os.geteuid()
            or stat.S_IMODE(staged.st_mode) != PRIVATE_FILE_MODE
            or staged.st_nlink != 2
            or staged.st_size != len(body)
        ):
            raise PrivatePathError(f"{label} immutable publication is inconsistent")

        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_name = None
        os.fsync(parent_descriptor)
        _validate_open_private_directory(
            parent,
            parent_descriptor,
            expected_identity=parent_identity,
            label=label,
        )
        if any(
            name.startswith(temporary_prefix)
            for name in os.listdir(parent_descriptor)
        ):
            raise PrivatePathError(
                f"{label} has incomplete immutable publication residue"
            )

        final_open = os.fstat(descriptor)
        final_named = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            (final_open.st_dev, final_open.st_ino)
            != (final_named.st_dev, final_named.st_ino)
            or not stat.S_ISREG(final_open.st_mode)
            or not stat.S_ISREG(final_named.st_mode)
            or final_open.st_uid != os.geteuid()
            or final_named.st_uid != os.geteuid()
            or stat.S_IMODE(final_open.st_mode) != PRIVATE_FILE_MODE
            or stat.S_IMODE(final_named.st_mode) != PRIVATE_FILE_MODE
            or final_open.st_nlink != 1
            or final_named.st_nlink != 1
            or final_open.st_size != len(body)
            or final_named.st_size != len(body)
        ):
            raise PrivatePathError(f"{label} changed during immutable publication")
        _validate_open_private_directory(
            parent,
            parent_descriptor,
            expected_identity=parent_identity,
            label=label,
        )
        return_named = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            int(return_named.st_dev),
            int(return_named.st_ino),
            int(return_named.st_size),
            int(return_named.st_uid),
            stat.S_IMODE(return_named.st_mode),
            int(return_named.st_nlink),
        ) != (
            int(final_open.st_dev),
            int(final_open.st_ino),
            len(body),
            os.geteuid(),
            PRIVATE_FILE_MODE,
            1,
        ):
            raise PrivatePathError(f"{label} changed during final validation")
        _validate_open_private_directory(
            parent,
            parent_descriptor,
            expected_identity=parent_identity,
            label=label,
        )
    except Exception:
        # Once the destination link exists, retain any remaining temporary link
        # so the result is visibly non-canonical instead of repairing a partial
        # publication into something that could be mistaken for valid evidence.
        if not destination_linked and temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)

    return candidate


def require_fresh_private_file_bundle(
    paths: Iterable[str | Path],
    *,
    label: str,
) -> tuple[Path, ...]:
    """Require every immutable output name in one private directory to be absent.

    This is an early, read-only bundle guard; each later publication must still
    use :func:`atomic_create_private_file` because another process can race the
    preflight.  Any destination or interrupted-publication residue means the
    caller must use a fresh evidence session rather than filling in a partial
    bundle.
    """

    candidates = tuple(
        _without_symlink_components(path, label=label) for path in paths
    )
    if not candidates:
        raise ValueError("private file bundle must contain at least one path")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise PrivatePathError(f"{label} contains duplicate output names")
    parents = {candidate.parent for candidate in candidates}
    if len(parents) != 1:
        raise PrivatePathError(f"{label} outputs must share one private directory")
    parent = ensure_private_directory(
        candidates[0].parent,
        label=f"{label} parent",
    )
    parent_before = parent.lstat()
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        parent_identity = (int(parent_before.st_dev), int(parent_before.st_ino))
        _validate_open_private_directory(
            parent,
            descriptor,
            expected_identity=parent_identity,
            label=label,
        )
        entries = set(os.listdir(descriptor))
        occupied = sorted(
            candidate.name
            for candidate in candidates
            if candidate.name in entries
            or any(
                name.startswith(f".{candidate.name}.tmp-") for name in entries
            )
        )
        if occupied:
            raise PrivatePathError(
                f"{label} is not fresh; start a new evidence session: {occupied}"
            )
        _validate_open_private_directory(
            parent,
            descriptor,
            expected_identity=parent_identity,
            label=label,
        )
    finally:
        os.close(descriptor)
    return tuple(parent / candidate.name for candidate in candidates)


def read_private_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read a bounded, stable private file through a no-follow checked handle.

    Identity, mode, link count, size, mtime, and ctime are checked around the
    read.  This detects concurrent in-call mutation; it does not make owner
    state immutable to a malicious process sharing the service UID.
    """

    limit = int(max_bytes)
    if limit < 1:
        raise ValueError("max_bytes must be positive")
    candidate = _without_symlink_components(path, label=label)
    validated = validate_private_file(candidate, label=label)
    expected = validated.lstat()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise PrivatePathError(f"{label} cannot be opened securely") from exc
    try:
        identity = (int(expected.st_dev), int(expected.st_ino))
        opened = _validate_open_private_file(
            descriptor,
            expected_identity=identity,
            label=label,
        )
        if opened.st_size > limit:
            raise PrivatePathError(f"{label} exceeds the {limit}-byte limit")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        final = _validate_open_private_file(
            descriptor,
            expected_identity=identity,
            label=label,
        )
        if (
            final.st_size != opened.st_size
            or final.st_mtime_ns != opened.st_mtime_ns
            or final.st_ctime_ns != opened.st_ctime_ns
            or len(payload) != final.st_size
        ):
            raise PrivatePathError(f"{label} changed while being read")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > limit:
        raise PrivatePathError(f"{label} exceeds the {limit}-byte limit")
    return payload


__all__ = [
    "atomic_create_private_file",
    "PRIVATE_DIRECTORY_MODE",
    "PRIVATE_FILE_MODE",
    "PrivatePathError",
    "atomic_write_private_file",
    "ensure_private_directory",
    "prepare_private_writable_file",
    "read_private_file",
    "require_fresh_private_file_bundle",
    "validate_private_file",
]
