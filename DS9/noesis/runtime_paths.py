"""Fail-closed import authority for DS9 native extensions."""

from __future__ import annotations

import importlib
import os
import sys
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType
from typing import Mapping, MutableSequence, Sequence


DS9_NATIVE_EXTENSION_MODULES: tuple[str, ...] = (
    "noesis_pose_meta_ext",
    "noesis_analytics_meta_ext",
    "noesis_v3dt_meta_ext",
    "noesis_reid_meta_ext",
    "noesis_latency_ext",
    "noesis_depth_meta_ext",
    "noesis_depth_tracking_tensor_ext",
)


class DS9NativeExtensionOriginError(RuntimeError):
    """Raised when a DS9 runtime could resolve a non-DS9 native extension."""


def configured_native_extension_dir(ds9_root: Path) -> Path:
    """Return the configured, existing DS9 native-extension directory."""

    raw = str(
        os.environ.get("NOESIS_NATIVE_EXT_DIR", ds9_root / "native_extensions")
        or ""
    ).strip()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise DS9NativeExtensionOriginError(
            f"NOESIS_NATIVE_EXT_DIR must be absolute: {raw or '<empty>'}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise DS9NativeExtensionOriginError(
            f"DS9 native-extension directory is unavailable: {candidate}"
        ) from exc
    if not resolved.is_dir():
        raise DS9NativeExtensionOriginError(
            f"DS9 native-extension path is not a directory: {resolved}"
        )
    return resolved


def _path_key(value: str) -> Path:
    raw = value if value else os.getcwd()
    return Path(raw).expanduser().resolve(strict=False)


def configure_ds9_runtime_import_paths(
    *,
    ds9_root: Path,
    repo_root: Path,
    system_site: Path | None = None,
    path_entries: MutableSequence[str] | None = None,
) -> Path:
    """Put DS9 native binaries ahead of every checkout-level import path."""

    native_dir = configured_native_extension_dir(ds9_root)
    ds9_root = ds9_root.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    preferred = [native_dir]
    if system_site is not None:
        system_candidate = system_site.expanduser().resolve(strict=False)
        if system_candidate.is_dir():
            preferred.append(system_candidate)
    preferred.extend((ds9_root, repo_root))

    target = sys.path if path_entries is None else path_entries
    preferred_keys = set(preferred)
    remaining: list[str] = []
    for entry in target:
        try:
            key = _path_key(str(entry))
        except (OSError, RuntimeError, ValueError):
            remaining.append(str(entry))
            continue
        if key not in preferred_keys:
            remaining.append(str(entry))
    target[:] = [str(path) for path in preferred] + remaining
    importlib.invalidate_caches()
    return native_dir


def _module_origin(module: object) -> str | None:
    raw = getattr(module, "__file__", None)
    if raw:
        return str(raw)
    spec = getattr(module, "__spec__", None)
    raw = getattr(spec, "origin", None)
    return str(raw) if raw else None


def _require_direct_owned_origin(
    module_name: str,
    raw_origin: str | None,
    native_dir: Path,
    *,
    source: str,
) -> Path:
    if not raw_origin:
        raise DS9NativeExtensionOriginError(
            f"{module_name} {source} has no concrete binary origin"
        )
    try:
        origin = Path(raw_origin).expanduser().resolve(strict=True)
    except OSError as exc:
        raise DS9NativeExtensionOriginError(
            f"{module_name} {source} origin is unavailable: {raw_origin}"
        ) from exc
    if origin.parent != native_dir:
        raise DS9NativeExtensionOriginError(
            f"{module_name} {source} resolved outside the DS9 native-extension "
            f"directory: {origin} (required parent: {native_dir})"
        )
    return origin


def require_ds9_native_extension_origins(
    native_dir: Path,
    *,
    module_names: Sequence[str] = DS9_NATIVE_EXTENSION_MODULES,
    search_path: Sequence[str] | None = None,
    loaded_modules: Mapping[str, ModuleType | object | None] | None = None,
) -> dict[str, Path]:
    """Require both cached and newly selected modules to be directly DS9-owned."""

    try:
        native_root = native_dir.expanduser().resolve(strict=True)
    except OSError as exc:
        raise DS9NativeExtensionOriginError(
            f"DS9 native-extension directory is unavailable: {native_dir}"
        ) from exc
    if not native_root.is_dir():
        raise DS9NativeExtensionOriginError(
            f"DS9 native-extension path is not a directory: {native_root}"
        )

    selected: dict[str, Path] = {}
    paths = list(sys.path if search_path is None else search_path)
    modules = sys.modules if loaded_modules is None else loaded_modules
    for module_name in module_names:
        if module_name in modules:
            cached = modules[module_name]
            cached_origin = _module_origin(cached) if cached is not None else None
            _require_direct_owned_origin(
                module_name,
                cached_origin,
                native_root,
                source="cached module",
            )

        spec = PathFinder.find_spec(module_name, paths)
        spec_origin = getattr(spec, "origin", None) if spec is not None else None
        selected[module_name] = _require_direct_owned_origin(
            module_name,
            str(spec_origin) if spec_origin else None,
            native_root,
            source="selected module",
        )
    return selected


def load_ds9_native_extensions(
    native_dir: Path,
    *,
    module_names: Sequence[str] = DS9_NATIVE_EXTENSION_MODULES,
    search_path: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Load every required native extension and revalidate its loaded origin.

    Selecting an owned extension spec is necessary but does not prove that the
    binary can be loaded against the active DeepStream/Python ABI.  Validate the
    selection before import so an already-cached foreign module fails closed,
    import every exact required module, then validate the concrete modules
    returned by the importer as well as the still-selected specs.
    """

    require_ds9_native_extension_origins(
        native_dir,
        module_names=module_names,
        search_path=search_path,
    )
    native_root = native_dir.expanduser().resolve(strict=True)

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise DS9NativeExtensionOriginError(
                f"{module_name} failed to import from the required DS9 "
                f"native-extension directory: {exc}"
            ) from exc
        _require_direct_owned_origin(
            module_name,
            _module_origin(module),
            native_root,
            source="imported module",
        )

    return require_ds9_native_extension_origins(
        native_dir,
        module_names=module_names,
        search_path=search_path,
    )


__all__ = [
    "DS9_NATIVE_EXTENSION_MODULES",
    "DS9NativeExtensionOriginError",
    "configure_ds9_runtime_import_paths",
    "configured_native_extension_dir",
    "load_ds9_native_extensions",
    "require_ds9_native_extension_origins",
]
