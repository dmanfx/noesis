"""Household identity state cutover helpers."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

_LEGACY_STATE_FILES = (
    "reid_gallery.npz",
    "reid_aliases.json",
    "sid_pool.json",
    "stable_id_state.json",
)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def is_household_identity_enabled() -> bool:
    # Household closed-world identity is the production ReID path.
    # Opt out explicitly with NOESIS_HOUSEHOLD_IDENTITY=0 for legacy debugging.
    return _truthy(os.environ.get("NOESIS_HOUSEHOLD_IDENTITY", "1"))


def is_household_archive_enabled() -> bool:
    if not is_household_identity_enabled():
        return False
    raw = os.environ.get("NOESIS_HOUSEHOLD_ARCHIVE_STATE")
    if raw is None or not str(raw).strip():
        return True
    return _truthy(raw)


def household_root() -> Path:
    return Path(os.path.expanduser("~/.noesis/household"))


def legacy_noesis_root() -> Path:
    return Path(os.path.expanduser("~/.noesis"))


def resolve_camera_topology_file(repo_root: Path) -> Path:
    raw = os.environ.get("NOESIS_CAMERA_TOPOLOGY_FILE", "config/camera_topology.yaml")
    path = Path(os.path.expanduser(str(raw)))
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_household_paths() -> Dict[str, str]:
    root = household_root()
    return {
        "gallery_persist_file": str(root / "reid_gallery.npz"),
        "alias_file": str(root / "reid_aliases.json"),
        "sid_pool_file": str(root / "sid_pool.json"),
        "residents_file": str(root / "residents.json"),
        # Must not share sid_pool.json — legacy free-list and visitor pool use different schemas.
        "visitor_pool_file": str(root / "visitor_pool.json"),
    }


def apply_household_env_defaults() -> None:
    if not is_household_identity_enabled():
        return
    os.environ.setdefault("NOESIS_REID_POSE_ENABLED", "1")
    os.environ.setdefault("NOESIS_REID_EMBEDS_PER_FRAME_MAX", "4")


def default_household_backup_dir(base_dir: Optional[str] = None) -> str:
    """Return a timestamped backup directory under ~/.noesis/household/backups/."""
    root = os.path.expanduser(base_dir or "~/.noesis/household/backups")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return os.path.join(root, stamp)


def archive_legacy_identity_state(
    backup_dir_or_logger: Any,
    legacy_paths: Optional[Iterable[str]] = None,
    *,
    logger: Optional[logging.Logger] = None,
    move: bool = True,
    legacy_root: Optional[Path] = None,
    backup_root: Optional[Path] = None,
) -> Any:
    """Archive legacy identity persistence files into a timestamped backup dir.

    Supports two call styles:
    - ``archive_legacy_identity_state(backup_dir, legacy_paths)`` (legacy API)
    - ``archive_legacy_identity_state(logger, move=True, ...)`` (runtime API)
    """
    if isinstance(backup_dir_or_logger, logging.Logger):
        log = backup_dir_or_logger
        src_root = legacy_root or legacy_noesis_root()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest_root = (backup_root or (household_root() / "backups" / ts)).resolve()
        paths = [str(src_root / name) for name in _LEGACY_STATE_FILES]
        archived = archive_legacy_identity_state(str(dest_root), paths)
        if archived:
            log.warning(
                "Household cutover archived %d legacy identity file(s) to %s",
                len(archived),
                dest_root,
            )
            return dest_root
        return None

    dest_root = os.path.expanduser(str(backup_dir_or_logger))
    os.makedirs(dest_root, exist_ok=True)
    archived: List[str] = []
    paths = list(legacy_paths or ())

    for raw_path in paths:
        if not raw_path:
            continue
        src = os.path.expanduser(str(raw_path))
        if not os.path.isfile(src):
            continue
        base = os.path.basename(src)
        dest = os.path.join(dest_root, base)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(base)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            dest = os.path.join(dest_root, f"{stem}_{stamp}{ext}")
        try:
            if move:
                shutil.move(src, dest)
            else:
                shutil.copy2(src, dest)
        except Exception:
            try:
                shutil.copy2(src, dest)
                if move:
                    os.unlink(src)
            except Exception as exc:
                (logger or logging.getLogger(__name__)).warning(
                    "failed to archive %s -> %s: %s",
                    src,
                    dest,
                    exc,
                )
                continue
        archived.append(dest)
        (logger or logging.getLogger(__name__)).info(
            "archived legacy identity state: %s -> %s",
            src,
            dest,
        )

    return archived


def prepare_household_stable_id_overrides(
    logger: logging.Logger,
    *,
    repo_root: Path,
    cos_sim_high_threshold: float,
    cos_sim_high_env_set: bool,
) -> Dict[str, Any]:
    apply_household_env_defaults()
    if is_household_archive_enabled():
        legacy_paths = [str(legacy_noesis_root() / name) for name in _LEGACY_STATE_FILES]
        backup_dir = default_household_backup_dir()
        archived = archive_legacy_identity_state(backup_dir, legacy_paths)
        if archived:
            logger.warning(
                "Household cutover archived %d legacy identity file(s) to %s",
                len(archived),
                backup_dir,
            )

    paths = resolve_household_paths()
    household_root().mkdir(parents=True, exist_ok=True)
    topology_file = resolve_camera_topology_file(repo_root)

    overrides: Dict[str, Any] = {
        "auto_merge_enabled": False,
        "allow_multi_zone_active": False,
        "gallery_persist_file": paths["gallery_persist_file"],
        "alias_file": paths["alias_file"],
        "sid_pool_file": paths["sid_pool_file"],
        "residents_file": paths["residents_file"],
        "visitor_pool_file": paths["visitor_pool_file"],
        "household_mode": True,
        "camera_topology_file": str(topology_file),
        "total_id_reuse": False,
    }
    if not cos_sim_high_env_set:
        overrides["cos_sim_high_threshold"] = 0.78
    else:
        overrides["cos_sim_high_threshold"] = cos_sim_high_threshold
    return overrides


def filter_kwargs_for_init(candidate: Mapping[str, Any], init_callable: Any) -> Dict[str, Any]:
    import inspect

    try:
        sig = inspect.signature(init_callable)
        valid = set(sig.parameters)
        valid.discard("self")
        return {key: val for key, val in candidate.items() if key in valid}
    except Exception:
        return dict(candidate)
