"""Scheduled backup of Menon config files.

This script copies Menon's persistent config files to
~/Menon/backup/config every 24 hours, but only if the content
changed since the last backup.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

LOGGER = logging.getLogger("backup_config")

MENON_CONFIG_ROOT = Path("~/Menon/public/config").expanduser()
MENON_FILES = [
    MENON_CONFIG_ROOT / "virtual-devices.json",
    MENON_CONFIG_ROOT / "room-zones.json",
]
BACKUP_ROOT = Path("~/Menon/backup/config").expanduser()
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

CHECK_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json_bytes(path: Path) -> bytes:
    if not path.exists():
        LOGGER.warning("Config file %s does not exist", path)
        return b"{}"
    return path.read_bytes()


def _find_latest_backup(stem: str) -> Path | None:
    candidates = sorted(BACKUP_ROOT.glob(f"{stem}_*.json"))
    return candidates[-1] if candidates else None


def _make_backup_filename(stem: str) -> Path:
    from datetime import datetime

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return BACKUP_ROOT / f"{stem}_{timestamp}.json"


def _should_backup(config_bytes: bytes, latest_backup: Path | None) -> bool:
    if latest_backup is None or not latest_backup.exists():
        return True
    return _hash_bytes(config_bytes) != _hash_bytes(latest_backup.read_bytes())


def run_backup_once() -> Dict[str, str]:
    summary: Dict[str, str] = {}
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

    for config_path in MENON_FILES:
        stem = config_path.stem
        data = _load_json_bytes(config_path)
        latest = _find_latest_backup(stem)
        if _should_backup(data, latest):
            target = _make_backup_filename(stem)
            target.write_bytes(data)
            summary[stem] = f"created {target.name}"
            LOGGER.info("Backed up %s -> %s", config_path.name, target.name)
        else:
            summary[stem] = "no changes"
            LOGGER.debug("No changes detected for %s", config_path.name)
    return summary


async def backup_loop():
    LOGGER.info("Starting backup loop; interval=%s seconds", CHECK_INTERVAL_SECONDS)
    while True:
        try:
            run_backup_once()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Backup run failed")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    try:
        asyncio.run(backup_loop())
    except KeyboardInterrupt:
        LOGGER.info("Backup loop interrupted; exiting")
