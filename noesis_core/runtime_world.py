from __future__ import annotations

import hashlib
import os
import socket
import uuid
from pathlib import Path
from typing import Any, Mapping

from noesis_core.contracts.base import ProducerRef
from noesis_core.health import CapabilityMonitor, CapabilityPolicy
from noesis_core.journal import AsyncContractJournal, ContractJournal
from noesis_core.runtime_secrets import public_pipeline_config
from noesis_core.world_service import (
    CanonicalWorldService,
    WorldArtifacts,
    fingerprint_payload,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_existing_file(value: str, repo_root: Path) -> Path | None:
    candidate = Path(value).expanduser()
    candidates = (candidate,) if candidate.is_absolute() else (repo_root / candidate, candidate)
    for item in candidates:
        try:
            resolved = item.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_file():
            return resolved
    return None


def _content_manifest(value: Any, repo_root: Path, *, key_path: str = "root") -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _content_manifest(item, repo_root, key_path=f"{key_path}.{key}")
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _content_manifest(item, repo_root, key_path=f"{key_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        path = _resolve_existing_file(value, repo_root)
        if path is not None:
            return {
                "configured_value": value,
                "content_sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return value


def _git_revision(repo_root: Path) -> str:
    override = str(os.environ.get("NOESIS_SOFTWARE_REVISION", "")).strip()
    if override:
        return override
    git_dir = repo_root / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_name = head.removeprefix("ref: ").strip()
            ref_path = git_dir / ref_name
            if ref_path.is_file():
                head = ref_path.read_text(encoding="utf-8").strip()
            else:
                for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith(("#", "^")):
                        sha, name = line.split(" ", 1)
                        if name == ref_name:
                            head = sha
                            break
        if len(head) >= 7:
            return f"{head}+workspace"
    except (OSError, ValueError):
        pass
    return "local-workspace-unversioned"


def create_runtime_world_service(
    *,
    runtime: str,
    pipeline_config: Mapping[str, Any],
    camera_labels: Mapping[int, str],
    calibration_provider: Any,
    repo_root: Path,
    run_id: str | None = None,
    instance_id: str | None = None,
    software_revision: str | None = None,
    journal_path: str | Path | None = None,
) -> CanonicalWorldService:
    """Create the canonical world owner from runtime-owned, content-addressed inputs."""

    root = Path(repo_root).resolve()
    producer = ProducerRef(
        runtime=runtime,  # type: ignore[arg-type]
        instance_id=(instance_id or socket.gethostname()).strip(),
        run_id=(run_id or os.environ.get("NOESIS_RUNTIME_RUN_ID") or str(uuid.uuid4())).strip(),
        software_revision=(software_revision or _git_revision(root)).strip(),
    )
    public_config = public_pipeline_config(pipeline_config)
    model_manifest = _content_manifest(
        {
            "models": public_config.get("models", {}),
            "tracker": public_config.get("tracker", {}),
        },
        root,
        key_path="model_manifest",
    )
    model_fingerprint = fingerprint_payload(
        "tracking_model_manifest",
        model_manifest,
        version=str(public_config.get("version") or "runtime"),
    )
    config_fingerprint = fingerprint_payload(
        "pipeline_config",
        public_config,
        version=str(public_config.get("version") or "runtime"),
    )

    def artifacts(source_id: int, metadata: Mapping[str, Any]) -> WorldArtifacts:
        camera_id = str(
            metadata.get("camera_id")
            or camera_labels.get(int(source_id))
            or f"camera_{int(source_id)}"
        )
        snapshot = calibration_provider.snapshot(int(source_id), camera_id)
        calibration_version = str(metadata.get("calibration_version") or "runtime")
        if snapshot is None:
            calibration_payload: Any = {
                "camera_id": camera_id,
                "source_id": int(source_id),
                "status": "unavailable",
            }
            calibration_role = "camera_calibration_unavailable"
        else:
            calibration_payload = snapshot
            calibration_role = "camera_calibration"
        return WorldArtifacts(
            calibration=fingerprint_payload(
                calibration_role,
                calibration_payload,
                version=calibration_version,
            ),
            model=model_fingerprint,
            config=config_fingerprint,
        )

    configured_journal = str(os.environ.get("NOESIS_WORLD_JOURNAL_PATH", "")).strip()
    resolved_journal = (
        Path(journal_path).expanduser()
        if journal_path is not None
        else (
            Path(configured_journal).expanduser()
            if configured_journal
            else Path.home() / ".local" / "state" / "noesis" / f"world_{runtime}.sqlite3"
        )
    )
    durable_journal = ContractJournal(
        resolved_journal,
        max_records=max(
            1,
            int(os.environ.get("NOESIS_WORLD_JOURNAL_MAX_RECORDS", "10000")),
        ),
        max_age_us=max(
            1,
            int(
                float(
                    os.environ.get(
                        "NOESIS_WORLD_JOURNAL_RETENTION_HOURS",
                        "24",
                    )
                )
                * 3_600_000_000
            ),
        ),
    )
    journal = AsyncContractJournal(
        durable_journal,
        max_pending_batches=max(
            1,
            int(os.environ.get("NOESIS_WORLD_JOURNAL_MAX_PENDING_BATCHES", "256")),
        ),
        max_transaction_batches=max(
            1,
            int(os.environ.get("NOESIS_WORLD_JOURNAL_TRANSACTION_BATCHES", "32")),
        ),
    )
    return CanonicalWorldService(producer=producer, artifacts=artifacts, journal=journal)


__all__ = ["create_runtime_world_service"]


def create_runtime_capability_monitor(world_service: CanonicalWorldService) -> CapabilityMonitor:
    return CapabilityMonitor(
        instance_id=world_service.producer.instance_id,
        run_id=world_service.producer.run_id,
        policies={
            "tracking_observations": CapabilityPolicy(
                stale_after_us=2_000_000,
                fail_after_us=10_000_000,
            ),
            "global_world": CapabilityPolicy(
                stale_after_us=2_000_000,
                fail_after_us=10_000_000,
            ),
        },
    )


__all__ = ["create_runtime_capability_monitor", "create_runtime_world_service"]
