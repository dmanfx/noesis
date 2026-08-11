#!/usr/bin/env python3
"""Atomically merge one DS9 engine run into the external realization overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_asset_manifest as validator  # noqa: E402
from engine_maintenance_common import (  # noqa: E402
    mapanything_quality_gate_from_source_contracts,
    validate_mapanything_functional_quality_receipt,
)


REALIZATION_CONTRACT = "noesis.ds9.asset_realization"
REALIZATION_FILENAME = "asset_realization.json"
SOURCE_CONTRACTS = REPO_ROOT / "DS9/config/engine_source_contracts.json"
ENGINE_ARTIFACT_IDS = {
    "yolo11_seg": "engine.yolo11_seg_alternate",
    "yolo26_m": "engine.yolo26_detect_m",
    "yolo26_seg_s": "engine.yolo26_seg_s",
    "reid_swin": "engine.reid_swin_tiny",
    "yolo26_pose_n": "engine.pose_yolo26",
    "depth_anything_v2_tracking": "engine.depth_tracking_dav2",
    "mapanything": "engine.mapanything",
    "wholebody49_s_masks": "engine.wholebody49_s_masks",
    "wholebody49_x_boxes": "engine.wholebody49_x_boxes",
    "bodypose3dnet": "engine.v3dt_bodypose",
    "v3dt_tracker_reid": "engine.v3dt_tracker_reid",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _artifact(payload: Mapping[str, Any], artifact_id: str) -> Mapping[str, Any]:
    rows = [row for row in payload.get("artifacts", []) if row.get("id") == artifact_id]
    if len(rows) != 1:
        raise ValueError(
            f"expected one base-manifest artifact {artifact_id}, found {len(rows)}"
        )
    if rows[0].get("kind") != "tensorrt_engine":
        raise ValueError(f"realization may only update TensorRT engines: {artifact_id}")
    return rows[0]


def _maintenance_paths(manifest: Path, artifact_root: Path) -> tuple[Path, Path]:
    evidence_root = (artifact_root / "models/engine_maintenance").resolve(strict=True)
    bounded = validator._bounded_private_path(
        manifest, evidence_root, "maintenance manifest"
    )
    relative = bounded.relative_to(evidence_root)
    return bounded, Path("DS9/models/engine_maintenance") / relative


def _tensor_contract(build_contract: Mapping[str, Any]) -> dict[str, Any]:
    raw = build_contract.get("onnx") or build_contract.get("tensor_contract")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("maintenance manifest lacks a structured tensor contract")
    result = dict(raw)
    result.pop("path", None)
    return result


def _new_realization(base_hash: str, source_contract_hash: str) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": 1,
        "contract": REALIZATION_CONTRACT,
        "base_manifest": {
            "path": "DS9/asset_manifest.yaml",
            "sha256": base_hash,
        },
        "source_contracts": {
            "path": "DS9/config/engine_source_contracts.json",
            "sha256": source_contract_hash,
        },
        "created_at_utc": now,
        "updated_at_utc": now,
        "artifacts": {},
    }


def _load_realization(
    path: Path, *, base_hash: str, source_contract_hash: str
) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return _new_realization(base_hash, source_contract_hash)
    payload = dict(validator._load_private_json(path, "asset realization"))
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != REALIZATION_CONTRACT
    ):
        raise ValueError("external asset realization contract mismatch")
    if payload.get("base_manifest") != {
        "path": "DS9/asset_manifest.yaml",
        "sha256": base_hash,
    }:
        raise ValueError("external asset realization base-manifest digest drift")
    if payload.get("source_contracts") != {
        "path": "DS9/config/engine_source_contracts.json",
        "sha256": source_contract_hash,
    }:
        raise ValueError("external asset realization source-contract digest drift")
    if not isinstance(payload.get("artifacts"), dict):
        raise ValueError("external asset realization artifacts must be a mapping")
    return payload


def _expected_realization_matches(path: Path, expected: str) -> bool:
    if expected == "missing":
        return not path.exists() and not path.is_symlink()
    return path.is_file() and not path.is_symlink() and _sha256(path) == expected


def reconcile(
    *,
    engine_name: str,
    maintenance_manifest: Path,
    gpu_memory_guard: Mapping[str, Any],
    artifact_root: Path,
    base_manifest_path: Path,
    expected_base_manifest_sha256: str,
    expected_source_contracts_sha256: str,
    expected_realization_sha256: str,
    dry_run: bool,
) -> dict[str, Any]:
    if engine_name not in ENGINE_ARTIFACT_IDS:
        raise ValueError(f"unknown reviewed engine name: {engine_name}")
    artifact_root = artifact_root.resolve(strict=True)
    realization_path = artifact_root / REALIZATION_FILENAME
    base_hash = _sha256(base_manifest_path)
    if base_hash != expected_base_manifest_sha256:
        raise ValueError(
            "portable base manifest changed before reconciliation: "
            f"expected={expected_base_manifest_sha256} observed={base_hash}"
        )
    source_contract_hash = _sha256(SOURCE_CONTRACTS)
    if source_contract_hash != expected_source_contracts_sha256:
        raise ValueError(
            "engine source contracts changed before reconciliation: "
            f"expected={expected_source_contracts_sha256} observed={source_contract_hash}"
        )
    if not _expected_realization_matches(realization_path, expected_realization_sha256):
        raise ValueError("external asset realization changed before reconciliation")

    base = yaml.safe_load(base_manifest_path.read_text(encoding="utf-8"))
    artifact_id = ENGINE_ARTIFACT_IDS[engine_name]
    artifact = _artifact(base, artifact_id)
    known_engine_ids = {
        row["id"]
        for row in base.get("artifacts", [])
        if row.get("kind") == "tensorrt_engine"
    }
    realization = _load_realization(
        realization_path,
        base_hash=base_hash,
        source_contract_hash=source_contract_hash,
    )
    unknown_ids = set(realization["artifacts"]) - known_engine_ids
    if unknown_ids:
        raise ValueError(
            "external asset realization contains unknown artifact IDs: "
            + ", ".join(sorted(unknown_ids))
        )
    for realized_id, realized in realization["artifacts"].items():
        if not isinstance(realized, Mapping) or set(realized) != {
            "state",
            "provenance",
        }:
            raise ValueError(
                f"external realization may override only state/provenance: {realized_id}"
            )
        if realized.get("state") not in {"staged_unverified", "validated"}:
            raise ValueError(f"external realization has invalid state: {realized_id}")
        if not isinstance(realized.get("provenance"), Mapping):
            raise ValueError(
                f"external realization provenance is invalid: {realized_id}"
            )

    maintenance_path, maintenance_relative = _maintenance_paths(
        maintenance_manifest, artifact_root
    )
    maintenance_payload, maintenance_raw = validator._load_private_json_with_bytes(
        maintenance_path, "maintenance manifest"
    )
    maintenance_hash = hashlib.sha256(maintenance_raw).hexdigest()
    if (
        maintenance_payload.get("contract") != "noesis.ds9.engine_maintenance"
        or maintenance_payload.get("status") != "complete"
        or maintenance_payload.get("engine") != engine_name
    ):
        raise ValueError(
            "maintenance manifest is not a complete run for selected engine"
        )
    maintenance_inputs = _required_mapping(
        maintenance_payload.get("inputs"), "maintenance inputs"
    )
    recorded_contract = _required_mapping(
        maintenance_inputs.get("source_contracts"),
        "maintenance source-contract input",
    )
    if recorded_contract.get("sha256") != expected_source_contracts_sha256:
        raise ValueError(
            "maintenance run did not execute the pre-build source-contract digest"
        )

    compatibility = _required_mapping(
        artifact.get("compatibility"), f"{artifact_id}.compatibility"
    )
    output = validator._physical_path(
        validator._relative_path(artifact["output"]), artifact_root
    )
    if output.is_symlink() or not output.is_file() or output.stat().st_size <= 0:
        raise ValueError(f"engine output is missing, empty, or a symlink: {output}")
    installed = _required_mapping(
        maintenance_payload.get("installed"), "maintenance installed record"
    )
    output_hash = _sha256(output)
    output_size = output.stat().st_size
    if (
        installed.get("sha256") != output_hash
        or installed.get("size_bytes") != output_size
    ):
        raise ValueError("maintenance installed hash/size differs from external output")
    if engine_name == "mapanything":
        quality_authority = mapanything_quality_gate_from_source_contracts(
            SOURCE_CONTRACTS
        )
        quality_fixture = artifact_root / Path(
            str(quality_authority["fixture"]["artifact_relative_path"])
        )
        validate_mapanything_functional_quality_receipt(
            maintenance_payload,
            authority=quality_authority,
            expected_engine_path=output,
            evidence_directory=maintenance_path.parent,
            fixture_path=quality_fixture,
        )

    metadata = _required_mapping(
        maintenance_payload.get("metadata"), "maintenance metadata"
    )
    platform = _required_mapping(metadata.get("platform"), "maintenance platform")
    build_contract = _required_mapping(
        metadata.get("build_contract"), "maintenance build contract"
    )
    for key in (
        "image",
        "image_id",
        "base_digest",
        "tensorrt_version",
        "cuda_version",
        "driver_version",
        "gpu_name",
        "gpu_uuid",
        "gpu_compute_capability",
        "gpu_memory_mib",
    ):
        if not str(platform.get(key) or "").strip():
            raise ValueError(f"maintenance platform is missing {key}")
    build_image = _required_mapping(
        base["target"].get("build_image"), "target build-image authority"
    )
    exact_platform = {
        "image": build_image.get("reference"),
        "image_id": build_image.get("image_id"),
        "base_digest": build_image.get("base_digest"),
        "tensorrt_version": build_image.get("tensorrt_version"),
        "cuda_version": build_image.get("cuda_version"),
    }
    for key, expected in exact_platform.items():
        if platform.get(key) != expected:
            raise ValueError(
                f"maintenance platform {key} differs from build-image authority"
            )
    if not isinstance(gpu_memory_guard, Mapping):
        raise ValueError(
            "reconciliation requires an independently validated GPU-memory guard"
        )
    validated_gpu_memory_guard = validator._validate_gpu_memory_guard_proof(
        artifact_root=artifact_root,
        expected_engine=engine_name,
        maintenance_payload=maintenance_payload,
        guard_record=gpu_memory_guard,
    )

    build_command = next(
        (
            row.get("command")
            for row in maintenance_payload.get("commands", [])
            if row.get("label") in {"build", "build-tracker-engine"}
        ),
        None,
    )
    if not isinstance(build_command, list) or not build_command:
        raise ValueError("maintenance manifest lacks the engine build command")
    source_hash = validator._hash_sources(
        list(artifact["sources"]), artifact_root=artifact_root
    )
    pinned_source_hash = str(
        (artifact.get("provenance") or {}).get("source_sha256") or ""
    ).strip()
    if not pinned_source_hash or pinned_source_hash != source_hash:
        raise ValueError(
            f"declared source trust anchor drifted for {artifact_id}: "
            f"expected={pinned_source_hash or '<missing>'} observed={source_hash}"
        )

    provenance = {
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "built_at_utc": maintenance_payload.get("completed_at_utc"),
        "build_host": (
            f"{platform['image']}@{platform['image_id']} "
            f"driver={platform['driver_version']} gpu={platform['gpu_name']} "
            f"uuid={platform['gpu_uuid']}"
        ),
        "command": shlex.join(str(value) for value in build_command),
        "maintenance": {
            "manifest": maintenance_relative.as_posix(),
            "manifest_sha256": maintenance_hash,
            "output_size_bytes": output_size,
            "image": str(platform["image"]),
            "image_id": str(platform["image_id"]),
            "base_digest": str(platform["base_digest"]),
            "tensorrt_version": str(platform["tensorrt_version"]),
            "cuda_version": str(platform["cuda_version"]),
            "driver_version": str(platform["driver_version"]),
            "gpu": {
                "name": str(platform["gpu_name"]),
                "uuid": str(platform["gpu_uuid"]),
                "compute_capability": str(platform["gpu_compute_capability"]),
                "memory_mib": int(platform["gpu_memory_mib"]),
            },
            "precision": str(compatibility["precision"]),
            "batch": int(compatibility["batch"]),
            "tensor_contract": _tensor_contract(build_contract),
            "gpu_memory_guard": validated_gpu_memory_guard,
        },
    }
    validator._validate_engine_maintenance_proof(
        artifact_id=artifact_id,
        artifact=artifact,
        provenance=provenance,
        maintenance_payload=maintenance_payload,
        target=base["target"],
        source_contracts_path=SOURCE_CONTRACTS,
        artifact_root=artifact_root,
        maintenance_manifest_path=maintenance_path,
        maintenance_manifest_sha256=maintenance_hash,
        output_sha256=output_hash,
        output_size_bytes=output_size,
    )
    realization["artifacts"][artifact_id] = {
        "state": "staged_unverified",
        "provenance": provenance,
    }
    realization["updated_at_utc"] = _utc_now()
    result = {
        "artifact_id": artifact_id,
        "state": "staged_unverified",
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "output_size_bytes": output_size,
        "maintenance_manifest": maintenance_relative.as_posix(),
        "realization": str(realization_path),
    }
    if dry_run:
        return result

    if _sha256(base_manifest_path) != expected_base_manifest_sha256:
        raise ValueError("portable base manifest changed during reconciliation")
    if _sha256(SOURCE_CONTRACTS) != source_contract_hash:
        raise ValueError("engine source contracts changed during reconciliation")
    if not _expected_realization_matches(realization_path, expected_realization_sha256):
        raise ValueError("external asset realization changed during reconciliation")
    if _sha256(maintenance_path) != maintenance_hash:
        raise ValueError("maintenance manifest changed during reconciliation")
    if (
        validator._validate_gpu_memory_guard_proof(
            artifact_root=artifact_root,
            expected_engine=engine_name,
            maintenance_payload=maintenance_payload,
            guard_record=validated_gpu_memory_guard,
        )
        != validated_gpu_memory_guard
    ):
        raise ValueError("GPU-memory guard changed during reconciliation")
    if (
        output.is_symlink()
        or not output.is_file()
        or output.stat().st_size != output_size
        or _sha256(output) != output_hash
    ):
        raise ValueError("engine output changed during reconciliation")
    temporary = artifact_root / f".{REALIZATION_FILENAME}.writing-{os.getpid()}"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(realization, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, realization_path)
        os.chmod(realization_path, 0o600)
        directory = os.open(artifact_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    result["realization_sha256"] = _sha256(realization_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=sorted(ENGINE_ARTIFACT_IDS), required=True)
    parser.add_argument("--maintenance-manifest", type=Path, required=True)
    parser.add_argument("--gpu-memory-guard-record", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=REPO_ROOT / "DS9/asset_manifest.yaml",
    )
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--expected-source-contracts-sha256", required=True)
    parser.add_argument("--expected-realization-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        print(
            "[FAIL] real reconciliation is internal to "
            "finalize_engine_realization.py commit",
            file=sys.stderr,
        )
        return 2
    try:
        result = reconcile(
            engine_name=args.engine,
            maintenance_manifest=args.maintenance_manifest.expanduser().absolute(),
            gpu_memory_guard=validator._load_private_json(
                args.gpu_memory_guard_record.expanduser().absolute(),
                "GPU-memory guard reconciliation record",
            ),
            artifact_root=args.artifact_root.resolve(),
            base_manifest_path=args.base_manifest.resolve(),
            expected_base_manifest_sha256=str(args.expected_base_manifest_sha256),
            expected_source_contracts_sha256=str(args.expected_source_contracts_sha256),
            expected_realization_sha256=str(args.expected_realization_sha256),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
