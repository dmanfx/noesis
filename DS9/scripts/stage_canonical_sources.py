#!/usr/bin/env python3
"""Stage canonical and promoted DS9 ONNX inputs in an explicit artifact root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import onnx


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from engine_maintenance_common import (  # noqa: E402
    load_source_contracts,
    validate_source_contract,
)

ROOT_MODELS = (REPO_ROOT / "models").resolve()
SOURCE_CONTRACTS = DS9_ROOT / "config" / "engine_source_contracts.json"
CONTRACT_PROFILES = {
    "canonical": (
        "yolo26_m",
        "reid_swin",
        "yolo26_pose_n",
        "depth_anything_v2_tracking",
        "mapanything",
    ),
    "v3dt": ("yolo26_seg_s", "bodypose3dnet", "v3dt_tracker_reid"),
}
DEFAULT_MIN_FREE_HEADROOM_BYTES = 10 * 1024 * 1024 * 1024

CANONICAL_SOURCES = (
    (DS9_ROOT / "models" / "coco_labels.txt", Path("models/coco_labels.txt")),
    (
        DS9_ROOT / "models" / "deimv2_wholebody49" / "classes.txt",
        Path("models/deimv2_wholebody49/classes.txt"),
    ),
    (
        ROOT_MODELS / "onnx" / "yolo11s-seg_cust_fused.onnx",
        Path("models/onnx/yolo11s-seg_cust_fused.onnx"),
    ),
    (
        ROOT_MODELS / "onnx" / "yolo26m.onnx",
        Path("models/onnx/yolo26m.onnx"),
    ),
    (
        ROOT_MODELS / "yolo26s-seg_fused.onnx",
        Path("models/onnx/yolo26s-seg_fused.onnx"),
    ),
    (
        ROOT_MODELS / "onnx" / "reid_swin_tiny_market1501_aicity156_featuredim256.onnx",
        Path("models/onnx/reid_swin_tiny_market1501_aicity156_featuredim256.onnx"),
    ),
    (
        ROOT_MODELS / "onnx" / "yolo26n-pose_b3.onnx",
        Path("models/onnx/yolo26n-pose_b3.onnx"),
    ),
    (
        ROOT_MODELS / "onnx" / "depth_anything_v2_metric_hypersim_vits_294x518_b3.onnx",
        Path("models/onnx/depth_anything_v2_metric_hypersim_vits_294x518_b3.onnx"),
    ),
    (
        ROOT_MODELS / "onnx" / "mapanything_images_294x518_b3.onnx",
        Path("models/onnx/mapanything_images_294x518_b3.onnx"),
    ),
    (
        ROOT_MODELS / "onnx" / "deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx",
        Path("models/onnx/deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx"),
    ),
    (
        ROOT_MODELS / "onnx" / "deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx",
        Path("models/onnx/deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx"),
    ),
    (
        ROOT_MODELS / "bodypose3dnet" / "bodypose3dnet_accuracy.onnx",
        Path("models/onnx/bodypose3dnet_accuracy.onnx"),
    ),
    (
        ROOT_MODELS / "tracker_reid" / "resnet50_market1501.etlt",
        Path("models/tracker_reid/resnet50_market1501.etlt"),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_root() -> Path:
    raw = os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "").strip()
    if not raw:
        raise ValueError(
            "NOESIS_DS9_ARTIFACT_ROOT must name an explicit machine-local artifact root"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"NOESIS_DS9_ARTIFACT_ROOT must be absolute: {raw}")
    resolved = path.resolve(strict=False)
    if resolved in {Path("/"), REPO_ROOT.resolve(), DS9_ROOT.resolve()}:
        raise ValueError(f"refusing unsafe artifact root: {resolved}")
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"artifact root must not be inside the checkout: {resolved}")
    return resolved


def _external_locations(onnx_path: Path) -> list[Path]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    locations: set[Path] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        metadata = {item.key: item.value for item in tensor.external_data}
        raw = str(metadata.get("location", "")).strip()
        relative = Path(raw)
        if not raw or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"unsafe ONNX external-data location in {onnx_path}: {raw!r}"
            )
        locations.add(relative)
    return sorted(locations, key=lambda value: value.as_posix())


def _source_plan() -> list[tuple[Path, Path]]:
    destinations: dict[Path, Path] = {}

    def add(source: Path, destination: Path) -> None:
        existing = destinations.get(destination)
        if existing is not None and existing.resolve(strict=False) != source.resolve(
            strict=False
        ):
            raise ValueError(
                f"canonical source destination collision: {destination} maps to {existing} and {source}"
            )
        destinations[destination] = source

    for source, destination in CANONICAL_SOURCES:
        add(source, destination)
        if source.suffix.lower() == ".onnx":
            for relative in _external_locations(source):
                add(source.parent / relative, destination.parent / relative)

        provenance = DS9_ROOT / destination.with_name(
            f"{destination.name}.provenance.json"
        )
        if provenance.is_file():
            add(provenance, destination.with_name(provenance.name))

    return [(source, destination) for destination, source in destinations.items()]


def _minimum_free_headroom() -> int:
    raw = os.environ.get(
        "NOESIS_DS9_MIN_FREE_HEADROOM_BYTES", str(DEFAULT_MIN_FREE_HEADROOM_BYTES)
    ).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"NOESIS_DS9_MIN_FREE_HEADROOM_BYTES must be an integer: {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError("NOESIS_DS9_MIN_FREE_HEADROOM_BYTES must not be negative")
    return value


def _bounded_destination(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact destination: {relative}")
    root_resolved = root.resolve(strict=False)
    destination = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"artifact destination contains a symlink: {current}")
    try:
        destination.resolve(strict=False).relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"artifact destination escapes root: {destination}") from exc
    return destination


def _require_staging_capacity(root: Path, plan: list[tuple[Path, Path]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    growth = 0
    largest_temporary = 0
    for source, relative_destination in plan:
        destination = _bounded_destination(root, relative_destination)
        source_size = source.stat().st_size
        destination_size = destination.stat().st_size if destination.is_file() else 0
        growth += max(0, source_size - destination_size)
        largest_temporary = max(largest_temporary, source_size)
    headroom = _minimum_free_headroom()
    required = growth + largest_temporary + headroom
    available = shutil.disk_usage(root).free
    if available < required:
        raise OSError(
            "insufficient artifact-root capacity: "
            f"available={available} required={required} "
            f"(growth={growth}, atomic_copy_peak={largest_temporary}, residual_headroom={headroom}) "
            f"root={root}"
        )


def _copy_verified(source: Path, destination: Path, *, root: Path) -> tuple[str, bool]:
    source_digest = _sha256(source)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        if _sha256(destination) == source_digest:
            return source_digest, False

    destination.parent.mkdir(parents=True, exist_ok=True)
    _bounded_destination(root, destination.relative_to(root))
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        _fsync_file(temporary)
        if _sha256(temporary) != source_digest:
            raise RuntimeError(
                f"copied bytes failed SHA-256 verification: {destination}"
            )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return source_digest, True


def _aggregate_digest(records: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["destination"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate staged bytes without copying missing files.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--only",
        default="",
        help="Comma-separated engine source contracts to verify without reading root sources.",
    )
    selection.add_argument(
        "--profile",
        choices=sorted(CONTRACT_PROFILES),
        help="Verify one reviewed engine-source profile without reading root sources.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Explicitly stage or verify the complete source inventory.",
    )
    args = parser.parse_args()

    selected_names = {
        value.strip() for value in str(args.only or "").split(",") if value.strip()
    }
    if args.profile:
        selected_names = set(CONTRACT_PROFILES[args.profile])
    if selected_names and not args.verify_only:
        print("[FAIL] scoped --only/--profile is supported only with --verify-only", file=sys.stderr)
        return 2
    if args.verify_only and selected_names:
        try:
            root = _artifact_root()
            contracts = load_source_contracts(SOURCE_CONTRACTS)
            unknown = selected_names - set(contracts)
            if unknown:
                raise ValueError(
                    "unknown engine source contract(s): " + ", ".join(sorted(unknown))
                )
            total_bytes = 0
            for name in sorted(selected_names):
                contract = contracts[name]
                destination = _bounded_destination(root, Path(str(contract["staged"])))
                record = validate_source_contract(name, destination, SOURCE_CONTRACTS)
                total_bytes += sum(
                    int(row["size_bytes"]) for row in record.get("files", [])
                )
        except Exception as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 4
        print(
            f"[OK] verified {len(selected_names)} pinned staged engine source contract(s) "
            f"({total_bytes} bytes; root sources not read)"
        )
        return 0

    try:
        root = _artifact_root()
        plan = _source_plan()
        contracts = load_source_contracts(SOURCE_CONTRACTS)
        contract_by_destination = {
            str(contract["staged"]): name for name, contract in contracts.items()
        }
        for source, destination in plan:
            if not source.is_file() or source.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"canonical source is missing or empty: {source}"
                )
            contract_name = contract_by_destination.get(destination.as_posix())
            if contract_name:
                validate_source_contract(contract_name, source, SOURCE_CONTRACTS)
        if not args.verify_only:
            _require_staging_capacity(root, plan)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    records: list[dict[str, object]] = []
    copied_count = 0
    for source, relative_destination in plan:
        destination = _bounded_destination(root, relative_destination)
        if args.verify_only:
            if not destination.is_file() or destination.stat().st_size <= 0:
                print(
                    f"[FAIL] staged canonical source is missing or empty: {destination}",
                    file=sys.stderr,
                )
                return 4
            source_digest = _sha256(source)
            if _sha256(destination) != source_digest:
                print(
                    f"[FAIL] staged source digest differs: {destination}",
                    file=sys.stderr,
                )
                return 4
            copied = False
        else:
            source_digest, copied = _copy_verified(source, destination, root=root)
        copied_count += int(copied)
        records.append(
            {
                "source_resolved": str(source.resolve()),
                "destination": relative_destination.as_posix(),
                "sha256": source_digest,
                "size_bytes": source.stat().st_size,
            }
        )

    for relative_destination, contract_name in sorted(
        (
            (Path(destination), name)
            for destination, name in contract_by_destination.items()
        ),
        key=lambda row: row[0].as_posix(),
    ):
        validate_source_contract(
            contract_name,
            _bounded_destination(root, relative_destination),
            SOURCE_CONTRACTS,
        )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "file_count": len(records),
        "total_bytes": sum(int(record["size_bytes"]) for record in records),
        "aggregate_sha256": _aggregate_digest(records),
        "files": records,
    }
    provenance_path = root / "canonical_sources.provenance.json"
    if not args.verify_only:
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = provenance_path.with_name(
            f".{provenance_path.name}.partial-{os.getpid()}"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, provenance_path)
            _fsync_directory(provenance_path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    mode = "verified" if args.verify_only else "staged"
    print(
        f"[OK] {mode} {len(records)} canonical source files "
        f"({payload['total_bytes']} bytes; copied={copied_count}; aggregate_sha256={payload['aggregate_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
