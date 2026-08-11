"""Read-only legacy inspection and idempotent identity-v2 cutover helpers.

The inspector deliberately separates durable resident metadata from biometric
material.  A resident can be migrated without embeddings, but an embedding is
never promoted to an enrollment anchor unless its model fingerprint and
dimension are both explicitly verified.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import struct
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from noesis_core.private_paths import PrivatePathError, ensure_private_directory
from noesis_core.strict_json import strict_json_loads

from .store import (
    IdentityStore,
    LegacyResidentImport,
    MigrationConflict,
    normalize_display_name,
)

_DEFAULT_GALLERY_NAMES = (
    "resident_gallery.npz",
    "reid_gallery.npz",
    "gallery.npz",
)
_DEFAULT_VISITOR_POOL_NAMES = ("visitor_pool.json", "sid_pool.json")
_DEFAULT_MANIFEST_NAMES = (
    "identity_manifest.json",
    "gallery_manifest.json",
    "migration_manifest.json",
)


class LegacyMigrationError(RuntimeError):
    pass


class LegacyMigrationBlocked(LegacyMigrationError):
    pass


@dataclass(frozen=True)
class MigrationIssue:
    code: str
    severity: str
    subject: str
    detail: str


@dataclass(frozen=True)
class LegacyResidentPlan:
    resident_uuid: str
    display_name: str
    normalized_name: str
    compatibility_sid: int
    recorded_embedding_count: int
    actual_embedding_count: int
    anchor_vectors: Tuple[Tuple[float, ...], ...]
    blocked_reasons: Tuple[str, ...]
    gallery_compatible: bool

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_reasons)


@dataclass(frozen=True)
class LegacyMigrationReport:
    source_root: str
    migration_key: str
    source_digest: str
    expected_model_fingerprint: str
    expected_embedding_dim: int
    source_model_fingerprint: Optional[str]
    source_embedding_dim: Optional[int]
    asserted_source_model_fingerprint: Optional[str]
    asserted_source_embedding_dim: Optional[int]
    residents_source: Optional[str]
    gallery_source: Optional[str]
    visitor_pool_source: Optional[str]
    manifest_source: Optional[str]
    residents: Tuple[LegacyResidentPlan, ...]
    issues: Tuple[MigrationIssue, ...]
    visitor_collisions: Tuple[int, ...]

    @property
    def blocked_resident_count(self) -> int:
        return sum(1 for resident in self.residents if resident.blocked)

    @property
    def migratable_resident_count(self) -> int:
        return sum(1 for resident in self.residents if not resident.blocked)

    @property
    def fatal(self) -> bool:
        return any(issue.severity == "fatal" for issue in self.issues)

    @property
    def error_codes(self) -> Tuple[str, ...]:
        return tuple(
            sorted({issue.code for issue in self.issues if issue.severity == "error"})
        )

    @property
    def residents_without_importable_anchors(self) -> Tuple[str, ...]:
        return tuple(
            resident.resident_uuid
            for resident in self.residents
            if not resident.blocked and not resident.anchor_vectors
        )

    @property
    def apply_disposition(self) -> str:
        if self.fatal:
            return "blocked"
        if (
            self.error_codes
            or self.blocked_resident_count
            or self.residents_without_importable_anchors
            or self.visitor_pool_source is not None
        ):
            return "partial"
        return "full"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_root": self.source_root,
            "migration_key": self.migration_key,
            "source_digest": self.source_digest,
            "expected_model_fingerprint": self.expected_model_fingerprint,
            "expected_embedding_dim": self.expected_embedding_dim,
            "source_model_fingerprint": self.source_model_fingerprint,
            "source_embedding_dim": self.source_embedding_dim,
            "asserted_source_model_fingerprint": self.asserted_source_model_fingerprint,
            "asserted_source_embedding_dim": self.asserted_source_embedding_dim,
            "residents_source": self.residents_source,
            "gallery_source": self.gallery_source,
            "visitor_pool_source": self.visitor_pool_source,
            "manifest_source": self.manifest_source,
            "blocked_resident_count": self.blocked_resident_count,
            "migratable_resident_count": self.migratable_resident_count,
            "apply_disposition": self.apply_disposition,
            "full_migration_safe": self.apply_disposition == "full",
            "requires_accept_partial": self.apply_disposition == "partial",
            "error_codes": list(self.error_codes),
            "excluded_resident_uuids": [
                resident.resident_uuid
                for resident in self.residents
                if resident.blocked
            ],
            "residents_without_importable_anchors": list(
                self.residents_without_importable_anchors
            ),
            "visitor_state_imported": False,
            "visitor_collisions": list(self.visitor_collisions),
            "residents": [
                {
                    "resident_uuid": row.resident_uuid,
                    "display_name": row.display_name,
                    "normalized_name": row.normalized_name,
                    "compatibility_sid": row.compatibility_sid,
                    "recorded_embedding_count": row.recorded_embedding_count,
                    "actual_embedding_count": row.actual_embedding_count,
                    "anchor_count": len(row.anchor_vectors),
                    "blocked_reasons": list(row.blocked_reasons),
                    "gallery_compatible": row.gallery_compatible,
                }
                for row in self.residents
            ],
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "subject": issue.subject,
                    "detail": issue.detail,
                }
                for issue in self.issues
            ],
        }


@dataclass(frozen=True)
class MigrationApplyResult:
    idempotent: bool
    residents_created: int
    residents_existing: int
    anchors_created: int
    blocked_resident_uuids: Tuple[str, ...]
    anchorless_resident_uuids: Tuple[str, ...]
    disposition: str
    source_error_codes: Tuple[str, ...]
    visitor_state_imported: bool = False
    pre_apply_archive_receipt: str = ""


@dataclass(frozen=True)
class LegacyArchiveResult:
    archive_path: str
    source_digest: str
    files_archived: Tuple[str, ...]
    sources_removed: Tuple[str, ...]
    idempotent: bool
    receipt: str


@dataclass(frozen=True)
class _ArrayData:
    shape: Tuple[int, ...]
    values: Tuple[float | int, ...]
    fortran_order: bool


def _issue(
    issues: list[MigrationIssue],
    code: str,
    severity: str,
    subject: str,
    detail: str,
) -> None:
    issues.append(MigrationIssue(code, severity, subject, detail))


def _read_json(path: Path) -> Any:
    return strict_json_loads(
        path.read_bytes(),
        label=f"legacy identity file {path.name}",
    )


def _existing_candidates(root: Path, names: Sequence[str]) -> Tuple[Path, ...]:
    return tuple(root / name for name in names if (root / name).is_file())


def _source_digest(
    root: Path,
    *,
    expected_model_fingerprint: str,
    expected_embedding_dim: int,
    asserted_source_model_fingerprint: Optional[str],
    asserted_source_embedding_dim: Optional[int],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"noesis-identity-v2-legacy-source\0")
    for name in sorted(
        {
            "residents.json",
            *_DEFAULT_GALLERY_NAMES,
            *_DEFAULT_VISITOR_POOL_NAMES,
            *_DEFAULT_MANIFEST_NAMES,
        }
    ):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0present\0" if path.is_file() else b"\0missing\0")
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    profile = json.dumps(
        {
            "expected_model_fingerprint": expected_model_fingerprint,
            "expected_embedding_dim": expected_embedding_dim,
            "asserted_source_model_fingerprint": asserted_source_model_fingerprint,
            "asserted_source_embedding_dim": asserted_source_embedding_dim,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest.update(profile.encode("utf-8"))
    return digest.hexdigest()


def _migration_key(root: Path) -> str:
    root_digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:24]
    return f"identity-v2:{root_digest}"


def _canonical_uuid(raw: Any) -> str:
    return str(uuid.UUID(str(raw)))


def _parse_npy(payload: bytes) -> _ArrayData:
    if not payload.startswith(b"\x93NUMPY") or len(payload) < 10:
        raise ValueError("invalid NPY magic")
    major = payload[6]
    if major == 1:
        header_size = 2
        header_length = struct.unpack("<H", payload[8:10])[0]
        header_start = 10
        encoding = "latin1"
    elif major in (2, 3):
        header_size = 4
        header_length = struct.unpack("<I", payload[8:12])[0]
        header_start = 12
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise ValueError(f"unsupported NPY version {major}")
    if len(payload) < 8 + header_size + header_length:
        raise ValueError("truncated NPY header")
    header_end = header_start + header_length
    try:
        header = ast.literal_eval(
            payload[header_start:header_end].decode(encoding).strip()
        )
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid NPY header") from exc
    if not isinstance(header, dict):
        raise ValueError("NPY header is not a mapping")
    descr = str(header.get("descr", ""))
    shape_raw = header.get("shape")
    if not isinstance(shape_raw, tuple) or not all(
        isinstance(value, int) and value >= 0 for value in shape_raw
    ):
        raise ValueError("invalid NPY shape")
    shape = tuple(int(value) for value in shape_raw)
    count = math.prod(shape) if shape else 1
    if len(descr) < 3 or descr[0] not in "<>=|":
        raise ValueError(f"unsupported NPY dtype {descr!r}")
    try:
        item_size = int(descr[2:])
    except ValueError as exc:
        raise ValueError(f"unsupported NPY dtype {descr!r}") from exc
    kind = descr[1]
    formats = {
        ("b", 1): "b",
        ("u", 1): "B",
        ("i", 2): "h",
        ("u", 2): "H",
        ("i", 4): "i",
        ("u", 4): "I",
        ("i", 8): "q",
        ("u", 8): "Q",
        ("f", 4): "f",
        ("f", 8): "d",
        ("?", 1): "?",
    }
    code = formats.get((kind, item_size))
    if code is None:
        raise ValueError(f"unsupported NPY dtype {descr!r}")
    byte_order = descr[0]
    if byte_order == "|":
        byte_order = "="
    data = payload[header_end:]
    expected_bytes = count * item_size
    if len(data) != expected_bytes:
        raise ValueError(
            f"NPY payload has {len(data)} bytes; expected {expected_bytes}"
        )
    values: Tuple[float | int, ...]
    if count:
        values = tuple(struct.unpack(f"{byte_order}{count}{code}", data))
    else:
        values = ()
    return _ArrayData(
        shape=shape,
        values=values,
        fortran_order=bool(header.get("fortran_order", False)),
    )


def _read_npz(path: Path) -> Dict[str, _ArrayData]:
    arrays: Dict[str, _ArrayData] = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.endswith(".npy"):
                continue
            key = Path(member.filename).name[:-4]
            if key in arrays:
                raise ValueError(f"duplicate NPZ array key {key!r}")
            arrays[key] = _parse_npy(archive.read(member))
    return arrays


def _matrix_rows(array: _ArrayData) -> Tuple[Tuple[float, ...], ...]:
    if len(array.shape) != 2:
        raise ValueError(f"expected a matrix; got shape {array.shape}")
    rows, columns = array.shape
    if array.fortran_order:
        return tuple(
            tuple(float(array.values[row + column * rows]) for column in range(columns))
            for row in range(rows)
        )
    return tuple(
        tuple(
            float(value) for value in array.values[row * columns : (row + 1) * columns]
        )
        for row in range(rows)
    )


def _flat_ints(array: _ArrayData) -> Tuple[int, ...]:
    if len(array.shape) > 1:
        raise ValueError(f"expected a scalar/vector; got shape {array.shape}")
    return tuple(int(value) for value in array.values)


def _select_single_source(
    root: Path,
    names: Sequence[str],
    *,
    source_kind: str,
    issues: list[MigrationIssue],
) -> Optional[Path]:
    matches = _existing_candidates(root, names)
    if len(matches) > 1:
        _issue(
            issues,
            f"multiple_{source_kind}_sources",
            "error",
            source_kind,
            "multiple legacy files exist; choose a single source before applying: "
            + ", ".join(path.name for path in matches),
        )
        return None
    return matches[0] if matches else None


def _manifest_profile(
    manifest_path: Optional[Path],
    issues: list[MigrationIssue],
) -> Tuple[Optional[str], Optional[int]]:
    if manifest_path is None:
        return None, None
    try:
        raw = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "manifest_invalid", "error", manifest_path.name, str(exc))
        return None, None
    if not isinstance(raw, Mapping):
        _issue(
            issues,
            "manifest_invalid",
            "error",
            manifest_path.name,
            "manifest root must be a JSON object",
        )
        return None, None
    raw_fingerprint = raw.get("model_fingerprint")
    fingerprint = str(raw_fingerprint).strip() if raw_fingerprint is not None else None
    if not fingerprint:
        fingerprint = None
    raw_dim = raw.get("embedding_dim")
    dimension: Optional[int]
    try:
        dimension = int(raw_dim) if raw_dim is not None else None
        if dimension is not None and dimension <= 0:
            raise ValueError
    except (TypeError, ValueError):
        _issue(
            issues,
            "manifest_dimension_invalid",
            "error",
            manifest_path.name,
            f"invalid embedding_dim {raw_dim!r}",
        )
        dimension = None
    return fingerprint, dimension


def _effective_source_profile(
    *,
    asserted_fingerprint: Optional[str],
    asserted_dimension: Optional[int],
    manifest_fingerprint: Optional[str],
    manifest_dimension: Optional[int],
    issues: list[MigrationIssue],
) -> Tuple[Optional[str], Optional[int]]:
    fingerprint = asserted_fingerprint or manifest_fingerprint
    dimension = (
        asserted_dimension if asserted_dimension is not None else manifest_dimension
    )
    if (
        asserted_fingerprint
        and manifest_fingerprint
        and asserted_fingerprint != manifest_fingerprint
    ):
        _issue(
            issues,
            "source_profile_conflict",
            "error",
            "model_fingerprint",
            "operator assertion and manifest fingerprint disagree",
        )
        fingerprint = None
    if (
        asserted_dimension is not None
        and manifest_dimension is not None
        and asserted_dimension != manifest_dimension
    ):
        _issue(
            issues,
            "source_profile_conflict",
            "error",
            "embedding_dim",
            "operator assertion and manifest embedding dimension disagree",
        )
        dimension = None
    return fingerprint, dimension


def _load_resident_rows(
    residents_path: Path,
    issues: list[MigrationIssue],
) -> list[Dict[str, Any]]:
    try:
        raw = _read_json(residents_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "residents_invalid", "fatal", residents_path.name, str(exc))
        return []
    if not isinstance(raw, Mapping) or not isinstance(raw.get("residents"), list):
        _issue(
            issues,
            "residents_invalid",
            "fatal",
            residents_path.name,
            "expected a JSON object containing a residents list",
        )
        return []
    return [
        row if isinstance(row, dict) else {"_invalid_row": row}
        for row in raw["residents"]
    ]


def _gallery_vectors(
    gallery_path: Optional[Path],
    issues: list[MigrationIssue],
) -> Tuple[Dict[int, Tuple[Tuple[float, ...], ...]], Tuple[int, ...]]:
    if gallery_path is None:
        return {}, ()
    try:
        arrays = _read_npz(gallery_path)
        if "sids" in arrays:
            sids = _flat_ints(arrays["sids"])
        else:
            sids = tuple(
                sorted(
                    int(key[4:])
                    for key in arrays
                    if key.startswith("emb_") and key[4:].isdigit()
                )
            )
        vectors: Dict[int, Tuple[Tuple[float, ...], ...]] = {}
        for sid in sids:
            key = f"emb_{sid}"
            if key not in arrays:
                _issue(
                    issues,
                    "gallery_entry_missing",
                    "error",
                    str(sid),
                    f"sids lists {sid}, but {key} is absent",
                )
                vectors[sid] = ()
                continue
            rows = _matrix_rows(arrays[key])
            if any(not all(math.isfinite(value) for value in row) for row in rows):
                _issue(
                    issues,
                    "gallery_non_finite",
                    "error",
                    str(sid),
                    "gallery contains NaN or infinite values",
                )
                vectors[sid] = ()
                continue
            vectors[sid] = rows
        return vectors, tuple(sorted(set(sids)))
    except (OSError, ValueError, zipfile.BadZipFile, struct.error) as exc:
        _issue(issues, "gallery_invalid", "error", gallery_path.name, str(exc))
        return {}, ()


def _visitor_collisions(
    visitor_pool_path: Optional[Path],
    *,
    gallery_sids: Iterable[int],
    visitor_id_min: int,
    visitor_id_max: int,
    issues: list[MigrationIssue],
) -> Tuple[int, ...]:
    if visitor_pool_path is None:
        return ()
    try:
        raw = _read_json(visitor_pool_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues, "visitor_pool_invalid", "error", visitor_pool_path.name, str(exc)
        )
        return ()
    if not isinstance(raw, Mapping):
        _issue(
            issues,
            "visitor_pool_invalid",
            "error",
            visitor_pool_path.name,
            "visitor pool root must be a JSON object",
        )
        return ()
    free_raw = raw.get("free_visitor_sids", raw.get("free_sids", []))
    last_seen_raw = raw.get("visitor_last_seen", {})
    free: list[int] = []
    active: set[int] = set()
    if isinstance(free_raw, list):
        for value in free_raw:
            try:
                sid = int(value)
            except (TypeError, ValueError):
                _issue(
                    issues,
                    "visitor_slot_invalid",
                    "warning",
                    repr(value),
                    "not an integer",
                )
                continue
            if visitor_id_min <= sid <= visitor_id_max:
                free.append(sid)
    else:
        _issue(
            issues,
            "visitor_pool_invalid",
            "error",
            visitor_pool_path.name,
            "free visitor slots must be a list",
        )
    if isinstance(last_seen_raw, Mapping):
        for key in last_seen_raw:
            try:
                sid = int(key)
            except (TypeError, ValueError):
                _issue(
                    issues,
                    "visitor_slot_invalid",
                    "warning",
                    repr(key),
                    "not an integer",
                )
                continue
            if visitor_id_min <= sid <= visitor_id_max:
                active.add(sid)
    else:
        _issue(
            issues,
            "visitor_pool_invalid",
            "error",
            visitor_pool_path.name,
            "visitor_last_seen must be a JSON object",
        )
    duplicate_free = {sid for sid in free if free.count(sid) > 1}
    free_set = set(free)
    gallery_visitors = {
        int(sid) for sid in gallery_sids if visitor_id_min <= int(sid) <= visitor_id_max
    }
    collisions = tuple(
        sorted(duplicate_free | (free_set & active) | (free_set & gallery_visitors))
    )
    for sid in collisions:
        reasons = []
        if sid in duplicate_free:
            reasons.append("listed free more than once")
        if sid in free_set and sid in active:
            reasons.append("listed both free and active")
        if sid in free_set and sid in gallery_visitors:
            reasons.append("listed free while gallery biometrics remain")
        _issue(
            issues,
            "visitor_slot_collision",
            "error",
            str(sid),
            "; ".join(reasons),
        )
    return collisions


def inspect_legacy_identity(
    source_root: str | Path,
    *,
    expected_model_fingerprint: str,
    expected_embedding_dim: int,
    source_model_fingerprint: Optional[str] = None,
    source_embedding_dim: Optional[int] = None,
    visitor_id_min: int = 1000,
    visitor_id_max: int = 1031,
) -> LegacyMigrationReport:
    """Inspect legacy state without changing it or creating a v2 database."""

    root = Path(source_root).expanduser().resolve()
    expected_fingerprint = str(expected_model_fingerprint or "").strip()
    expected_dim = int(expected_embedding_dim)
    if not expected_fingerprint or expected_dim <= 0:
        raise ValueError(
            "expected model fingerprint and embedding dimension are required"
        )
    asserted_fingerprint = (
        str(source_model_fingerprint).strip()
        if source_model_fingerprint is not None
        else None
    )
    if asserted_fingerprint == "":
        asserted_fingerprint = None
    asserted_dim = (
        int(source_embedding_dim) if source_embedding_dim is not None else None
    )
    if asserted_dim is not None and asserted_dim <= 0:
        raise ValueError("source_embedding_dim must be positive")
    if int(visitor_id_min) <= 0 or int(visitor_id_max) < int(visitor_id_min):
        raise ValueError("invalid visitor slot range")

    issues: list[MigrationIssue] = []
    residents_path = root / "residents.json"
    if not residents_path.is_file():
        _issue(
            issues,
            "residents_missing",
            "fatal",
            residents_path.name,
            "legacy resident enrollment file does not exist",
        )
    gallery_path = _select_single_source(
        root,
        _DEFAULT_GALLERY_NAMES,
        source_kind="gallery",
        issues=issues,
    )
    visitor_path = _select_single_source(
        root,
        _DEFAULT_VISITOR_POOL_NAMES,
        source_kind="visitor_pool",
        issues=issues,
    )
    manifest_path = _select_single_source(
        root,
        _DEFAULT_MANIFEST_NAMES,
        source_kind="manifest",
        issues=issues,
    )

    manifest_fingerprint, manifest_dim = _manifest_profile(manifest_path, issues)
    effective_fingerprint, effective_dim = _effective_source_profile(
        asserted_fingerprint=asserted_fingerprint,
        asserted_dimension=asserted_dim,
        manifest_fingerprint=manifest_fingerprint,
        manifest_dimension=manifest_dim,
        issues=issues,
    )
    fingerprint_verified = effective_fingerprint == expected_fingerprint
    dimension_profile_conflict = any(
        issue.code == "source_profile_conflict" and issue.subject == "embedding_dim"
        for issue in issues
    )
    dimension_profile_verified = not dimension_profile_conflict and (
        effective_dim is None or effective_dim == expected_dim
    )
    if gallery_path is not None and effective_fingerprint is None:
        _issue(
            issues,
            "model_fingerprint_unknown",
            "error",
            gallery_path.name,
            "legacy embeddings require a manifest or explicit source fingerprint",
        )
    elif gallery_path is not None and not fingerprint_verified:
        _issue(
            issues,
            "model_fingerprint_mismatch",
            "error",
            gallery_path.name,
            f"source {effective_fingerprint!r} does not match expected {expected_fingerprint!r}",
        )
    if (
        gallery_path is not None
        and effective_dim is not None
        and not dimension_profile_verified
    ):
        _issue(
            issues,
            "model_dimension_mismatch",
            "error",
            gallery_path.name,
            f"source dimension {effective_dim} does not match expected {expected_dim}",
        )

    gallery, gallery_sids = _gallery_vectors(gallery_path, issues)
    if gallery_path is None and not any(
        issue.code == "multiple_gallery_sources" for issue in issues
    ):
        _issue(
            issues,
            "gallery_missing",
            "warning",
            "gallery",
            "no legacy gallery file exists; residents will migrate without anchors",
        )
    elif gallery_path is not None and not any(gallery.values()):
        _issue(
            issues,
            "gallery_empty",
            "warning",
            gallery_path.name,
            "legacy gallery contains no usable embeddings",
        )
    collisions = _visitor_collisions(
        visitor_path,
        gallery_sids=gallery_sids,
        visitor_id_min=int(visitor_id_min),
        visitor_id_max=int(visitor_id_max),
        issues=issues,
    )
    rows = (
        _load_resident_rows(residents_path, issues) if residents_path.is_file() else []
    )

    mutable_plans: list[Dict[str, Any]] = []
    for index, raw in enumerate(rows):
        subject = f"resident[{index}]"
        blocked: list[str] = []
        if "_invalid_row" in raw:
            _issue(
                issues,
                "resident_invalid",
                "error",
                subject,
                "resident row is not an object",
            )
            mutable_plans.append(
                {
                    "resident_uuid": f"invalid-row-{index}",
                    "display_name": "",
                    "normalized_name": "",
                    "compatibility_sid": -1,
                    "recorded_embedding_count": 0,
                    "actual_embedding_count": 0,
                    "anchor_vectors": (),
                    "blocked_reasons": ["resident_invalid"],
                    "gallery_compatible": False,
                }
            )
            continue
        try:
            resident_uuid = _canonical_uuid(raw.get("uuid"))
        except (TypeError, ValueError, AttributeError):
            resident_uuid = f"invalid-uuid-{index}"
            blocked.append("resident_uuid_invalid")
            _issue(
                issues,
                "resident_uuid_invalid",
                "error",
                subject,
                f"invalid UUID {raw.get('uuid')!r}",
            )
        display_name = " ".join(str(raw.get("display_name", "")).split())
        try:
            normalized_name = normalize_display_name(display_name)
        except ValueError:
            normalized_name = ""
            blocked.append("display_name_empty")
            _issue(
                issues,
                "display_name_empty",
                "error",
                resident_uuid,
                "display name is empty",
            )
        try:
            sid = int(raw.get("stable_id"))
            if sid <= 0:
                raise ValueError
        except (TypeError, ValueError):
            sid = -1
            blocked.append("compatibility_sid_invalid")
            _issue(
                issues,
                "compatibility_sid_invalid",
                "error",
                resident_uuid,
                f"invalid stable_id {raw.get('stable_id')!r}",
            )
        try:
            recorded_count = max(0, int(raw.get("embedding_count", 0)))
        except (TypeError, ValueError):
            recorded_count = 0
            _issue(
                issues,
                "embedding_count_invalid",
                "warning",
                resident_uuid,
                f"invalid embedding_count {raw.get('embedding_count')!r}; treated as zero",
            )
        vectors = gallery.get(sid, ()) if sid > 0 else ()
        actual_count = len(vectors)
        if recorded_count != actual_count:
            _issue(
                issues,
                "stale_embedding_count",
                "warning",
                resident_uuid,
                f"recorded {recorded_count}; gallery contains {actual_count}",
            )
        if actual_count == 0:
            _issue(
                issues,
                "empty_resident_gallery",
                "warning",
                resident_uuid,
                "resident has no usable legacy embeddings",
            )
        vector_dimensions = {len(vector) for vector in vectors}
        vector_dim_ok = not vector_dimensions or vector_dimensions == {expected_dim}
        if not vector_dim_ok:
            _issue(
                issues,
                "embedding_dimension_mismatch",
                "error",
                resident_uuid,
                f"gallery dimensions {sorted(vector_dimensions)} do not match expected {expected_dim}",
            )
        gallery_compatible = bool(
            actual_count
            and fingerprint_verified
            and dimension_profile_verified
            and vector_dim_ok
        )
        mutable_plans.append(
            {
                "resident_uuid": resident_uuid,
                "display_name": display_name,
                "normalized_name": normalized_name,
                "compatibility_sid": sid,
                "recorded_embedding_count": recorded_count,
                "actual_embedding_count": actual_count,
                "anchor_vectors": vectors if gallery_compatible else (),
                "blocked_reasons": blocked,
                "gallery_compatible": gallery_compatible,
            }
        )

    duplicate_fields = (
        ("normalized_name", "duplicate_normalized_name"),
        ("compatibility_sid", "duplicate_compatibility_sid"),
        ("resident_uuid", "duplicate_resident_uuid"),
    )
    for field, code in duplicate_fields:
        groups: Dict[Any, list[int]] = {}
        for index, plan in enumerate(mutable_plans):
            value = plan[field]
            if value in ("", -1) or str(value).startswith("invalid-"):
                continue
            groups.setdefault(value, []).append(index)
        for value, indices in groups.items():
            if len(indices) < 2:
                continue
            for index in indices:
                if code not in mutable_plans[index]["blocked_reasons"]:
                    mutable_plans[index]["blocked_reasons"].append(code)
            _issue(
                issues,
                code,
                "error",
                str(value),
                "ambiguous legacy records remain blocked and are not auto-merged",
            )

    plans = tuple(
        LegacyResidentPlan(
            resident_uuid=str(plan["resident_uuid"]),
            display_name=str(plan["display_name"]),
            normalized_name=str(plan["normalized_name"]),
            compatibility_sid=int(plan["compatibility_sid"]),
            recorded_embedding_count=int(plan["recorded_embedding_count"]),
            actual_embedding_count=int(plan["actual_embedding_count"]),
            anchor_vectors=tuple(
                tuple(float(value) for value in vector)
                for vector in plan["anchor_vectors"]
            ),
            blocked_reasons=tuple(sorted(set(plan["blocked_reasons"]))),
            gallery_compatible=bool(plan["gallery_compatible"]),
        )
        for plan in sorted(
            mutable_plans,
            key=lambda row: (int(row["compatibility_sid"]), str(row["resident_uuid"])),
        )
    )
    source_digest = _source_digest(
        root,
        expected_model_fingerprint=expected_fingerprint,
        expected_embedding_dim=expected_dim,
        asserted_source_model_fingerprint=asserted_fingerprint,
        asserted_source_embedding_dim=asserted_dim,
    )
    return LegacyMigrationReport(
        source_root=str(root),
        migration_key=_migration_key(root),
        source_digest=source_digest,
        expected_model_fingerprint=expected_fingerprint,
        expected_embedding_dim=expected_dim,
        source_model_fingerprint=effective_fingerprint,
        source_embedding_dim=effective_dim,
        asserted_source_model_fingerprint=asserted_fingerprint,
        asserted_source_embedding_dim=asserted_dim,
        residents_source=str(residents_path) if residents_path.is_file() else None,
        gallery_source=str(gallery_path) if gallery_path is not None else None,
        visitor_pool_source=str(visitor_path) if visitor_path is not None else None,
        manifest_source=str(manifest_path) if manifest_path is not None else None,
        residents=plans,
        issues=tuple(
            sorted(
                issues,
                key=lambda issue: (
                    {"fatal": 0, "error": 1, "warning": 2}.get(issue.severity, 3),
                    issue.code,
                    issue.subject,
                    issue.detail,
                ),
            )
        ),
        visitor_collisions=collisions,
    )


def apply_legacy_identity(
    report: LegacyMigrationReport,
    store: IdentityStore,
    *,
    archive_root: str | Path,
    archive_receipt: str,
    accept_partial: bool = False,
    now: Optional[float] = None,
) -> MigrationApplyResult:
    """Apply one inspected snapshot; blocked rows are deliberately omitted."""

    if report.fatal:
        fatal_codes = sorted(
            {issue.code for issue in report.issues if issue.severity == "fatal"}
        )
        raise LegacyMigrationBlocked(
            "legacy migration has fatal inspection issues: " + ", ".join(fatal_codes)
        )
    if report.apply_disposition == "partial" and not accept_partial:
        raise LegacyMigrationBlocked(
            "partial migration requires explicit accept_partial=True after reviewing exclusions"
        )
    verified_archive = verify_legacy_archive(
        report,
        archive_root,
        expected_receipt=archive_receipt,
    )
    imports = tuple(
        LegacyResidentImport(
            resident_uuid=resident.resident_uuid,
            display_name=resident.display_name,
            compatibility_sid=resident.compatibility_sid,
            model_fingerprint=report.expected_model_fingerprint,
            embedding_dim=report.expected_embedding_dim,
            anchor_vectors=resident.anchor_vectors
            if resident.gallery_compatible
            else (),
        )
        for resident in report.residents
        if not resident.blocked
    )
    result = store.import_legacy_residents_once(
        migration_key=report.migration_key,
        source_digest=report.source_digest,
        residents=imports,
        now=now,
    )
    return MigrationApplyResult(
        idempotent=result.idempotent,
        residents_created=result.residents_created,
        residents_existing=result.residents_existing,
        anchors_created=result.anchors_created,
        blocked_resident_uuids=tuple(
            resident.resident_uuid for resident in report.residents if resident.blocked
        ),
        anchorless_resident_uuids=tuple(
            resident.resident_uuid
            for resident in report.residents
            if not resident.blocked and not resident.anchor_vectors
        ),
        disposition=report.apply_disposition,
        source_error_codes=report.error_codes,
        visitor_state_imported=False,
        pre_apply_archive_receipt=verified_archive.receipt,
    )


def _legacy_source_paths(root: Path) -> Tuple[Path, ...]:
    names = {
        "residents.json",
        *_DEFAULT_GALLERY_NAMES,
        *_DEFAULT_VISITOR_POOL_NAMES,
        *_DEFAULT_MANIFEST_NAMES,
    }
    return tuple(root / name for name in sorted(names) if (root / name).is_file())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_existing_archive(
    destination: Path,
    report: LegacyMigrationReport,
) -> LegacyArchiveResult:
    manifest_path = destination / "archive_manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LegacyMigrationError(
            f"existing archive manifest is invalid: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise LegacyMigrationError("existing archive manifest is not a JSON object")
    if (
        str(manifest.get("source_digest")) != report.source_digest
        or str(manifest.get("migration_key")) != report.migration_key
    ):
        raise LegacyMigrationError(
            "existing archive belongs to a different legacy snapshot"
        )
    file_rows = manifest.get("files")
    if not isinstance(file_rows, list):
        raise LegacyMigrationError("existing archive manifest has no file inventory")
    names = []
    for row in file_rows:
        if not isinstance(row, Mapping):
            raise LegacyMigrationError("existing archive file inventory is malformed")
        name = str(row.get("name", ""))
        expected_hash = str(row.get("sha256", ""))
        archived = destination / name
        if (
            not name
            or not archived.is_file()
            or _file_sha256(archived) != expected_hash
        ):
            raise LegacyMigrationError(
                f"existing archive file failed verification: {name!r}"
            )
        names.append(name)
    return LegacyArchiveResult(
        archive_path=str(destination),
        source_digest=report.source_digest,
        files_archived=tuple(sorted(names)),
        sources_removed=(),
        idempotent=True,
        receipt=_file_sha256(manifest_path),
    )


def archive_legacy_identity(
    report: LegacyMigrationReport,
    archive_root: str | Path,
    *,
    remove_source: bool = False,
    now: Optional[float] = None,
) -> LegacyArchiveResult:
    """Create a verified private archive before any migration is applied."""
    root = Path(report.source_root).expanduser().resolve()
    try:
        archive_base = ensure_private_directory(
            Path(archive_root).expanduser(),
            label="identity legacy archive root",
        )
    except PrivatePathError as exc:
        raise LegacyMigrationError(str(exc)) from exc
    destination = archive_base / f"identity-v2-legacy-{report.source_digest[:16]}"
    if destination.exists():
        try:
            ensure_private_directory(
                destination,
                label="identity legacy archive",
            )
        except PrivatePathError as exc:
            raise LegacyMigrationError(str(exc)) from exc
        result = _verify_existing_archive(destination, report)
        if remove_source:
            removed = []
            archived_hashes = {
                path.name: _file_sha256(path)
                for path in destination.iterdir()
                if path.is_file() and path.name != "archive_manifest.json"
            }
            for source in _legacy_source_paths(root):
                if archived_hashes.get(source.name) != _file_sha256(source):
                    raise LegacyMigrationError(
                        f"source changed after archive; refusing to remove {source.name}"
                    )
                source.unlink()
                removed.append(source.name)
            return LegacyArchiveResult(
                archive_path=result.archive_path,
                source_digest=result.source_digest,
                files_archived=result.files_archived,
                sources_removed=tuple(sorted(removed)),
                idempotent=True,
                receipt=result.receipt,
            )
        return result

    current_digest = _source_digest(
        root,
        expected_model_fingerprint=report.expected_model_fingerprint,
        expected_embedding_dim=report.expected_embedding_dim,
        asserted_source_model_fingerprint=report.asserted_source_model_fingerprint,
        asserted_source_embedding_dim=report.asserted_source_embedding_dim,
    )
    if current_digest != report.source_digest:
        raise LegacyMigrationError(
            "legacy source changed after inspection; refusing to archive"
        )
    sources = _legacy_source_paths(root)
    if not sources:
        raise LegacyMigrationError("legacy source contains no files to archive")

    temporary = archive_base / f".{destination.name}.tmp-{uuid.uuid4()}"
    temporary.mkdir(mode=0o700)
    ensure_private_directory(temporary, label="identity legacy archive temporary")
    inventory = []
    try:
        for source in sources:
            target = temporary / source.name
            target_descriptor = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with source.open("rb") as source_handle, os.fdopen(
                target_descriptor, "wb"
            ) as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            source_hash = _file_sha256(source)
            target_hash = _file_sha256(target)
            if source_hash != target_hash:
                raise LegacyMigrationError(
                    f"archive copy verification failed for {source.name}"
                )
            inventory.append(
                {
                    "name": source.name,
                    "bytes": int(target.stat().st_size),
                    "sha256": target_hash,
                }
            )
        manifest = {
            "version": 1,
            "archived_at": float(time.time() if now is None else now),
            "source_root": str(root),
            "migration_key": report.migration_key,
            "source_digest": report.source_digest,
            "expected_model_fingerprint": report.expected_model_fingerprint,
            "expected_embedding_dim": report.expected_embedding_dim,
            "asserted_source_model_fingerprint": report.asserted_source_model_fingerprint,
            "asserted_source_embedding_dim": report.asserted_source_embedding_dim,
            "files": sorted(inventory, key=lambda row: str(row["name"])),
        }
        _write_private_json(temporary / "archive_manifest.json", manifest)
        receipt = _file_sha256(temporary / "archive_manifest.json")
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(archive_base)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    removed = []
    if remove_source:
        archived_hashes = {str(row["name"]): str(row["sha256"]) for row in inventory}
        for source in sources:
            if _file_sha256(source) != archived_hashes[source.name]:
                raise LegacyMigrationError(
                    f"source changed after archive; refusing to remove {source.name}"
                )
        for source in sources:
            source.unlink()
            removed.append(source.name)
    return LegacyArchiveResult(
        archive_path=str(destination),
        source_digest=report.source_digest,
        files_archived=tuple(sorted(source.name for source in sources)),
        sources_removed=tuple(sorted(removed)),
        idempotent=False,
        receipt=receipt,
    )


def verify_legacy_archive(
    report: LegacyMigrationReport,
    archive_root: str | Path,
    *,
    expected_receipt: str,
) -> LegacyArchiveResult:
    receipt = str(expected_receipt or "").strip().lower()
    if len(receipt) != 64:
        raise LegacyMigrationError("archive receipt must be a SHA-256 digest")
    archive_base = Path(archive_root).expanduser().resolve()
    destination = archive_base / f"identity-v2-legacy-{report.source_digest[:16]}"
    if not destination.is_dir():
        raise LegacyMigrationBlocked(
            "verified pre-apply archive is missing for this source digest"
        )
    result = _verify_existing_archive(destination, report)
    if result.receipt.lower() != receipt:
        raise LegacyMigrationError(
            "archive receipt does not match the verified pre-apply archive manifest"
        )
    return result


__all__ = [
    "LegacyMigrationBlocked",
    "LegacyMigrationError",
    "LegacyMigrationReport",
    "LegacyArchiveResult",
    "LegacyResidentPlan",
    "MigrationApplyResult",
    "MigrationConflict",
    "MigrationIssue",
    "apply_legacy_identity",
    "archive_legacy_identity",
    "inspect_legacy_identity",
    "verify_legacy_archive",
]
