"""Resolve an exact, review-only PCF Scene Prior for a phone scan.

The catalog is configuration, not a source of inferred fallbacks.  This
resolver reads only the bounded catalog entries needed for configured camera
bindings and verifies the selected immutable manifest and points artifact on
every resolution.  It deliberately returns the catalog's raw frame binding;
the binding descriptor hash used by the public selector is a hash of that raw
JSON object, not the transform digest contained inside it.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from noesis_core.contracts.scene_prior import (
    ScenePriorCatalog,
    ScenePriorCatalogEntry,
    ScenePriorFrameBinding,
    ScenePriorRevision,
)
from noesis_core.scene_files import (
    SceneFileError,
    absolute_path_without_resolving,
    load_strict_json,
    read_scene_file,
    read_scene_root_file,
)


MAX_CATALOG_JSON_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_JSON_BYTES = 2 * 1024 * 1024
MAX_POINTS_BYTES = 64 * 1024 * 1024
MAX_CONFIGURED_REVISIONS = 64
MAX_CONFIGURED_CAMERA_BINDINGS = 64

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_FIELDS = frozenset(
    {"kind", "prior_id", "manifest_sha256", "camera_id", "frame_binding_sha256"}
)


class _NoMatchingSource(Exception):
    """The candidate is valid configuration for another source scan."""


class _InvalidCandidate(Exception):
    """A configured candidate was found but cannot be selected."""


def _json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("frame binding is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _bytes_fingerprint(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _read_json_file(path: Path, *, label: str, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    try:
        verified = read_scene_file(path, label=label, max_bytes=max_bytes)
        payload = load_strict_json(verified.data, label=label)
    except (SceneFileError, TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, verified.data


def _camera_label(camera_id: str) -> str:
    readable = " ".join(part.capitalize() for part in camera_id.replace("_", "-").split("-"))
    return f"Noesis PCF · {readable or camera_id}"


def _selection_shape(selection: Any) -> dict[str, str]:
    if not isinstance(selection, Mapping) or set(selection) != _SELECTION_FIELDS:
        raise ValueError(
            "PCF reference must contain its exact revision, camera and digests"
        )
    if selection.get("kind") != "scene_prior_pcf":
        raise ValueError("Unsupported path reference kind")
    for field in ("prior_id", "camera_id", "manifest_sha256", "frame_binding_sha256"):
        value = selection.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid PCF reference {field}")
    for field in ("manifest_sha256", "frame_binding_sha256"):
        if _SHA256.fullmatch(selection[field]) is None:
            raise ValueError(f"Invalid PCF reference {field}")
    return {field: str(selection[field]) for field in _SELECTION_FIELDS}


class PcfPathReferences:
    """Bounded read-only resolver for active conditioned PCF references."""

    def __init__(self, catalog_path: Path | None):
        self.catalog_path = (
            absolute_path_without_resolving(catalog_path)
            if catalog_path is not None
            else None
        )

    def _load_catalog(
        self,
    ) -> tuple[dict[str, Any], ScenePriorCatalog, bytes]:
        if self.catalog_path is None:
            raise ValueError("No Scene Prior catalog is configured")
        raw, catalog_bytes = _read_json_file(
            self.catalog_path,
            label="scene-prior catalog",
            max_bytes=MAX_CATALOG_JSON_BYTES,
        )
        revisions = raw.get("revisions")
        bindings = raw.get("camera_bindings")
        if not isinstance(revisions, list) or len(revisions) > MAX_CONFIGURED_REVISIONS:
            raise ValueError(
                f"scene-prior catalog exceeds the {MAX_CONFIGURED_REVISIONS}-revision resolver bound"
            )
        if not isinstance(bindings, list) or len(bindings) > MAX_CONFIGURED_CAMERA_BINDINGS:
            raise ValueError(
                f"scene-prior catalog exceeds the {MAX_CONFIGURED_CAMERA_BINDINGS}-camera resolver bound"
            )
        try:
            catalog = ScenePriorCatalog.model_validate(raw)
        except ValueError as exc:
            raise ValueError(f"scene-prior catalog contract is invalid: {exc}") from exc
        return raw, catalog, catalog_bytes

    @staticmethod
    def _raw_binding(raw_catalog: Mapping[str, Any], camera_id: str) -> dict[str, Any]:
        bindings = raw_catalog.get("camera_bindings")
        if not isinstance(bindings, list):
            raise _InvalidCandidate("catalog camera bindings are unavailable")
        for item in bindings:
            if isinstance(item, dict) and item.get("camera_id") == camera_id:
                return item
        raise _InvalidCandidate("raw catalog camera binding is unavailable")

    @staticmethod
    def _raw_frame_binding(raw_binding: Mapping[str, Any]) -> dict[str, Any]:
        value = raw_binding.get("frame_binding")
        if not isinstance(value, dict):
            raise _InvalidCandidate("active camera binding has no frame binding")
        try:
            ScenePriorFrameBinding.model_validate(value)
        except ValueError as exc:
            raise _InvalidCandidate(f"camera frame binding contract is invalid: {exc}") from exc
        return value

    def _read_manifest(
        self,
        entry: ScenePriorCatalogEntry,
    ) -> tuple[dict[str, Any], ScenePriorRevision, bytes, Path]:
        assert self.catalog_path is not None
        manifest_path = self.catalog_path.parent / entry.manifest_path
        try:
            verified = read_scene_root_file(
                self.catalog_path.parent,
                entry.manifest_path,
                label=f"scene-prior manifest {entry.prior_id}",
                max_bytes=MAX_MANIFEST_JSON_BYTES,
            )
            raw = load_strict_json(
                verified.data,
                label=f"scene-prior manifest {entry.prior_id}",
            )
        except (SceneFileError, TypeError, ValueError) as exc:
            raise _InvalidCandidate(str(exc)) from exc
        if not isinstance(raw, dict):
            raise _InvalidCandidate("scene-prior manifest must be a JSON object")
        try:
            manifest = ScenePriorRevision.model_validate(raw)
        except ValueError as exc:
            raise _InvalidCandidate(f"scene-prior manifest contract is invalid: {exc}") from exc
        return raw, manifest, verified.data, manifest_path

    def _resolve_binding(
        self,
        raw_catalog: Mapping[str, Any],
        binding: Any,
        entry: ScenePriorCatalogEntry,
        scan_id: str,
        catalog_bytes: bytes,
        catalog_site_id: str,
        *,
        source_may_not_match: bool,
    ) -> dict[str, Any]:
        if binding.frame_binding is None:
            raise _NoMatchingSource()
        raw_binding = self._raw_binding(raw_catalog, binding.camera_id)
        raw_frame_binding = self._raw_frame_binding(raw_binding)
        raw_manifest, manifest, manifest_bytes, manifest_path = self._read_manifest(entry)

        source = raw_manifest.get("source")
        source_capture_id = source.get("capture_id") if isinstance(source, dict) else None
        if source_capture_id != scan_id:
            if source_may_not_match:
                raise _NoMatchingSource()
            raise _InvalidCandidate(
                "manifest source capture_id does not match the requested source scan"
            )

        actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if (
            len(manifest_bytes) != entry.manifest_size_bytes
            or actual_manifest_sha256 != entry.manifest_sha256
        ):
            raise _InvalidCandidate(
                "manifest fingerprint or size does not match the catalog entry"
            )

        if manifest.prior_id != entry.prior_id:
            raise _InvalidCandidate("manifest prior_id does not match the catalog revision")
        if manifest.site_id != catalog_site_id:
            raise _InvalidCandidate("manifest site_id does not match the catalog site")
        if manifest.space_id != entry.space_id or binding.space_id != entry.space_id:
            raise _InvalidCandidate("manifest and camera binding space identities disagree")
        if manifest.preview is None:
            raise _InvalidCandidate("scene-prior manifest has no preview camera provenance")
        if manifest.preview.reference_camera_id != binding.camera_id:
            raise _InvalidCandidate("manifest preview camera does not match the catalog binding")

        source_model = manifest.source
        if source_model.source_type != "conditioned_multimodel_room_walk":
            raise _InvalidCandidate("scene-prior source is not conditioned multimodel PCF")
        if "prior_conditioned_consensus_da3_carrier" not in source_model.model:
            raise _InvalidCandidate(
                "scene-prior source model is not prior_conditioned_consensus_da3_carrier"
            )
        if source_model.bundle_schema != "noesis.reference.room_scan_bundle.v1":
            raise _InvalidCandidate("scene-prior source bundle schema is unsupported")
        if manifest.quality.passed is not True:
            raise _InvalidCandidate("scene-prior quality gate did not pass")
        if manifest.quality.alignment_status != "passed":
            raise _InvalidCandidate("scene-prior alignment status is not passed")

        preview = manifest.preview
        frame_binding = binding.frame_binding
        assert frame_binding is not None
        if frame_binding.source_camera_calibration_sha256 != preview.camera_calibration.sha256:
            raise _InvalidCandidate(
                "frame binding calibration digest does not match manifest preview"
            )
        if frame_binding.source_world_alignment_sha256 != manifest.world_to_scene.sha256:
            raise _InvalidCandidate(
                "frame binding world-alignment digest does not match manifest"
            )
        if (
            frame_binding.target_revision_metadata_sha256
            != preview.target_revision_metadata.sha256
        ):
            raise _InvalidCandidate(
                "frame binding target revision metadata digest does not match manifest preview"
            )

        points_artifact = next(
            (artifact for artifact in manifest.artifacts if artifact.role == "points_glb"),
            None,
        )
        if points_artifact is None:
            raise _InvalidCandidate("scene-prior points_glb artifact is not listed")
        if points_artifact.size_bytes > MAX_POINTS_BYTES:
            raise _InvalidCandidate("scene-prior points_glb exceeds the resolver size bound")
        if Path(points_artifact.relative_path).suffix.lower() != ".glb":
            raise _InvalidCandidate("scene-prior points_glb artifact is not a GLB path")
        try:
            points_verified = read_scene_root_file(
                manifest_path.parent,
                points_artifact.relative_path,
                label=f"scene-prior {entry.prior_id} points_glb",
                max_bytes=MAX_POINTS_BYTES,
            )
        except SceneFileError as exc:
            raise _InvalidCandidate(str(exc)) from exc
        if (
            points_verified.size_bytes != points_artifact.size_bytes
            or points_verified.sha256 != points_artifact.sha256
        ):
            raise _InvalidCandidate(
                "points_glb fingerprint or size does not match the manifest"
            )

        assert self.catalog_path is not None
        catalog_fingerprint = _bytes_fingerprint(self.catalog_path, catalog_bytes)
        manifest_fingerprint = _bytes_fingerprint(manifest_path, manifest_bytes)
        points_path = manifest_path.parent / points_artifact.relative_path
        points_fingerprint = _bytes_fingerprint(points_path, points_verified.data)
        selection = {
            "kind": "scene_prior_pcf",
            "prior_id": entry.prior_id,
            "manifest_sha256": entry.manifest_sha256,
            "camera_id": binding.camera_id,
            "frame_binding_sha256": _json_sha256(raw_frame_binding),
        }
        return {
            "selection": selection,
            "label": _camera_label(binding.camera_id),
            "source_scan_id": source_model.capture_id,
            "manifest": deepcopy(raw_manifest),
            "frame_binding": deepcopy(raw_frame_binding),
            "evidence": {
                "catalog": catalog_fingerprint,
                "manifest": manifest_fingerprint,
                "points": points_fingerprint,
            },
            "assets": {
                "manifest": deepcopy(manifest_fingerprint),
                "points": deepcopy(points_fingerprint),
            },
        }

    def _candidates_for_scan(self, scan_id: str) -> tuple[list[dict[str, Any]], list[str]]:
        if not isinstance(scan_id, str) or not scan_id:
            raise ValueError("source scan id is required")
        raw_catalog, catalog, catalog_bytes = self._load_catalog()
        entries = {entry.prior_id: entry for entry in catalog.revisions}
        available: list[dict[str, Any]] = []
        invalid: list[str] = []
        for binding in catalog.camera_bindings:
            if binding.frame_binding is None:
                continue
            entry = entries.get(binding.prior_id)
            if entry is None:
                invalid.append(f"camera {binding.camera_id}: referenced revision is missing")
                continue
            try:
                available.append(
                    self._resolve_binding(
                        raw_catalog,
                        binding,
                        entry,
                        scan_id,
                        catalog_bytes,
                        catalog.site_id,
                        source_may_not_match=True,
                    )
                )
            except _NoMatchingSource:
                continue
            except _InvalidCandidate as exc:
                # A candidate is only relevant when its manifest identifies this
                # scan.  _resolve_binding performs that check before this path.
                invalid.append(f"camera {binding.camera_id}: {exc}")
        return available, invalid

    def for_scan(self, scan_id: str) -> dict[str, Any] | None:
        """Return the public exact selector, or an unavailable status.

        No candidate for the requested source scan is represented by ``None``.
        A matching candidate that fails any contract or fingerprint gate is
        represented explicitly so callers cannot silently fall back to raw
        reconstruction geometry.
        """
        if self.catalog_path is None:
            return None
        try:
            available, invalid = self._candidates_for_scan(scan_id)
        except (OSError, ValueError, TypeError) as exc:
            return {"status": "unavailable", "reason": f"PCF catalog is unavailable: {exc}"}
        if invalid:
            return {
                "status": "unavailable",
                "reason": "; ".join(invalid)[:4000],
            }
        if not available:
            return None
        if len(available) != 1:
            return {
                "status": "unavailable",
                "reason": "PCF source scan is ambiguous: multiple active camera bindings match",
            }
        resolved = available[0]
        return {
            "status": "available",
            "label": resolved["label"],
            "source_scan_id": resolved["source_scan_id"],
            "selection": deepcopy(resolved["selection"]),
        }

    def resolve(self, scan_id: str, selection: dict[str, Any]) -> dict[str, Any]:
        """Resolve and revalidate one exact public selection descriptor."""
        normalized = _selection_shape(selection)
        if self.catalog_path is None:
            raise ValueError("No Scene Prior catalog is configured")
        raw_catalog, catalog, catalog_bytes = self._load_catalog()
        entries = {entry.prior_id: entry for entry in catalog.revisions}
        matches = [
            binding
            for binding in catalog.camera_bindings
            if binding.camera_id == normalized["camera_id"]
            and binding.prior_id == normalized["prior_id"]
            and binding.frame_binding is not None
        ]
        if len(matches) != 1:
            raise ValueError("selection does not identify exactly one active PCF camera binding")
        entry = entries.get(matches[0].prior_id)
        if entry is None:
            raise ValueError("selection references a missing catalog revision")
        try:
            resolved = self._resolve_binding(
                raw_catalog,
                matches[0],
                entry,
                scan_id,
                catalog_bytes,
                catalog.site_id,
                source_may_not_match=False,
            )
        except (_InvalidCandidate, _NoMatchingSource) as exc:
            raise ValueError(str(exc) or "selected PCF reference is unavailable") from exc
        if resolved["selection"] != normalized:
            raise ValueError("selection does not match the exact active PCF reference")
        return resolved


__all__ = ["PcfPathReferences"]
