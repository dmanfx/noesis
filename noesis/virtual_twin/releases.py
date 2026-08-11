from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

from pydantic import ValidationError

from noesis.virtual_twin.store import VirtualTwinStore, VirtualTwinStoreError
from noesis_core.contracts.base import ArtifactFingerprint
from noesis_core.contracts.scene import SceneArtifact, SceneCameraRevision, SceneRelease
from noesis_core.scene_files import (
    MAX_AUTHORED_SCENE_BYTES,
    MAX_MTL_BYTES,
    MAX_MTL_LINES,
    MAX_OBJ_LINES,
    MAX_SCENE_ARTIFACT_BYTES,
    MAX_SCENE_AUTHORED_DEPENDENCIES,
    MAX_SCENE_CAMERAS,
    MAX_SCENE_CAMERA_ARTIFACTS,
    MAX_SCENE_MANIFEST_BYTES,
    MAX_SCENE_RELEASE_BYTES,
    MAX_SCENE_RELEASE_FILES,
    MAX_SCENE_TEXT_LINE_BYTES,
    SceneFileError,
    VerifiedSceneFile,
    absolute_path_without_resolving,
    ensure_scene_directory,
    load_strict_json,
    materialize_scene_file,
    materialize_scene_tree,
    normalized_scene_relative_path,
    read_scene_file,
    read_scene_root_file,
)
from noesis_core.world_service import fingerprint_payload


class SceneReleaseBuildError(RuntimeError):
    pass


RELEASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
ROLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
MTL_TEXTURE_DIRECTIVES = frozenset({"bump", "disp", "decal", "refl", "norm"})
MTL_FIXED_OPTIONS = {
    "-blendu": 1,
    "-blendv": 1,
    "-boost": 1,
    "-bm": 1,
    "-cc": 1,
    "-clamp": 1,
    "-colorspace": 1,
    "-imfchan": 1,
    "-mm": 2,
    "-texres": 1,
    "-type": 1,
}
MTL_VECTOR_OPTIONS = frozenset({"-o", "-s", "-t"})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _bounded_text_lines(
    payload: bytes,
    *,
    label: str,
    max_lines: int,
) -> list[str]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SceneReleaseBuildError(f"{label} is not valid UTF-8") from exc
    lines = text.splitlines()
    if len(lines) > int(max_lines):
        raise SceneReleaseBuildError(
            f"{label} exceeds the {int(max_lines)}-line limit"
        )
    for line_number, line in enumerate(lines, start=1):
        if len(line.encode("utf-8")) > MAX_SCENE_TEXT_LINE_BYTES:
            raise SceneReleaseBuildError(
                f"{label} line {line_number} exceeds the "
                f"{MAX_SCENE_TEXT_LINE_BYTES}-byte limit"
            )
    return lines


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise SceneReleaseBuildError(f"{label} is not a SHA-256 fingerprint")
    return normalized


def _manifest_model_fingerprint(manifest: Mapping[str, Any]) -> str:
    models = manifest.get("model_fingerprints")
    if not isinstance(models, Mapping) or not models:
        raise SceneReleaseBuildError(
            f"revision {manifest.get('revision_id')!r} has no model_fingerprints"
        )
    return _sha256_bytes(_canonical_bytes(models))


def _artifact_inventory(
    virtual_twins: VirtualTwinStore,
    revision_id: str,
    manifest: Mapping[str, Any],
    *,
    max_total_bytes: int,
) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise SceneReleaseBuildError(f"revision {revision_id!r} has no artifact manifest")
    if len(artifacts) > MAX_SCENE_CAMERA_ARTIFACTS:
        raise SceneReleaseBuildError(
            f"revision {revision_id!r} exceeds the "
            f"{MAX_SCENE_CAMERA_ARTIFACTS}-artifact limit"
        )
    inventory: list[dict[str, Any]] = []
    total_bytes = 0
    for role, rel_path in sorted(artifacts.items(), key=lambda row: str(row[0])):
        normalized_role = str(role)
        if not ROLE_RE.fullmatch(normalized_role):
            raise SceneReleaseBuildError(
                f"revision {revision_id!r} has invalid artifact role {role!r}"
            )
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise SceneReleaseBuildError(
                f"revision {revision_id!r} artifact {role!r} has no path"
            )
        try:
            normalized_path = normalized_scene_relative_path(
                rel_path,
                label=f"revision {revision_id!r} artifact {role!r}",
            )
            verified = read_scene_root_file(
                virtual_twins.revisions_root,
                (PurePosixPath(revision_id) / normalized_path).as_posix(),
                label=f"revision {revision_id!r} artifact {role!r}",
                max_bytes=MAX_SCENE_ARTIFACT_BYTES,
            )
        except SceneFileError as exc:
            raise SceneReleaseBuildError(str(exc)) from exc
        total_bytes += verified.size_bytes
        if total_bytes > int(max_total_bytes):
            raise SceneReleaseBuildError(
                f"revision {revision_id!r} exceeds the remaining scene byte budget"
            )
        inventory.append(
            {
                "role": normalized_role,
                "path": normalized_path,
                "sha256": verified.sha256,
                "size_bytes": verified.size_bytes,
            }
        )
    return inventory


def _authored_relative_path(
    raw_path: str,
    *,
    label: str,
    base: PurePosixPath | None = None,
) -> str:
    raw = str(raw_path or "").strip()
    if (
        not raw
        or raw.startswith("/")
        or "\\" in raw
        or "\x00" in raw
        or "//" in raw
        or raw.endswith("/")
    ):
        raise SceneReleaseBuildError(f"{label} must be scene-relative: {raw_path!r}")
    parts = list((base or PurePosixPath()).parts)
    for part in raw.split("/"):
        if part in {"", "."}:
            raise SceneReleaseBuildError(
                f"{label} contains an unsafe path component"
            )
        if part == "..":
            if not parts:
                raise SceneReleaseBuildError(
                    f"{label} escapes the authored scene root"
                )
            parts.pop()
            continue
        try:
            normalized_scene_relative_path(part, label=label)
        except SceneFileError as exc:
            raise SceneReleaseBuildError(str(exc)) from exc
        parts.append(part)
    if not parts:
        raise SceneReleaseBuildError(f"{label} must name a file")
    return PurePosixPath(*parts).as_posix()


def _read_authored_dependency(
    root: Path,
    raw_path: str,
    *,
    label: str,
    max_bytes: int,
    base: PurePosixPath | None = None,
) -> tuple[str, VerifiedSceneFile]:
    relative = _authored_relative_path(raw_path, label=label, base=base)
    try:
        verified = read_scene_root_file(
            root,
            relative,
            label=label,
            max_bytes=max_bytes,
        )
    except SceneFileError as exc:
        raise SceneReleaseBuildError(str(exc)) from exc
    return relative, verified


def _mtl_texture_path(
    tokens: list[str],
    *,
    material_path: str,
    line_number: int,
) -> str:
    index = 0
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index].lower()
        index += 1
        if option in MTL_FIXED_OPTIONS:
            count = MTL_FIXED_OPTIONS[option]
            if len(tokens) - index < count:
                raise SceneReleaseBuildError(
                    f"invalid MTL option {option} at {material_path}:{line_number}"
                )
            index += count
            continue
        if option in MTL_VECTOR_OPTIONS:
            consumed = 0
            while index < len(tokens) and consumed < 3:
                try:
                    float(tokens[index])
                except ValueError:
                    break
                index += 1
                consumed += 1
            if consumed == 0:
                raise SceneReleaseBuildError(
                    f"invalid MTL option {option} at {material_path}:{line_number}"
                )
            continue
        raise SceneReleaseBuildError(
            f"unsupported MTL map option {option} at {material_path}:{line_number}"
        )
    path = " ".join(tokens[index:]).strip()
    if not path:
        raise SceneReleaseBuildError(
            f"empty MTL texture dependency {material_path}:{line_number}"
        )
    return path


def _texture_reference(
    root: Path,
    tokens: list[str],
    *,
    material_path: str,
    line_number: int,
) -> tuple[str, VerifiedSceneFile]:
    raw_path = _mtl_texture_path(
        tokens,
        material_path=material_path,
        line_number=line_number,
    )
    return _read_authored_dependency(
        root,
        raw_path,
        label=f"MTL texture {material_path}:{line_number}",
        max_bytes=MAX_SCENE_ARTIFACT_BYTES,
        base=PurePosixPath(material_path).parent,
    )


def _authored_dependency_inventory(
    scene_path: Path,
    scene_bytes: bytes,
    *,
    max_total_bytes: int,
) -> list[dict[str, Any]]:
    if scene_path.suffix.lower() != ".obj":
        return []
    root = absolute_path_without_resolving(scene_path.parent)
    obj_lines = _bounded_text_lines(
        scene_bytes,
        label="authored OBJ",
        max_lines=MAX_OBJ_LINES,
    )
    material_paths: dict[str, VerifiedSceneFile] = {}
    dependency_bytes = 0
    for line_number, raw_line in enumerate(obj_lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        keyword = parts[0]
        remainder = parts[1] if len(parts) == 2 else ""
        if keyword.lower() != "mtllib":
            continue
        try:
            references = shlex.split(remainder, comments=True, posix=True)
        except ValueError as exc:
            raise SceneReleaseBuildError(
                f"invalid OBJ mtllib directive at line {line_number}: {exc}"
            ) from exc
        if not references:
            raise SceneReleaseBuildError(
                f"empty OBJ mtllib directive at line {line_number}"
            )
        for reference in references:
            relative, material = _read_authored_dependency(
                root,
                reference,
                label=f"OBJ material library line {line_number}",
                max_bytes=MAX_MTL_BYTES,
            )
            existing = material_paths.get(relative)
            if existing is not None and existing.sha256 != material.sha256:
                raise SceneReleaseBuildError(
                    f"authored material {relative!r} changed during dependency discovery"
                )
            if existing is None:
                dependency_bytes += material.size_bytes
                if dependency_bytes > int(max_total_bytes):
                    raise SceneReleaseBuildError(
                        "authored dependencies exceed the remaining scene byte budget"
                    )
                material_paths[relative] = material
            if len(material_paths) > MAX_SCENE_AUTHORED_DEPENDENCIES:
                raise SceneReleaseBuildError(
                    "authored OBJ exceeds the material/dependency count limit"
                )

    texture_paths: dict[str, VerifiedSceneFile] = {}
    for relative, material_file in sorted(material_paths.items()):
        lines = _bounded_text_lines(
            material_file.data,
            label=f"authored MTL {relative}",
            max_lines=MAX_MTL_LINES,
        )
        for line_number, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=1)
            keyword = parts[0]
            remainder = parts[1] if len(parts) == 2 else ""
            normalized_keyword = keyword.lower()
            if not (
                normalized_keyword.startswith("map_")
                or normalized_keyword in MTL_TEXTURE_DIRECTIVES
            ):
                continue
            try:
                tokens = shlex.split(remainder, comments=True, posix=True)
            except ValueError as exc:
                raise SceneReleaseBuildError(
                    f"invalid MTL map directive {relative}:{line_number}: {exc}"
                ) from exc
            texture_relative, texture = _texture_reference(
                root,
                tokens,
                material_path=relative,
                line_number=line_number,
            )
            existing_texture = texture_paths.get(texture_relative)
            if (
                existing_texture is not None
                and existing_texture.sha256 != texture.sha256
            ):
                raise SceneReleaseBuildError(
                    f"authored texture {texture_relative!r} changed during dependency discovery"
                )
            if existing_texture is None:
                dependency_bytes += texture.size_bytes
                if dependency_bytes > int(max_total_bytes):
                    raise SceneReleaseBuildError(
                        "authored dependencies exceed the remaining scene byte budget"
                    )
                texture_paths[texture_relative] = texture
            if (
                len(material_paths) + len(texture_paths)
                > MAX_SCENE_AUTHORED_DEPENDENCIES
            ):
                raise SceneReleaseBuildError(
                    "authored OBJ/MTL dependency count exceeds the "
                    f"{MAX_SCENE_AUTHORED_DEPENDENCIES}-file limit"
                )

    inventory: list[dict[str, Any]] = []
    for index, (relative, verified) in enumerate(sorted(material_paths.items())):
        inventory.append(
            {
                "role": f"material_library_{index:03d}",
                "path": relative,
                "payload": verified.data,
                "sha256": verified.sha256,
                "size_bytes": verified.size_bytes,
            }
        )
    for index, (relative, verified) in enumerate(sorted(texture_paths.items())):
        inventory.append(
            {
                "role": f"material_texture_{index:03d}",
                "path": relative,
                "payload": verified.data,
                "sha256": verified.sha256,
                "size_bytes": verified.size_bytes,
            }
        )
    return inventory


def build_scene_release(
    *,
    virtual_twins: VirtualTwinStore,
    revision_ids: Iterable[str],
    release_id: str,
    created_by: str,
    authored_scene_path: str | Path,
    cohort_max_delta_us: int,
    created_at_us: int | None = None,
) -> tuple[SceneRelease, dict[str, Any]]:
    normalized_release_id = str(release_id).strip()
    if not RELEASE_ID_RE.fullmatch(normalized_release_id):
        raise SceneReleaseBuildError(f"invalid scene release id: {release_id!r}")
    normalized_creator = str(created_by).strip()
    if not normalized_creator:
        raise SceneReleaseBuildError("created_by is required")
    selected = tuple(str(item).strip() for item in revision_ids if str(item).strip())
    if not selected:
        raise SceneReleaseBuildError("at least one explicit revision is required")
    if len(selected) != len(set(selected)):
        raise SceneReleaseBuildError("revision selection contains duplicates")
    if len(selected) > MAX_SCENE_CAMERAS:
        raise SceneReleaseBuildError(
            f"revision selection exceeds the {MAX_SCENE_CAMERAS}-camera limit"
        )
    if int(cohort_max_delta_us) < 0:
        raise SceneReleaseBuildError("cohort_max_delta_us cannot be negative")
    scene_path = absolute_path_without_resolving(authored_scene_path)
    try:
        scene_name = normalized_scene_relative_path(
            scene_path.name,
            label="authored scene filename",
        )
        scene_file = read_scene_file(
            scene_path,
            label="authored scene",
            max_bytes=MAX_AUTHORED_SCENE_BYTES,
        )
    except SceneFileError as exc:
        raise SceneReleaseBuildError(str(exc)) from exc

    rows: list[dict[str, Any]] = []
    camera_ids: set[str] = set()
    camera_bytes = 0
    camera_artifact_count = 0
    for revision_id in selected:
        try:
            virtual_twins.revision_dir(revision_id)
            manifest_file = read_scene_root_file(
                virtual_twins.revisions_root,
                (PurePosixPath(revision_id) / "manifest.json").as_posix(),
                label=f"revision manifest {revision_id!r}",
                max_bytes=MAX_SCENE_MANIFEST_BYTES,
            )
        except (SceneFileError, VirtualTwinStoreError) as exc:
            raise SceneReleaseBuildError(
                f"unable to read revision manifest {revision_id!r}: {exc}"
            ) from exc
        try:
            manifest = load_strict_json(
                manifest_file.data,
                label=f"revision manifest {revision_id!r}",
            )
        except SceneFileError as exc:
            raise SceneReleaseBuildError(
                f"unable to parse revision manifest {revision_id!r}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise SceneReleaseBuildError(f"revision {revision_id!r} manifest is not an object")
        if str(manifest.get("revision_id") or "") != revision_id:
            raise SceneReleaseBuildError(f"revision {revision_id!r} manifest identity mismatch")
        camera_id = str(manifest.get("camera") or "").strip()
        if not camera_id:
            raise SceneReleaseBuildError(f"revision {revision_id!r} has no camera")
        if camera_id in camera_ids:
            raise SceneReleaseBuildError(f"multiple revisions selected for camera {camera_id!r}")
        camera_ids.add(camera_id)
        try:
            captured_at_us = int(manifest.get("created_ts_us"))
        except Exception as exc:
            raise SceneReleaseBuildError(
                f"revision {revision_id!r} has no valid created_ts_us"
            ) from exc
        if captured_at_us <= 0:
            raise SceneReleaseBuildError(
                f"revision {revision_id!r} has no valid created_ts_us"
            )
        calibration_map = manifest.get("calibration_fingerprints")
        if not isinstance(calibration_map, Mapping):
            raise SceneReleaseBuildError(
                f"revision {revision_id!r} has no calibration_fingerprints"
            )
        calibration_sha = _require_sha256(
            calibration_map.get(camera_id),
            f"revision {revision_id!r} camera calibration",
        )
        declared_artifacts = manifest.get("artifacts")
        if isinstance(declared_artifacts, Mapping):
            prospective_count = camera_artifact_count + len(declared_artifacts)
            if 2 + len(selected) + prospective_count > MAX_SCENE_RELEASE_FILES:
                raise SceneReleaseBuildError(
                    f"selected revisions exceed the {MAX_SCENE_RELEASE_FILES}-file limit"
                )
        inventory = _artifact_inventory(
            virtual_twins,
            revision_id,
            manifest,
            max_total_bytes=(
                MAX_SCENE_RELEASE_BYTES - scene_file.size_bytes - camera_bytes
            ),
        )
        camera_bytes += sum(int(item["size_bytes"]) for item in inventory)
        camera_artifact_count += len(inventory)
        rows.append(
            {
                "camera_id": camera_id,
                "revision_id": revision_id,
                "captured_at_us": captured_at_us,
                "calibration_sha256": calibration_sha,
                "model_sha256": _manifest_model_fingerprint(manifest),
                "manifest_sha256": manifest_file.sha256,
                "artifacts": inventory,
            }
        )
        declared_files = 2 + len(selected) + sum(
            len(row["artifacts"]) for row in rows
        )
        if declared_files > MAX_SCENE_RELEASE_FILES:
            raise SceneReleaseBuildError(
                f"selected revisions exceed the {MAX_SCENE_RELEASE_FILES}-file limit"
            )
        if camera_bytes > MAX_SCENE_RELEASE_BYTES:
            raise SceneReleaseBuildError(
                f"selected revisions exceed the {MAX_SCENE_RELEASE_BYTES}-byte limit"
            )

    captured = [int(row["captured_at_us"]) for row in rows]
    actual_delta = max(captured) - min(captured)
    if actual_delta > int(cohort_max_delta_us):
        raise SceneReleaseBuildError(
            f"selected revisions exceed cohort: delta_us={actual_delta} limit={cohort_max_delta_us}"
        )
    calibration_bundle = fingerprint_payload(
        "scene_calibration_bundle",
        {row["camera_id"]: row["calibration_sha256"] for row in rows},
        version="v1",
    )
    model_bundle = fingerprint_payload(
        "scene_model_bundle",
        {row["camera_id"]: row["model_sha256"] for row in rows},
        version="v1",
    )
    config = fingerprint_payload(
        "scene_release_config",
        {
            "release_version": 1,
            "revision_ids": list(selected),
            "cohort_max_delta_us": int(cohort_max_delta_us),
        },
        version="v1",
    )
    authored_scene = ArtifactFingerprint(
        role="authored_home_scene",
        sha256=scene_file.sha256,
        version=scene_name,
    )
    authored_dependencies = _authored_dependency_inventory(
        scene_path,
        scene_file.data,
        max_total_bytes=(
            MAX_SCENE_RELEASE_BYTES - scene_file.size_bytes - camera_bytes
        ),
    )
    if (
        2
        + len(selected)
        + sum(len(row["artifacts"]) for row in rows)
        + len(authored_dependencies)
        > MAX_SCENE_RELEASE_FILES
    ):
        raise SceneReleaseBuildError(
            f"scene release exceeds the {MAX_SCENE_RELEASE_FILES}-file limit"
        )
    if (
        scene_file.size_bytes
        + sum(int(item["size_bytes"]) for item in authored_dependencies)
        + sum(
            int(artifact["size_bytes"])
            for row in rows
            for artifact in row["artifacts"]
        )
        > MAX_SCENE_RELEASE_BYTES
    ):
        raise SceneReleaseBuildError(
            f"scene release exceeds the {MAX_SCENE_RELEASE_BYTES}-byte limit"
        )
    release_directory = PurePosixPath("releases", normalized_release_id)
    release_authored_scene_path = (release_directory / scene_name).as_posix()
    dependency_contract_rows = [
        {
            "role": str(item["role"]),
            "path": (release_directory / str(item["path"])).as_posix(),
            "sha256": str(item["sha256"]),
            "size_bytes": int(item["size_bytes"]),
        }
        for item in authored_dependencies
    ]
    validation_report: dict[str, Any] = {
        "schema": "noesis.scene.validation.v1",
        "release_id": normalized_release_id,
        "cohort": {
            "max_delta_us": int(cohort_max_delta_us),
            "actual_delta_us": actual_delta,
            "captured_start_us": min(captured),
            "captured_end_us": max(captured),
        },
        "calibration_bundle_sha256": calibration_bundle.sha256,
        "model_bundle_sha256": model_bundle.sha256,
        "authored_scene": {
            "path": release_authored_scene_path,
            "sha256": authored_scene.sha256,
            "size_bytes": scene_file.size_bytes,
            "dependencies": dependency_contract_rows,
        },
        "cameras": rows,
        "checks": {
            "unique_cameras": True,
            "manifest_identity": True,
            "all_declared_artifacts_present_and_hashed": True,
            "authored_scene_dependencies_present_and_hashed": True,
            "cohort_within_limit": True,
        },
    }
    validation_bytes = _canonical_bytes(validation_report)
    validation_sha = _sha256_bytes(validation_bytes)
    validation_report_path = PurePosixPath(
        "releases",
        normalized_release_id,
        f"validation-{validation_sha}.json",
    ).as_posix()
    timestamp = int(created_at_us if created_at_us is not None else time.time_ns() // 1_000)
    try:
        release = SceneRelease(
            contract="noesis.scene.release",
            contract_version=1,
            release_id=normalized_release_id,
            release_version=1,
            created_at_us=max(timestamp, max(captured)),
            created_by=normalized_creator,
            calibration=calibration_bundle,
            model=model_bundle,
            config=config,
            authored_scene=authored_scene,
            authored_scene_path=release_authored_scene_path,
            authored_scene_size_bytes=scene_file.size_bytes,
            authored_scene_dependencies=tuple(
                SceneArtifact(
                    role=str(item["role"]),
                    relative_path=str(item["path"]),
                    sha256=str(item["sha256"]),
                    size_bytes=int(item["size_bytes"]),
                )
                for item in dependency_contract_rows
            ),
            cohort_max_delta_us=int(cohort_max_delta_us),
            cameras=tuple(
                SceneCameraRevision(
                    camera_id=str(row["camera_id"]),
                    revision_id=str(row["revision_id"]),
                    captured_at_us=int(row["captured_at_us"]),
                    calibration_bundle_sha256=calibration_bundle.sha256,
                    calibration_sha256=str(row["calibration_sha256"]),
                    model_bundle_sha256=model_bundle.sha256,
                    model_sha256=str(row["model_sha256"]),
                    manifest_sha256=str(row["manifest_sha256"]),
                    artifact_path=str(row["revision_id"]),
                    artifacts=tuple(
                        SceneArtifact(
                            role=str(artifact["role"]),
                            relative_path=str(artifact["path"]),
                            sha256=str(artifact["sha256"]),
                            size_bytes=int(artifact["size_bytes"]),
                        )
                        for artifact in row["artifacts"]
                    ),
                )
                for row in sorted(rows, key=lambda item: str(item["camera_id"]))
            ),
            validation_report_sha256=validation_sha,
            validation_report_path=validation_report_path,
            validation_report_size_bytes=len(validation_bytes),
        )
    except ValidationError as exc:
        raise SceneReleaseBuildError(f"scene release contract is invalid: {exc}") from exc

    bundle_files: dict[str, bytes] = {
        scene_name: scene_file.data,
        PurePosixPath(validation_report_path).name: validation_bytes,
    }
    for dependency in authored_dependencies:
        relative = str(dependency["path"])
        if relative in bundle_files:
            raise SceneReleaseBuildError(
                f"authored scene bundle path is duplicated: {relative}"
            )
        bundle_files[relative] = bytes(dependency["payload"])
    try:
        materialize_scene_tree(
            absolute_path_without_resolving(virtual_twins.root)
            / "releases"
            / normalized_release_id,
            bundle_files,
            label=f"scene release {normalized_release_id}",
        )
    except SceneFileError as exc:
        raise SceneReleaseBuildError(str(exc)) from exc
    return release, validation_report


def write_scene_release_bundle(
    output_directory: str | Path,
    release: SceneRelease,
    validation_report: Mapping[str, Any],
) -> tuple[Path, Path]:
    try:
        output = ensure_scene_directory(
            output_directory,
            label="scene release output",
        )
    except SceneFileError as exc:
        raise SceneReleaseBuildError(str(exc)) from exc
    release_path = output / f"{release.release_id}.json"
    validation_path = output / f"{release.release_id}.validation.json"
    validation_bytes = _canonical_bytes(validation_report)
    if _sha256_bytes(validation_bytes) != release.validation_report_sha256:
        raise SceneReleaseBuildError(
            "validation report bytes do not match the release fingerprint"
        )
    release_bytes = (release.model_dump_json(indent=2) + "\n").encode("utf-8")
    try:
        # The release JSON is the pair's commit marker, so publish validation
        # first and never overwrite either path.
        materialize_scene_file(
            validation_path,
            validation_bytes,
            label="scene release validation output",
            expected_sha256=release.validation_report_sha256,
        )
        materialize_scene_file(
            release_path,
            release_bytes,
            label="scene release output",
            expected_sha256=_sha256_bytes(release_bytes),
        )
    except SceneFileError as exc:
        raise SceneReleaseBuildError(str(exc)) from exc
    return release_path, validation_path


__all__ = [
    "SceneReleaseBuildError",
    "build_scene_release",
    "write_scene_release_bundle",
]
