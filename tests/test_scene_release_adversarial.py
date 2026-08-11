from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import noesis.virtual_twin.releases as release_builder
import noesis_core.scene_files as scene_files
from noesis.virtual_twin.releases import SceneReleaseBuildError, build_scene_release
from noesis.virtual_twin.store import VirtualTwinStore
from noesis_core.scene_files import (
    MAX_AUTHORED_SCENE_BYTES,
    MAX_MTL_BYTES,
    MAX_SCENE_ARTIFACT_BYTES,
    MAX_SCENE_CAMERA_ARTIFACTS,
    MAX_SCENE_MANIFEST_BYTES,
    MAX_SCENE_TEXT_LINE_BYTES,
    SceneFileError,
    materialize_scene_tree,
    read_scene_file,
    read_scene_root_file,
)


def _revision(
    root: Path,
    *,
    revision_id: str = "kitchen-r1",
    camera_id: str = "kitchen",
    artifact_count: int = 1,
) -> Path:
    directory = root / "revisions" / revision_id
    directory.mkdir(parents=True)
    artifacts: dict[str, str] = {}
    for index in range(artifact_count):
        name = f"surface-{index:03d}.glb"
        (directory / name).write_bytes(f"surface:{index}".encode())
        artifacts[f"surface_{index:03d}"] = name
    manifest = {
        "schema": "noesis.virtual_twin.revision.v1",
        "revision_id": revision_id,
        "camera": camera_id,
        "created_ts_us": 1_000_000,
        "calibration_fingerprints": {
            camera_id: hashlib.sha256(camera_id.encode()).hexdigest()
        },
        "model_fingerprints": {"mapanything": {"checkpoint": "model-a"}},
        "artifacts": artifacts,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def _build(root: Path, authored: Path, *, release_id: str = "home-1"):
    return build_scene_release(
        virtual_twins=VirtualTwinStore(root),
        revision_ids=("kitchen-r1",),
        release_id=release_id,
        created_by="owner",
        authored_scene_path=authored,
        cohort_max_delta_us=100_000,
        created_at_us=2_000_000,
    )


def test_verified_scene_reads_reject_links_nonregular_and_component_symlinks(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(target)
    with pytest.raises(SceneFileError, match="without following links"):
        read_scene_file(symlink, label="symlink", max_bytes=64)

    hardlink_source = tmp_path / "hardlink-source.bin"
    hardlink_source.write_bytes(b"hardlink")
    hardlink = tmp_path / "hardlink.bin"
    os.link(hardlink_source, hardlink)
    with pytest.raises(SceneFileError, match="exactly one hard link"):
        read_scene_file(hardlink, label="hardlink", max_bytes=64)

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(SceneFileError, match="regular file"):
        read_scene_file(fifo, label="fifo", max_bytes=64)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "asset.bin").write_bytes(b"outside")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked-directory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SceneFileError, match="symlink|non-directory"):
        read_scene_root_file(
            root,
            "linked-directory/asset.bin",
            label="component",
            max_bytes=64,
        )
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SceneFileError, match="symlink|non-directory"):
        read_scene_root_file(
            linked_root,
            "asset.bin",
            label="linked root",
            max_bytes=64,
        )
    with pytest.raises(SceneFileError, match="normalized relative path"):
        read_scene_root_file(
            root,
            "../outside/asset.bin",
            label="traversal",
            max_bytes=64,
        )


def test_verified_scene_read_detects_replace_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original-bytes")
    original_read = scene_files.os.read
    replaced = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        block = original_read(descriptor, count)
        if block and not replaced:
            replaced = True
            artifact.rename(tmp_path / "artifact.old")
            artifact.write_bytes(b"replacement!!!")
        return block

    monkeypatch.setattr(scene_files.os, "read", racing_read)
    with pytest.raises(SceneFileError, match="changed while it was being read"):
        read_scene_file(artifact, label="racing artifact", max_bytes=64)


def test_builder_rejects_oversized_manifest_and_artifact(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifest-case"
    revision = _revision(manifest_root)
    (revision / "manifest.json").write_bytes(b"{")
    with (revision / "manifest.json").open("r+b") as handle:
        handle.truncate(MAX_SCENE_MANIFEST_BYTES + 1)
    authored = manifest_root / "home.obj"
    authored.write_text("o home\n", encoding="utf-8")
    with pytest.raises(SceneReleaseBuildError, match="byte limit"):
        _build(manifest_root, authored)

    artifact_root = tmp_path / "artifact-case"
    revision = _revision(artifact_root)
    oversized = revision / "surface-000.glb"
    with oversized.open("r+b") as handle:
        handle.truncate(MAX_SCENE_ARTIFACT_BYTES + 1)
    authored = artifact_root / "home.obj"
    authored.write_text("o home\n", encoding="utf-8")
    with pytest.raises(SceneReleaseBuildError, match="byte limit"):
        _build(artifact_root, authored)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda text: text.replace(
                '"revision_id": "kitchen-r1"',
                '"revision_id": "kitchen-r1", "revision_id": "duplicate"',
            ),
            "duplicate key",
        ),
        (lambda text: text.replace("1000000", "NaN"), "non-finite JSON"),
    ],
)
def test_builder_rejects_noncanonical_manifest_json(
    tmp_path: Path,
    mutate,
    expected: str,
) -> None:
    root = tmp_path / expected.replace(" ", "-")
    revision = _revision(root)
    manifest = revision / "manifest.json"
    manifest.write_text(mutate(manifest.read_text(encoding="utf-8")), encoding="utf-8")
    authored = root / "home.obj"
    authored.write_text("o home\n", encoding="utf-8")
    with pytest.raises(SceneReleaseBuildError, match=expected):
        _build(root, authored)


def test_builder_rejects_oversized_or_malformed_obj_and_mtl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized_root = tmp_path / "oversized"
    _revision(oversized_root)
    oversized_obj = oversized_root / "home.obj"
    oversized_obj.write_bytes(b"o")
    with oversized_obj.open("r+b") as handle:
        handle.truncate(MAX_AUTHORED_SCENE_BYTES + 1)
    with pytest.raises(SceneReleaseBuildError, match="byte limit"):
        _build(oversized_root, oversized_obj)

    malformed_root = tmp_path / "malformed"
    _revision(malformed_root)
    malformed_obj = malformed_root / "home.obj"
    malformed_obj.write_bytes(b"o home\n\xff")
    with pytest.raises(SceneReleaseBuildError, match="valid UTF-8"):
        _build(malformed_root, malformed_obj)

    line_root = tmp_path / "line"
    _revision(line_root)
    line_obj = line_root / "home.obj"
    line_mtl = line_root / "home.mtl"
    line_obj.write_text("mtllib home.mtl\n", encoding="utf-8")
    line_mtl.write_text(
        "map_Kd " + ("x" * (MAX_SCENE_TEXT_LINE_BYTES + 1)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SceneReleaseBuildError, match="line 1 exceeds"):
        _build(line_root, line_obj)

    option_root = tmp_path / "option"
    _revision(option_root)
    option_obj = option_root / "home.obj"
    option_mtl = option_root / "home.mtl"
    option_obj.write_text("mtllib home.mtl\n", encoding="utf-8")
    option_mtl.write_text(
        "map_Kd -unknown-option texture.jpg\n",
        encoding="utf-8",
    )
    with pytest.raises(SceneReleaseBuildError, match="unsupported MTL map option"):
        _build(option_root, option_obj)

    mtl_size_root = tmp_path / "mtl-size"
    _revision(mtl_size_root)
    mtl_size_obj = mtl_size_root / "home.obj"
    oversized_mtl = mtl_size_root / "home.mtl"
    mtl_size_obj.write_text("mtllib home.mtl\n", encoding="utf-8")
    oversized_mtl.write_bytes(b"n")
    with oversized_mtl.open("r+b") as handle:
        handle.truncate(MAX_MTL_BYTES + 1)
    with pytest.raises(SceneReleaseBuildError, match="byte limit"):
        _build(mtl_size_root, mtl_size_obj)

    obj_lines_root = tmp_path / "obj-lines"
    _revision(obj_lines_root)
    obj_lines = obj_lines_root / "home.obj"
    obj_lines.write_text("o one\no two\no three\n", encoding="utf-8")
    monkeypatch.setattr(release_builder, "MAX_OBJ_LINES", 2)
    with pytest.raises(SceneReleaseBuildError, match="2-line limit"):
        _build(obj_lines_root, obj_lines)

    mtl_lines_root = tmp_path / "mtl-lines"
    _revision(mtl_lines_root)
    mtl_lines_obj = mtl_lines_root / "home.obj"
    mtl_lines = mtl_lines_root / "home.mtl"
    mtl_lines_obj.write_text("mtllib home.mtl\n", encoding="utf-8")
    mtl_lines.write_text("newmtl one\nnewmtl two\nnewmtl three\n", encoding="utf-8")
    monkeypatch.setattr(release_builder, "MAX_MTL_LINES", 2)
    with pytest.raises(SceneReleaseBuildError, match="2-line limit"):
        _build(mtl_lines_root, mtl_lines_obj)


def test_builder_rejects_dependency_and_artifact_bombs(tmp_path: Path) -> None:
    dependency_root = tmp_path / "dependencies"
    _revision(dependency_root)
    references: list[str] = []
    for index in range(65):
        name = f"material-{index:03d}.mtl"
        (dependency_root / name).write_text("newmtl home\n", encoding="utf-8")
        references.append(name)
    authored = dependency_root / "home.obj"
    authored.write_text(f"mtllib {' '.join(references)}\n", encoding="utf-8")
    with pytest.raises(SceneReleaseBuildError, match="dependency count limit"):
        _build(dependency_root, authored)

    artifact_root = tmp_path / "artifacts"
    _revision(
        artifact_root,
        artifact_count=MAX_SCENE_CAMERA_ARTIFACTS + 1,
    )
    authored = artifact_root / "home.obj"
    authored.write_text("o home\n", encoding="utf-8")
    with pytest.raises(SceneReleaseBuildError, match="artifact limit"):
        _build(artifact_root, authored)


def test_scene_tree_publication_is_atomic_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "release"
    original_rename = scene_files._rename_noreplace

    def fail_publication(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        if destination_name == destination.name:
            raise SceneFileError("injected atomic publication failure")
        original_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(scene_files, "_rename_noreplace", fail_publication)
    with pytest.raises(SceneFileError, match="injected atomic publication failure"):
        materialize_scene_tree(
            destination,
            {"home.obj": b"home", "textures/wall.jpg": b"wall"},
            label="release",
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".release.stage-*"))


def test_existing_scene_tree_rejects_extra_content(tmp_path: Path) -> None:
    destination = tmp_path / "release"
    files = {"home.obj": b"home", "textures/wall.jpg": b"wall"}
    materialize_scene_tree(destination, files, label="release")
    (destination / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(SceneFileError, match="inventory does not match"):
        materialize_scene_tree(destination, files, label="release")
