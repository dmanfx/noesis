from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from noesis.virtual_twin.releases import (
    SceneReleaseBuildError,
    build_scene_release,
    write_scene_release_bundle,
)
from noesis.virtual_twin.store import VirtualTwinStore


def _revision(root: Path, revision_id: str, camera_id: str, created_at_us: int) -> None:
    directory = root / "revisions" / revision_id
    directory.mkdir(parents=True)
    (directory / "surface.glb").write_bytes(f"surface:{camera_id}".encode())
    manifest = {
        "schema": "noesis.virtual_twin.revision.v1",
        "revision_id": revision_id,
        "camera": camera_id,
        "created_ts_us": created_at_us,
        "calibration_fingerprints": {camera_id: hashlib.sha256(camera_id.encode()).hexdigest()},
        "model_fingerprints": {"mapanything": {"checkpoint": "model-a"}},
        "artifacts": {"surface": "surface.glb"},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_builds_explicit_multi_camera_release_and_hashed_validation(tmp_path: Path) -> None:
    _revision(tmp_path, "kitchen-r1", "kitchen", 1_000_000)
    _revision(tmp_path, "hall-r1", "hall", 1_050_000)
    authored = tmp_path / "home.obj"
    authored.write_text("o home\n", encoding="utf-8")

    release, report = build_scene_release(
        virtual_twins=VirtualTwinStore(tmp_path),
        revision_ids=("kitchen-r1", "hall-r1"),
        release_id="home-1",
        created_by="owner",
        authored_scene_path=authored,
        cohort_max_delta_us=100_000,
        created_at_us=2_000_000,
    )

    assert [camera.camera_id for camera in release.cameras] == ["hall", "kitchen"]
    assert len({camera.calibration_sha256 for camera in release.cameras}) == 2
    assert all(
        camera.calibration_bundle_sha256 == release.calibration.sha256
        for camera in release.cameras
    )
    assert report["checks"]["all_declared_artifacts_present_and_hashed"] is True
    output_release, output_validation = write_scene_release_bundle(
        tmp_path / "releases",
        release,
        report,
    )
    assert json.loads(output_release.read_text())["release_id"] == "home-1"
    assert hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() == release.validation_report_sha256
    assert (
        hashlib.sha256(output_validation.read_bytes()).hexdigest()
        == release.validation_report_sha256
    )
    assert output_validation.is_file()


def test_builds_immutable_obj_material_and_texture_dependency_bundle(
    tmp_path: Path,
) -> None:
    _revision(tmp_path, "kitchen-r1", "kitchen", 1_000_000)
    authored = tmp_path / "authored" / "home.obj"
    material = tmp_path / "authored" / "materials" / "home.mtl"
    texture = tmp_path / "authored" / "textures" / "wall color.jpg"
    authored.parent.mkdir()
    material.parent.mkdir()
    texture.parent.mkdir()
    authored.write_text("mtllib materials/home.mtl\no home\n", encoding="utf-8")
    material.write_text(
        "newmtl wall\nmap_Pr ../textures/wall color.jpg # PBR extension\n",
        encoding="utf-8",
    )
    texture.write_bytes(b"jpeg texture")

    release, report = build_scene_release(
        virtual_twins=VirtualTwinStore(tmp_path),
        revision_ids=("kitchen-r1",),
        release_id="home-materials-1",
        created_by="owner",
        authored_scene_path=authored,
        cohort_max_delta_us=100_000,
        created_at_us=2_000_000,
    )

    dependencies = {item.role: item for item in release.authored_scene_dependencies}
    assert set(dependencies) == {"material_library_000", "material_texture_000"}
    assert dependencies["material_library_000"].relative_path.endswith(
        "/materials/home.mtl"
    )
    assert dependencies["material_texture_000"].relative_path.endswith(
        "/textures/wall color.jpg"
    )
    for dependency in dependencies.values():
        materialized = tmp_path / dependency.relative_path
        assert materialized.is_file()
        assert hashlib.sha256(materialized.read_bytes()).hexdigest() == dependency.sha256
        assert materialized.stat().st_size == dependency.size_bytes
    assert report["authored_scene"]["dependencies"] == [
        {
            "role": item.role,
            "path": item.relative_path,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in release.authored_scene_dependencies
    ]


@pytest.mark.parametrize(
    ("obj_text", "expected"),
    [
        ("mtllib missing.mtl\no home\n", "is missing"),
        ("mtllib ../outside.mtl\no home\n", "escapes the authored scene root"),
    ],
)
def test_rejects_missing_or_escaping_authored_dependencies(
    tmp_path: Path,
    obj_text: str,
    expected: str,
) -> None:
    _revision(tmp_path, "kitchen-r1", "kitchen", 1_000_000)
    authored_root = tmp_path / "authored"
    authored_root.mkdir()
    authored = authored_root / "home.obj"
    authored.write_text(obj_text, encoding="utf-8")
    (tmp_path / "outside.mtl").write_text("newmtl outside\n", encoding="utf-8")

    with pytest.raises(SceneReleaseBuildError, match=expected):
        build_scene_release(
            virtual_twins=VirtualTwinStore(tmp_path),
            revision_ids=("kitchen-r1",),
            release_id="home-bad-dependency",
            created_by="owner",
            authored_scene_path=authored,
            cohort_max_delta_us=100_000,
        )


def test_bundle_writer_rejects_validation_content_drift(tmp_path: Path) -> None:
    _revision(tmp_path, "kitchen-r1", "kitchen", 1_000_000)
    authored = tmp_path / "home.obj"
    authored.write_text("o home\n", encoding="utf-8")
    release, report = build_scene_release(
        virtual_twins=VirtualTwinStore(tmp_path),
        revision_ids=("kitchen-r1",),
        release_id="home-1",
        created_by="owner",
        authored_scene_path=authored,
        cohort_max_delta_us=100_000,
        created_at_us=2_000_000,
    )
    drifted = dict(report)
    drifted["release_id"] = "different"

    with pytest.raises(SceneReleaseBuildError, match="do not match"):
        write_scene_release_bundle(tmp_path / "output", release, drifted)


def test_bundle_writer_never_overwrites_an_existing_release_id(tmp_path: Path) -> None:
    _revision(tmp_path, "kitchen-r1", "kitchen", 1_000_000)
    authored = tmp_path / "home.obj"
    authored.write_text("o home\n", encoding="utf-8")
    release, report = build_scene_release(
        virtual_twins=VirtualTwinStore(tmp_path),
        revision_ids=("kitchen-r1",),
        release_id="home-1",
        created_by="owner",
        authored_scene_path=authored,
        cohort_max_delta_us=100_000,
        created_at_us=2_000_000,
    )
    output = tmp_path / "output"
    write_scene_release_bundle(output, release, report)

    changed = release.model_copy(update={"created_by": "different-owner"})
    with pytest.raises(SceneReleaseBuildError, match="different content"):
        write_scene_release_bundle(output, changed, report)


def test_rejects_implicit_or_incoherent_revision_selection(tmp_path: Path) -> None:
    _revision(tmp_path, "kitchen-r1", "kitchen", 1_000_000)
    _revision(tmp_path, "hall-r1", "hall", 2_000_000)
    authored = tmp_path / "home.obj"
    authored.write_text("o home\n", encoding="utf-8")

    with pytest.raises(SceneReleaseBuildError, match="at least one explicit"):
        build_scene_release(
            virtual_twins=VirtualTwinStore(tmp_path),
            revision_ids=(),
            release_id="home-1",
            created_by="owner",
            authored_scene_path=authored,
            cohort_max_delta_us=10,
        )
    with pytest.raises(SceneReleaseBuildError, match="exceed cohort"):
        build_scene_release(
            virtual_twins=VirtualTwinStore(tmp_path),
            revision_ids=("kitchen-r1", "hall-r1"),
            release_id="home-1",
            created_by="owner",
            authored_scene_path=authored,
            cohort_max_delta_us=10,
        )
