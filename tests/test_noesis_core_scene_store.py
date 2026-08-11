from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from noesis_core.contracts.base import ArtifactFingerprint
from noesis_core.contracts.scene import SceneArtifact, SceneCameraRevision, SceneRelease
from noesis_core.scene_store import SceneReleaseConflict, SceneReleaseStore, SceneReleaseStoreError


SHA_A = "a" * 64
SHA_B = "b" * 64


def _fingerprint(role: str, sha: str = SHA_A) -> ArtifactFingerprint:
    return ArtifactFingerprint(role=role, sha256=sha, version="v1")


def _release(
    release_id: str,
    artifact_path: str,
    *,
    captured_at_us: int = 100,
    manifest_sha256: str = SHA_B,
    artifact_sha256: str = SHA_A,
    artifact_size_bytes: int = 7,
) -> SceneRelease:
    return SceneRelease(
        contract="noesis.scene.release",
        contract_version=1,
        release_id=release_id,
        release_version=1,
        created_at_us=captured_at_us + 100,
        created_by="owner",
        calibration=_fingerprint("calibration"),
        model=_fingerprint("model"),
        config=_fingerprint("config"),
        authored_scene=_fingerprint("authored_scene"),
        authored_scene_path=f"releases/{release_id}/home.obj",
        authored_scene_size_bytes=4,
        authored_scene_dependencies=(),
        cohort_max_delta_us=100,
        cameras=(
            SceneCameraRevision(
                camera_id="kitchen",
                revision_id=f"{release_id}-camera",
                captured_at_us=captured_at_us,
                calibration_bundle_sha256=SHA_A,
                calibration_sha256=SHA_A,
                model_bundle_sha256=SHA_A,
                model_sha256=SHA_A,
                manifest_sha256=manifest_sha256,
                artifact_path=artifact_path,
                artifacts=(
                    SceneArtifact(
                        role="surface",
                        relative_path="surface.glb",
                        sha256=artifact_sha256,
                        size_bytes=artifact_size_bytes,
                    ),
                ),
            ),
        ),
        validation_report_sha256=SHA_B,
        validation_report_path=f"releases/{release_id}/validation.json",
        validation_report_size_bytes=2,
    )


def test_register_promote_restart_and_rollback_are_atomic(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / "kitchen" / "one").mkdir(parents=True)
    (artifacts / "kitchen" / "two").mkdir(parents=True)
    one_manifest = artifacts / "kitchen" / "one" / "manifest.json"
    two_manifest = artifacts / "kitchen" / "two" / "manifest.json"
    one_surface = artifacts / "kitchen" / "one" / "surface.glb"
    two_surface = artifacts / "kitchen" / "two" / "surface.glb"
    one_surface.write_bytes(b"surface-one")
    two_surface.write_bytes(b"surface-two")
    one_manifest.write_text(
        '{"artifacts":{"surface":"surface.glb"},"camera":"kitchen","revision_id":"one-camera"}',
        encoding="utf-8",
    )
    two_manifest.write_text(
        '{"artifacts":{"surface":"surface.glb"},"camera":"kitchen","revision_id":"two-camera"}',
        encoding="utf-8",
    )
    db = tmp_path / "state" / "scene.sqlite3"
    store = SceneReleaseStore(db, artifact_root=artifacts)
    one = _release(
        "one",
        "kitchen/one",
        manifest_sha256=hashlib.sha256(one_manifest.read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256(one_surface.read_bytes()).hexdigest(),
        artifact_size_bytes=one_surface.stat().st_size,
    )
    two = _release(
        "two",
        "kitchen/two",
        captured_at_us=200,
        manifest_sha256=hashlib.sha256(two_manifest.read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256(two_surface.read_bytes()).hexdigest(),
        artifact_size_bytes=two_surface.stat().st_size,
    )
    store.register(one)
    store.register(two)
    first = store.promote(
        "one",
        actor_id="owner-1",
        occurred_at_us=1_000,
        expected_current_release_id=None,
    )
    assert first.previous_release_id is None
    assert store.current() == one

    restarted = SceneReleaseStore(db, artifact_root=artifacts)
    second = restarted.promote(
        "two",
        actor_id="owner-1",
        occurred_at_us=2_000,
        expected_current_release_id="one",
    )
    assert second.previous_release_id == "one"
    rollback = restarted.rollback(
        "one",
        actor_id="owner-1",
        occurred_at_us=3_000,
        expected_current_release_id="two",
    )
    assert rollback.event == "rollback"
    assert restarted.current() == one
    assert [release.release_id for release in restarted.list_releases()] == ["two", "one"]
    assert [entry.event for entry in restarted.history()] == ["promote", "promote", "rollback"]
    assert db.stat().st_mode & 0o777 == 0o600
    assert db.parent.stat().st_mode & 0o777 == 0o700


def test_release_ids_are_immutable_and_promotion_is_compare_and_swap(tmp_path: Path) -> None:
    store = SceneReleaseStore(tmp_path / "scene.sqlite3")
    release = _release("one", "kitchen/one")
    first_hash = store.register(release)
    assert store.register(release) == first_hash
    different = release.model_copy(update={"created_by": "someone-else"})
    with pytest.raises(SceneReleaseConflict, match="different immutable content"):
        store.register(different)
    with pytest.raises(SceneReleaseConflict, match="compare-and-swap"):
        store.promote(
            "one",
            actor_id="owner",
            occurred_at_us=1_000,
            expected_current_release_id="not-current",
        )


def test_scene_artifact_paths_are_safe_and_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="relative path"):
        _release("bad", "../escape")
    store = SceneReleaseStore(tmp_path / "scene.sqlite3", artifact_root=tmp_path / "artifacts")
    with pytest.raises(SceneReleaseStoreError, match="directory is missing|is missing"):
        store.register(_release("missing", "kitchen/missing"))


def test_promotion_verifies_manifest_content_and_revision_identity(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    revision_dir = artifacts / "kitchen" / "one"
    revision_dir.mkdir(parents=True)
    manifest = revision_dir / "manifest.json"
    manifest.write_text(
        '{"artifacts":{"surface":"surface.glb"},"camera":"kitchen","revision_id":"wrong-revision"}',
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    store = SceneReleaseStore(tmp_path / "scene.sqlite3", artifact_root=artifacts)
    with pytest.raises(SceneReleaseStoreError, match="revision mismatch"):
        store.register(
            _release("one", "kitchen/one", manifest_sha256=manifest_sha)
        )

    manifest.write_text(
        '{"artifacts":{"surface":"surface.glb"},"camera":"kitchen","revision_id":"one-camera"}',
        encoding="utf-8",
    )
    surface = revision_dir / "surface.glb"
    surface.write_bytes(b"surface")
    store.register(
        _release(
            "one",
            "kitchen/one",
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            artifact_sha256=hashlib.sha256(surface.read_bytes()).hexdigest(),
            artifact_size_bytes=surface.stat().st_size,
        )
    )
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(SceneReleaseStoreError, match="fingerprint mismatch"):
        store.promote(
            "one",
            actor_id="owner",
            occurred_at_us=1_001,
            expected_current_release_id=None,
        )


def test_promotion_and_reads_reject_mutated_render_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "virtual-twin"
    revisions = root / "revisions"
    revision_dir = revisions / "one-camera"
    revision_dir.mkdir(parents=True)
    surface = revision_dir / "surface.glb"
    surface.write_bytes(b"immutable-surface")
    manifest = revision_dir / "manifest.json"
    manifest.write_text(
        '{"artifacts":{"surface":"surface.glb"},"camera":"kitchen","revision_id":"one-camera"}',
        encoding="utf-8",
    )
    bundle = root / "releases" / "one"
    bundle.mkdir(parents=True)
    authored = bundle / "home.obj"
    validation = bundle / "validation.json"
    authored.write_bytes(b"home")
    validation.write_bytes(b"{}")
    release = _release(
        "one",
        "one-camera",
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256(surface.read_bytes()).hexdigest(),
        artifact_size_bytes=surface.stat().st_size,
    ).model_copy(
        update={
            "authored_scene": _fingerprint(
                "authored_scene", hashlib.sha256(authored.read_bytes()).hexdigest()
            ),
            "validation_report_sha256": hashlib.sha256(
                validation.read_bytes()
            ).hexdigest(),
        }
    )
    store = SceneReleaseStore(
        tmp_path / "scene.sqlite3",
        artifact_root=revisions,
        bundle_root=root,
    )
    store.register(release)
    store.promote(
        "one",
        actor_id="owner",
        occurred_at_us=1_000,
        expected_current_release_id=None,
    )

    surface.write_bytes(b"mutated-surface")
    with pytest.raises(SceneReleaseStoreError, match="size mismatch|fingerprint mismatch"):
        store.validate_artifacts(release)


def test_promotion_and_reads_reject_mutated_authored_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "virtual-twin"
    bundle = root / "releases" / "one"
    bundle.mkdir(parents=True)
    authored = bundle / "home.obj"
    material = bundle / "home.mtl"
    validation = bundle / "validation.json"
    authored.write_bytes(b"mtllib home.mtl\n")
    material.write_bytes(b"newmtl home\n")
    validation.write_bytes(b"{}")
    release = _release("one", "unused").model_copy(
        update={
            "authored_scene": _fingerprint(
                "authored_scene", hashlib.sha256(authored.read_bytes()).hexdigest()
            ),
            "authored_scene_size_bytes": authored.stat().st_size,
            "authored_scene_dependencies": (
                SceneArtifact(
                    role="material_library_000",
                    relative_path="releases/one/home.mtl",
                    sha256=hashlib.sha256(material.read_bytes()).hexdigest(),
                    size_bytes=material.stat().st_size,
                ),
            ),
            "validation_report_sha256": hashlib.sha256(
                validation.read_bytes()
            ).hexdigest(),
        }
    )
    store = SceneReleaseStore(
        tmp_path / "state" / "scene.sqlite3",
        bundle_root=root,
    )
    store.register(release)
    store.promote(
        "one",
        actor_id="owner",
        occurred_at_us=1_000,
        expected_current_release_id=None,
    )

    material.write_bytes(b"newmtl mutated\n")
    with pytest.raises(
        SceneReleaseStoreError,
        match="authored scene dependency.*size mismatch|fingerprint mismatch",
    ):
        store.validate_artifacts(release)
