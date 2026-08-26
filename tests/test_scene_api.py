from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server.scene_api import router
from noesis_core.scene_store import SceneReleaseStore


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _client(tmp_path: Path, monkeypatch) -> tuple[TestClient, dict[str, object]]:
    root = tmp_path / "virtual-twin"
    revision = root / "revisions" / "revision-1"
    revision.mkdir(parents=True)
    manifest = {
        "schema": "noesis.virtual_twin.revision.v1",
        "revision_id": "revision-1",
        "camera": "kitchen",
        "created_ts_us": 100,
        "artifacts": {"surfaces_glb": "surfaces.glb"},
    }
    manifest_path = revision / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    surface_path = revision / "surfaces.glb"
    surface_path.write_bytes(b"glTF")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    release_assets = root / "releases" / "release-1"
    release_assets.mkdir(parents=True)
    authored_path = release_assets / "home.obj"
    material_path = release_assets / "home.mtl"
    texture_path = release_assets / "textures" / "wall.jpg"
    texture_path.parent.mkdir(parents=True)
    validation_path = release_assets / "validation.json"
    authored_path.write_bytes(b"mtllib home.mtl\no authored-home\n")
    material_path.write_bytes(b"newmtl wall\nmap_Kd textures/wall.jpg\n")
    texture_path.write_bytes(b"jpeg-wall")
    validation_path.write_bytes(b"{}")
    authored_sha = hashlib.sha256(authored_path.read_bytes()).hexdigest()
    material_sha = hashlib.sha256(material_path.read_bytes()).hexdigest()
    texture_sha = hashlib.sha256(texture_path.read_bytes()).hexdigest()
    validation_sha = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    monkeypatch.setenv("NOESIS_VIRTUAL_TWIN_ROOT", str(root))
    monkeypatch.setenv("NOESIS_SCENE_STORE_PATH", str(tmp_path / "state" / "scenes.sqlite3"))
    release: dict[str, object] = {
        "contract": "noesis.scene.release",
        "contract_version": 1,
        "release_id": "release-1",
        "release_version": 1,
        "created_at_us": 200,
        "created_by": "owner-1",
        "calibration": {"role": "calibration_bundle", "sha256": SHA_A},
        "model": {"role": "model_bundle", "sha256": SHA_B},
        "config": {"role": "scene_config", "sha256": SHA_C},
        "authored_scene": {"role": "authored_home", "sha256": authored_sha},
        "authored_scene_path": "releases/release-1/home.obj",
        "authored_scene_size_bytes": authored_path.stat().st_size,
        "authored_scene_dependencies": [
            {
                "role": "material_library_000",
                "relative_path": "releases/release-1/home.mtl",
                "sha256": material_sha,
                "size_bytes": material_path.stat().st_size,
            },
            {
                "role": "material_texture_000",
                "relative_path": "releases/release-1/textures/wall.jpg",
                "sha256": texture_sha,
                "size_bytes": texture_path.stat().st_size,
            },
        ],
        "cohort_max_delta_us": 100,
        "cameras": [
            {
                "camera_id": "kitchen",
                "revision_id": "revision-1",
                "captured_at_us": 100,
                "calibration_bundle_sha256": SHA_A,
                "calibration_sha256": SHA_C,
                "model_bundle_sha256": SHA_B,
                "model_sha256": SHA_B,
                "manifest_sha256": manifest_sha,
                "artifact_path": "revision-1",
                "artifacts": [
                    {
                        "role": "surfaces_glb",
                        "relative_path": "surfaces.glb",
                        "sha256": hashlib.sha256(surface_path.read_bytes()).hexdigest(),
                        "size_bytes": surface_path.stat().st_size,
                    }
                ],
            }
        ],
        "validation_report_sha256": validation_sha,
        "validation_report_path": "releases/release-1/validation.json",
        "validation_report_size_bytes": validation_path.stat().st_size,
    }
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), release


def test_register_promote_current_payload_and_rollback_contract(tmp_path: Path, monkeypatch) -> None:
    client, release = _client(tmp_path, monkeypatch)
    registered = client.post("/api/v1/scenes/releases", json=release)
    assert registered.status_code == 201
    assert registered.json()["release_id"] == "release-1"

    promoted = client.post(
        "/api/v1/scenes/releases/release-1/promote",
        json={"actor_id": "owner-1", "expected_current_release_id": None},
    )
    assert promoted.status_code == 200
    assert promoted.json()["event"] == "promote"

    current = client.get("/api/v1/scenes/current/payload")
    assert current.status_code == 200
    payload = current.json()
    assert payload["release"]["release_id"] == "release-1"
    assert payload["cameras"][0]["manifest_sha256"] == release["cameras"][0]["manifest_sha256"]
    assert payload["cameras"][0]["artifact_urls"]["surfaces_glb"].endswith(
        "/artifacts/surfaces_glb"
    )
    assert client.get(payload["cameras"][0]["artifact_urls"]["surfaces_glb"]).content == b"glTF"
    assert client.get(payload["authored_scene_url"]).content == (
        b"mtllib home.mtl\no authored-home\n"
    )
    dependency_urls = payload["authored_scene_dependency_urls"]
    assert client.get(dependency_urls["material_library_000"]).content == (
        b"newmtl wall\nmap_Kd textures/wall.jpg\n"
    )
    assert client.get(dependency_urls["material_texture_000"]).content == b"jpeg-wall"
    assert client.get(
        "/api/v1/scenes/current/authored-dependencies/not-present"
    ).status_code == 404
    assert client.get(payload["validation_report_url"]).json() == {}

    missing_expected = client.post(
        "/api/v1/scenes/releases/release-1/rollback",
        json={"actor_id": "owner-1"},
    )
    assert missing_expected.status_code == 422


def test_promotion_is_compare_and_swap_and_manifest_verified(tmp_path: Path, monkeypatch) -> None:
    client, release = _client(tmp_path, monkeypatch)
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    conflict = client.post(
        "/api/v1/scenes/releases/release-1/promote",
        json={"actor_id": "owner-1", "expected_current_release_id": "other"},
    )
    assert conflict.status_code == 409

    root = tmp_path / "virtual-twin"
    (root / "revisions" / "revision-1" / "manifest.json").write_text("{}", encoding="utf-8")
    mismatch = client.post(
        "/api/v1/scenes/releases/release-1/promote",
        json={"actor_id": "owner-1", "expected_current_release_id": None},
    )
    assert mismatch.status_code == 400
    assert "fingerprint mismatch" in mismatch.json()["detail"]


def test_current_scene_artifact_route_rejects_mutated_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, release = _client(tmp_path, monkeypatch)
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    assert client.post(
        "/api/v1/scenes/releases/release-1/promote",
        json={"actor_id": "owner-1", "expected_current_release_id": None},
    ).status_code == 200
    surface = (
        tmp_path
        / "virtual-twin"
        / "revisions"
        / "revision-1"
        / "surfaces.glb"
    )
    surface.write_bytes(b"evil")

    response = client.get(
        "/api/v1/scenes/current/cameras/kitchen/artifacts/surfaces_glb"
    )
    assert response.status_code == 409
    assert "fingerprint mismatch" in response.json()["detail"]


def test_current_authored_dependency_route_rejects_mutated_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, release = _client(tmp_path, monkeypatch)
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    assert client.post(
        "/api/v1/scenes/releases/release-1/promote",
        json={"actor_id": "owner-1", "expected_current_release_id": None},
    ).status_code == 200
    material = tmp_path / "virtual-twin" / "releases" / "release-1" / "home.mtl"
    material.write_bytes(b"mutated material")

    response = client.get(
        "/api/v1/scenes/current/authored-dependencies/material_library_000"
    )
    assert response.status_code == 409
    assert "size mismatch" in response.json()["detail"]


def test_artifact_route_hashes_only_the_selected_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, release = _client(tmp_path, monkeypatch)
    calls = {"validate": 0, "artifact": 0}
    original_validate = SceneReleaseStore.validate_artifacts
    original_read = SceneReleaseStore.read_camera_artifact

    def counted_validate(self, selected_release):
        calls["validate"] += 1
        return original_validate(self, selected_release)

    def counted_read(self, camera, artifact):
        calls["artifact"] += 1
        return original_read(self, camera, artifact)

    monkeypatch.setattr(SceneReleaseStore, "validate_artifacts", counted_validate)
    monkeypatch.setattr(SceneReleaseStore, "read_camera_artifact", counted_read)
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    assert client.post(
        "/api/v1/scenes/releases/release-1/promote",
        json={"actor_id": "owner-1", "expected_current_release_id": None},
    ).status_code == 200
    calls.update(validate=0, artifact=0)

    response = client.get(
        "/api/v1/scenes/current/cameras/kitchen/artifacts/surfaces_glb"
    )
    assert response.status_code == 200
    assert response.content == b"glTF"
    assert calls == {"validate": 0, "artifact": 1}


def test_artifact_response_is_the_verified_snapshot_not_a_path_reopen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, release = _client(tmp_path, monkeypatch)
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    assert client.post(
        "/api/v1/scenes/releases/release-1/promote",
        json={"actor_id": "owner-1", "expected_current_release_id": None},
    ).status_code == 200
    surface = (
        tmp_path
        / "virtual-twin"
        / "revisions"
        / "revision-1"
        / "surfaces.glb"
    )
    original_read = SceneReleaseStore.read_camera_artifact

    def replace_after_read(self, camera, artifact):
        verified = original_read(self, camera, artifact)
        surface.write_bytes(b"evil")
        return verified

    monkeypatch.setattr(
        SceneReleaseStore,
        "read_camera_artifact",
        replace_after_read,
    )
    response = client.get(
        "/api/v1/scenes/current/cameras/kitchen/artifacts/surfaces_glb"
    )
    assert response.status_code == 200
    assert response.content == b"glTF"
    assert response.headers["x-noesis-scene-release"] == "release-1"
    assert response.headers["etag"] == (
        f'"sha256-{hashlib.sha256(b"glTF").hexdigest()}"'
    )


def test_artifact_route_rejects_symlink_and_hardlink_substitution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, release = _client(tmp_path, monkeypatch)
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    assert client.post(
        "/api/v1/scenes/releases/release-1/promote",
        json={"actor_id": "owner-1", "expected_current_release_id": None},
    ).status_code == 200
    surface = (
        tmp_path
        / "virtual-twin"
        / "revisions"
        / "revision-1"
        / "surfaces.glb"
    )
    outside = tmp_path / "outside.glb"
    outside.write_bytes(b"glTF")
    surface.unlink()
    surface.symlink_to(outside)
    symlink_response = client.get(
        "/api/v1/scenes/current/cameras/kitchen/artifacts/surfaces_glb"
    )
    assert symlink_response.status_code == 409
    assert "following links" in symlink_response.json()["detail"]

    surface.unlink()
    os.link(outside, surface)
    hardlink_response = client.get(
        "/api/v1/scenes/current/cameras/kitchen/artifacts/surfaces_glb"
    )
    assert hardlink_response.status_code == 409
    assert "hard link" in hardlink_response.json()["detail"]


def test_current_payload_rejects_any_mutated_cohort_member(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, release = _client(tmp_path, monkeypatch)
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    assert client.post(
        "/api/v1/scenes/releases/release-1/promote",
        json={"actor_id": "owner-1", "expected_current_release_id": None},
    ).status_code == 200
    material = tmp_path / "virtual-twin" / "releases" / "release-1" / "home.mtl"
    material.write_bytes(b"mutated")
    response = client.get("/api/v1/scenes/current/payload")
    assert response.status_code == 409


def test_ds9_mounts_the_shared_scene_router() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in ("DS9/noesis/ds9_runtime_core.py",):
        source = (root / relative).read_text(encoding="utf-8")
        assert "scene_api" in source
        assert "app.include_router(scene_api.router)" in source
