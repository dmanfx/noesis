from __future__ import annotations

import json
import os
import secrets
import stat
from pathlib import Path
from urllib.parse import urlunsplit

import pytest
import yaml

from mapanything_config import ServiceConfig, load_service_config
from noesis_core.runtime_secrets import (
    RuntimeSecretError,
    load_camera_uri_registry,
    load_mapanything_api_key,
    materialize_pipeline_config,
    public_pipeline_config,
    redact_runtime_secrets,
    source_provenance_ref,
)


def _camera_uri(name: str) -> str:
    return urlunsplit(("rtsp", "camera.invalid", f"/{name}/private-token", "", ""))


def _write_private(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    return path


def _camera_payload(uri: str | None = None) -> bytes:
    return (
        json.dumps(
            {"version": 1, "sources": {"living-room": uri or _camera_uri("living")}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _authority_key() -> bytes:
    return secrets.token_urlsafe(48).encode("ascii")


@pytest.mark.parametrize("loader,env_name,payload", [
    (load_mapanything_api_key, "NOESIS_MAPANYTHING_API_KEY_FILE", _authority_key()),
    (load_camera_uri_registry, "NOESIS_CAMERA_SECRETS_FILE", _camera_payload()),
])
def test_runtime_secret_missing_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader,
    env_name: str,
    payload: bytes,
) -> None:
    del payload
    missing = tmp_path / "private" / "missing"
    missing.parent.mkdir(mode=0o700)
    monkeypatch.setenv(env_name, str(missing))
    with pytest.raises(RuntimeSecretError, match="missing"):
        loader()


@pytest.mark.parametrize("loader,env_name,payload", [
    (load_mapanything_api_key, "NOESIS_MAPANYTHING_API_KEY_FILE", _authority_key()),
    (load_camera_uri_registry, "NOESIS_CAMERA_SECRETS_FILE", _camera_payload()),
])
def test_runtime_secret_insecure_file_is_rejected_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader,
    env_name: str,
    payload: bytes,
) -> None:
    path = _write_private(tmp_path / "private" / "secret", payload)
    os.chmod(path, 0o644)
    monkeypatch.setenv(env_name, str(path))
    with pytest.raises(RuntimeSecretError, match="mode must be 0600"):
        loader()
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


@pytest.mark.parametrize("loader,env_name,payload", [
    (load_mapanything_api_key, "NOESIS_MAPANYTHING_API_KEY_FILE", _authority_key()),
    (load_camera_uri_registry, "NOESIS_CAMERA_SECRETS_FILE", _camera_payload()),
])
def test_runtime_secret_symlink_and_hardlink_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader,
    env_name: str,
    payload: bytes,
) -> None:
    target = _write_private(tmp_path / "private" / "target", payload)
    symlink = target.parent / "symlink"
    symlink.symlink_to(target)
    monkeypatch.setenv(env_name, str(symlink))
    with pytest.raises(RuntimeSecretError, match="symlink"):
        loader()

    symlink.unlink()
    hardlink = target.parent / "hardlink"
    os.link(target, hardlink)
    monkeypatch.setenv(env_name, str(target))
    with pytest.raises(RuntimeSecretError, match="exactly one hard link"):
        loader()


def test_runtime_secret_insecure_parent_is_rejected_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    os.chmod(parent, 0o755)
    path = parent / "key"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, _authority_key())
    finally:
        os.close(descriptor)
    monkeypatch.setenv("NOESIS_MAPANYTHING_API_KEY_FILE", str(path))
    with pytest.raises(RuntimeSecretError, match="directory mode must be 0700"):
        load_mapanything_api_key()
    assert stat.S_IMODE(parent.stat().st_mode) == 0o755


def test_camera_registry_rejects_duplicate_uri_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uri = _camera_uri("duplicate")
    payload = (
        json.dumps(
            {
                "version": 1,
                "sources": {"living-room": uri, "kitchen": uri},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    path = _write_private(tmp_path / "private" / "camera_sources.json", payload)
    monkeypatch.setenv("NOESIS_CAMERA_SECRETS_FILE", str(path))
    with pytest.raises(RuntimeSecretError, match="duplicate URI values"):
        load_camera_uri_registry()


@pytest.mark.parametrize("payload", [b"short", b"a" * 42, b"a" * 43 + b"\n", b"a" * 42 + b"!"])
def test_mapanything_key_requires_authority_grade_urlsafe_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    path = _write_private(tmp_path / "private" / "key", payload)
    monkeypatch.setenv("NOESIS_MAPANYTHING_API_KEY_FILE", str(path))
    with pytest.raises(RuntimeSecretError, match="43 URL-safe"):
        load_mapanything_api_key()


def test_mapanything_config_rejects_inline_key_and_repr_hides_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _authority_key()
    key_path = _write_private(tmp_path / "private" / "key", key)
    monkeypatch.setenv("NOESIS_MAPANYTHING_API_KEY_FILE", str(key_path))
    ini = tmp_path / "mapanything.ini"
    ini.write_text("[service]\nhost=127.0.0.1\nport=8001\napi_key=inline\n", encoding="utf-8")
    with pytest.raises(RuntimeSecretError, match="must not contain an inline api_key"):
        load_service_config(ini)

    ini.write_text("[service]\nhost=127.0.0.1\nport=8001\n", encoding="utf-8")
    config = load_service_config(ini)
    assert isinstance(config, ServiceConfig)
    assert key.decode("ascii") not in repr(config)


def test_pipeline_materialization_rejects_inline_and_missing_camera_secrets() -> None:
    inline = {"sources": [{"element": "nvurisrcbin", "uri": _camera_uri("inline")}]}
    with pytest.raises(RuntimeSecretError, match="inline RTSP URI"):
        materialize_pipeline_config(inline, camera_registry={})

    missing = {"sources": [{"element": "nvurisrcbin", "uri_secret": "living-room"}]}
    with pytest.raises(RuntimeSecretError, match="missing camera secret"):
        materialize_pipeline_config(missing, camera_registry={})

    ambiguous = {
        "sources": [
            {
                "element": "nvurisrcbin",
                "uri_secret": "living-room",
                "uri": _camera_uri("inline"),
            }
        ]
    }
    with pytest.raises(RuntimeSecretError, match="must not combine"):
        materialize_pipeline_config(
            ambiguous,
            camera_registry={"living-room": _camera_uri("registry")},
        )


def test_public_pipeline_and_provenance_never_serialize_camera_uri() -> None:
    secret_uri = _camera_uri("living")
    materialized = materialize_pipeline_config(
        {"version": 1, "sources": [{"uri_secret": "living-room"}]},
        camera_registry={"living-room": secret_uri},
    )
    assert materialized["sources"][0]["uri"] == secret_uri
    public = public_pipeline_config(materialized)
    assert public == {"version": 1, "sources": [{"uri_secret": "living-room"}]}
    assert secret_uri not in yaml.safe_dump(public)
    assert source_provenance_ref(materialized["sources"][0], source_id=0) == (
        "camera-secret:living-room"
    )
    rendered_error = redact_runtime_secrets(f"source failed: {secret_uri}")
    assert secret_uri not in rendered_error
    assert "rtsp://<redacted>" in rendered_error


def test_ds8_and_ds9_pipeline_components_do_not_receive_uri_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from noesis.pipelines import ds8_pipeline as ds8
    from DS9.noesis.pipelines import ds8_pipeline as ds9

    monkeypatch.setenv("NOESIS_DS8_STUB_PIPELINE", "1")
    monkeypatch.setenv("NOESIS_BUILD_DIR", str(tmp_path / "build"))
    ds9_engine = tmp_path / "ds9.engine"
    ds9_engine.write_bytes(b"engine")
    ds9_infer = tmp_path / "ds9.ini"
    ds9_infer.write_text(
        "[property]\n"
        "onnx-file=offline-source.onnx\n"
        f"model-engine-file={ds9_engine}\n",
        encoding="utf-8",
    )
    ds9_config = tmp_path / "ds9.yaml"
    ds9_config.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sources": [
                    {
                        "element": "nvurisrcbin",
                        "uri_secret": "living-room",
                    }
                ],
                "models": {
                    "pgie": {
                        "config-file-path": str(ds9_infer),
                        "engine": str(ds9_engine),
                    }
                },
                "tracker": {},
                "analytics": {"enable": False},
                "sinks": [{"name": "sink", "type": "fakesink"}],
            }
        ),
        encoding="utf-8",
    )
    for module, path in (
        (ds8, Path("config/infer.yaml")),
        (ds9, ds9_config),
    ):
        module._PIPELINE_SINGLETON = None
        graph = module.build_pipeline(path)
        source_components = [
            component
            for component in graph.components.values()
            if component.element == "nvurisrcbin"
        ]
        assert source_components
        assert all("uri_secret" not in component.config for component in source_components)
        assert all(str(component.config.get("uri") or "").startswith("rtsp") for component in source_components)
        module._PIPELINE_SINGLETON = None
