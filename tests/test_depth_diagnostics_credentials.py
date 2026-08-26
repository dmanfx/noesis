from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import geometry.depth_publisher as depth_publisher
from geometry.depth_publisher import (
    DepthDiagnosticsPublisher,
    DiagnosticsConfig,
    DiagnosticsConfigurationError,
    DiagnosticsSettings,
    load_private_secret,
)


def _secret(path: Path, *, mode: int = 0o600, value: str = "s" * 32) -> Path:
    path.write_text(f"{value}\n", encoding="utf-8")
    path.chmod(mode)
    return path


def _config(**overrides: object) -> DiagnosticsConfig:
    base = DiagnosticsConfig(
        mqtt_enabled=False,
        influx_enabled=False,
        base_topic="noesis/occupancy",
        mqtt_host="127.0.0.1",
        mqtt_port=1883,
        mqtt_username="noesis",
        mqtt_password_file=None,
        mqtt_qos=1,
        mqtt_retain=True,
        influx_url="http://127.0.0.1:8086",
        influx_org="test-org",
        influx_token_file=None,
        influx_bucket="test-bucket",
    )
    return replace(base, **overrides)


def test_diagnostics_settings_are_disabled_and_contain_only_secret_paths() -> None:
    settings = DiagnosticsSettings()
    assert settings.mqtt_enabled is False
    assert settings.influx_enabled is False
    assert settings.mqtt_password_file is None
    assert settings.influx_token_file is None
    assert not hasattr(settings, "mqtt_password")
    assert not hasattr(settings, "influx_token")


def test_disabled_settings_do_not_open_configured_secret_paths(tmp_path: Path) -> None:
    settings = replace(
        DiagnosticsSettings(),
        mqtt_password_file=tmp_path / "does-not-exist",
        influx_token_file=tmp_path / "also-does-not-exist",
    )
    assert DepthDiagnosticsPublisher.from_settings(settings, environ={}) is None


@pytest.mark.parametrize("mode", [0o400, 0o600])
def test_private_secret_accepts_only_owner_readable_modes(tmp_path: Path, mode: int) -> None:
    path = _secret(tmp_path / "secret", mode=mode)
    assert len(load_private_secret(path, purpose="test")) == 32
    assert path.stat().st_mode & 0o777 == mode


@pytest.mark.parametrize("mode", [0o000, 0o440, 0o640, 0o660, 0o700, 0o644])
def test_private_secret_rejects_insecure_or_unusable_modes(
    tmp_path: Path,
    mode: int,
) -> None:
    path = _secret(tmp_path / "secret", mode=mode)
    with pytest.raises(DiagnosticsConfigurationError, match="0400 or 0600"):
        load_private_secret(path, purpose="test")
    assert path.stat().st_mode & 0o777 == mode


def test_private_secret_rejects_missing_symlink_hardlink_and_wrong_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(DiagnosticsConfigurationError, match="missing"):
        load_private_secret(tmp_path / "missing", purpose="test")

    target = _secret(tmp_path / "target")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    with pytest.raises(DiagnosticsConfigurationError, match="symlink"):
        load_private_secret(symlink, purpose="test")

    hardlink = tmp_path / "hardlink"
    os.link(target, hardlink)
    with pytest.raises(DiagnosticsConfigurationError, match="hard link"):
        load_private_secret(target, purpose="test")
    hardlink.unlink()

    monkeypatch.setattr(depth_publisher.os, "geteuid", lambda: target.stat().st_uid + 1)
    with pytest.raises(DiagnosticsConfigurationError, match="service user"):
        load_private_secret(target, purpose="test")


def test_private_secret_rejects_directory_oversize_and_non_utf8(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir(mode=0o700)
    with pytest.raises(DiagnosticsConfigurationError, match="regular file"):
        load_private_secret(directory, purpose="test")

    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"x" * (16 * 1024 + 1))
    oversized.chmod(0o600)
    with pytest.raises(DiagnosticsConfigurationError, match="too large"):
        load_private_secret(oversized, purpose="test")

    binary = tmp_path / "binary"
    binary.write_bytes(b"\xff" * 32)
    binary.chmod(0o600)
    with pytest.raises(DiagnosticsConfigurationError, match="UTF-8"):
        load_private_secret(binary, purpose="test")


def test_private_secret_reader_never_repairs_parent_permissions(tmp_path: Path) -> None:
    parent = tmp_path / "secrets"
    parent.mkdir(mode=0o750)
    path = _secret(parent / "secret")
    assert len(load_private_secret(path, purpose="test")) == 32
    assert parent.stat().st_mode & 0o777 == 0o750


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("short", "too short"),
        ("x" * 16 + "\n" + "y" * 16, "one line"),
        (" " + "x" * 31, "surrounding whitespace"),
    ],
)
def test_private_secret_rejects_ambiguous_content(
    tmp_path: Path,
    value: str,
    error: str,
) -> None:
    path = _secret(tmp_path / "secret", value=value)
    with pytest.raises(DiagnosticsConfigurationError, match=error):
        load_private_secret(path, purpose="test")


def test_file_path_environment_overrides_are_explicit_but_plaintext_env_is_rejected(
    tmp_path: Path,
) -> None:
    settings = replace(
        DiagnosticsSettings(),
        mqtt_password_file=tmp_path / "settings-password",
        influx_token_file=tmp_path / "settings-token",
    )
    config = DiagnosticsConfig.from_settings(
        settings,
        environ={
            "NOESIS_MQTT_PASSWORD_FILE": str(tmp_path / "environment-password"),
            "NOESIS_INFLUX_TOKEN_FILE": str(tmp_path / "environment-token"),
        },
    )
    assert config.mqtt_password_file == tmp_path / "environment-password"
    assert config.influx_token_file == tmp_path / "environment-token"

    with pytest.raises(DiagnosticsConfigurationError, match="plaintext"):
        DiagnosticsConfig.from_settings(settings, environ={"NOESIS_INFLUX_TOKEN": "not-used"})
    with pytest.raises(DiagnosticsConfigurationError, match="plaintext"):
        DiagnosticsConfig.from_settings(settings, environ={"NOESIS_MQTT_PASSWORD": "not-used"})


def test_explicit_mqtt_enable_fails_closed_without_secret_file() -> None:
    with pytest.raises(DiagnosticsConfigurationError, match="password file is required"):
        DepthDiagnosticsPublisher(_config(mqtt_enabled=True))


def test_explicit_influx_enable_fails_closed_on_insecure_secret_file(tmp_path: Path) -> None:
    token_file = _secret(tmp_path / "token", mode=0o644)
    with pytest.raises(DiagnosticsConfigurationError, match="0400 or 0600"):
        DepthDiagnosticsPublisher(
            _config(influx_enabled=True, influx_token_file=token_file)
        )


def test_enabled_mqtt_loads_secret_only_into_client_and_connects_synchronously(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password_file = _secret(tmp_path / "password")
    client = SimpleNamespace(
        credential_length=0,
        connected=False,
        loop_started=False,
        disconnected=False,
    )

    def username_pw_set(username: str, password: str) -> None:
        assert username == "noesis"
        client.credential_length = len(password)

    def connect(host: str, port: int, keepalive: int) -> int:
        assert (host, port, keepalive) == ("127.0.0.1", 1883, 60)
        client.connected = True
        return 0

    client.username_pw_set = username_pw_set
    client.connect = connect
    client.loop_start = lambda: setattr(client, "loop_started", True)
    client.loop_stop = lambda: setattr(client, "loop_started", False)
    client.disconnect = lambda: setattr(client, "disconnected", True)
    monkeypatch.setattr(
        depth_publisher,
        "mqtt",
        SimpleNamespace(Client=lambda **_kwargs: client),
    )

    publisher = DepthDiagnosticsPublisher(
        _config(mqtt_enabled=True, mqtt_password_file=password_file)
    )
    assert publisher.client is client
    assert client.credential_length == 32
    assert client.connected is True
    assert client.loop_started is True
    publisher.close()
    assert client.loop_started is False
    assert client.disconnected is True


def test_multi_sink_initialization_is_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password_file = _secret(tmp_path / "password")
    token_file = _secret(tmp_path / "token")
    client = SimpleNamespace(loop_stopped=False, disconnected=False)
    client.username_pw_set = lambda *_args: None
    client.connect = lambda *_args: 0
    client.loop_start = lambda: None
    client.loop_stop = lambda: setattr(client, "loop_stopped", True)
    client.disconnect = lambda: setattr(client, "disconnected", True)
    monkeypatch.setattr(
        depth_publisher,
        "mqtt",
        SimpleNamespace(Client=lambda **_kwargs: client),
    )
    monkeypatch.setattr(depth_publisher, "InfluxDBClient", None)

    with pytest.raises(DiagnosticsConfigurationError, match="not installed"):
        DepthDiagnosticsPublisher(
            _config(
                mqtt_enabled=True,
                influx_enabled=True,
                mqtt_password_file=password_file,
                influx_token_file=token_file,
            )
        )
    assert client.loop_stopped is True
    assert client.disconnected is True


def test_client_failures_do_not_echo_secret_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_file = _secret(tmp_path / "token", value="sentinel-secret-material")

    def fail_client(**kwargs: object) -> None:
        raise RuntimeError(str(kwargs.get("token")))

    monkeypatch.setattr(depth_publisher, "InfluxDBClient", fail_client)
    with pytest.raises(DiagnosticsConfigurationError) as raised:
        DepthDiagnosticsPublisher(
            _config(influx_enabled=True, influx_token_file=token_file)
        )
    assert "sentinel-secret-material" not in str(raised.value)
