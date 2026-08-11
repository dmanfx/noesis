from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from websockets.legacy.client import connect as legacy_websocket_connect

from noesis.server.internal_auth import (
    InternalAuthConfig,
    InternalAuthConfigurationError,
    configure_internal_rest_app,
    load_or_create_internal_token,
    validate_internal_auth_listener,
)
from websocket_server import WebSocketServer


def _app(env: dict[str, str]) -> FastAPI:
    app = FastAPI()

    @app.get("/api/v1/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    configure_internal_rest_app(app, env=env)
    return app


def test_required_internal_auth_creates_private_token_and_rejects_direct_calls(tmp_path: Path) -> None:
    token_file = tmp_path / "state" / "gateway-token"
    app = _app(
        {
            "NOESIS_INTERNAL_AUTH_MODE": "required",
            "NOESIS_INTERNAL_AUTH_TOKEN_FILE": str(token_file),
        }
    )
    token = token_file.read_text(encoding="utf-8").strip()
    assert token_file.stat().st_mode & 0o777 == 0o600
    assert token_file.parent.stat().st_mode & 0o777 == 0o700

    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 401
        assert client.get("/api/v1/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.get("/api/v1/health", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_internal_token_is_stable_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "gateway-token"
    first = load_or_create_internal_token(path)
    second = load_or_create_internal_token(path)
    assert first == second
    assert len(first) >= 43


def test_disabled_mode_must_be_explicit_and_invalid_modes_fail(tmp_path: Path) -> None:
    app = _app({"NOESIS_INTERNAL_AUTH_MODE": "disabled"})
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
    with pytest.raises(InternalAuthConfigurationError, match="must be"):
        _app({"NOESIS_INTERNAL_AUTH_MODE": "optional"})


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "[::1]", "localhost"])
def test_disabled_internal_auth_accepts_only_explicit_loopback(host: str) -> None:
    validate_internal_auth_listener("disabled", host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "appliance.local", ""])
def test_disabled_internal_auth_rejects_lan_or_wildcard_listener(host: str) -> None:
    with pytest.raises(InternalAuthConfigurationError, match="loopback"):
        validate_internal_auth_listener("disabled", host)


def test_short_or_symlinked_token_files_fail_loudly(tmp_path: Path) -> None:
    short = tmp_path / "short"
    short.write_text("guessable\n", encoding="utf-8")
    short.chmod(0o600)
    with pytest.raises(InternalAuthConfigurationError, match="too short"):
        load_or_create_internal_token(short)
    target = tmp_path / "target"
    target.write_text("x" * 64, encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(InternalAuthConfigurationError, match="symlink"):
        load_or_create_internal_token(link)


def test_ds8_v3dt_and_ds9_composition_roots_install_the_same_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "noesis/ds8_runtime.py",
        "noesis/ds8_runtime_v3dt_reimpl.py",
        "DS9/noesis/ds9_runtime_core.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "configure_internal_rest_app(app)" in source
        assert "validate_internal_auth_listener" in source
        assert "NOESIS_REST_CORS_ALLOW_ALL is forbidden" in source


def test_websocket_boundary_requires_the_same_internal_bearer() -> None:
    config = InternalAuthConfig(mode="required", token_file=None, token="t" * 64)
    server = WebSocketServer(internal_auth_config=config)
    denied = asyncio.run(server._process_request("/", {}))
    assert denied is not None
    assert int(denied[0]) == 401
    assert asyncio.run(
        server._process_request("/", {"Authorization": f"Bearer {'t' * 64}"})
    ) is None


def test_disabled_websocket_auth_rejects_non_loopback_listener() -> None:
    config = InternalAuthConfig(mode="disabled", token_file=None, token=None)
    with pytest.raises(InternalAuthConfigurationError, match="loopback"):
        WebSocketServer(host="0.0.0.0", internal_auth_config=config)


def test_websocket_health_path_never_registers_a_full_telemetry_client() -> None:
    class HealthSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed: tuple[int, str] | None = None

        async def send(self, payload: str) -> None:
            self.sent.append(payload)

        async def close(self, *, code: int, reason: str) -> None:
            self.closed = (code, reason)

    config = InternalAuthConfig(mode="required", token_file=None, token="t" * 64)
    server = WebSocketServer(internal_auth_config=config)
    server.detection_config_getter = lambda: (_ for _ in ()).throw(AssertionError("must not run"))
    socket = HealthSocket()
    asyncio.run(server.handle_client(socket, "/healthz"))

    assert socket.sent == [WebSocketServer.HEALTH_PAYLOAD]
    assert socket.closed == (1000, "health_complete")
    assert not server.connected_clients


def test_authenticated_websocket_health_exchange_uses_the_exact_path_and_frame() -> None:
    async def scenario() -> None:
        token = "t" * 64
        config = InternalAuthConfig(mode="required", token_file=None, token=token)
        server = WebSocketServer(host="127.0.0.1", port=0, internal_auth_config=config)
        await server.start()
        try:
            async with legacy_websocket_connect(
                f"ws://127.0.0.1:{server.port}/healthz",
                extra_headers={"Authorization": f"Bearer {token}"},
                max_size=4096,
            ) as socket:
                assert await socket.recv() == WebSocketServer.HEALTH_PAYLOAD
            assert not server.connected_clients
        finally:
            await server.stop()

    asyncio.run(scenario())
