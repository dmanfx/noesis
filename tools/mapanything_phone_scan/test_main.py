from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tools.mapanything_phone_scan import __main__ as launcher


def _lifespan_app(events):
    @asynccontextmanager
    async def lifespan(_app):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    return SimpleNamespace(router=SimpleNamespace(lifespan_context=lifespan))


def test_server_configs_share_app_and_use_one_lifespan(tmp_path, monkeypatch):
    certificate = tmp_path / "appliance-cert.pem"
    key = tmp_path / "appliance-key.pem"
    certificate.write_text("certificate", encoding="utf-8")
    key.write_text("key", encoding="utf-8")
    monkeypatch.setenv("NOESIS_PHONE_SCAN_PORT", "18888")
    monkeypatch.setenv("NOESIS_PHONE_SCAN_HTTPS_PORT", "18889")
    monkeypatch.setenv("NOESIS_PHONE_SCAN_TLS_CERT_FILE", str(certificate))
    monkeypatch.setenv("NOESIS_PHONE_SCAN_TLS_KEY_FILE", str(key))
    monkeypatch.setattr(launcher, "_validate_tls_material", lambda *_paths: None)
    app = object()

    http_config, https_config = launcher._server_configs(app)

    assert http_config.app is app
    assert https_config.app is app
    assert (http_config.host, http_config.port) == ("0.0.0.0", 18888)
    assert (https_config.host, https_config.port) == ("0.0.0.0", 18889)
    assert http_config.lifespan == "off"
    assert https_config.lifespan == "off"
    assert https_config.ssl_certfile == str(certificate.resolve())
    assert https_config.ssl_keyfile == str(key.resolve())


def test_menon_state_directory_is_already_the_menon_state_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MENON_STATE_DIR", str(tmp_path / "menon-state"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)

    assert launcher._default_menon_tls_directory() == tmp_path / "menon-state" / "tls"


def test_ca_certificate_path_has_a_separate_public_override(tmp_path, monkeypatch):
    ca = tmp_path / "ca.pem"
    monkeypatch.setenv("NOESIS_PHONE_SCAN_TLS_CA_CERT_FILE", str(ca))

    assert launcher._ca_certificate_path() == ca.resolve()


def test_server_configs_fail_closed_when_tls_material_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("NOESIS_PHONE_SCAN_TLS_CERT_FILE", str(tmp_path / "missing-cert.pem"))
    monkeypatch.setenv("NOESIS_PHONE_SCAN_TLS_KEY_FILE", str(tmp_path / "missing-key.pem"))

    with pytest.raises(RuntimeError, match="TLS material is missing"):
        launcher._server_configs(object())


def test_run_servers_stops_peer_and_reports_listener_failure():
    class FakeServer:
        def __init__(self, failure: bool):
            self.failure = failure
            self.should_exit = False
            self.started = False

        async def serve(self):
            self.started = True
            if self.failure:
                raise OSError("bind failed")
            while not self.should_exit:
                await asyncio.sleep(0.001)

    failing = FakeServer(True)
    peer = FakeServer(False)
    events = []

    with pytest.raises(OSError, match="bind failed"):
        asyncio.run(launcher._run_servers([failing, peer], _lifespan_app(events)))

    assert failing.should_exit is True
    assert peer.should_exit is True
    assert events == ["enter", "exit"]


def test_run_servers_starts_both_listeners_inside_one_lifespan():
    class FakeServer:
        def __init__(self):
            self.should_exit = False
            self.started = False

        async def serve(self):
            self.started = True
            while not self.should_exit:
                await asyncio.sleep(0.001)

    servers = [FakeServer(), FakeServer()]
    events = []

    async def stop_after_startup():
        while not all(server.started for server in servers):
            await asyncio.sleep(0.001)
        for server in servers:
            server.should_exit = True

    async def run():
        stopper = asyncio.create_task(stop_after_startup())
        await launcher._run_servers(servers, _lifespan_app(events))
        await stopper

    asyncio.run(run())

    assert all(server.started for server in servers)
    assert events == ["enter", "exit"]
