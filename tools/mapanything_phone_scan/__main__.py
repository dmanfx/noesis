from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import socket
import ssl
from typing import Any

from uvicorn import Config, Server


def _lan_ip() -> str | None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))
        address = str(probe.getsockname()[0])
        return address if address and not address.startswith("127.") else None
    except OSError:
        return None
    finally:
        probe.close()


def _env_port(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer between 1 and 65535") from exc
    if not 1 <= port <= 65_535:
        raise ValueError(f"{name} must be an integer between 1 and 65535")
    return port


def _default_menon_tls_directory() -> Path:
    configured_state = os.environ.get("MENON_STATE_DIR", "").strip()
    if configured_state:
        return Path(configured_state).expanduser() / "tls"
    xdg_state = os.environ.get("XDG_STATE_HOME", "").strip()
    state_root = (
        Path(xdg_state).expanduser()
        if xdg_state
        else Path.home() / ".local" / "state"
    )
    return state_root / "menon" / "tls"


def _tls_paths() -> tuple[Path, Path]:
    tls_directory = _default_menon_tls_directory()
    certificate = Path(
        os.environ.get(
            "NOESIS_PHONE_SCAN_TLS_CERT_FILE",
            str(tls_directory / "TauntonMainframe.local-cert.pem"),
        )
    ).expanduser()
    key = Path(
        os.environ.get(
            "NOESIS_PHONE_SCAN_TLS_KEY_FILE",
            str(tls_directory / "TauntonMainframe.local-key.pem"),
        )
    ).expanduser()
    return certificate.resolve(), key.resolve()


def _ca_certificate_path() -> Path:
    configured = os.environ.get("NOESIS_PHONE_SCAN_TLS_CA_CERT_FILE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (_default_menon_tls_directory() / "menon-local-ca-cert.pem").resolve()


def _certificate_fingerprint(certificate: Path) -> str:
    pem = certificate.read_bytes().decode("ascii")
    der = ssl.PEM_cert_to_DER_cert(pem)
    return hashlib.sha256(der).hexdigest()


def _validate_tls_material(certificate: Path, key: Path) -> None:
    missing = [str(path) for path in (certificate, key) if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Phone-scan HTTPS is enabled but its TLS material is missing: "
            f"{joined}. Set NOESIS_PHONE_SCAN_TLS_CERT_FILE and "
            "NOESIS_PHONE_SCAN_TLS_KEY_FILE, or install the Menon appliance "
            "certificate under the XDG state directory."
        )
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(certificate), keyfile=str(key))
    except (OSError, ssl.SSLError) as exc:
        raise RuntimeError(
            "Phone-scan HTTPS TLS certificate and key could not be loaded; "
            "verify that they are a matching, readable certificate pair."
        ) from exc


def _server_configs(app: Any) -> tuple[Config, Config]:
    host = os.environ.get("NOESIS_PHONE_SCAN_HOST", "0.0.0.0").strip() or "0.0.0.0"
    https_host = (
        os.environ.get("NOESIS_PHONE_SCAN_HTTPS_HOST", host).strip() or host
    )
    http_port = _env_port("NOESIS_PHONE_SCAN_PORT", 8788)
    https_port = _env_port("NOESIS_PHONE_SCAN_HTTPS_PORT", 8789)
    if (host, http_port) == (https_host, https_port):
        raise ValueError(
            "NOESIS_PHONE_SCAN_PORT and NOESIS_PHONE_SCAN_HTTPS_PORT must "
            "identify different listeners"
        )
    log_level = os.environ.get("NOESIS_PHONE_SCAN_LOG_LEVEL", "info")
    common = {
        "log_level": log_level,
        "access_log": False,
        # Native paired capture renews its lease every ten seconds. Keep its
        # verified HTTPS connection across that interval instead of forcing
        # hostname resolution and a new handshake for each heartbeat.
        "timeout_keep_alive": 30,
        "timeout_graceful_shutdown": 30,
    }
    certificate, key = _tls_paths()
    _validate_tls_material(certificate, key)
    return (
        Config(
            app,
            host=host,
            port=http_port,
            lifespan="off",
            **common,
        ),
        Config(
            app,
            host=https_host,
            port=https_port,
            lifespan="off",
            ssl_certfile=str(certificate),
            ssl_keyfile=str(key),
            **common,
        ),
    )


def _stop_servers(servers: list[Server]) -> None:
    for server in servers:
        server.should_exit = True


class _CoordinatedServer(Server):
    """Make either Uvicorn listener request shutdown of its sibling."""

    peers: tuple["_CoordinatedServer", ...] = ()

    def __init__(self, config: Config, *, signal_owner: bool):
        super().__init__(config)
        self.signal_owner = signal_owner

    @contextmanager
    def capture_signals(self):
        if not self.signal_owner:
            yield
            return
        with super().capture_signals():
            yield

    def handle_exit(self, sig: int, frame: Any) -> None:
        super().handle_exit(sig, frame)
        for peer in self.peers:
            peer.should_exit = True


async def _run_servers(servers: list[Server], app: Any) -> None:
    if len(servers) != 2:
        raise ValueError("Phone-scan launcher requires exactly two listeners")
    http_server, https_server = servers
    async with app.router.lifespan_context(app):
        http_task = asyncio.create_task(http_server.serve())
        https_task: asyncio.Task[None] | None = None
        try:
            # The shared app lifespan has already recovered state. Wait until
            # the HTTP listener has bound before exposing HTTPS; both configs
            # use lifespan=off so this context is entered exactly once.
            while not http_server.started and not http_task.done():
                await asyncio.sleep(0.01)
            if http_task.done():
                await http_task
                return
            https_task = asyncio.create_task(https_server.serve())
            done, pending = await asyncio.wait(
                (http_task, https_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            _stop_servers(servers)
            for task in done:
                task.result()
            await asyncio.gather(*pending)
        except BaseException:
            _stop_servers(servers)
            tasks = [http_task]
            if https_task is not None:
                tasks.append(https_task)
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            # Drain both listeners before the outer app lifespan invokes
            # PhoneScanService.shutdown().
            _stop_servers(servers)
            tasks = [http_task]
            if https_task is not None:
                tasks.append(https_task)
            await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    from .app import app

    host = os.environ.get("NOESIS_PHONE_SCAN_HOST", "0.0.0.0")
    port = _env_port("NOESIS_PHONE_SCAN_PORT", 8788)
    https_host = os.environ.get("NOESIS_PHONE_SCAN_HTTPS_HOST", host)
    https_port = _env_port("NOESIS_PHONE_SCAN_HTTPS_PORT", 8789)
    tls_hostname = (
        os.environ.get("NOESIS_PHONE_SCAN_TLS_HOSTNAME", "TauntonMainframe.local")
        .strip()
        or "TauntonMainframe.local"
    )
    address = _lan_ip()
    print("Noesis multi-view phone scan is starting.", flush=True)
    if address:
        print(f"Phone URL: http://{address}:{port}", flush=True)
    else:
        print(f"Open this machine's LAN address on port {port}.", flush=True)
    print(
        f"Phone HTTPS URL: https://{tls_hostname}:{https_port}",
        flush=True,
    )
    print(
        "Use the HTTPS hostname after trusting the configured appliance CA; "
        "the certificate does not authorize the raw LAN IP.",
        flush=True,
    )
    if address:
        print(
            f"CA download URL: http://{address}:{port}/api/browser-capture/ca-certificate",
            flush=True,
        )
        try:
            ca_digest = _certificate_fingerprint(_ca_certificate_path())
        except (OSError, UnicodeDecodeError, ValueError):
            print(
                "CA SHA-256: unavailable; configured CA certificate is missing or invalid.",
                flush=True,
            )
        else:
            print(f"CA SHA-256: {ca_digest}", flush=True)
    print("Keep this terminal running while recording, uploading, and reviewing.", flush=True)
    if (host, port) == (https_host, https_port):
        raise ValueError(
            "NOESIS_PHONE_SCAN_PORT and NOESIS_PHONE_SCAN_HTTPS_PORT must "
            "identify different listeners"
        )
    servers = [
        _CoordinatedServer(config, signal_owner=index == 0)
        for index, config in enumerate(_server_configs(app))
    ]
    peers = tuple(servers)
    for server in servers:
        server.peers = peers
    asyncio.run(_run_servers(servers, app))


if __name__ == "__main__":
    main()
