from __future__ import annotations

import argparse
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from noesis.server.internal_auth import load_internal_token

DEFAULT_INTERNAL_AUTH_TOKEN_FILE = (
    Path.home() / ".local" / "state" / "noesis" / "gateway-token"
)


@dataclass(frozen=True)
class RequiredInternalAuth:
    token_file: Path
    _token: str = field(repr=False)

    def authorization_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def __str__(self) -> str:
        return f"RequiredInternalAuth(token_file={self.token_file})"


def add_auth_token_file_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auth-token-file",
        type=Path,
        default=None,
        help=(
            "Owner-only internal bearer file. Defaults to "
            "NOESIS_INTERNAL_AUTH_TOKEN_FILE, then the appliance gateway-token path."
        ),
    )


def resolve_auth_token_file(
    configured: str | Path | None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if env is None else env
    if configured is not None and str(configured).strip():
        return Path(configured).expanduser().absolute()
    from_env = str(values.get("NOESIS_INTERNAL_AUTH_TOKEN_FILE", "") or "").strip()
    if from_env:
        return Path(from_env).expanduser().absolute()
    return DEFAULT_INTERNAL_AUTH_TOKEN_FILE.expanduser().absolute()


def load_required_internal_auth(
    configured: str | Path | None,
    *,
    env: Mapping[str, str] | None = None,
) -> RequiredInternalAuth:
    token_file = resolve_auth_token_file(configured, env=env)
    token = load_internal_token(token_file)
    return RequiredInternalAuth(token_file=token_file, _token=token)


def configure_required_auth_environment(
    env: MutableMapping[str, str], auth: RequiredInternalAuth
) -> None:
    env["NOESIS_INTERNAL_AUTH_MODE"] = "required"
    env["NOESIS_INTERNAL_AUTH_TOKEN_FILE"] = str(auth.token_file)


def connect_required_websocket(
    uri: str,
    auth: RequiredInternalAuth,
    **kwargs: Any,
) -> Any:
    if "additional_headers" in kwargs or "extra_headers" in kwargs:
        raise ValueError("WebSocket authorization headers are owned by internal auth")
    import websockets

    return websockets.connect(
        uri,
        additional_headers=auth.authorization_headers(),
        **kwargs,
    )


def build_required_auth_request(
    url: str,
    auth: RequiredInternalAuth,
    *,
    method: str,
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
) -> urllib.request.Request:
    merged = dict(headers or {})
    if any(str(key).lower() == "authorization" for key in merged):
        raise ValueError("HTTP authorization header is owned by internal auth")
    merged.update(auth.authorization_headers())
    return urllib.request.Request(
        url,
        data=data,
        headers=merged,
        method=method,
    )


__all__ = [
    "DEFAULT_INTERNAL_AUTH_TOKEN_FILE",
    "RequiredInternalAuth",
    "add_auth_token_file_argument",
    "build_required_auth_request",
    "configure_required_auth_environment",
    "connect_required_websocket",
    "load_required_internal_auth",
    "resolve_auth_token_file",
]
