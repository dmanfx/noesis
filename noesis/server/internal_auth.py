from __future__ import annotations

import hmac
import ipaddress
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from noesis_core.private_paths import (
    PrivatePathError,
    ensure_private_directory,
    read_private_file,
    validate_private_file,
)


class InternalAuthConfigurationError(RuntimeError):
    pass


def validate_internal_auth_listener(mode: str, host: str) -> None:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"required", "disabled"}:
        raise InternalAuthConfigurationError(
            "internal auth listener validation requires a known auth mode"
        )
    if normalized_mode == "required":
        return
    normalized_host = str(host or "").strip().lower().strip("[]")
    if normalized_host == "localhost":
        return
    try:
        address = ipaddress.ip_address(normalized_host.split("%", 1)[0])
    except ValueError as exc:
        raise InternalAuthConfigurationError(
            "disabled internal auth is allowed only on an explicit loopback listener"
        ) from exc
    if not address.is_loopback:
        raise InternalAuthConfigurationError(
            "disabled internal auth is allowed only on an explicit loopback listener"
        )


@dataclass(frozen=True)
class InternalAuthConfig:
    mode: str
    token_file: Path | None
    token: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "InternalAuthConfig":
        values = os.environ if env is None else env
        mode = (
            str(values.get("NOESIS_INTERNAL_AUTH_MODE", "required") or "")
            .strip()
            .lower()
        )
        if mode not in {"required", "disabled"}:
            raise InternalAuthConfigurationError(
                "NOESIS_INTERNAL_AUTH_MODE must be 'required' or explicit loopback-development 'disabled'"
            )
        if mode == "disabled":
            return cls(mode=mode, token_file=None, token=None)
        raw_path = str(values.get("NOESIS_INTERNAL_AUTH_TOKEN_FILE", "") or "").strip()
        configured_token_file = (
            Path(raw_path).expanduser()
            if raw_path
            else (Path.home() / ".local" / "state" / "noesis" / "gateway-token")
        )
        token = load_or_create_internal_token(configured_token_file)
        return cls(mode=mode, token_file=configured_token_file.resolve(), token=token)


def load_or_create_internal_token(path: str | Path) -> str:
    configured_path = Path(path).expanduser().absolute()
    if configured_path.is_symlink():
        raise InternalAuthConfigurationError(
            "internal auth token file must not be a symlink"
        )
    if configured_path.exists():
        try:
            token_path = validate_private_file(
                configured_path,
                label="internal auth token file",
            )
        except PrivatePathError as exc:
            raise InternalAuthConfigurationError(str(exc)) from exc
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(configured_path, flags)
        except OSError as exc:
            raise InternalAuthConfigurationError(
                "internal auth token cannot be opened securely"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            expected = token_path.stat()
            if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
                raise InternalAuthConfigurationError(
                    "internal auth token changed while opening"
                )
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise InternalAuthConfigurationError(
                    "internal auth token must be a single-link regular file"
                )
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                descriptor = -1
                raw = stream.read(4097)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(raw) > 4096:
            raise InternalAuthConfigurationError(
                "internal auth token file is too large"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InternalAuthConfigurationError(
                "internal auth token must contain UTF-8 text"
            ) from exc
        token = text.rstrip("\r\n")
        if "\n" in token or "\r" in token or token != token.strip():
            raise InternalAuthConfigurationError(
                "internal auth token must contain exactly one unpadded line"
            )
        if len(token) < 43:
            raise InternalAuthConfigurationError(
                "internal auth token is missing or too short"
            )
        return token

    try:
        ensure_private_directory(
            configured_path.parent,
            label="internal auth token parent",
        )
    except PrivatePathError as exc:
        raise InternalAuthConfigurationError(str(exc)) from exc

    token = secrets.token_urlsafe(48)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(configured_path, flags, 0o600)
    except FileExistsError as exc:
        raise InternalAuthConfigurationError(
            "internal auth token creation is already in progress"
        ) from exc
    except OSError as exc:
        raise InternalAuthConfigurationError(
            "internal auth token cannot be created securely"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as stream:
            descriptor = -1
            stream.write(token)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        try:
            configured_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise InternalAuthConfigurationError(
            "internal auth token cannot be persisted"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        validate_private_file(configured_path, label="internal auth token file")
        parent_descriptor = os.open(
            configured_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except (OSError, PrivatePathError) as exc:
        raise InternalAuthConfigurationError(
            "internal auth token persistence could not be verified"
        ) from exc
    return token


def load_internal_token(path: str | Path) -> str:
    """Load an existing owner-only internal bearer without creating state."""

    try:
        raw = read_private_file(
            path,
            label="internal auth token file",
            max_bytes=4096,
        )
    except PrivatePathError as exc:
        raise InternalAuthConfigurationError(str(exc)) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InternalAuthConfigurationError(
            "internal auth token must contain UTF-8 text"
        ) from exc
    token = text.rstrip("\r\n")
    if "\n" in token or "\r" in token or token != token.strip():
        raise InternalAuthConfigurationError(
            "internal auth token must contain exactly one unpadded line"
        )
    if len(token) < 43:
        raise InternalAuthConfigurationError(
            "internal auth token is missing or too short"
        )
    return token


class InternalBearerAuthMiddleware:
    """Require the appliance gateway token on every internal HTTP request."""

    def __init__(self, app: Any, *, token: str) -> None:
        if len(str(token)) < 43:
            raise InternalAuthConfigurationError("internal bearer token is too short")
        self.app = app
        self.token = str(token)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        authorization = headers.get("authorization", "")
        prefix = "Bearer "
        supplied = (
            authorization[len(prefix) :].strip()
            if authorization.startswith(prefix)
            else ""
        )
        if supplied and hmac.compare_digest(supplied, self.token):
            await self.app(scope, receive, send)
            return
        body = json.dumps(
            {
                "error": "internal_auth_required",
                "message": "Authenticated appliance gateway required.",
            },
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"cache-control", b"no-store"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def configure_internal_rest_app(
    app: Any, *, env: Mapping[str, str] | None = None
) -> InternalAuthConfig:
    config = InternalAuthConfig.from_env(env)
    if config.mode == "required":
        assert config.token is not None
        app.add_middleware(InternalBearerAuthMiddleware, token=config.token)
    app.state.noesis_internal_auth = {
        "mode": config.mode,
        "token_file": str(config.token_file) if config.token_file is not None else None,
    }
    return config


__all__ = [
    "InternalAuthConfig",
    "InternalAuthConfigurationError",
    "configure_internal_rest_app",
    "load_internal_token",
    "load_or_create_internal_token",
    "validate_internal_auth_listener",
]
