from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import websockets

from noesis.server.internal_auth import InternalAuthConfigurationError
from scripts.internal_auth_client import (
    build_required_auth_request,
    configure_required_auth_environment,
    connect_required_websocket,
    load_required_internal_auth,
    resolve_auth_token_file,
)
from scripts.zero_copy_smoke_test import _request_depth_refresh

SECRET = "strict-smoke-client-secret-" + ("x" * 48)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_token(path: Path, token: str = SECRET) -> Path:
    path.write_text(token + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def test_repository_scripts_package_wins_over_installed_name_collision() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "import scripts, scripts.internal_auth_client as client; "
                "print(Path(scripts.__file__).resolve()); "
                "print(Path(client.__file__).resolve())"
            ),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == [
        str((REPO_ROOT / "scripts" / "__init__.py").resolve()),
        str((REPO_ROOT / "scripts" / "internal_auth_client.py").resolve()),
    ]


def test_required_client_auth_loads_existing_file_without_disclosing_secret(
    tmp_path: Path,
) -> None:
    token_file = _write_token(tmp_path / "gateway-token")

    auth = load_required_internal_auth(token_file)

    assert auth.token_file == token_file.absolute()
    assert auth.authorization_headers() == {"Authorization": f"Bearer {SECRET}"}
    assert SECRET not in repr(auth)
    assert SECRET not in str(auth)

    env = {"UNRELATED": "preserved"}
    configure_required_auth_environment(env, auth)
    assert env == {
        "UNRELATED": "preserved",
        "NOESIS_INTERNAL_AUTH_MODE": "required",
        "NOESIS_INTERNAL_AUTH_TOKEN_FILE": str(token_file.absolute()),
    }
    assert SECRET not in json.dumps(env)


def test_required_client_auth_is_read_only_and_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing-token"

    with pytest.raises(InternalAuthConfigurationError) as raised:
        load_required_internal_auth(missing)

    assert not missing.exists()
    assert SECRET not in str(raised.value)


def test_auth_token_resolution_prefers_cli_then_environment(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit-token"
    from_env = tmp_path / "environment-token"

    assert resolve_auth_token_file(explicit, env={}) == explicit.absolute()
    assert (
        resolve_auth_token_file(
            None, env={"NOESIS_INTERNAL_AUTH_TOKEN_FILE": str(from_env)}
        )
        == from_env.absolute()
    )


def test_websocket_connector_owns_bearer_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth = load_required_internal_auth(_write_token(tmp_path / "gateway-token"))
    observed: dict[str, object] = {}
    marker = object()

    def fake_connect(uri: str, **kwargs: object) -> object:
        observed["uri"] = uri
        observed.update(kwargs)
        return marker

    monkeypatch.setattr(websockets, "connect", fake_connect)

    result = connect_required_websocket(
        "ws://127.0.0.1:6008",
        auth,
        max_size=None,
    )

    assert result is marker
    assert observed["uri"] == "ws://127.0.0.1:6008"
    assert observed["additional_headers"] == {"Authorization": f"Bearer {SECRET}"}
    with pytest.raises(ValueError, match="owned by internal auth"):
        connect_required_websocket(
            "ws://127.0.0.1:6008",
            auth,
            additional_headers={"Authorization": "Bearer attacker"},
        )


def test_http_request_owns_bearer_header_without_query_credentials(
    tmp_path: Path,
) -> None:
    auth = load_required_internal_auth(_write_token(tmp_path / "gateway-token"))
    request = build_required_auth_request(
        "http://127.0.0.1:8080/api/v1/depth/refresh?seconds=20",
        auth,
        method="POST",
        headers={"Accept": "application/json"},
    )

    assert request.method == "POST"
    assert request.get_header("Authorization") == f"Bearer {SECRET}"
    assert request.get_header("Accept") == "application/json"
    assert SECRET not in request.full_url
    with pytest.raises(ValueError, match="owned by internal auth"):
        build_required_auth_request(
            "http://127.0.0.1:8080/api/v1/health",
            auth,
            method="GET",
            headers={"authorization": "Bearer attacker"},
        )


def test_zero_copy_rest_smoke_posts_authenticated_refresh_without_secret_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth = load_required_internal_auth(_write_token(tmp_path / "gateway-token"))
    observed: dict[str, object] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"enabled":true}'

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        observed["request"] = request
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = _request_depth_refresh(
        "http://127.0.0.1:8080/api/v1/depth/refresh",
        20,
        auth,
    )

    request = observed["request"]
    assert getattr(request, "method") == "POST"
    assert request.get_header("Authorization") == f"Bearer {SECRET}"
    assert result["ok"] is True
    assert result["url"].endswith("?seconds=20")
    assert SECRET not in json.dumps(result)
