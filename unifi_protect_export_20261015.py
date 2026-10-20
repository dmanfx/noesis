#!/usr/bin/env python3
"""Fetch the two requested Protect clips using a certificate-pinned local session."""

from __future__ import annotations

import getpass
import hashlib
import http.client
import json
import os
import ssl
import sys
from datetime import datetime
from http.cookies import Morsel, SimpleCookie
from pathlib import Path
from zoneinfo import ZoneInfo


# Python 3.12 rejects the entire Set-Cookie header when it contains this flag.
# UniFi sends Partitioned cookies; uiprotect uses the same compatibility fix.
if "partitioned" not in Morsel._reserved:
    Morsel._reserved["partitioned"] = "Partitioned"
    Morsel._flags.add("partitioned")


HOST = "192.168.3.214"
PORT = 443
CERT_SHA256 = "830fba578c0e87a971cbd65fbe329f1a57e3eb82e9070f3a6b11f2a5fc27f609"
DOWNLOADS = Path.home() / "Downloads"
LOCAL_TZ = ZoneInfo("America/New_York")
END = datetime(2026, 10, 15, 8, 10, 59, tzinfo=LOCAL_TZ)
CAMERAS = [
    ("Family Room", "67fae5c2028cfa03e4096cc7", 3),
    ("Kitchen", "602ebdbd036e7a03e70003eb", 4),
]
MAX_BYTES_PER_CLIP = 1_500 * 1024 * 1024
MAX_LOGIN_RESPONSE_BYTES = 64 * 1024
MAX_BOOTSTRAP_RESPONSE_BYTES = 4 * 1024 * 1024
KEYRING_SCHEMA_NAME = "com.noesis.unifi-protect.local-export"
KEYRING_ATTRIBUTES = {"console": HOST, "purpose": "recording-export"}
KEYRING_LABEL = "UniFi Protect local export at 192.168.3.214"


def load_secret_service():
    import gi

    gi.require_version("Secret", "1")
    from gi.repository import Secret

    schema = Secret.Schema.new(
        KEYRING_SCHEMA_NAME,
        Secret.SchemaFlags.NONE,
        {
            "console": Secret.SchemaAttributeType.STRING,
            "purpose": Secret.SchemaAttributeType.STRING,
        },
    )
    return Secret, schema


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        super().connect()
        peer_cert = self.sock.getpeercert(binary_form=True)
        actual = hashlib.sha256(peer_cert).hexdigest()
        if actual != CERT_SHA256:
            self.close()
            raise ssl.SSLError("NVR certificate fingerprint does not match the inspected certificate")


def open_request(path: str, headers: dict[str, str], body: bytes | None = None):
    connection = PinnedHTTPSConnection(
        HOST,
        PORT,
        timeout=60,
        context=ssl._create_unverified_context(),
    )
    try:
        connection.request("POST" if body is not None else "GET", path, body=body, headers=headers)
        return connection, connection.getresponse()
    except BaseException:
        connection.close()
        raise


def mfa_token_field(payload: dict) -> str | None:
    """Select the observed UniFi OS or legacy SSO challenge contract."""
    errors = payload.get("errors", [])
    legacy_cookie = payload.get("token")
    if (
        isinstance(errors, list)
        and any(isinstance(error, str) and "ubic_2fa_token" in error.lower() for error in errors)
    ) or (isinstance(legacy_cookie, str) and legacy_cookie.startswith("UBIC_2FA=")):
        return "ubic_2fa_token"
    data = payload.get("data")
    meta = payload.get("meta")
    if (
        payload.get("code") == "MFA_AUTH_REQUIRED"
        or (isinstance(data, dict) and data.get("required") in {"2fa", "mfa"})
        or (isinstance(meta, dict) and meta.get("msg") == "api.err.Ubic2faTokenRequired")
    ):
        return "token"
    return None


def validate_protect_session(session_headers: dict[str, str]) -> None:
    """Confirm the cookie grants Protect access without logging private data."""
    probe_headers = dict(session_headers)
    probe_headers["Accept"] = "application/json"
    connection, response = open_request("/proxy/protect/api/bootstrap", probe_headers)
    try:
        if response.status != 200:
            raise RuntimeError(f"The login session could not access Protect (HTTP {response.status})")
        raw_body = response.read(MAX_BOOTSTRAP_RESPONSE_BYTES + 1)
        if len(raw_body) > MAX_BOOTSTRAP_RESPONSE_BYTES:
            raise RuntimeError("The Protect session check exceeded the allowed response size")
        try:
            bootstrap = json.loads(raw_body)
        except (ValueError, UnicodeError):
            raise RuntimeError("The Protect session check did not return valid JSON") from None
        if (
            not isinstance(bootstrap, dict)
            or not isinstance(bootstrap.get("authUserId"), str)
            or not bootstrap["authUserId"]
            or not isinstance(bootstrap.get("nvr"), dict)
            or not isinstance(bootstrap.get("cameras"), list)
        ):
            raise RuntimeError("The Protect session check did not return an authenticated bootstrap")
        visible_ids = {
            camera.get("id") for camera in bootstrap["cameras"] if isinstance(camera, dict)
        }
        if not all(camera_id in visible_ids for _, camera_id, _ in CAMERAS):
            raise RuntimeError("The authenticated Protect account cannot see both requested cameras")
        cookie = SimpleCookie()
        cookie.load(session_headers["Cookie"])
        for key, value in response.getheaders():
            if key.lower() == "set-cookie":
                cookie.load(value)
            elif key.lower() in {"x-csrf-token", "x-updated-csrf-token"}:
                session_headers["x-csrf-token"] = value
        if "UBIC_2FA" in cookie:
            del cookie["UBIC_2FA"]
        session_headers["Cookie"] = "; ".join(f"{item.key}={item.value}" for item in cookie.values())
    finally:
        response.close()
        connection.close()


def authenticate(username: str, password: str, prompt_code=None) -> dict[str, str]:
    """Allow one MFA verification request; keep all challenge data in memory."""
    if prompt_code is None:
        prompt_code = getpass.getpass
    cookie = SimpleCookie()
    csrf_token = None
    login_payload = {"username": username, "password": password, "rememberMe": False}
    try:
        for attempt in range(2):
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if cookie:
                headers["Cookie"] = "; ".join(f"{item.key}={item.value}" for item in cookie.values())
            if csrf_token:
                headers["x-csrf-token"] = csrf_token
            connection, response = open_request(
                "/api/auth/login", headers, json.dumps(login_payload).encode()
            )
            try:
                status = response.status
                cookie_header_count = 0
                partitioned_cookie_seen = False
                for key, value in response.getheaders():
                    if key.lower() == "set-cookie":
                        cookie_header_count += 1
                        partitioned_cookie_seen |= any(
                            attr.strip().lower() == "partitioned" for attr in value.split(";")[1:]
                        )
                        cookie.load(value)
                    elif key.lower() in {"x-csrf-token", "x-updated-csrf-token"}:
                        csrf_token = value
                raw_body = response.read(MAX_LOGIN_RESPONSE_BYTES + 1)
                if len(raw_body) > MAX_LOGIN_RESPONSE_BYTES:
                    raise RuntimeError("The NVR login response exceeded the allowed size")
                try:
                    payload = json.loads(raw_body)
                except (ValueError, UnicodeError):
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                del raw_body
            finally:
                response.close()
                connection.close()

            if status == 200:
                if not any(name in cookie and cookie[name].value for name in ("TOKEN", "UOS_TOKEN")):
                    raise RuntimeError(
                        "Login returned no authenticated UniFi OS session cookie "
                        f"(HTTP 200; Set-Cookie headers={cookie_header_count}; "
                        f"Partitioned flag={partitioned_cookie_seen}; parsed cookies={len(cookie)})"
                    )
                if "UBIC_2FA" in cookie:
                    del cookie["UBIC_2FA"]
                session_headers = {
                    "Cookie": "; ".join(f"{item.key}={item.value}" for item in cookie.values()),
                    "Accept": "video/mp4, application/octet-stream",
                }
                if csrf_token:
                    session_headers["x-csrf-token"] = csrf_token
                validate_protect_session(session_headers)
                return session_headers

            if attempt == 1:
                raise RuntimeError(
                    f"MFA verification failed (HTTP {status}). Rerun for a new challenge and use its latest code."
                )
            token_field = mfa_token_field(payload) if status == 499 else None
            if token_field is None:
                raise RuntimeError(f"Protect login failed (HTTP {status}); no supported MFA challenge was returned")

            data = payload.get("data")
            challenge_cookie = data.get("mfaCookie") if isinstance(data, dict) else None
            if challenge_cookie is None:
                challenge_cookie = payload.get("token")
            if challenge_cookie:
                if not isinstance(challenge_cookie, str) or not challenge_cookie.startswith("UBIC_2FA="):
                    raise RuntimeError("The NVR returned an unsupported MFA challenge cookie")
                challenge = SimpleCookie()
                challenge.load(challenge_cookie)
                if "UBIC_2FA" not in challenge or not challenge["UBIC_2FA"].value:
                    raise RuntimeError("The NVR returned an empty MFA challenge cookie")
                cookie["UBIC_2FA"] = challenge["UBIC_2FA"].value
            del payload, challenge_cookie
            print("UniFi requires MFA. Enter the latest email or authenticator code locally.", flush=True)
            code = prompt_code("UniFi MFA code (hidden; not saved): ").strip()
            if not code or len(code) > 64 or not code.isascii() or not code.isalnum():
                raise RuntimeError("No valid MFA code was entered")
            login_payload[token_field] = code
            del code
        raise RuntimeError("Protect login did not complete")
    finally:
        login_payload.clear()
        cookie.clear()


def main() -> int:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for room, camera_id, preroll_seconds in CAMERAS:
        start = datetime(2026, 10, 15, 7, 45, 0, tzinfo=LOCAL_TZ).timestamp() - preroll_seconds
        end = END.timestamp()
        target = DOWNLOADS / f"{room.replace(' ', '')}_2026-10-15_0745-0810.mp4"
        partial = target.with_suffix(target.suffix + ".part")
        if target.exists() or partial.exists():
            raise RuntimeError(f"Refusing to overwrite an existing output: {target}")
        jobs.append((room, camera_id, int(start * 1000), int(end * 1000), target, partial))

    Secret, schema = load_secret_service()
    cached = Secret.password_lookup_sync(schema, KEYRING_ATTRIBUTES, None)
    username = None
    password = None
    cache_new_login = False
    if cached:
        try:
            saved = json.loads(cached)
            username = saved["username"]
            password = saved["password"]
        except (TypeError, ValueError, KeyError):
            raise RuntimeError("The saved keyring entry is malformed; remove it through the desktop keyring manager")
        finally:
            del cached
        del saved
        print("Using the local Protect login from GNOME Keyring.", flush=True)
    else:
        username = input("Local Protect username: ").strip()
        password = getpass.getpass("Local Protect password (hidden): ")
        if not username or not password:
            raise RuntimeError("Username and password are required")
        save_answer = input("Save this login in GNOME Keyring for future runs? [y/N]: ").strip().casefold()
        cache_new_login = save_answer in {"y", "yes"}
    if not isinstance(username, str) or not username or not isinstance(password, str) or not password:
        raise RuntimeError("Username and password are required")

    print("Checking the pinned NVR certificate and signing in...", flush=True)
    request_headers = authenticate(username, password)
    print("Protect sign-in succeeded.", flush=True)
    if cache_new_login:
        try:
            saved = json.dumps({"username": username, "password": password})
            stored = Secret.password_store_sync(
                schema,
                KEYRING_ATTRIBUTES,
                Secret.COLLECTION_DEFAULT,
                KEYRING_LABEL,
                saved,
                None,
            )
            del saved
            print(
                "Saved the Protect login in GNOME Keyring."
                if stored else "GNOME Keyring declined to save the login; continuing the export.",
                flush=True,
            )
        except Exception:
            print("GNOME Keyring could not save the login; continuing the export.", flush=True)
    del password, username

    try:
        for room, camera_id, start_ms, end_ms, target, partial in jobs:
            query = f"camera={camera_id}&start={start_ms}&end={end_ms}&channel=0"
            print(f"Exporting {room} to {target.name}...", flush=True)
            connection, response = open_request(
                f"/proxy/protect/api/video/export?{query}", request_headers
            )
            try:
                if response.status != 200:
                    raise RuntimeError(f"{room} export failed (HTTP {response.status})")
                content_type = (response.getheader("Content-Type") or "").split(";", 1)[0].lower()
                if content_type not in {"video/mp4", "application/octet-stream", "application/mp4"}:
                    raise RuntimeError(f"{room} export returned unexpected content type: {content_type or 'missing'}")
                length = response.getheader("Content-Length")
                if length and int(length) > MAX_BYTES_PER_CLIP:
                    raise RuntimeError(f"{room} export exceeds the 1.5 GiB per-clip limit")
                count = 0
                next_report = 64 * 1024 * 1024
                with partial.open("xb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > MAX_BYTES_PER_CLIP:
                            raise RuntimeError(f"{room} export exceeded the 1.5 GiB per-clip limit")
                        output.write(block)
                        if count >= next_report:
                            print(f"  {room}: {count / (1024 * 1024):.0f} MiB received", flush=True)
                            next_report += 64 * 1024 * 1024
                if count == 0:
                    raise RuntimeError(f"{room} export was empty")
                os.replace(partial, target)
                print(f"Saved {target} ({count / (1024 * 1024):.1f} MiB)", flush=True)
            finally:
                response.close()
                connection.close()
        print("Both requested exports finished.", flush=True)
        return 0
    finally:
        request_headers.clear()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; any incomplete .part file is retained for inspection.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Export stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
