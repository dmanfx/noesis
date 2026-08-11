#!/usr/bin/env python3
"""
Replay a captured browser WebRTC SDP offer against the Noesis H.264-AU WebRTC gateway.

This is an SDP-level regression gate to ensure the gateway never replies with
`a=inactive` for the video m= section on browser-style offers.

Notes:
  - This does NOT attempt to complete ICE/DTLS/media (captured offers contain
    browser-specific ICE credentials and DTLS fingerprints).
  - Use `scripts/webrtc_gateway_smoke_test.py` to validate RTP + decoded frames.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import websockets

from noesis_core.private_paths import (
    PrivatePathError,
    ensure_private_directory,
    read_private_file,
    validate_private_file,
)


MAX_OFFER_BYTES = 1024 * 1024


def _sdp_video_direction(sdp: str) -> Optional[str]:
    in_video = False
    for line in (sdp or "").splitlines():
        line = line.strip()
        if line.startswith("m="):
            in_video = line.startswith("m=video ")
            continue
        if not in_video:
            continue
        if line in ("a=sendonly", "a=recvonly", "a=sendrecv", "a=inactive"):
            return line.split("=", 1)[1]
    return None


def _default_offer_path() -> Optional[Path]:
    configured = os.environ.get("NOESIS_WEBRTC_OFFER_DIR", "").strip()
    offer_dir = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".local" / "state" / "noesis" / "webrtc_offers"
    )
    if not offer_dir.exists():
        return None
    private_dir = ensure_private_directory(offer_dir, label="WebRTC offer directory")
    offers = [
        validate_private_file(path, label="WebRTC offer")
        for path in private_dir.glob("offer_*.sdp")
    ]
    offers.sort(key=lambda path: path.lstat().st_mtime_ns, reverse=True)
    return offers[0] if offers else None


async def _run(ws_url: str, offer_sdp: str, timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "webrtc_offer", "sdp": offer_sdp}))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("FAIL: timed out waiting for webrtc_answer/webrtc_error", file=sys.stderr)
                return 2

            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            msg_type = msg.get("type")
            if msg_type == "webrtc_error":
                print(f"FAIL: server webrtc_error: {msg.get('error')}", file=sys.stderr)
                return 3

            if msg_type != "webrtc_answer":
                continue

            answer_sdp = msg.get("sdp") or ""
            direction = _sdp_video_direction(answer_sdp)
            if direction != "sendonly":
                print(f"FAIL: answer video_direction={direction!r} (expected 'sendonly')", file=sys.stderr)
                return 4

            if "m=video" not in answer_sdp:
                print("FAIL: answer SDP missing m=video", file=sys.stderr)
                return 5

            print("PASS: answer has video_direction='sendonly'")
            return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket signaling URL")
    ap.add_argument("--offer", type=str, default=None, help="Path to captured offer SDP (.sdp)")
    ap.add_argument("--timeout", type=float, default=5.0, help="Timeout seconds")
    args = ap.parse_args()

    try:
        offer_path = (
            validate_private_file(Path(args.offer), label="WebRTC offer")
            if args.offer
            else _default_offer_path()
        )
    except PrivatePathError as exc:
        print(f"FAIL: unsafe WebRTC offer path: {exc}", file=sys.stderr)
        return 1
    if offer_path is None:
        print(
            "FAIL: no offer provided and no owner-protected captured offers found",
            file=sys.stderr,
        )
        return 1
    try:
        offer_sdp = read_private_file(
            offer_path,
            label="WebRTC offer",
            max_bytes=MAX_OFFER_BYTES,
        ).decode("utf-8")
    except (PrivatePathError, UnicodeDecodeError) as exc:
        print(f"FAIL: unable to read WebRTC offer safely: {exc}", file=sys.stderr)
        return 1
    if not offer_sdp.strip():
        print(f"FAIL: offer file is empty: {offer_path}", file=sys.stderr)
        return 1

    print(f"Replaying offer: {offer_path}")
    return asyncio.run(_run(args.ws, offer_sdp, float(args.timeout)))


if __name__ == "__main__":
    raise SystemExit(main())
