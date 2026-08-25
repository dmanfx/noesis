from __future__ import annotations

import asyncio
import json

from websocket_server import WebSocketServer


class _Socket:
    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        self.sent: list[object] = []
        self.remote_address = ("127.0.0.1", 6008)
        self._messages = iter(json.dumps(item) for item in (messages or []))

    async def send(self, payload: object) -> None:
        self.sent.append(payload)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _BlockingSocket(_Socket):
    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        super().__init__(messages)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __anext__(self) -> str:
        self.entered.set()
        if self._messages is not None:
            try:
                return next(self._messages)
            except StopIteration:
                pass
        await self.release.wait()
        raise StopAsyncIteration


def _toggle_messages(socket: _Socket) -> list[dict[str, object]]:
    return [
        json.loads(str(payload))
        for payload in socket.sent
        if isinstance(payload, str)
        and json.loads(str(payload)).get("toggle_name")
        == WebSocketServer.LOCALIZATION_DETAILS_TOGGLE
    ]


def test_localization_details_is_off_by_default_and_connection_scoped() -> None:
    callbacks: list[tuple[str, bool]] = []

    async def _run() -> tuple[WebSocketServer, _BlockingSocket, _Socket]:
        server = WebSocketServer(
            stats_callback=None,
            toggle_callback=lambda name, enabled: callbacks.append(
                (name, enabled)
            ),
        )
        observer = _BlockingSocket()
        observer_task = asyncio.create_task(server.handle_client(observer))
        await asyncio.wait_for(observer.entered.wait(), timeout=1.0)

        requester = _Socket(
            [
                {
                    "type": "set_vis_toggle",
                    "toggle_name": WebSocketServer.LOCALIZATION_DETAILS_TOGGLE,
                    "enabled": True,
                }
            ]
        )
        await server.handle_client(requester)
        assert server.localization_details_enabled is False
        observer.release.set()
        await observer_task
        return server, observer, requester

    server, observer, requester = asyncio.run(_run())
    assert server._localization_details_requesters == set()
    assert callbacks == [
        (WebSocketServer.LOCALIZATION_DETAILS_TOGGLE, True),
        (WebSocketServer.LOCALIZATION_DETAILS_TOGGLE, False),
    ]
    assert _toggle_messages(requester) == [
        {
            "type": "toggle_update",
            "toggle_name": WebSocketServer.LOCALIZATION_DETAILS_TOGGLE,
            "enabled": False,
        },
        {
            "type": "toggle_update",
            "toggle_name": WebSocketServer.LOCALIZATION_DETAILS_TOGGLE,
            "enabled": True,
        },
    ]
    assert [message["enabled"] for message in _toggle_messages(observer)] == [
        False,
        True,
        False,
    ]


def test_one_requester_keeps_details_enabled_until_it_disconnects() -> None:
    callbacks: list[bool] = []
    server = WebSocketServer(
        stats_callback=None,
        toggle_callback=lambda name, enabled: callbacks.append(bool(enabled)),
    )
    first = object()
    second = object()

    assert server.localization_details_enabled is False
    assert server._set_localization_details_preference(first, True) is True
    assert server._set_localization_details_preference(second, True) is False
    assert server.localization_details_enabled is True
    assert server._remove_localization_details_client(first) is False
    assert server.localization_details_enabled is True
    assert server._remove_localization_details_client(second) is True
    assert server.localization_details_enabled is False
    assert callbacks == [True, False]
