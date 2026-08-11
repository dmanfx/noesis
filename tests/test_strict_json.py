from __future__ import annotations

import pytest

from noesis_core.strict_json import StrictJSONError, strict_json_loads


@pytest.mark.parametrize(
    "payload",
    (
        b'{"selector":"first","selector":"second"}',
        b'{"outer":{"verdict":false,"verdict":true}}',
    ),
)
def test_strict_json_rejects_duplicate_keys_at_every_depth(payload: bytes) -> None:
    with pytest.raises(StrictJSONError, match="duplicate JSON object key") as captured:
        strict_json_loads(payload, label="authority document")
    assert captured.value.reason == "duplicate_key"


@pytest.mark.parametrize(
    "payload",
    (
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
        b'{"value":1e309}',
        b'{"value":-1e309}',
    ),
)
def test_strict_json_rejects_nonfinite_and_overflow_numbers(payload: bytes) -> None:
    with pytest.raises(StrictJSONError, match="non-finite JSON number") as captured:
        strict_json_loads(payload, label="authority document")
    assert captured.value.reason == "nonfinite_number"


def test_strict_json_rejects_invalid_utf8_without_echoing_input() -> None:
    secret = b'\xffowner-secret-value'
    with pytest.raises(StrictJSONError) as captured:
        strict_json_loads(secret, label="private authority")
    assert captured.value.reason == "invalid_utf8"
    assert "owner-secret-value" not in str(captured.value)


def test_strict_json_duplicate_error_does_not_echo_key_or_values() -> None:
    secret_key = "owner_secret_selector"
    secret_value = "private-resident-token"
    payload = (
        f'{{"{secret_key}":"first","{secret_key}":"{secret_value}"}}'
    )
    with pytest.raises(StrictJSONError) as captured:
        strict_json_loads(payload, label="private authority")
    message = str(captured.value)
    assert secret_key not in message
    assert secret_value not in message


def test_strict_json_exposes_only_a_safe_invalid_json_reason() -> None:
    with pytest.raises(StrictJSONError) as captured:
        strict_json_loads('{"secret":"unterminated}', label="private authority")
    assert captured.value.reason == "invalid_json"
    assert "unterminated" not in str(captured.value)


def test_strict_json_normalizes_excessive_nesting_to_safe_invalid_reason() -> None:
    payload = "[" * 20_000 + "0" + "]" * 20_000
    with pytest.raises(StrictJSONError) as captured:
        strict_json_loads(payload, label="private authority")
    assert captured.value.reason == "invalid_json"
    assert "recursion" not in str(captured.value).lower()


def test_strict_json_preserves_finite_json_types() -> None:
    assert strict_json_loads(
        b'{"enabled":true,"count":2,"score":0.25,"items":[null,"ok"]}',
        label="authority document",
    ) == {
        "enabled": True,
        "count": 2,
        "score": 0.25,
        "items": [None, "ok"],
    }
