from __future__ import annotations

import threading

import pytest

from noesis_core.startup_lifecycle import (
    RuntimeStartupTransaction,
    StartupLifecycleError,
    StartupMainGuard,
    StartupOwnershipAmbiguous,
    StartupPhase,
    StartupResourceRegistrationError,
)


def test_reversible_abort_is_exactly_once_and_reverse_ordered() -> None:
    calls: list[str] = []
    transaction = RuntimeStartupTransaction(on_ambiguous=lambda _reason: None)
    transaction.register("storage", lambda: calls.append("storage"))
    transaction.register("world", lambda: calls.append("world"))
    transaction.register("identity", lambda: calls.append("identity"))

    receipt = transaction.abort_reversible()

    assert calls == ["identity", "world", "storage"]
    assert receipt.completed is True
    assert [row.name for row in receipt.cleanup] == ["identity", "world", "storage"]
    assert transaction.abort_reversible().cleanup == ()
    assert calls == ["identity", "world", "storage"]


def test_cleanup_receipt_rejection_is_visible() -> None:
    transaction = RuntimeStartupTransaction(on_ambiguous=lambda _reason: None)
    transaction.register(
        "storage",
        lambda: {"completed": False},
        validator=lambda receipt: bool(receipt["completed"]),
    )

    receipt = transaction.abort_reversible()

    assert receipt.completed is False
    assert receipt.cleanup[0].error_type == "CleanupReceiptRejected"


def test_activation_ambiguity_closes_only_ingress_and_never_dependencies() -> None:
    calls: list[str] = []
    transaction = RuntimeStartupTransaction(
        on_ambiguous=lambda reason: calls.append(f"ambiguous:{reason}")
    )
    transaction.register("storage", lambda: calls.append("storage"))
    transaction.register(
        "websocket",
        lambda: calls.append("websocket"),
        ingress=True,
    )
    transaction.mark_prepared()
    transaction.mark_activation_attempted()

    with pytest.raises(
        StartupOwnershipAmbiguous,
        match="partial activation",
    ) as exc_info:
        transaction.abort_ambiguous("partial activation")

    assert calls == ["ambiguous:partial activation", "websocket"]
    assert exc_info.value.ingress_quiesced is True
    assert [row.name for row in exc_info.value.ingress_cleanup] == ["websocket"]


def test_wait_owner_is_required_before_handoff() -> None:
    reasons: list[str] = []
    transaction = RuntimeStartupTransaction(on_ambiguous=reasons.append)
    transaction.mark_activation_attempted()

    with pytest.raises(StartupOwnershipAmbiguous, match="wait ownership"):
        transaction.claim_wait_owner(None)
    assert reasons == ["Service Maker wait ownership was not proven"]


def test_live_wait_owner_allows_handoff_and_normal_quiescence() -> None:
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    transaction = RuntimeStartupTransaction(on_ambiguous=lambda _reason: None)
    try:
        transaction.mark_prepared()
        transaction.mark_activation_attempted()
        transaction.claim_wait_owner(thread)
        transaction.handoff_to_runtime()
        assert transaction.phase is StartupPhase.HANDED_OFF
        with pytest.raises(StartupLifecycleError, match="no longer reversibly"):
            transaction.abort_reversible()
        transaction.mark_quiesced()
        assert transaction.phase is StartupPhase.QUIESCED
    finally:
        release.set()
        thread.join(timeout=1.0)


def test_acquire_rolls_back_resource_when_registration_fails() -> None:
    calls: list[str] = []
    resource = object()
    transaction = RuntimeStartupTransaction(on_ambiguous=lambda _reason: None)
    transaction.register("resource", lambda: calls.append("original"))

    with pytest.raises(StartupResourceRegistrationError) as exc_info:
        transaction.acquire(
            "resource",
            lambda: resource,
            lambda value: calls.append("rollback") if value is resource else None,
        )

    assert calls == ["rollback"]
    assert exc_info.value.rollback_completed is True
    receipt = transaction.abort_reversible()
    assert receipt.completed is True
    assert calls == ["rollback", "original"]


def test_acquire_exposes_unproven_registration_rollback() -> None:
    transaction = RuntimeStartupTransaction(on_ambiguous=lambda _reason: None)
    transaction.mark_activation_attempted()

    with pytest.raises(StartupResourceRegistrationError) as exc_info:
        transaction.acquire(
            "late",
            object,
            lambda _resource: False,
            validator=bool,
        )

    assert exc_info.value.rollback_completed is False
    assert exc_info.value.rollback[0].name == "late"
    assert exc_info.value.rollback[0].completed is False


def test_bind_registers_cleanup_before_install_and_rolls_back_once() -> None:
    calls: list[str] = []
    transaction = RuntimeStartupTransaction(on_ambiguous=lambda _reason: None)
    transaction.register("storage", lambda: calls.append("storage"))

    def _install() -> None:
        calls.append("install")
        raise RuntimeError("partial binding")

    with pytest.raises(StartupResourceRegistrationError) as exc_info:
        transaction.bind(
            "api",
            _install,
            lambda: calls.append("api_cleanup"),
        )

    assert exc_info.value.rollback_completed is True
    assert calls == ["install", "api_cleanup"]
    transaction.abort_reversible()
    assert calls == ["install", "api_cleanup", "storage"]


def test_bind_never_installs_when_cleanup_registration_is_rejected() -> None:
    calls: list[str] = []
    transaction = RuntimeStartupTransaction(on_ambiguous=lambda _reason: None)
    transaction.register("api", lambda: calls.append("existing"))

    with pytest.raises(StartupLifecycleError, match="already registered"):
        transaction.bind(
            "api",
            lambda: calls.append("install"),
            lambda: calls.append("cleanup"),
        )

    assert calls == []


def test_ambiguity_receipt_rejects_unproven_ingress_cleanup() -> None:
    transaction = RuntimeStartupTransaction(on_ambiguous=lambda _reason: None)
    transaction.register(
        "websocket",
        lambda: False,
        validator=bool,
        ingress=True,
    )
    transaction.mark_activation_attempted()

    with pytest.raises(StartupOwnershipAmbiguous) as exc_info:
        transaction.abort_ambiguous("activation uncertain")

    assert exc_info.value.ingress_quiesced is False
    assert exc_info.value.ingress_cleanup[0].error_type == "CleanupReceiptRejected"


def test_startup_main_guard_handles_only_while_armed() -> None:
    guard = StartupMainGuard()
    observed: list[str] = []
    guard.arm(lambda exc: observed.append(type(exc).__name__) or 17)

    assert guard.run(lambda: (_ for _ in ()).throw(RuntimeError("assembly"))) == 17
    assert observed == ["RuntimeError"]

    with pytest.raises(RuntimeError, match="after handoff"):
        guard.run(
            lambda: (_ for _ in ()).throw(RuntimeError("after handoff"))
        )
