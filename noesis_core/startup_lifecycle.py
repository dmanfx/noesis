"""Fail-closed ownership transaction for runtime startup resources."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar


_T = TypeVar("_T")


class StartupPhase(str, Enum):
    ASSEMBLING = "assembling"
    PREPARED = "prepared"
    ACTIVATION_ATTEMPTED = "activation_attempted"
    ACTIVE_WAIT_OWNED = "active_wait_owned"
    HANDED_OFF = "handed_off"
    ABORTED = "aborted"
    QUIESCED = "quiesced"


class StartupLifecycleError(RuntimeError):
    pass


class StartupOwnershipAmbiguous(StartupLifecycleError):
    """Native ownership is ambiguous and ingress cleanup evidence is attached."""

    def __init__(
        self,
        message: str,
        *,
        ingress_cleanup: tuple["StartupCleanupResult", ...] = (),
    ) -> None:
        self.ingress_cleanup = tuple(ingress_cleanup)
        super().__init__(str(message))

    @property
    def ingress_quiesced(self) -> bool:
        return all(result.completed for result in self.ingress_cleanup)


class StartupResourceRegistrationError(StartupLifecycleError):
    """An acquired/bound resource was rolled back after ownership setup failed."""

    def __init__(
        self,
        message: str,
        *,
        rollback: tuple["StartupCleanupResult", ...],
    ) -> None:
        self.rollback = tuple(rollback)
        super().__init__(str(message))

    @property
    def rollback_completed(self) -> bool:
        return bool(self.rollback) and all(result.completed for result in self.rollback)


@dataclass(frozen=True)
class StartupCleanupResult:
    name: str
    completed: bool
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class StartupAbortReceipt:
    phase_before: StartupPhase
    phase_after: StartupPhase
    completed: bool
    cleanup: tuple[StartupCleanupResult, ...]


@dataclass(frozen=True)
class _Cleanup:
    name: str
    callback: Callable[[], Any]
    validator: Callable[[Any], bool]
    ingress: bool


def _truthy_result(_result: Any) -> bool:
    return True


class StartupMainGuard:
    """Catch otherwise-unhandled exceptions only while startup owns cleanup.

    The handler is armed from inside the runtime after its transaction and
    watchdog callbacks exist.  It is deliberately disarmed immediately after
    native wait ownership is handed to the normal runtime shutdown path.
    """

    def __init__(self) -> None:
        self._handler: Callable[[BaseException], int] | None = None

    def arm(self, handler: Callable[[BaseException], int]) -> None:
        if not callable(handler):
            raise TypeError("startup main exception handler must be callable")
        self._handler = handler

    def disarm(self) -> None:
        self._handler = None

    def run(self, callback: Callable[[], int]) -> int:
        if not callable(callback):
            raise TypeError("startup main callback must be callable")
        try:
            return int(callback())
        except BaseException as exc:
            handler = self._handler
            if handler is None:
                raise
            return int(handler(exc))
        finally:
            self.disarm()


class RuntimeStartupTransaction:
    """Own reversible startup work until native wait ownership is proven.

    Python-owned dependencies may be unwound only before activation is
    attempted. Once activation begins, only external ingress may close; native
    callback dependencies remain alive for the process-level fail-closed path.
    """

    def __init__(
        self,
        *,
        on_ambiguous: Callable[[str], None],
    ) -> None:
        if not callable(on_ambiguous):
            raise TypeError("on_ambiguous must be callable")
        self._on_ambiguous = on_ambiguous
        self._phase = StartupPhase.ASSEMBLING
        self._cleanup: list[_Cleanup] = []
        self._lock = threading.RLock()

    @property
    def phase(self) -> StartupPhase:
        with self._lock:
            return self._phase

    def register(
        self,
        name: str,
        callback: Callable[[], Any],
        *,
        validator: Callable[[Any], bool] | None = None,
        ingress: bool = False,
    ) -> None:
        normalized = str(name or "").strip()
        if not normalized or not callable(callback):
            raise ValueError("startup cleanup requires a name and callable")
        check = validator or _truthy_result
        if not callable(check):
            raise TypeError("startup cleanup validator must be callable")
        with self._lock:
            if self._phase not in {StartupPhase.ASSEMBLING, StartupPhase.PREPARED}:
                raise StartupLifecycleError(
                    f"cannot register startup cleanup during {self._phase.value}"
                )
            if any(entry.name == normalized for entry in self._cleanup):
                raise StartupLifecycleError(
                    f"startup cleanup is already registered: {normalized}"
                )
            self._cleanup.append(
                _Cleanup(
                    name=normalized,
                    callback=callback,
                    validator=check,
                    ingress=bool(ingress),
                )
            )

    def acquire(
        self,
        name: str,
        factory: Callable[[], _T],
        cleanup: Callable[[_T], Any],
        *,
        validator: Callable[[Any], bool] | None = None,
        ingress: bool = False,
    ) -> _T:
        """Create a resource and atomically place its cleanup in the ledger.

        If registration fails after construction, the just-created resource is
        rolled back immediately and the exact rollback result is attached to
        the raised error.  The resource can therefore never exist unowned.
        """

        if not callable(factory) or not callable(cleanup):
            raise TypeError("startup acquisition requires factory and cleanup callables")
        resource = factory()

        def _cleanup() -> Any:
            return cleanup(resource)

        try:
            self.register(
                name,
                _cleanup,
                validator=validator,
                ingress=ingress,
            )
        except BaseException as exc:
            check = validator or _truthy_result
            rollback = self._run(
                [
                    _Cleanup(
                        name=str(name or "").strip() or "unregistered_resource",
                        callback=_cleanup,
                        validator=check,
                        ingress=bool(ingress),
                    )
                ]
            )
            raise StartupResourceRegistrationError(
                f"startup resource registration failed: {name}",
                rollback=rollback,
            ) from exc
        return resource

    def bind(
        self,
        name: str,
        installer: Callable[[], _T],
        cleanup: Callable[[], Any],
        *,
        validator: Callable[[Any], bool] | None = None,
        ingress: bool = False,
    ) -> _T:
        """Register idempotent rollback before mutating a process-global binding."""

        if not callable(installer) or not callable(cleanup):
            raise TypeError("startup binding requires installer and cleanup callables")
        self.register(
            name,
            cleanup,
            validator=validator,
            ingress=ingress,
        )
        try:
            return installer()
        except BaseException as exc:
            entry = self._pop_cleanup(name)
            rollback = self._run([entry]) if entry is not None else ()
            raise StartupResourceRegistrationError(
                f"startup binding installation failed: {name}",
                rollback=rollback,
            ) from exc

    def _pop_cleanup(self, name: str) -> _Cleanup | None:
        normalized = str(name or "").strip()
        with self._lock:
            for index in range(len(self._cleanup) - 1, -1, -1):
                if self._cleanup[index].name == normalized:
                    return self._cleanup.pop(index)
        return None

    def mark_prepared(self) -> None:
        with self._lock:
            if self._phase is not StartupPhase.ASSEMBLING:
                raise StartupLifecycleError(
                    f"cannot mark prepared from {self._phase.value}"
                )
            self._phase = StartupPhase.PREPARED

    def mark_activation_attempted(self) -> None:
        with self._lock:
            if self._phase not in {StartupPhase.ASSEMBLING, StartupPhase.PREPARED}:
                raise StartupLifecycleError(
                    f"cannot attempt activation from {self._phase.value}"
                )
            self._phase = StartupPhase.ACTIVATION_ATTEMPTED

    def claim_wait_owner(self, thread: threading.Thread | None) -> None:
        with self._lock:
            if self._phase is not StartupPhase.ACTIVATION_ATTEMPTED:
                raise StartupLifecycleError(
                    f"cannot claim wait ownership from {self._phase.value}"
                )
            if thread is None or not thread.is_alive():
                self._abort_ambiguous_locked("Service Maker wait ownership was not proven")
            self._phase = StartupPhase.ACTIVE_WAIT_OWNED

    def handoff_to_runtime(self) -> None:
        with self._lock:
            if self._phase is not StartupPhase.ACTIVE_WAIT_OWNED:
                raise StartupLifecycleError(
                    f"cannot hand off startup ownership from {self._phase.value}"
                )
            self._cleanup.clear()
            self._phase = StartupPhase.HANDED_OFF

    def mark_quiesced(self) -> None:
        with self._lock:
            if self._phase is not StartupPhase.HANDED_OFF:
                raise StartupLifecycleError(
                    f"cannot mark runtime quiesced from {self._phase.value}"
                )
            self._phase = StartupPhase.QUIESCED

    @staticmethod
    def _run(entries: list[_Cleanup]) -> tuple[StartupCleanupResult, ...]:
        results: list[StartupCleanupResult] = []
        for entry in reversed(entries):
            try:
                value = entry.callback()
                completed = bool(entry.validator(value))
                results.append(
                    StartupCleanupResult(
                        name=entry.name,
                        completed=completed,
                        error_type=None if completed else "CleanupReceiptRejected",
                        error_message=None
                        if completed
                        else "cleanup receipt did not prove completion",
                    )
                )
            except BaseException as exc:
                results.append(
                    StartupCleanupResult(
                        name=entry.name,
                        completed=False,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
        return tuple(results)

    def abort_reversible(self) -> StartupAbortReceipt:
        with self._lock:
            phase_before = self._phase
            if phase_before is StartupPhase.ABORTED:
                return StartupAbortReceipt(
                    phase_before=phase_before,
                    phase_after=phase_before,
                    completed=True,
                    cleanup=(),
                )
            if phase_before in {
                StartupPhase.ACTIVATION_ATTEMPTED,
                StartupPhase.ACTIVE_WAIT_OWNED,
                StartupPhase.HANDED_OFF,
                StartupPhase.QUIESCED,
            }:
                raise StartupLifecycleError(
                    f"startup is no longer reversibly abortable: {phase_before.value}"
                )
            entries = list(self._cleanup)
            self._cleanup.clear()
            self._phase = StartupPhase.ABORTED
        cleanup = self._run(entries)
        return StartupAbortReceipt(
            phase_before=phase_before,
            phase_after=StartupPhase.ABORTED,
            completed=all(result.completed for result in cleanup),
            cleanup=cleanup,
        )

    def abort_ambiguous(self, reason: str) -> None:
        with self._lock:
            self._abort_ambiguous_locked(reason)

    def _abort_ambiguous_locked(self, reason: str) -> None:
        if self._phase not in {
            StartupPhase.ACTIVATION_ATTEMPTED,
            StartupPhase.ACTIVE_WAIT_OWNED,
        }:
            raise StartupLifecycleError(
                f"startup ownership is not activation-ambiguous: {self._phase.value}"
            )
        message = str(reason or "startup native ownership is ambiguous").strip()
        callback_error: BaseException | None = None
        try:
            self._on_ambiguous(message)
        except BaseException as exc:
            callback_error = exc
        ingress = [entry for entry in self._cleanup if entry.ingress]
        cleanup = self._run(ingress)
        error = StartupOwnershipAmbiguous(
            message,
            ingress_cleanup=cleanup,
        )
        if callback_error is not None:
            raise error from callback_error
        raise error


__all__ = [
    "RuntimeStartupTransaction",
    "StartupMainGuard",
    "StartupAbortReceipt",
    "StartupCleanupResult",
    "StartupLifecycleError",
    "StartupOwnershipAmbiguous",
    "StartupPhase",
    "StartupResourceRegistrationError",
]
