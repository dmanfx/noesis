"""Small runtime-neutral receipts for MapAnything worker lifecycle barriers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MapAnythingIdleReceipt:
    """Proof that a non-closing worker barrier observed no admitted work."""

    active_captures: int
    unfinished_tasks: int
    worker_started: bool
    worker_alive: bool
    accepting: bool

    def __post_init__(self) -> None:
        if type(self.active_captures) is not int or self.active_captures < 0:  # noqa: E721
            raise ValueError("active_captures must be a non-negative integer")
        if type(self.unfinished_tasks) is not int or self.unfinished_tasks < 0:  # noqa: E721
            raise ValueError("unfinished_tasks must be a non-negative integer")
        if self.active_captures != 0 or self.unfinished_tasks != 0:
            raise ValueError("an idle receipt cannot contain unfinished work")
        if (
            type(self.worker_started) is not bool  # noqa: E721
            or type(self.worker_alive) is not bool  # noqa: E721
            or type(self.accepting) is not bool  # noqa: E721
        ):
            raise ValueError("worker lifecycle flags must be booleans")
        if self.worker_alive and not self.worker_started:
            raise ValueError("an unstarted worker cannot be alive")
        if not self.accepting:
            raise ValueError(
                "wait_idle is a non-closing barrier and requires admission"
            )


__all__ = ["MapAnythingIdleReceipt"]
