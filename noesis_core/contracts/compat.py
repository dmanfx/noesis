from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractCompatibilityError(ValueError):
    contract: str
    received_version: int
    supported_versions: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"unsupported {self.contract} contract version {self.received_version}; "
            f"supported={list(self.supported_versions)}"
        )


def require_supported_contract(
    payload: Mapping[str, Any],
    *,
    contract: str,
    supported_versions: tuple[int, ...],
) -> int:
    received_contract = str(payload.get("contract") or "")
    try:
        received_version = int(payload.get("contract_version"))
    except Exception as exc:
        raise ContractCompatibilityError(contract, -1, supported_versions) from exc
    if received_contract != contract or received_version not in supported_versions:
        raise ContractCompatibilityError(contract, received_version, supported_versions)
    return received_version
