"""Strict, SDK-neutral control for orderly Service Maker pipeline EOS."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


class OrderlyEosError(RuntimeError):
    """Raised when the native EOS bridge is absent or rejects a request."""


@dataclass(frozen=True)
class OrderlyEosEvidence:
    component_name: str
    request_sequence: int
    accepted_sequence: int
    last_request_ok: bool


SYNTHETIC_STUB_BACKEND: Final = "synthetic_stub"
_SYNTHETIC_STUB_LIFECYCLE_EVIDENCE: Final = MappingProxyType(
    {
        "backend": SYNTHETIC_STUB_BACKEND,
        "native_runtime": False,
        "promotable": False,
    }
)


@dataclass(frozen=True, slots=True)
class SyntheticStubEosMessage:
    """Typed EOS callback used only by the explicit pure-Python test backend.

    This deliberately is not, and must never masquerade as, a Service Maker
    ``EOSMessage``.  Runtime message handlers may recognize it only after also
    checking the emitting pipeline's exact non-promotable lifecycle evidence.
    """

    request_sequence: int
    backend: str = SYNTHETIC_STUB_BACKEND
    native_runtime: bool = False
    promotable: bool = False


def synthetic_stub_lifecycle_evidence() -> dict[str, object]:
    """Return a fresh, explicit non-native/non-promotable evidence record."""

    return dict(_SYNTHETIC_STUB_LIFECYCLE_EVIDENCE)


def is_synthetic_stub_pipeline(pipeline: Any) -> bool:
    """Recognize only the exact explicit synthetic backend evidence contract."""

    evidence = getattr(pipeline, "lifecycle_evidence", None)
    return isinstance(evidence, Mapping) and dict(evidence) == dict(
        _SYNTHETIC_STUB_LIFECYCLE_EVIDENCE
    )


def validate_synthetic_stub_eos_message(pipeline: Any, message: Any) -> int:
    """Validate the exact synthetic callback/pipeline pair and return its sequence."""

    if type(message) is not SyntheticStubEosMessage:
        raise TypeError("message is not an exact SyntheticStubEosMessage")
    if not is_synthetic_stub_pipeline(pipeline):
        raise OrderlyEosError(
            "synthetic EOS callback came from a pipeline without exact stub evidence"
        )
    if (
        message.backend != SYNTHETIC_STUB_BACKEND
        or message.native_runtime is not False
        or message.promotable is not False
    ):
        raise OrderlyEosError("synthetic EOS callback lifecycle marker drifted")
    sequence = message.request_sequence
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise OrderlyEosError("synthetic EOS callback sequence is not an integer")
    if sequence <= 0 or sequence > 0xFFFFFFFF:
        raise OrderlyEosError("synthetic EOS callback sequence is outside uint32")
    return sequence


def pipeline_expects_finite_source_eos(pipeline: Any) -> bool:
    """Return true only for an explicitly finite, non-looping local-file graph."""

    config = getattr(pipeline, "config", None)
    components = getattr(pipeline, "components", None)
    if not isinstance(config, Mapping) or not isinstance(components, dict):
        return False

    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        return False
    for source in sources:
        if not isinstance(source, Mapping):
            return False
        uri = str(source.get("uri") or "").strip().lower()
        if not uri.startswith("file:"):
            return False

    streammux = components.get("streammux")
    streammux_config = getattr(streammux, "config", None)
    if not isinstance(streammux_config, Mapping):
        return False
    live_source = streammux_config.get(
        "live-source", streammux_config.get("live_source", 1)
    )
    try:
        is_live = bool(int(live_source))
    except Exception:
        is_live = str(live_source).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if is_live or bool(streammux_config.get("file-loop", False)):
        return False
    if bool(streammux_config.get("drop-pipeline-eos", False)):
        return False

    for component in components.values():
        component_config = getattr(component, "config", None)
        if isinstance(component_config, Mapping) and bool(
            component_config.get("file-loop", False)
        ):
            return False
    return True


def request_orderly_eos(
    pipeline: Any,
    *,
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.01,
) -> OrderlyEosEvidence:
    """Push and synchronously acknowledge one downstream EOS request.

    ``pipeline`` is intentionally duck-typed so the shared contract has no
    import-time dependency on pyservicemaker or either DeepStream runtime.
    Returning from this function proves only that the downstream event path
    accepted EOS.  Callers must separately require the Service Maker EOS
    callback and ``Pipeline.wait()`` completion before releasing callback-owned
    resources.
    """

    component_name = str(
        getattr(pipeline, "shutdown_eos_component_name", "orderly_eos_control")
    )
    components = getattr(pipeline, "components", None)
    if not isinstance(components, dict):
        raise OrderlyEosError("pipeline component registry is unavailable")
    component = components.get(component_name)
    if component is None or getattr(component, "element", None) != "noesiseos":
        raise OrderlyEosError(
            f"required noesiseos component is missing: {component_name}"
        )

    ds_pipeline = getattr(pipeline, "ds_pipeline", None)
    if ds_pipeline is None:
        raise OrderlyEosError("Service Maker pipeline is unavailable")
    try:
        node = ds_pipeline[component_name]
        current_request = int(node.get("request-sequence"))
        current_accepted = int(node.get("accepted-sequence"))
    except Exception as exc:
        raise OrderlyEosError("failed to read orderly EOS bridge state") from exc

    request_sequence = max(current_request, current_accepted) + 1
    if request_sequence > 0xFFFFFFFF:
        raise OrderlyEosError("orderly EOS request sequence is exhausted")

    try:
        node.set({"request-sequence": request_sequence})
    except Exception as exc:
        raise OrderlyEosError("orderly EOS bridge request failed") from exc

    deadline = time.monotonic() + max(0.01, float(timeout_s))
    observed_request = request_sequence
    accepted_sequence = current_accepted
    last_request_ok = False
    while True:
        try:
            observed_request = int(node.get("request-sequence"))
            accepted_sequence = int(node.get("accepted-sequence"))
            last_request_ok = bool(node.get("last-request-ok"))
        except Exception as exc:
            raise OrderlyEosError("failed to read orderly EOS acknowledgement") from exc
        if (
            observed_request == request_sequence
            and accepted_sequence == request_sequence
            and last_request_ok
        ):
            break
        if observed_request != request_sequence or accepted_sequence > request_sequence:
            raise OrderlyEosError(
                "orderly EOS bridge returned inconsistent sequence state "
                f"(requested={request_sequence}, observed={observed_request}, "
                f"accepted={accepted_sequence})"
            )
        if time.monotonic() >= deadline:
            raise OrderlyEosError(
                "orderly EOS bridge acknowledgement timed out "
                f"for request {request_sequence} (observed={observed_request}, "
                f"accepted={accepted_sequence}, ok={last_request_ok})"
            )
        time.sleep(max(0.001, float(poll_interval_s)))

    return OrderlyEosEvidence(
        component_name=component_name,
        request_sequence=request_sequence,
        accepted_sequence=accepted_sequence,
        last_request_ok=last_request_ok,
    )
