"""Runtime-neutral Noesis product core.

This package must remain importable without DeepStream, Service Maker, CUDA,
GStreamer, or browser dependencies. The DS9.1 adapter consumes the contracts
and domain services from here.
"""

from .contracts import CONTRACT_MODELS
from .replay import ReplayArchive, ReplayHeader, ReplayValidationError, read_replay, write_replay
from .strict_json import StrictJSONError, StrictJSONReason, strict_json_loads

__all__ = [
    "CONTRACT_MODELS",
    "ReplayArchive",
    "ReplayHeader",
    "ReplayValidationError",
    "StrictJSONError",
    "StrictJSONReason",
    "read_replay",
    "strict_json_loads",
    "write_replay",
]
