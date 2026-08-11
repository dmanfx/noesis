"""DS9 telemetry adapters plus single-sourced telemetry product logic.

The person-ground estimator and canonical BEV renderer remain single-sourced in
the repository-level ``noesis.telemetry`` package. Extending this package path
lets the DS9 SDK adapter import those implementations without local shadows;
DS9-specific modules in this directory remain first in the adapter search path.
"""

from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)
