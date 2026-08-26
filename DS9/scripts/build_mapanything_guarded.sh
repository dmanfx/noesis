#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the focused entrypoint, but route it through the canonical native-host
# maintenance path used by every active DS9.1 engine.
exec "${SCRIPT_DIR}/run_canonical_engine_maintenance_host.sh" --only mapanything "$@"
