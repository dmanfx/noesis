#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the historical entrypoint, but route it through the same explicit
# secondary-daemon, external-artifact, idle-GPU, timeout, and memory guards as
# every other canonical DS9 engine.
exec "${SCRIPT_DIR}/run_canonical_engine_maintenance.sh" --only mapanything "$@"
