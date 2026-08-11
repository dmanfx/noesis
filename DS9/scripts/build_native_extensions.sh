#!/usr/bin/env bash
set -euo pipefail

DS9_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep one authoritative compiler/linker contract.  The per-module builder
# owns all DS9/CUDA/header checks and the aggregate script owns the exact six
# extension kinds; this compatibility entry point only delegates.
exec "${DS9_ROOT}/scripts/build_all_native_ds9.sh"
