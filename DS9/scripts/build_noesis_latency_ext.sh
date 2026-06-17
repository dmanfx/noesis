#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/build_native_ext_ds9.sh" \
  noesis_latency_ext \
  "${ROOT}/native/noesis_latency_ext.cpp"
