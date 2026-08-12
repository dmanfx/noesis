#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/build_native_ext_ds9.sh" \
  noesis_analytics_meta_ext \
  "${ROOT}/native/noesis_analytics_meta_ext.cpp"
