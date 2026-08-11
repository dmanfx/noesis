#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python3 - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"Noesis CPU contract gates require Python 3.12; found {sys.version.split()[0]}"
    )
PY

command -v rg >/dev/null 2>&1 || {
  echo "Noesis CPU contract gates require ripgrep (rg)." >&2
  exit 1
}

python3 scripts/export_noesis_core_schemas.py --check
python3 scripts/check_agents_docs_consistency.py
bash DS9/scripts/run_static_prep_checks.sh
pytest -q
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" pytest -q DS9/tests
git diff --check
