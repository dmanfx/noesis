from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_static_prep_launcher_owns_shared_import_path() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        ["bash", "DS9/scripts/run_static_prep_checks.sh"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK] DS9 static prep checks passed" in result.stdout
