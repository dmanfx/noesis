from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "DS9" / "noesis" / "ds9_runtime.py"
THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _bootstrap_environment(
    extra_env: dict[str, str] | None = None,
) -> dict[str, str | None]:
    env = os.environ.copy()
    for name in THREAD_ENV_VARS + ("NOESIS_CPU_MATH_THREADS",):
        env.pop(name, None)
    env.update(extra_env or {})
    code = f"""
import json
import os
from pathlib import Path

source = Path({str(LAUNCHER)!r}).read_text(encoding="utf-8")
prefix = source.split("DS9_ROOT =", 1)[0]
namespace = {{"__file__": {str(LAUNCHER)!r}}}
exec(compile(prefix, {str(LAUNCHER)!r}, "exec"), namespace)
print(json.dumps({{name: os.environ.get(name) for name in namespace["_CPU_MATH_THREAD_ENV_VARS"]}}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout.strip())


def test_ds9_runtime_caps_cpu_math_threads_before_runtime_imports() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert source.index("_configure_cpu_math_threads()") < source.index(
        "from noesis.runtime_paths"
    )
    assert _bootstrap_environment() == {name: "1" for name in THREAD_ENV_VARS}


def test_ds9_runtime_preserves_explicit_cpu_math_thread_values() -> None:
    values = _bootstrap_environment(
        {
            "NOESIS_CPU_MATH_THREADS": "3",
            "OPENBLAS_NUM_THREADS": "7",
        }
    )
    assert values["OPENBLAS_NUM_THREADS"] == "7"
    assert all(values[name] == "3" for name in THREAD_ENV_VARS[1:])
