from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sanity_check_v3dt_calibration.py"
SPEC = importlib.util.spec_from_file_location("v3dt_sanity_paths_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sanity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sanity
SPEC.loader.exec_module(sanity)


def test_ds9_repo_relative_paths_resolve_from_the_real_repository_root() -> None:
    ds9_pipeline = REPO_ROOT / "DS9" / "config" / "infer_v3dt.yaml"

    assert sanity.resolve_like_noesis(
        ds9_pipeline, "DS9/config/v3dt/nvtracker_v3dt.yaml"
    ) == REPO_ROOT / "DS9" / "config" / "v3dt" / "nvtracker_v3dt.yaml"


def test_local_relative_path_still_resolves_beside_its_owner_yaml() -> None:
    owner = REPO_ROOT / "DS9" / "config" / "v3dt" / "owner.yaml"
    assert sanity.resolve_like_noesis(owner, "camInfo_living-room.yml") == (
        owner.parent / "camInfo_living-room.yml"
    )
