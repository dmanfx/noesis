from __future__ import annotations

from importlib import import_module
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

SHARED_MODULES = (
    "noesis.adapters.mapanything",
    "noesis.calibration.depth_registration",
    "noesis.calibration.depth_registration_builder",
    "noesis.calibration.geometry",
    "noesis.calibration.manager",
    "noesis.metadata.depth_result",
    "noesis.metadata.object_depth",
    "noesis.metadata.pose_features",
    "noesis.models",
    "noesis.semantic_capture.manager",
    "noesis.server.analytics_api",
    "noesis.server.boundary_metrics",
    "noesis.server.depth_api",
    "noesis.server.reid_api",
    "noesis.server.scene_prior_api",
    "noesis.server.semantic_seg_api",
    "noesis.server.websocket",
    "noesis.telemetry.motion_smoothing",
    "noesis.yolo26_seg_materialization",
)


def test_shared_modules_resolve_to_the_single_root_owner() -> None:
    shared_root = (REPO_ROOT / "noesis").resolve()
    ds9_shadow_root = (REPO_ROOT / "DS9" / "noesis").resolve()

    for name in SHARED_MODULES:
        module = import_module(name)
        origin = Path(module.__file__).resolve()
        assert origin.is_relative_to(shared_root), (name, origin)
        assert not origin.is_relative_to(ds9_shadow_root), (name, origin)


def test_removed_shared_shadows_do_not_reappear_under_ds9() -> None:
    for name in SHARED_MODULES:
        relative = Path(*name.split(".")[1:]).with_suffix(".py")
        assert not (REPO_ROOT / "DS9" / "noesis" / relative).exists(), name
