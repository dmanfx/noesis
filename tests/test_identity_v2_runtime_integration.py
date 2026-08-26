from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_ds9_hooks_use_the_shared_identity_product_adapter() -> None:
    path = ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py"
    source = path.read_text(encoding="utf-8")
    assert "from noesis.identity_v2_service import IdentityFramePrimitive" in source
    assert source.count("service.process_source_frame(") == 1
    assert "IdentityV2Runtime(" not in source
    assert "IdentityStore(" not in source
    analytics = source.split("class _AnalyticsTelemetryProcessor:", 1)[1]
    handler = analytics.split(
        "def _handle_servicemaker_frame_admitted(self, frame_meta: Any) -> None:",
        1,
    )[1]
    handler = handler.split("\n    def handle_frame(self, frame_meta: Any) -> None:", 1)[0]
    assert handler.index("identity_v2_authoritative = bool(") < handler.index(
        "stable_id = self._maybe_assign_stable_id("
    )
    assert "stable_id = None\n                if not identity_v2_authoritative:" in handler
    assert "stable_id=0 if identity_v2_authoritative else stable_id_int" in handler
    assert "if callable(observe_fn) and not identity_v2_authoritative:" in handler
    assert "if not identity_v2_authoritative:\n                self._maintain_stable_ids" in handler


def test_canonical_runtime_mounts_authenticated_v2_api_and_world_run_owner() -> None:
    path = ROOT / "DS9" / "noesis" / "ds9_runtime_core.py"
    source = path.read_text(encoding="utf-8")
    assert "app.include_router(reid_v2_api.router)" in source
    assert source.index("app.include_router(reid_v2_api.router)") < source.index(
        "configure_internal_rest_app(app)"
    )
    assert "create_identity_v2_service(" in source
    assert "run_id=world_service.producer.run_id" in source
    assert "register_identity_v2_runtime_getter(" in source
    assert "identity_v2_service.close()" in source


def test_canonical_runtime_configs_declare_exact_reid_output_profile() -> None:
    expected = {
        ROOT / "DS9" / "config" / "infer.yaml": ("fc_pred", 256),
    }
    for path, (layer, dimension) in expected.items():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        reid = config["models"]["reid"]
        assert reid["enable"] is True
        assert reid["layer"] == layer
        assert reid["embedding_dim"] == dimension


def test_identity_v2_product_logic_is_not_copied_into_ds9() -> None:
    assert not (ROOT / "DS9" / "noesis" / "identity_v2_service.py").exists()
    assert (ROOT / "noesis" / "identity_v2_service.py").is_file()
