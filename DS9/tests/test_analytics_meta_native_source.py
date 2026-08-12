from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_analytics_bridge_uses_public_servicemaker_metadata() -> None:
    source = (ROOT / "native" / "noesis_analytics_meta_ext.cpp").read_text(
        encoding="utf-8"
    )

    assert "deepstream::ObjectMetadata" in source
    assert "deepstream::AnalyticsObjInfo" in source
    assert "obj_meta.iterate" in source
    assert "NVDS_USER_OBJ_META_NVDSANALYTICS" in source
    assert "analytics.getUniqueId()" in source
    assert "analytics.getOcStatus()" in source
    assert "analytics.getRoiStatus()" in source
    assert "NvDsObjectMeta" not in source
    assert "nvds_get_user_meta_type" not in source
    assert "pyds" not in source


def test_runtime_requires_the_analytics_bridge() -> None:
    runtime_paths = (ROOT / "noesis" / "runtime_paths.py").read_text(encoding="utf-8")
    hooks = (ROOT / "noesis" / "pipelines" / "hooks.py").read_text(encoding="utf-8")

    assert '"noesis_analytics_meta_ext"' in runtime_paths
    assert "noesis_analytics_meta_ext.extract_analytics is required" in hooks


def test_analytics_bridge_records_the_ds91_rebuild() -> None:
    manifest = (ROOT / "native_artifact_manifest.yaml").read_text(encoding="utf-8")

    assert "contract: noesis.ds9.native_artifact_manifest" in manifest
    assert "id: native.analytics_meta" in manifest
    assert "state: validated" in manifest
    assert (
        "output_sha256: "
        "f535287518ec36ed17a357d80a8b4237549f183e2bbdab79d2e95f5208de8240"
    ) in manifest
    assert "noesis-ds9-dev@sha256:88d80ad35f12ec3a574cf2555a8242d33ac4110abdcc5f88a6cbdee40dfcf872" in manifest
