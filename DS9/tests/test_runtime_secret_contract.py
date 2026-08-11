from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


class RuntimeSecretContractTests(unittest.TestCase):
    def test_ds9_pipeline_uses_only_camera_secret_references(self) -> None:
        config = yaml.safe_load((ROOT / "DS9/config/infer.yaml").read_text(encoding="utf-8"))
        sources = config["sources"]
        self.assertTrue(sources)
        self.assertTrue(all(source.get("uri_secret") for source in sources))
        self.assertTrue(all("uri" not in source for source in sources))

    def test_engine_plan_has_no_runtime_secret_dependency(self) -> None:
        paths = (
            ROOT / "DS9/scripts/run_canonical_engine_maintenance.sh",
            ROOT / "DS9/scripts/rebuild_engines.py",
        )
        forbidden = (
            "NOESIS_CAMERA_SECRETS_FILE",
            "NOESIS_MAPANYTHING_API_KEY_FILE",
            "camera_sources.json",
            "mapanything_rpc.key",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path} unexpectedly depends on {token}")

    def test_no_gpu_secret_mount_check_is_fail_closed(self) -> None:
        text = (ROOT / "DS9/scripts/validate_runtime_secrets_container.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "--runtime=runc",
            "--network=none",
            "--read-only",
            "NVIDIA_VISIBLE_DEVICES=void",
            "NOESIS_CAMERA_SECRETS_FILE=/run/noesis-secrets/camera_sources.json",
            "NOESIS_MAPANYTHING_API_KEY_FILE=/run/noesis-secrets/mapanything_rpc.key",
            "NOESIS_INTERNAL_AUTH_TOKEN_FILE=/run/noesis-secrets/gateway-token",
            "camera_sources.json,readonly",
            "mapanything_rpc.key,readonly",
            "gateway-token,readonly",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
