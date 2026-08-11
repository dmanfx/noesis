from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


class Yolo26SegProfileTests(unittest.TestCase):
    def test_ds9_runtime_resolver_covers_every_supported_size(self) -> None:
        script = r'''
import json
import sys
from pathlib import Path

ds9 = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path = [str(ds9), str(repo)] + [
    value for value in sys.path
    if value and Path(value).resolve() not in {ds9, repo}
]
from noesis import ds9_runtime_core as runtime

payload = {}
for size in ("n", "s", "m"):
    assets = runtime._resolve_yolo26_assets(size)
    payload[size] = {key: str(value) for key, value in assets.items()}
print(json.dumps(payload, sort_keys=True))
'''
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(DS9_ROOT),
                str(REPO_ROOT),
            ],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(set(payload), {"n", "s", "m"})
        for size, assets in payload.items():
            self.assertIn(f"yolo26{size}-seg", Path(assets["onnx"]).name)
            self.assertTrue(str(assets["engine"]).startswith(str(DS9_ROOT / "models")))


if __name__ == "__main__":
    unittest.main()
