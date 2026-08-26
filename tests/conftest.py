from __future__ import annotations

import atexit
import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlunsplit


ROOT = Path(__file__).resolve().parents[1]
DS9_ROOT = ROOT / "DS9"
DS9_NATIVE_EXTENSIONS = DS9_ROOT / "native_extensions"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DS9_ROOT) in sys.path:
    sys.path.remove(str(DS9_ROOT))
sys.path.insert(0, str(DS9_ROOT))
if str(DS9_NATIVE_EXTENSIONS) in sys.path:
    sys.path.remove(str(DS9_NATIVE_EXTENSIONS))
sys.path.insert(0, str(DS9_NATIVE_EXTENSIONS))

from noesis_core.private_paths import atomic_write_private_file


_TEST_RUNTIME_SECRET_DIR = Path(tempfile.mkdtemp(prefix="noesis-test-secrets-"))
atexit.register(shutil.rmtree, _TEST_RUNTIME_SECRET_DIR, True)
_TEST_MAPANYTHING_KEY = _TEST_RUNTIME_SECRET_DIR / "mapanything_rpc.key"
_TEST_CAMERA_SECRETS = _TEST_RUNTIME_SECRET_DIR / "camera_sources.json"
atomic_write_private_file(
    _TEST_MAPANYTHING_KEY,
    secrets.token_urlsafe(48).encode("ascii"),
    label="test MapAnything RPC key",
)
_camera_sources = {
    name: urlunsplit(("rtsp", "camera.invalid", f"/{name}", "", ""))
    for name in ("living-room", "kitchen", "family-room")
}
atomic_write_private_file(
    _TEST_CAMERA_SECRETS,
    json.dumps(
        {"version": 1, "sources": _camera_sources},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    + b"\n",
    label="test camera source secrets",
)
os.environ["NOESIS_MAPANYTHING_API_KEY_FILE"] = str(_TEST_MAPANYTHING_KEY)
os.environ["NOESIS_CAMERA_SECRETS_FILE"] = str(_TEST_CAMERA_SECRETS)
