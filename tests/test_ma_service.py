import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

try:
    from fastapi.testclient import TestClient
except (RuntimeError, ImportError):
    class _StubResponse:
        def __init__(self, status_code: int, body: bytes, headers: List[Tuple[bytes, bytes]]):
            self.status_code = status_code
            self._body = body
            self._headers = {k.decode('latin-1'): v.decode('latin-1') for k, v in headers}

        def json(self) -> Any:
            if not self._body:
                return None
            return json.loads(self._body.decode('utf-8'))

        def text(self) -> str:
            return self._body.decode('utf-8')

    class TestClient:  # type: ignore[override]
        def __init__(self, app):
            self.app = app

        def post(self, path: str, *, json: Any = None, headers: Dict[str, str] | None = None) -> _StubResponse:
            return self._request('POST', path, json=json, headers=headers)

        def _request(self, method: str, path: str, *, json: Any = None, headers: Dict[str, str] | None = None) -> _StubResponse:
            headers = {k.lower(): v for k, v in (headers or {}).items()}
            body = b''
            if json is not None:
                body = json.dumps(json, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
                headers.setdefault('content-type', 'application/json')
            headers.setdefault('content-length', str(len(body)))

            scope = {
                'type': 'http',
                'asgi': {'version': '3.0'},
                'http_version': '1.1',
                'method': method,
                'scheme': 'http',
                'path': path,
                'raw_path': path.encode('ascii'),
                'query_string': b'',
                'headers': [(k.encode('latin-1'), v.encode('latin-1')) for k, v in headers.items()],
                'client': ('testclient', 0),
                'server': ('testserver', 80),
            }

            async def receive() -> Dict[str, Any]:
                nonlocal body
                if receive.has_sent:  # type: ignore[attr-defined]
                    return {'type': 'http.disconnect'}
                receive.has_sent = True  # type: ignore[attr-defined]
                payload, body = body, b''
                return {'type': 'http.request', 'body': payload, 'more_body': False}

            response_body = bytearray()
            status_code = 500
            response_headers: List[Tuple[bytes, bytes]] = []

            async def send(message: Dict[str, Any]) -> None:
                nonlocal status_code, response_headers, response_body
                if message['type'] == 'http.response.start':
                    status_code = message['status']
                    response_headers = message.get('headers', [])
                elif message['type'] == 'http.response.body':
                    response_body.extend(message.get('body', b''))

            receive.has_sent = False  # type: ignore[attr-defined]
            asyncio.run(self.app(scope, receive, send))
            return _StubResponse(status_code=status_code, body=bytes(response_body), headers=response_headers)

from services.mapanything_svc import server


class DummyModel:
    """Simulates a model that returns a flat 2m depth map for every view."""

    def infer(self, views: List[Dict[str, Any]], **kwargs) -> List[Dict[str, torch.Tensor]]:
        predictions: List[Dict[str, torch.Tensor]] = []
        for view in views:
            img_tensor: torch.Tensor = view['img']
            _, _, height, width = img_tensor.shape
            depth = torch.full((1, height, width, 1), 2.0, dtype=torch.float32)
            conf = torch.ones((1, height, width), dtype=torch.float32)
            mask = torch.ones((1, height, width, 1), dtype=torch.bool)
            predictions.append(
                {
                    'depth_z': depth,
                    'conf': conf,
                    'mask': mask,
                    'camera_poses': torch.eye(4, dtype=torch.float32).reshape(1, 4, 4),
                    'intrinsics': torch.eye(3, dtype=torch.float32).reshape(1, 3, 3),
                    'metric_scaling_factor': torch.tensor(1.0, dtype=torch.float32),
                }
            )
        return predictions


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    dummy = DummyModel()

    async def fake_ensure_model_loaded(self):
        self.model = dummy
        return dummy

    async def run_without_retries(callable_, *, max_attempts: int = 3, base_delay: float = 0.5):
        return callable_()

    server.state.scene_queues.clear()
    monkeypatch.setattr(server.ServiceState, 'ensure_model_loaded', fake_ensure_model_loaded, raising=False)
    monkeypatch.setattr(server.state, 'ensure_model_loaded', lambda: fake_ensure_model_loaded(server.state))
    monkeypatch.setattr(server, '_run_with_retries', run_without_retries)
    monkeypatch.setattr(server, 'preprocess_inputs', None)
    monkeypatch.setattr(server, 'verify_weights_checksum', lambda: None)
    monkeypatch.setattr(server.torch.cuda, 'is_available', lambda: False)
    server.state.model = dummy
    yield
    server.state.model = None


def encode_image(rgb: np.ndarray, cam_id: str) -> Dict[str, Any]:
    return {
        'cam_id': cam_id,
        'img': rgb.tolist(),
    }


def _build_flat_depth_rgb(height: int = 256, width: int = 256) -> np.ndarray:
    """Create a synthetic RGB frame used to represent a flat 2m scene."""
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    rgb = np.tile(gradient, (height, 1))
    rgb = np.stack([rgb, np.flipud(rgb), np.full_like(rgb, 128)], axis=-1)
    return rgb


def test_infer_mono_returns_flat_depth_map():
    client = TestClient(server.app)
    rgb = _build_flat_depth_rgb()
    response = client.post(
        '/infer_mono',
        json={'view': encode_image(rgb, 'mono-cam')},
        headers={'X-API-Key': server.config.service.api_key},
    )

    assert response.status_code == 200
    payload = response.json()
    depth = np.array(payload['depth_z'], dtype=np.float32)

    assert depth.shape == (256, 256)
    assert np.allclose(depth, 2.0, atol=0.01)


def test_infer_multi_returns_consistent_flat_depth_maps():
    client = TestClient(server.app)
    rgb_a = _build_flat_depth_rgb()
    rgb_b = np.rot90(rgb_a)
    response = client.post(
        '/infer_multi',
        json={
            'scene_id': 'synthetic-scene',
            'views': [
                encode_image(rgb_a, 'cam-a'),
                encode_image(rgb_b, 'cam-b'),
            ],
        },
        headers={'X-API-Key': server.config.service.api_key},
    )

    assert response.status_code == 200
    payload = response.json()
    for cam_id in ('cam-a', 'cam-b'):
        depth = np.array(payload['depth_z'][cam_id], dtype=np.float32)
        assert depth.shape == (256, 256)
        assert np.allclose(depth, 2.0, atol=0.01)
