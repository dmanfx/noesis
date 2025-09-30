import base64
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

try:
    from fastapi.testclient import TestClient
except RuntimeError:
    TestClient = None

from services.mapanything_svc import server


class DummyModel:
    def infer(self, views: List[Dict[str, Any]], **kwargs) -> List[Dict[str, torch.Tensor]]:
        h = 2
        w = 2
        depth = torch.full((1, h, w, 1), 2.0, dtype=torch.float32)
        conf = torch.full((1, h, w), 0.85, dtype=torch.float32)
        mask = torch.ones((1, h, w, 1), dtype=torch.bool)
        return [{'depth_z': depth, 'conf': conf, 'mask': mask}]

    def infer_multi(self, views: List[Dict[str, Any]], **kwargs) -> List[Dict[str, torch.Tensor]]:
        outputs = []
        for idx, _ in enumerate(views):
            depth = torch.full((1, 3, 3, 1), 1.5 + idx, dtype=torch.float32)
            conf = torch.full((1, 3, 3), 0.9 - idx * 0.1, dtype=torch.float32)
            mask = torch.ones((1, 3, 3, 1), dtype=torch.bool)
            pose = torch.eye(4, dtype=torch.float32)
            outputs.append({
                'depth_z': depth,
                'conf': conf,
                'mask': mask,
                'camera_poses': pose.unsqueeze(0),
                'intrinsics': torch.eye(3, dtype=torch.float32).unsqueeze(0)
            })
        return outputs


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    dummy = DummyModel()

    async def fake_ensure_model_loaded():
        server.state.model = dummy
        return dummy

    monkeypatch.setattr(server, 'preprocess_inputs', None)
    monkeypatch.setattr(server, 'verify_weights_checksum', lambda: None)
    monkeypatch.setattr(server.state, 'ensure_model_loaded', fake_ensure_model_loaded)
    server.state.model = dummy
    yield
    server.state.model = None


def encode_image(rgb: np.ndarray, cam_id: str = 'living-room') -> Dict[str, Any]:
    shape = rgb.shape
    payload = base64.b64encode(rgb.tobytes()).decode('ascii')
    return {
        'cam_id': cam_id,
        'img_b64': payload,
        'shape': list(shape)
    }


def test_infer_mono_endpoint():
    if TestClient is None:
        pytest.skip('httpx not installed for TestClient')
    client = TestClient(server.app)
    rgb = np.full((2, 2, 3), 128, dtype=np.uint8)
    view = encode_image(rgb, 'living-room')
    response = client.post(
        '/infer_mono',
        json={'view': view},
        headers={'X-API-Key': server.config.service.api_key}
    )
    assert response.status_code == 200
    data = response.json()
    assert data['cam_id'] == 'living-room'
    assert data['shape'] == [2, 2]
    assert len(data['depth_z']) == 2
    assert len(data['conf']) == 2


def test_infer_multi_endpoint():
    if TestClient is None:
        pytest.skip('httpx not installed for TestClient')
    client = TestClient(server.app)
    rgb_one = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb_two = np.ones((3, 3, 3), dtype=np.uint8) * 255
    views = [
        encode_image(rgb_one, 'living-room'),
        encode_image(rgb_two, 'kitchen')
    ]
    response = client.post(
        '/infer_multi',
        json={'scene_id': 'test', 'views': views},
        headers={'X-API-Key': server.config.service.api_key}
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data['depth_z'].keys()) == {'living-room', 'kitchen'}
    assert isinstance(data['scale'], float)
