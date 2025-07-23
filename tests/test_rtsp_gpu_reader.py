#!/usr/bin/env python3
"""
Smoke-tests for the refactored GPU-only RTSP reader.
These tests exercise the *single* modern PyNvVideoCodec code-path and
verify that a real RTSP stream from the global config can be decoded
into a CUDA tensor with the expected geometry.

Legacy tests that mocked PyNvDecoder / PySurfaceConverter and the
multi-signature fall-back logic were removed because that logic no
longer exists.
"""

import time
from typing import Optional

import pytest
import torch

from config import AppConfig

try:
    from nvdec_rtsp_gpu_reader import GPUOnlyRTSPReader, PYNVVIDEOCODEC_AVAILABLE
    GPU_READER_AVAILABLE = PYNVVIDEOCODEC_AVAILABLE
except ImportError:
    GPU_READER_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GPU_READER_AVAILABLE,
    reason="PyNvVideoCodec or GPU reader not available on this machine",
)


@pytest.fixture(scope="session")
def rtsp_url() -> str:
    """Return RTSP test stream URL pulled from the project config."""
    cfg = AppConfig()
    if cfg.cameras.RTSP_STREAMS:
        return cfg.cameras.RTSP_STREAMS[0]["url"]
    pytest.skip("No RTSP stream configured in AppConfig for testing")


def test_gpu_only_rtsp_reader_smoke(rtsp_url: str):
    """Ensure first frame decodes correctly and tensor geometry matches reader metadata."""
    reader = GPUOnlyRTSPReader(rtsp_url, gpu_id=0, queue_size=5)

    try:
        assert reader.start(), "reader.start() returned False"

        # Give the decoder a short time budget to emit the first frame
        tensor: Optional[torch.Tensor] = None
        deadline = time.time() + 10.0  # 10 seconds max
        while time.time() < deadline and tensor is None:
            tensor = reader.read()

        assert tensor is not None, "Did not receive a frame from the GPU reader"
        assert tensor.device.type == "cuda", "Tensor is not on GPU"
        assert tensor.ndim == 3, "Expected CHW tensor"
        c, h, w = tensor.shape
        assert c == 3, "Expected 3-channel RGB tensor"
        assert h == reader.height and w == reader.width, "Tensor geometry mismatch with reader metadata"

    finally:
        reader.close()


if __name__ == '__main__':
    # Configure test environment
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA not available - some tests will be skipped")
    
    if GPU_READER_AVAILABLE:
        print("GPU reader available - running full test suite")
    else:
        print("GPU reader not available - running limited test suite")
    
    # Run tests
    pytest.main(verbosity=2) 