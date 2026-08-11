import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.ctypeslib import ndpointer


REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = REPO_ROOT / "pipelines" / "mapanything_preprocess_fused"
BUILD_DIR = LIB_DIR / "build"
LIB_PATH = LIB_DIR / "libmapanything_preprocess_fused.so"
HAS_FUSED_PREPROCESS_SRC = (LIB_DIR / "CMakeLists.txt").exists()

pytestmark = pytest.mark.skipif(
    not HAS_FUSED_PREPROCESS_SRC,
    reason="Fused preprocess source tree is unavailable (missing CMakeLists.txt).",
)


def _build_library() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "-S", str(LIB_DIR), "-B", str(BUILD_DIR)], check=True)
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "--parallel"], check=True)


@pytest.fixture(scope="session")
def fused_lib() -> ctypes.CDLL:
    if not LIB_PATH.exists():
        _build_library()
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.mapanything_preprocess_fused_test_pack_rgb_fp32.argtypes = [
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ]
    lib.mapanything_preprocess_fused_test_pack_rgb_fp32.restype = None

    lib.mapanything_preprocess_fused_test_pack_rgb_fp16.argtypes = [
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS"),
    ]
    lib.mapanything_preprocess_fused_test_pack_rgb_fp16.restype = None
    return lib


def _expected_rgb_tensor(image: np.ndarray, height: int, width: int, norm: float) -> np.ndarray:
    reshaped = image.reshape(height * width, 3)
    channels = reshaped.T.reshape(3, height, width)
    return channels.astype(np.float32) * norm


def test_pack_rgb_fp32(fused_lib: ctypes.CDLL) -> None:
    height, width = 2, 3
    norm = np.float32(1.0 / 255.0)
    image = np.arange(height * width * 3, dtype=np.uint8)
    intrinsics = np.linspace(0.1, 0.9, 9, dtype=np.float32)
    output = np.zeros(12 * height * width, dtype=np.float32)

    fused_lib.mapanything_preprocess_fused_test_pack_rgb_fp32(
        image, height, width, norm, intrinsics, output
    )

    tensor = output.reshape(12, height, width)
    expected_rgb = _expected_rgb_tensor(image, height, width, norm)

    np.testing.assert_allclose(tensor[0:3], expected_rgb, rtol=1e-6, atol=1e-6)
    for idx, value in enumerate(intrinsics):
        np.testing.assert_allclose(tensor[3 + idx], value, rtol=1e-6, atol=1e-6)


def test_pack_rgb_fp16(fused_lib: ctypes.CDLL) -> None:
    height, width = 4, 5
    norm = np.float32(0.25)
    image = (np.arange(height * width * 3, dtype=np.uint8) % 255).astype(np.uint8)
    intrinsics = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.25, 0.0, 0.0, 1.0], dtype=np.float32)
    output = np.zeros(12 * height * width, dtype=np.uint16)

    fused_lib.mapanything_preprocess_fused_test_pack_rgb_fp16(
        image, height, width, norm, intrinsics, output
    )

    tensor = output.view(np.float16).astype(np.float32).reshape(12, height, width)
    expected_rgb = _expected_rgb_tensor(image, height, width, norm)

    np.testing.assert_allclose(tensor[0:3], expected_rgb, rtol=1e-3, atol=1e-3)
    for idx, value in enumerate(intrinsics):
        np.testing.assert_allclose(tensor[3 + idx], value, rtol=1e-3, atol=1e-3)
