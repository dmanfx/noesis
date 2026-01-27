"""Build and locate the custom segmentation parser shared library."""

from __future__ import annotations

import subprocess
from pathlib import Path


CUSTOM_PARSER_DIR = Path(__file__).resolve().parent / "custom"
CUSTOM_PARSER_SRC = CUSTOM_PARSER_DIR / "bidnet_segparser.cpp"
CUSTOM_PARSER_LIB = CUSTOM_PARSER_DIR / "libbidnet_segparser.so"

DEEPSTREAM_INCLUDE = Path("/opt/nvidia/deepstream/deepstream-8.0/sources/includes")
CUDA_INCLUDE = Path("/usr/local/cuda/include")


def build_custom_parser() -> Path:
    if not CUSTOM_PARSER_SRC.exists():
        raise FileNotFoundError(f"Custom parser source not found: {CUSTOM_PARSER_SRC}")
    if not DEEPSTREAM_INCLUDE.exists():
        raise FileNotFoundError(f"DeepStream includes not found: {DEEPSTREAM_INCLUDE}")
    if not CUDA_INCLUDE.exists():
        raise FileNotFoundError(f"CUDA includes not found: {CUDA_INCLUDE}")

    CUSTOM_PARSER_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        f"-I{DEEPSTREAM_INCLUDE}",
        f"-I{CUDA_INCLUDE}",
        "-o",
        str(CUSTOM_PARSER_LIB),
        str(CUSTOM_PARSER_SRC),
    ]
    subprocess.run(cmd, check=True)
    return CUSTOM_PARSER_LIB


def ensure_custom_parser_lib() -> Path:
    if CUSTOM_PARSER_LIB.exists():
        try:
            if CUSTOM_PARSER_LIB.stat().st_mtime >= CUSTOM_PARSER_SRC.stat().st_mtime:
                return CUSTOM_PARSER_LIB
        except FileNotFoundError:
            pass
    return build_custom_parser()
