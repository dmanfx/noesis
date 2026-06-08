"""Build and locate the DEIMv2 Wholebody49 custom parser."""

from __future__ import annotations

import subprocess
from pathlib import Path


PARSER_DIR = Path(__file__).resolve().parent / "parser"
PARSER_SRC = PARSER_DIR / "deimv2_wholebody49_parser.cpp"
PARSER_LIB = PARSER_DIR / "libdeimv2_wholebody49_parser.so"

DEEPSTREAM_INCLUDE = Path("/opt/nvidia/deepstream/deepstream-8.0/sources/includes")
CUDA_INCLUDE = Path("/usr/local/cuda/include")


def build_custom_parser() -> Path:
    if not PARSER_SRC.exists():
        raise FileNotFoundError(f"Parser source not found: {PARSER_SRC}")
    if not DEEPSTREAM_INCLUDE.exists():
        raise FileNotFoundError(f"DeepStream includes not found: {DEEPSTREAM_INCLUDE}")
    if not CUDA_INCLUDE.exists():
        raise FileNotFoundError(f"CUDA includes not found: {CUDA_INCLUDE}")
    subprocess.run(["make", "-C", str(PARSER_DIR)], check=True)
    return PARSER_LIB


def ensure_custom_parser_lib() -> Path:
    if PARSER_LIB.exists() and PARSER_SRC.exists() and PARSER_LIB.stat().st_mtime >= PARSER_SRC.stat().st_mtime:
        return PARSER_LIB
    return build_custom_parser()
