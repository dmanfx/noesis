"""Build and locate the YOLO26 semantic DeepStream parser."""

from __future__ import annotations

import subprocess
from pathlib import Path


PARSER_DIR = Path(__file__).resolve().parent / "custom"
PARSER_SOURCE = PARSER_DIR / "yolo26_sem_parser.cpp"
PARSER_LIBRARY = PARSER_DIR / "libnvdsinfer_yolo26_sem_ade20k.so"
DEEPSTREAM_INCLUDE = Path("/opt/nvidia/deepstream/deepstream-8.0/sources/includes")


def build_parser() -> Path:
    if not PARSER_SOURCE.is_file():
        raise FileNotFoundError(f"Parser source missing: {PARSER_SOURCE}")
    if not DEEPSTREAM_INCLUDE.is_dir():
        raise FileNotFoundError(f"DeepStream includes missing: {DEEPSTREAM_INCLUDE}")
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{DEEPSTREAM_INCLUDE}",
            "-o",
            str(PARSER_LIBRARY),
            str(PARSER_SOURCE),
        ],
        check=True,
    )
    return PARSER_LIBRARY


def ensure_parser() -> Path:
    if PARSER_LIBRARY.is_file() and PARSER_LIBRARY.stat().st_mtime >= PARSER_SOURCE.stat().st_mtime:
        return PARSER_LIBRARY
    return build_parser()
