from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DS8_SOURCE = (
    REPO_ROOT
    / "pipelines/nvdsinfer_deimv2_wholebody49/nvdsinfer_deimv2_wholebody49.cpp"
)
DS9_SOURCE = (
    REPO_ROOT
    / "DS9/pipelines/nvdsinfer_deimv2_wholebody49/nvdsinfer_deimv2_wholebody49.cpp"
)
HARNESS_ROOT = REPO_ROOT / "tests/cpp/wholebody49_parser"
HARNESS = HARNESS_ROOT / "wholebody49_parser_harness.cpp"
DS8_LIBRARY = (
    REPO_ROOT
    / "pipelines/nvdsinfer_deimv2_wholebody49/libnvdsinfer_deimv2_wholebody49.so"
)
DS9_LIBRARY = (
    REPO_ROOT
    / "DS9/pipelines/nvdsinfer_deimv2_wholebody49/libnvdsinfer_deimv2_wholebody49.so"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ds8_and_ds9_wholebody49_parser_sources_are_byte_identical() -> None:
    assert DS8_SOURCE.read_bytes() == DS9_SOURCE.read_bytes()
    text = DS8_SOURCE.read_text(encoding="utf-8")
    assert "resolve_exact_layers" in text
    assert "set_normalized_bbox" in text
    assert "normalized_boxes" not in text
    assert "outputLayersInfo[0]" not in text
    assert "outputLayersInfo[1]" not in text
    assert "kQueryCount = 1240" in text
    assert "kMaskHeight = 80" in text


@pytest.mark.parametrize("source", [DS8_SOURCE, DS9_SOURCE], ids=("ds8", "ds9"))
def test_compiled_fake_nvdsinfer_harness_enforces_adversarial_contracts(
    source: Path, tmp_path: Path
) -> None:
    executable = tmp_path / f"wholebody49-{source.parents[2].name}"
    compile_result = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O1",
            "-g",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-D_GLIBCXX_ASSERTIONS",
            "-I",
            str(HARNESS_ROOT),
            str(source),
            str(HARNESS),
            "-o",
            str(executable),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    run_result = subprocess.run(
        [str(executable)],
        cwd=REPO_ROOT,
        env={**os.environ, "MALLOC_PERTURB_": "165"},
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "[OK] strict Wholebody49 parser contract" in run_result.stdout
    assert _sha256(DS8_SOURCE) == _sha256(DS9_SOURCE)


@pytest.mark.parametrize(
    ("library", "deepstream9"),
    [(DS8_LIBRARY, False), (DS9_LIBRARY, True)],
    ids=("installed-ds8", "installed-ds9"),
)
def test_installed_parser_binary_matches_sdk_abi(
    library: Path, deepstream9: bool, tmp_path: Path
) -> None:
    assert library.is_file(), f"required installed parser is missing: {library}"
    executable = tmp_path / f"wholebody49-{library.parents[2].name}"
    command = [
        "g++",
        "-std=c++17",
        "-O1",
        "-g",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-D_GLIBCXX_ASSERTIONS",
    ]
    if deepstream9:
        command.append("-DNOESIS_DEEPSTREAM_9_ABI")
    command.extend(
        [
            "-I",
            str(HARNESS_ROOT),
            str(HARNESS),
            str(library),
            "-Wl,-z,defs",
            f"-Wl,-rpath,{library.parent}",
            "-o",
            str(executable),
        ]
    )
    compile_result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    run_result = subprocess.run(
        [str(executable)],
        cwd=REPO_ROOT,
        env={**os.environ, "MALLOC_PERTURB_": "165"},
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "[OK] strict Wholebody49 parser contract" in run_result.stdout
