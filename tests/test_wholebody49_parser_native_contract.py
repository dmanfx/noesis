from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    REPO_ROOT
    / "DS9/pipelines/nvdsinfer_deimv2_wholebody49/nvdsinfer_deimv2_wholebody49.cpp"
)
HARNESS_ROOT = REPO_ROOT / "tests/cpp/wholebody49_parser"
HARNESS = HARNESS_ROOT / "wholebody49_parser_harness.cpp"
LIBRARY = (
    REPO_ROOT
    / "DS9/pipelines/nvdsinfer_deimv2_wholebody49/libnvdsinfer_deimv2_wholebody49.so"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ds9_wholebody49_parser_source_enforces_strict_contract() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "resolve_exact_layers" in text
    assert "set_normalized_bbox" in text
    assert "normalized_boxes" not in text
    assert "outputLayersInfo[0]" not in text
    assert "outputLayersInfo[1]" not in text
    assert "kQueryCount = 1240" in text
    assert "kMaskHeight = 80" in text


def test_compiled_fake_nvdsinfer_harness_enforces_adversarial_contracts(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "wholebody49-ds9"
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
            str(SOURCE),
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
    assert len(_sha256(SOURCE)) == 64


def test_installed_ds9_parser_binary_matches_sdk_abi(tmp_path: Path) -> None:
    assert LIBRARY.is_file(), f"required installed parser is missing: {LIBRARY}"
    executable = tmp_path / "wholebody49-ds9"
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
    command.append("-DNOESIS_DEEPSTREAM_9_ABI")
    command.extend(
        [
            "-I",
            str(HARNESS_ROOT),
            str(HARNESS),
            str(LIBRARY),
            "-Wl,-z,defs",
            f"-Wl,-rpath,{LIBRARY.parent}",
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
