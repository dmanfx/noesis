"""Build only the pinned CPU adapter in a new isolated storage directory.

Example: python3 build.py --storage-parent /mnt/noesis_storage
Requires host CMake, a C++17 compiler, and TBB development headers/library.
No installer, service operation, GPU operation, or global environment edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SOURCES = {
    "basalt": (
        "https://api.github.com/repos/VladyslavUsenko/basalt/tarball/6d8637b9d68ea18a1a63c1baa72818779e156932",
        "765830c29affd7ab7cd7582acd749b2d5c53284a23696fd0a08de93e256a12d9",
    ),
    "basalt-headers": (
        "https://gitlab.com/VladyslavUsenko/basalt-headers/-/archive/aa441ba3e51050c47ba1902537792a2e4db7e43d/basalt-headers-aa441ba3e51050c47ba1902537792a2e4db7e43d.tar.gz",
        "0024bf2601bd8c16ff6e073ae4f33e4907a081fe4e5e9cbd59344686afaaa766",
    ),
    "eigen": (
        "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz",
        "e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec",
    ),
    "sophus": (
        "https://github.com/strasdat/Sophus/archive/refs/tags/1.22.10.tar.gz",
        "eb1da440e6250c5efc7637a0611a5b8888875ce6ac22bf7ff6b6769bbc958082",
    ),
    "cereal": (
        "https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz",
        "16a7ad9b31ba5880dac55d62b5d6f243c3ebc8d46a3514149e56b5e7ea81f85f",
    ),
    "json": (
        "https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz",
        "0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406",
    ),
    "fmt": (
        "https://github.com/fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz",
        "1250e4cc58bf06ee631567523f48848dc4596133e163f02615c97f78bab6c811",
    ),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-parent", type=Path, required=True)
    parser.add_argument(
        "--archive-cache",
        type=Path,
        help="optional existing directory of exact hash-pinned archives; extracted sources are never reused",
    )
    args = parser.parse_args()
    parent = args.storage_parent.resolve()
    if not parent.is_dir():
        parser.error("storage-parent must be an existing directory")
    work = Path(tempfile.mkdtemp(prefix="roomwalk-calibration-build.", dir=parent))
    print(f"Build workspace: {work}", flush=True)
    receipt = {"schema": "roomwalk.calibration_build.v1", "sources": {}, "commands": []}
    for name, (url, digest) in SOURCES.items():
        archive = (
            args.archive_cache.resolve() if args.archive_cache else work
        ) / f"{name}.tar.gz"
        if args.archive_cache:
            if not archive.is_file() or archive.stat().st_size > 64 * 1024 * 1024:
                raise RuntimeError("cached pinned archive missing or over size bound")
        else:
            with (
                urllib.request.urlopen(url, timeout=60) as source,
                archive.open("xb") as target,
            ):
                size = 0
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > 64 * 1024 * 1024:
                        raise RuntimeError("dependency archive exceeds limit")
                    target.write(chunk)
        if sha(archive) != digest:
            raise RuntimeError(f"pinned archive hash mismatch: {name}")
        extraction = work / f".{name}-unpack"
        extraction.mkdir()
        with tarfile.open(archive) as source:
            members = source.getmembers()
            if len(members) > 40000 or sum(m.size for m in members) > 256 * 1024 * 1024:
                raise RuntimeError("expanded dependency exceeds limits")
            source.extractall(extraction, filter="data")
        roots = list(extraction.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise RuntimeError("dependency archive has no unique root")
        roots[0].rename(work / name)
        extraction.rmdir()
        receipt["sources"][name] = {
            "url": url,
            "sha256": digest,
            "bytes": archive.stat().st_size,
        }
    native = Path(__file__).resolve().parent
    commands = [
        [
            "cmake",
            "-S",
            str(native),
            "-B",
            str(work / "build"),
            f"-DDEPENDENCY_ROOT={work}",
            f"-DEIGEN_ROOT={work / 'eigen'}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", str(work / "build"), "-j1"],
    ]
    for index, command in enumerate(commands):
        receipt["commands"].append(command)
        with (work / f"build-{index}.log").open("x") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900
            )
    executable = work / "build/roomwalk_calibrate_imu"
    receipt["executable"] = {
        "path": str(executable),
        "sha256": sha(executable),
        "version": subprocess.check_output(
            [str(executable), "--version"], text=True, timeout=10
        ).strip(),
    }
    receipt["adapter_sources"] = {
        p.name: sha(p)
        for p in (
            native / "runner.cpp",
            native / "CMakeLists.txt",
            Path(__file__).resolve(),
        )
    }
    receipt["functional_validation"] = (
        "not_run_by_builder; run synthetic.py or native pytest cases"
    )
    (work / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["executable"], indent=2))


if __name__ == "__main__":
    main()
