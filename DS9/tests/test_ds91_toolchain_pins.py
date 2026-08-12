from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = ROOT / "DS9"
DS_HOME = "/opt/nvidia/deepstream/deepstream-9.1"
CUDA_HOME = "/usr/local/cuda-13.2"

PARSER_MAKEFILES = (
    "nvdsinfer_yolo26_seg/Makefile",
    "nvdsinfer_yolo11_seg/Makefile",
    "nvdsinfer_yolo_detect/Makefile",
    "nvdsinfer_rfdetr/Makefile",
    "nvdsinfer_rfdetr_seg/Makefile",
    "nvdsinfer_rfdetr_keypoint/Makefile",
    "nvdsinfer_deimv2_wholebody49/Makefile",
)
TRACKER_CONFIGS = (
    "infer.yaml",
    "infer_v3dt.yaml",
    "infer_v3dt_living_family_phone_optimized.yaml",
    "infer_v3dt_living_kitchen_tracking_candidate.yaml",
    "infer_v3dt_living_room_optimized.yaml",
)
PREPROCESS_CONFIGS = (
    "config_preproc.ini",
    "config_preproc_rfdetr_312.ini",
    "config_preproc_rfdetr_384.ini",
    "config_preproc_rfdetr_432.ini",
    "config_preproc_rfdetr_detect_384.ini",
    "config_preproc_rfdetr_detect_512.ini",
    "config_preproc_rfdetr_detect_576.ini",
    "config_preproc_yolo26_m.ini",
    "config_preproc_yolo26_seg_s.ini",
)


def _read(relative: str) -> str:
    return (DS9_ROOT / relative).read_text(encoding="utf-8")


def test_development_image_pins_ds91_and_installs_service_maker() -> None:
    dockerfile = _read("docker/Dockerfile")
    expected_base = (
        "nvcr.io/nvidia/deepstream:9.1-triton-multiarch@"
        "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994"
    )
    assert f"FROM {expected_base}" in dockerfile
    assert "service-maker/python/pyservicemaker*.whl" in dockerfile
    assert "import pyservicemaker" in dockerfile
    assert "9.0-triton-multiarch" not in dockerfile


def test_build_defaults_pin_ds91_and_cuda132() -> None:
    build_env = _read("scripts/ds9_build_env.sh")
    assert DS_HOME in build_env
    assert CUDA_HOME in build_env
    assert "NVDS_VERSION_MINOR[[:space:]]+1" in build_env

    trt_build = _read("scripts/build_trt_plugins.sh")
    assert DS_HOME in trt_build
    assert 'CUDA_VER="${CUDA_VER:-13.2}"' in trt_build

    for relative in PARSER_MAKEFILES:
        source = _read(f"pipelines/{relative}")
        assert f"DS_HOME ?= {DS_HOME}" in source
        assert f"CUDA_HOME ?= {CUDA_HOME}" in source
        assert "NVDS_VERSION_MINOR[[:space:]]+1" in source


def test_runtime_configs_bind_only_to_ds91_sdk_paths() -> None:
    tracker_library = f"{DS_HOME}/lib/libnvds_nvmultiobjecttracker.so"
    for relative in TRACKER_CONFIGS:
        payload = yaml.safe_load(_read(f"config/{relative}"))
        assert payload["tracker"]["ll-lib-file"] == tracker_library

    preprocess_library = f"{DS_HOME}/lib/gst-plugins/libcustom2d_preprocess.so"
    for relative in PREPROCESS_CONFIGS:
        source = _read(f"pipelines/{relative}")
        assert f"custom-lib-path={preprocess_library}" in source


def test_preflight_uses_exact_ds91_driver_floor() -> None:
    preflight = _read("scripts/ds9_preflight.py")
    assert 'DS9_MIN_DRIVER_VERSION = (595, 58, 3)' in preflight
    assert 'DS9_MIN_DRIVER_LABEL = "595.58.03"' in preflight
    assert DS_HOME in preflight
