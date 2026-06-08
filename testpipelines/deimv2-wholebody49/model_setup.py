"""Download, normalize-wrap, and build DEIMv2 Wholebody49 artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "models" / "deimv2_wholebody49"
ENGINE_ROOT = REPO_ROOT / "models" / "engines"
BUILD_ROOT = PIPELINE_ROOT / "build"

SOURCE_REPO_URL = "https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49"
RESOURCE_ARCHIVE_URL = (
    "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/"
    "488_DEIMv2-Wholebody49/resources.tar.gz"
)

BASE_ONNX_NAME = "deimv2_dinov3_s_wholebody49_ins_s08_maskhead256x3_center_1240query_masks_n_batch.onnx"
DS8_ONNX_NAME = "deimv2_dinov3_s_wholebody49_ins_s08_maskhead256x3_center_1240query_masks_n_batch_ds8norm.onnx"
BOXES_ONNX_NAME = "deimv2_dinov3_s_wholebody49_ins_s08_maskhead256x3_center_1240query_boxes_n_batch_ds8norm.onnx"
ENGINE_NAME = "deimv2_dinov3_s_wholebody49_masks_n_batch_ds8norm_b3_fp16.engine"
BOXES_ENGINE_NAME = "deimv2_dinov3_s_wholebody49_boxes_n_batch_ds8norm_b3_fp16.engine"
X_BASE_ONNX_NAME = "deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query_masks_n_batch.onnx"
X_DS8_ONNX_NAME = "deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query_masks_n_batch_ds8norm.onnx"
X_BOXES_ONNX_NAME = "deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query_boxes_n_batch_ds8norm.onnx"
X_ENGINE_NAME = "deimv2_dinov3_x_wholebody49_masks_n_batch_ds8norm_b3_fp16.engine"
X_BOXES_ENGINE_NAME = "deimv2_dinov3_x_wholebody49_boxes_n_batch_ds8norm_b3_fp16.engine"
X_BOXES_INT8_ENGINE_NAME = "deimv2_dinov3_x_wholebody49_boxes_n_batch_ds8norm_b3_int8.engine"
X_BOXES_INT8_CALIBRATION_CACHE_NAME = "deimv2_dinov3_x_wholebody49_boxes_n_batch_ds8norm_b3_int8.cache"

BASE_ONNX_PATH = MODEL_ROOT / BASE_ONNX_NAME
DS8_ONNX_PATH = MODEL_ROOT / DS8_ONNX_NAME
BOXES_ONNX_PATH = MODEL_ROOT / BOXES_ONNX_NAME
ENGINE_PATH = ENGINE_ROOT / ENGINE_NAME
BOXES_ENGINE_PATH = ENGINE_ROOT / BOXES_ENGINE_NAME
X_BASE_ONNX_PATH = MODEL_ROOT / X_BASE_ONNX_NAME
X_DS8_ONNX_PATH = MODEL_ROOT / X_DS8_ONNX_NAME
X_BOXES_ONNX_PATH = MODEL_ROOT / X_BOXES_ONNX_NAME
X_ENGINE_PATH = ENGINE_ROOT / X_ENGINE_NAME
X_BOXES_ENGINE_PATH = ENGINE_ROOT / X_BOXES_ENGINE_NAME
X_BOXES_INT8_ENGINE_PATH = ENGINE_ROOT / X_BOXES_INT8_ENGINE_NAME
X_BOXES_INT8_CALIBRATION_CACHE_PATH = MODEL_ROOT / X_BOXES_INT8_CALIBRATION_CACHE_NAME
LABELS_PATH = PIPELINE_ROOT / "classes.txt"
MODEL_INFO_PATH = MODEL_ROOT / "model_info_ds8norm.json"
BOXES_MODEL_INFO_PATH = MODEL_ROOT / "model_info_ds8norm_boxes.json"
X_MODEL_INFO_PATH = MODEL_ROOT / "model_info_ds8norm_dinov3_x.json"
X_BOXES_MODEL_INFO_PATH = MODEL_ROOT / "model_info_ds8norm_dinov3_x_boxes.json"
X_BOXES_INT8_MODEL_INFO_PATH = MODEL_ROOT / "model_info_ds8norm_dinov3_x_boxes_int8.json"
NVINFER_TEMPLATE_PATH = PIPELINE_ROOT / "config_infer_deimv2_wholebody49.template.ini"
BOXES_NVINFER_TEMPLATE_PATH = PIPELINE_ROOT / "config_infer_deimv2_wholebody49_boxes.template.ini"

INPUT_NAME = "images"
INPUT_HEIGHT = 640
INPUT_WIDTH = 640
MIN_BATCH = 1
OPT_BATCH = 3
MAX_BATCH = 3
GIE_UNIQUE_ID = 1
OUTPUT_NAMES = ("label_xyxy_score", "masks")
BOXES_OUTPUT_NAMES = ("label_xyxy_score",)
INT8_IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".webp")
INT8_CALIBRATION_BATCH_SIZE = OPT_BATCH

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


@dataclass(frozen=True)
class ModelVariantSpec:
    name: str
    legacy_variant_name: str
    family: str
    display_name: str
    mask_base_onnx_name: str
    ds8_onnx_name: str
    engine_name: str
    model_info_name: str
    output_names: tuple[str, ...]
    has_instance_masks: bool
    config_suffix: str
    aliases: tuple[str, ...] = ()
    precision: str = "fp16"
    network_mode: int = 2
    calibration_cache_name: Optional[str] = None

    @property
    def mask_base_onnx_path(self) -> Path:
        return MODEL_ROOT / self.mask_base_onnx_name

    @property
    def ds8_onnx_path(self) -> Path:
        return MODEL_ROOT / self.ds8_onnx_name

    @property
    def engine_path(self) -> Path:
        return ENGINE_ROOT / self.engine_name

    @property
    def model_info_path(self) -> Path:
        return MODEL_ROOT / self.model_info_name

    @property
    def calibration_cache_path(self) -> Optional[Path]:
        if self.calibration_cache_name is None:
            return None
        return MODEL_ROOT / self.calibration_cache_name


VARIANT_SPECS: Dict[str, ModelVariantSpec] = {
    "dinov3_s_masks": ModelVariantSpec(
        name="dinov3_s_masks",
        legacy_variant_name="masks",
        family="dinov3_s",
        display_name="DEIMv2 DINOv3-S Wholebody49",
        mask_base_onnx_name=BASE_ONNX_NAME,
        ds8_onnx_name=DS8_ONNX_NAME,
        engine_name=ENGINE_NAME,
        model_info_name=MODEL_INFO_PATH.name,
        output_names=OUTPUT_NAMES,
        has_instance_masks=True,
        config_suffix="masks",
        aliases=("masks", "s_masks", "dinov3_s_mask"),
    ),
    "dinov3_s_boxes": ModelVariantSpec(
        name="dinov3_s_boxes",
        legacy_variant_name="boxes",
        family="dinov3_s",
        display_name="DEIMv2 DINOv3-S Wholebody49",
        mask_base_onnx_name=BASE_ONNX_NAME,
        ds8_onnx_name=BOXES_ONNX_NAME,
        engine_name=BOXES_ENGINE_NAME,
        model_info_name=BOXES_MODEL_INFO_PATH.name,
        output_names=BOXES_OUTPUT_NAMES,
        has_instance_masks=False,
        config_suffix="boxes",
        aliases=("boxes", "s_boxes", "dinov3_s_box"),
    ),
    "dinov3_x_masks": ModelVariantSpec(
        name="dinov3_x_masks",
        legacy_variant_name="dinov3_x_masks",
        family="dinov3_x",
        display_name="DEIMv2 DINOv3-X Wholebody49",
        mask_base_onnx_name=X_BASE_ONNX_NAME,
        ds8_onnx_name=X_DS8_ONNX_NAME,
        engine_name=X_ENGINE_NAME,
        model_info_name=X_MODEL_INFO_PATH.name,
        output_names=OUTPUT_NAMES,
        has_instance_masks=True,
        config_suffix="dinov3_x_masks",
        aliases=("x_masks", "dinov3_x_mask"),
    ),
    "dinov3_x_boxes": ModelVariantSpec(
        name="dinov3_x_boxes",
        legacy_variant_name="dinov3_x_boxes",
        family="dinov3_x",
        display_name="DEIMv2 DINOv3-X Wholebody49",
        mask_base_onnx_name=X_BASE_ONNX_NAME,
        ds8_onnx_name=X_BOXES_ONNX_NAME,
        engine_name=X_BOXES_ENGINE_NAME,
        model_info_name=X_BOXES_MODEL_INFO_PATH.name,
        output_names=BOXES_OUTPUT_NAMES,
        has_instance_masks=False,
        config_suffix="dinov3_x_boxes",
        aliases=("x_boxes", "dinov3_x_box"),
    ),
    "dinov3_x_boxes_int8": ModelVariantSpec(
        name="dinov3_x_boxes_int8",
        legacy_variant_name="dinov3_x_boxes_int8",
        family="dinov3_x",
        display_name="DEIMv2 DINOv3-X Wholebody49",
        mask_base_onnx_name=X_BASE_ONNX_NAME,
        ds8_onnx_name=X_BOXES_ONNX_NAME,
        engine_name=X_BOXES_INT8_ENGINE_NAME,
        model_info_name=X_BOXES_INT8_MODEL_INFO_PATH.name,
        output_names=BOXES_OUTPUT_NAMES,
        has_instance_masks=False,
        config_suffix="dinov3_x_boxes_int8",
        aliases=("x_boxes_int8", "x_int8", "dinov3_x_int8", "dinov3_x_box_int8"),
        precision="int8",
        network_mode=1,
        calibration_cache_name=X_BOXES_INT8_CALIBRATION_CACHE_NAME,
    ),
}

MODEL_VARIANT_ALIASES: Dict[str, str] = {
    alias: name
    for name, spec in VARIANT_SPECS.items()
    for alias in spec.aliases
}
UNAVAILABLE_MODEL_VARIANTS: Dict[str, str] = {
    "dinov3_l_masks": (
        "DINOv3-L is not present in the official 488_DEIMv2-Wholebody49 resources archive; "
        "available DINOv3 families in this prototype are S and X."
    ),
    "dinov3_l_boxes": (
        "DINOv3-L is not present in the official 488_DEIMv2-Wholebody49 resources archive; "
        "available DINOv3 label-only families in this prototype are S and X."
    ),
    "l_masks": (
        "DINOv3-L is not present in the official 488_DEIMv2-Wholebody49 resources archive; "
        "available DINOv3 families in this prototype are S and X."
    ),
    "l_boxes": (
        "DINOv3-L is not present in the official 488_DEIMv2-Wholebody49 resources archive; "
        "available DINOv3 label-only families in this prototype are S and X."
    ),
    "dinov3_l_boxes_int8": (
        "DINOv3-L is not present in the official 488_DEIMv2-Wholebody49 resources archive; "
        "available DINOv3 label-only families in this prototype are S and X."
    ),
    "l_boxes_int8": (
        "DINOv3-L is not present in the official 488_DEIMv2-Wholebody49 resources archive; "
        "available DINOv3 label-only families in this prototype are S and X."
    ),
    "l_int8": (
        "DINOv3-L is not present in the official 488_DEIMv2-Wholebody49 resources archive; "
        "available DINOv3 label-only families in this prototype are S and X."
    ),
}
MODEL_VARIANT_CHOICES = tuple(sorted(set(VARIANT_SPECS) | set(MODEL_VARIANT_ALIASES)))


@dataclass(frozen=True)
class ModelArtifacts:
    base_onnx_path: Path
    ds8_onnx_path: Path
    engine_path: Path
    labels_path: Path
    model_info_path: Path
    calibration_cache_path: Optional[Path] = None


def _variant_key(variant: str) -> str:
    return variant.strip().lower().replace("-", "_")


def normalize_variant(variant: str) -> str:
    key = _variant_key(variant)
    if key in VARIANT_SPECS:
        return key
    if key in MODEL_VARIANT_ALIASES:
        return MODEL_VARIANT_ALIASES[key]
    if key in UNAVAILABLE_MODEL_VARIANTS:
        raise ValueError(UNAVAILABLE_MODEL_VARIANTS[key])
    choices = ", ".join(MODEL_VARIANT_CHOICES)
    raise ValueError(f"Unsupported DEIMv2 model variant: {variant!r}; expected one of: {choices}")


def variant_spec(variant: str) -> ModelVariantSpec:
    return VARIANT_SPECS[normalize_variant(variant)]


def variant_onnx_path(variant: str) -> Path:
    return variant_spec(variant).ds8_onnx_path


def variant_engine_path(variant: str) -> Path:
    return variant_spec(variant).engine_path


def variant_model_info_path(variant: str) -> Path:
    return variant_spec(variant).model_info_path


def variant_output_names(variant: str) -> tuple[str, ...]:
    return variant_spec(variant).output_names


def variant_has_instance_masks(variant: str) -> bool:
    return variant_spec(variant).has_instance_masks


def _family_mask_ds8_onnx_path(spec: ModelVariantSpec) -> Path:
    return MODEL_ROOT / spec.mask_base_onnx_name.replace(".onnx", "_ds8norm.onnx")


def _ensure_dirs() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    ENGINE_ROOT.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)


def load_classes(path: Path = LABELS_PATH) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Class labels not found: {path}")
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(classes) != 49:
        raise RuntimeError(f"Expected 49 Wholebody classes, got {len(classes)} from {path}")
    return classes


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest_dir / member.name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
        tar.extractall(dest_dir)


def ensure_base_model(*, allow_download: bool = True, variant: str = "masks") -> Path:
    spec = variant_spec(variant)
    base_path = spec.mask_base_onnx_path
    _ensure_dirs()
    if base_path.exists():
        return base_path
    if not allow_download:
        raise FileNotFoundError(f"Base DEIMv2 ONNX missing for {spec.name}: {base_path}")

    with tempfile.TemporaryDirectory(prefix="deimv2_download_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        archive_path = tmp_dir / "resources.tar.gz"
        urllib.request.urlretrieve(RESOURCE_ARCHIVE_URL, archive_path)
        _safe_extract_tar(archive_path, MODEL_ROOT)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Downloaded resources did not contain expected ONNX for {spec.name}: {base_path}"
        )
    return base_path


def _valid_onnx(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
    except Exception:
        return False
    return True


def ensure_ds8_normalized_onnx(*, allow_export: bool = True, variant: str = "masks") -> Path:
    spec = variant_spec(variant)
    base_path = ensure_base_model(allow_download=allow_export, variant=spec.name)
    output_path = _family_mask_ds8_onnx_path(spec)
    if _valid_onnx(output_path) and output_path.stat().st_mtime >= base_path.stat().st_mtime:
        return output_path
    if not allow_export:
        raise FileNotFoundError(f"DS8 normalization-wrapped ONNX missing: {output_path}")

    model = onnx.load(str(base_path), load_external_data=False)
    graph = model.graph
    normalized_name = "images_imagenet_normalized"
    centered_name = "images_minus_imagenet_mean"

    for node in graph.node:
        for idx, value in enumerate(node.input):
            if value == INPUT_NAME:
                node.input[idx] = normalized_name

    graph.initializer.extend(
        [
            numpy_helper.from_array(IMAGENET_MEAN, name="deimv2_imagenet_mean"),
            numpy_helper.from_array(IMAGENET_STD, name="deimv2_imagenet_std"),
        ]
    )
    sub_node = helper.make_node(
        "Sub",
        inputs=[INPUT_NAME, "deimv2_imagenet_mean"],
        outputs=[centered_name],
        name="NoesisImageNetMeanSub",
    )
    div_node = helper.make_node(
        "Div",
        inputs=[centered_name, "deimv2_imagenet_std"],
        outputs=[normalized_name],
        name="NoesisImageNetStdDiv",
    )

    original_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend([sub_node, div_node])
    graph.node.extend(original_nodes)

    model.producer_name = "Noesis DS8 DEIMv2 wrapper"
    metadata = model.metadata_props.add()
    metadata.key = "noesis.preprocess"
    metadata.value = "DeepStream RGB scale 1/255, ONNX ImageNet mean/std"
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))
    return output_path


def ensure_boxes_onnx(*, allow_export: bool = True, variant: str = "boxes") -> Path:
    spec = variant_spec(variant)
    if spec.has_instance_masks:
        raise ValueError(f"Variant {spec.name} already includes masks; use ensure_ds8_normalized_onnx")
    masks_path = ensure_ds8_normalized_onnx(allow_export=allow_export, variant=spec.name)
    output_path = spec.ds8_onnx_path
    if _valid_onnx(output_path) and output_path.stat().st_mtime >= masks_path.stat().st_mtime:
        return output_path
    if not allow_export:
        raise FileNotFoundError(f"Label-only DEIMv2 ONNX missing: {output_path}")

    onnx.utils.extract_model(
        str(masks_path),
        str(output_path),
        input_names=[INPUT_NAME],
        output_names=["label_xyxy_score"],
    )
    onnx.checker.check_model(str(output_path))
    return output_path


def ensure_variant_onnx(variant: str, *, allow_export: bool = True) -> Path:
    spec = variant_spec(variant)
    if spec.has_instance_masks:
        return ensure_ds8_normalized_onnx(allow_export=allow_export, variant=spec.name)
    return ensure_boxes_onnx(allow_export=allow_export, variant=spec.name)


def _engine_is_current(engine_path: Path, onnx_path: Path) -> bool:
    return engine_path.exists() and onnx_path.exists() and engine_path.stat().st_mtime >= onnx_path.stat().st_mtime


def _shape_args() -> tuple[str, str, str]:
    min_shape = f"{INPUT_NAME}:{MIN_BATCH}x3x{INPUT_HEIGHT}x{INPUT_WIDTH}"
    opt_shape = f"{INPUT_NAME}:{OPT_BATCH}x3x{INPUT_HEIGHT}x{INPUT_WIDTH}"
    max_shape = f"{INPUT_NAME}:{MAX_BATCH}x3x{INPUT_HEIGHT}x{INPUT_WIDTH}"
    return min_shape, opt_shape, max_shape


def _list_calibration_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"INT8 calibration image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise RuntimeError(f"INT8 calibration image path is not a directory: {image_dir}")
    images = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in INT8_IMAGE_EXTENSIONS
    ]
    if not images:
        extensions = ", ".join(INT8_IMAGE_EXTENSIONS)
        raise FileNotFoundError(f"No calibration images found in {image_dir} with extensions: {extensions}")
    return images


def _preprocess_calibration_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read INT8 calibration image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) * (1.0 / 255.0)
    return np.transpose(image, (2, 0, 1))


def _check_cuda(result: tuple[object, ...], action: str) -> tuple[object, ...]:
    from cuda.bindings import driver as cuda

    err = result[0]
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA {action} failed: {err}")
    return result[1:]


class _ImageEntropyCalibrator:
    """TensorRT entropy calibrator over representative RGB calibration images."""

    def __init__(self, image_paths: Sequence[Path], *, cache_path: Path, batch_size: int) -> None:
        import tensorrt as trt
        from cuda.bindings import driver as cuda

        class _Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, outer: _ImageEntropyCalibrator) -> None:
                trt.IInt8EntropyCalibrator2.__init__(self)
                self._outer = outer

            def get_batch_size(self) -> int:
                return self._outer.batch_size

            def get_batch(self, names: Sequence[str]) -> Optional[List[int]]:
                return self._outer.get_batch(names)

            def read_calibration_cache(self) -> Optional[bytes]:
                return self._outer.read_calibration_cache()

            def write_calibration_cache(self, cache: bytes) -> None:
                self._outer.write_calibration_cache(cache)

        self.image_paths = list(image_paths)
        self.cache_path = cache_path
        self.batch_size = int(batch_size)
        self.cursor = 0
        self.input_nbytes = self.batch_size * 3 * INPUT_HEIGHT * INPUT_WIDTH * np.dtype(np.float32).itemsize
        _check_cuda(cuda.cuInit(0), "init")
        (device,) = _check_cuda(cuda.cuDeviceGet(0), "get device")
        (self.context,) = _check_cuda(cuda.cuCtxCreate(None, 0, device), "create context")
        (self.device_input,) = _check_cuda(cuda.cuMemAlloc(self.input_nbytes), "allocate calibration input")
        self.calibrator = _Calibrator(self)

    def get_batch(self, names: Sequence[str]) -> Optional[List[int]]:
        from cuda.bindings import driver as cuda

        if self.cursor >= len(self.image_paths):
            return None
        batch_paths = self.image_paths[self.cursor : self.cursor + self.batch_size]
        self.cursor += len(batch_paths)
        while len(batch_paths) < self.batch_size:
            batch_paths.append(batch_paths[-1])
        batch = np.ascontiguousarray(
            np.stack([_preprocess_calibration_image(path) for path in batch_paths], axis=0),
            dtype=np.float32,
        )
        _check_cuda(cuda.cuMemcpyHtoD(self.device_input, batch, batch.nbytes), "copy calibration batch")
        return [int(self.device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)

    def close(self) -> None:
        from cuda.bindings import driver as cuda

        device_input = getattr(self, "device_input", None)
        if device_input is not None:
            cuda.cuMemFree(device_input)
            self.device_input = None
        context = getattr(self, "context", None)
        if context is not None:
            cuda.cuCtxDestroy(context)
            self.context = None


def _build_int8_engine_from_images(
    *,
    onnx_path: Path,
    engine_path: Path,
    calibration_image_dir: Path,
    calibration_cache_path: Path,
) -> Path:
    import tensorrt as trt

    image_paths = _list_calibration_images(calibration_image_dir)
    calibrator = _ImageEntropyCalibrator(
        image_paths,
        cache_path=calibration_cache_path,
        batch_size=INT8_CALIBRATION_BATCH_SIZE,
    )
    try:
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = [str(parser.get_error(idx)) for idx in range(parser.num_errors)]
            raise RuntimeError("Failed to parse ONNX for INT8 build:\n" + "\n".join(errors))

        profile = builder.create_optimization_profile()
        profile.set_shape(
            INPUT_NAME,
            (MIN_BATCH, 3, INPUT_HEIGHT, INPUT_WIDTH),
            (OPT_BATCH, 3, INPUT_HEIGHT, INPUT_WIDTH),
            (MAX_BATCH, 3, INPUT_HEIGHT, INPUT_WIDTH),
        )
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        config.set_calibration_profile(profile)
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)

        config.int8_calibrator = calibrator.calibrator
        serialized = builder.build_serialized_network(network, config)
    finally:
        calibrator.close()
    if serialized is None:
        raise RuntimeError("TensorRT INT8 engine build failed; no serialized engine was produced")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return engine_path


def _build_int8_engine_with_cache(
    *,
    onnx_path: Path,
    engine_path: Path,
    calibration_cache_path: Path,
) -> Path:
    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise FileNotFoundError("trtexec not found on PATH; cannot build TensorRT engine")
    min_shape, opt_shape, max_shape = _shape_args()
    subprocess.run(
        [
            trtexec,
            f"--onnx={onnx_path}",
            f"--minShapes={min_shape}",
            f"--optShapes={opt_shape}",
            f"--maxShapes={max_shape}",
            f"--saveEngine={engine_path}",
            "--int8",
            "--fp16",
            f"--calib={calibration_cache_path}",
            "--skipInference",
        ],
        check=True,
    )
    return engine_path


def _resolve_int8_calibration_cache(
    spec: ModelVariantSpec,
    calibration_cache_path: Optional[Path],
) -> Optional[Path]:
    candidate = calibration_cache_path or spec.calibration_cache_path
    if candidate is None:
        return None
    return candidate if candidate.exists() else None


def build_engine(
    *,
    allow_build: bool = True,
    variant: str = "masks",
    calibration_cache_path: Optional[Path] = None,
    calibration_images_dir: Optional[Path] = None,
) -> Path:
    spec = variant_spec(variant)
    onnx_path = ensure_variant_onnx(spec.name, allow_export=allow_build)
    engine_path = spec.engine_path
    if _engine_is_current(engine_path, onnx_path):
        return engine_path
    if not allow_build:
        raise FileNotFoundError(f"TensorRT engine missing or stale: {engine_path}")
    if spec.precision == "int8":
        default_cache_path = spec.calibration_cache_path
        if default_cache_path is None:
            raise RuntimeError(f"INT8 variant {spec.name} has no calibration cache path configured")
        explicit_cache_path = _resolve_int8_calibration_cache(spec, calibration_cache_path)
        if calibration_images_dir is not None:
            return _build_int8_engine_from_images(
                onnx_path=onnx_path,
                engine_path=engine_path,
                calibration_image_dir=calibration_images_dir,
                calibration_cache_path=calibration_cache_path or default_cache_path,
            )
        if explicit_cache_path is not None:
            return _build_int8_engine_with_cache(
                onnx_path=onnx_path,
                engine_path=engine_path,
                calibration_cache_path=explicit_cache_path,
            )
        raise FileNotFoundError(
            "INT8 engine is missing or stale and no calibration data was provided. "
            f"Variant {spec.name} requires representative calibration images via "
            "--int8-calibration-images or an existing TensorRT calibration cache via "
            f"--int8-calibration-cache. Default cache path: {default_cache_path}"
        )
    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise FileNotFoundError("trtexec not found on PATH; cannot build TensorRT engine")

    min_shape, opt_shape, max_shape = _shape_args()
    subprocess.run(
        [
            trtexec,
            f"--onnx={onnx_path}",
            f"--minShapes={min_shape}",
            f"--optShapes={opt_shape}",
            f"--maxShapes={max_shape}",
            f"--saveEngine={engine_path}",
            "--fp16",
            "--skipInference",
        ],
        check=True,
    )
    return engine_path


def _build_model_info(variant: str = "masks") -> Dict[str, object]:
    spec = variant_spec(variant)
    onnx_path = spec.ds8_onnx_path
    engine_path = spec.engine_path
    outputs: Dict[str, object] = {
        "label_xyxy_score": {
            "shape": ["N", 1240, 6],
            "columns": ["class_id", "x1", "y1", "x2", "y2", "score"],
            "coordinates": "normalized xyxy",
        },
    }
    if spec.has_instance_masks:
        outputs["masks"] = {
            "shape": ["N", 1240, 80, 80],
            "values": "probability mask per query",
        }
    inference: Dict[str, object] = {
        "gie_unique_id": GIE_UNIQUE_ID,
        "precision": spec.precision,
        "network_mode": spec.network_mode,
        "engine": str(engine_path.relative_to(REPO_ROOT)),
    }
    if spec.calibration_cache_path is not None:
        inference["int8_calibration_cache"] = str(spec.calibration_cache_path.relative_to(REPO_ROOT))
        inference["int8_requires_representative_calibration"] = True
    return {
        "model": {
            "name": spec.display_name,
            "variant": spec.legacy_variant_name,
            "variant_key": spec.name,
            "family": spec.family,
            "source_repo": SOURCE_REPO_URL,
            "resource_archive": RESOURCE_ARCHIVE_URL,
            "base_onnx": spec.mask_base_onnx_name,
            "ds8_onnx": onnx_path.name,
            "task": (
                "whole-body detection, attribute boxes, keypoint boxes, bone boxes, instance masks"
                if spec.has_instance_masks
                else "whole-body detection, attribute boxes, keypoint boxes, and bone boxes"
            ),
        },
        "input": {
            "name": INPUT_NAME,
            "channels": 3,
            "height": INPUT_HEIGHT,
            "width": INPUT_WIDTH,
            "min_batch": MIN_BATCH,
            "opt_batch": OPT_BATCH,
            "max_batch": MAX_BATCH,
            "deepstream_preprocess": "RGB scale 1/255",
            "onnx_preprocess": "ImageNet mean/std normalization baked into graph",
        },
        "outputs": outputs,
        "inference": inference,
        "classes": load_classes(),
    }


def write_model_info(variant: str = "masks") -> Path:
    spec = variant_spec(variant)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_info_path = spec.model_info_path
    model_info_path.write_text(json.dumps(_build_model_info(spec.name), indent=2), encoding="utf-8")
    return model_info_path


def ensure_model_artifacts(
    *,
    allow_download: bool = True,
    allow_export: bool = True,
    allow_engine_build: bool = True,
    variant: str = "masks",
    calibration_cache_path: Optional[Path] = None,
    calibration_images_dir: Optional[Path] = None,
) -> ModelArtifacts:
    spec = variant_spec(variant)
    _ensure_dirs()
    base_path = ensure_base_model(allow_download=allow_download, variant=spec.name)
    onnx_path = ensure_variant_onnx(spec.name, allow_export=allow_export)
    engine_path = build_engine(
        allow_build=allow_engine_build,
        variant=spec.name,
        calibration_cache_path=calibration_cache_path,
        calibration_images_dir=calibration_images_dir,
    )
    model_info_path = write_model_info(spec.name)
    return ModelArtifacts(
        base_onnx_path=base_path,
        ds8_onnx_path=onnx_path,
        engine_path=engine_path,
        labels_path=LABELS_PATH,
        model_info_path=model_info_path,
        calibration_cache_path=spec.calibration_cache_path,
    )


def load_model_info(path: Optional[Path] = None, *, variant: str = "masks") -> Dict[str, object]:
    if path is None:
        path = variant_model_info_path(variant)
    if not path.exists():
        raise FileNotFoundError(f"Model info not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_nvinfer_config(
    *,
    batch_size: int,
    parser_lib_path: Optional[Path] = None,
    score_threshold: float = 0.35,
    mask_threshold: float = 0.50,
    infer_interval: int = 0,
    variant: str = "masks",
) -> Path:
    spec = variant_spec(variant)
    if batch_size < MIN_BATCH or batch_size > MAX_BATCH:
        raise ValueError(f"batch_size must be between {MIN_BATCH} and {MAX_BATCH}, got {batch_size}")
    if int(infer_interval) < 0:
        raise ValueError(f"infer_interval must be >= 0, got {infer_interval}")
    if spec.has_instance_masks:
        if parser_lib_path is None or not parser_lib_path.exists():
            raise FileNotFoundError(f"DEIMv2 parser library missing: {parser_lib_path}")
        template_path = NVINFER_TEMPLATE_PATH
    else:
        template_path = BOXES_NVINFER_TEMPLATE_PATH
    template = template_path.read_text(encoding="utf-8")
    onnx_path = spec.ds8_onnx_path
    engine_path = spec.engine_path
    replacements = {
        "@ONNX_PATH@": str(onnx_path.resolve()),
        "@ENGINE_PATH@": str(engine_path.resolve()),
        "@LABELS_PATH@": str(LABELS_PATH.resolve()),
        "@CUSTOM_LIB@": "" if parser_lib_path is None else str(parser_lib_path.resolve()),
        "@BATCH_SIZE@": str(int(batch_size)),
        "@GIE_UNIQUE_ID@": str(GIE_UNIQUE_ID),
        "@INFER_INTERVAL@": str(int(infer_interval)),
        "@NETWORK_MODE@": str(spec.network_mode),
        "@SCORE_THRESHOLD@": f"{float(score_threshold):.6f}",
        "@MASK_THRESHOLD@": f"{float(mask_threshold):.6f}",
        "@INT8_CALIB_FILE@": (
            f"int8-calib-file={spec.calibration_cache_path.resolve()}"
            if spec.calibration_cache_path is not None and spec.calibration_cache_path.exists()
            else ""
        ),
    }
    contents = template
    for key, value in replacements.items():
        contents = contents.replace(key, value)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = BUILD_ROOT / f"config_infer_deimv2_wholebody49_{spec.config_suffix}_b{batch_size}.ini"
    output_path.write_text(contents, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DEIMv2 Wholebody49 model artifacts")
    parser.add_argument(
        "--model-variant",
        default="masks",
        help=f"Model variant or alias. Available: {', '.join(MODEL_VARIANT_CHOICES)}",
    )
    parser.add_argument(
        "--int8-calibration-cache",
        type=Path,
        default=None,
        help="Existing TensorRT INT8 calibration cache for INT8 variants",
    )
    parser.add_argument(
        "--int8-calibration-images",
        type=Path,
        default=None,
        help="Representative image directory used to build an INT8 calibration cache and engine",
    )
    args = parser.parse_args()
    try:
        variant = normalize_variant(args.model_variant)
    except ValueError as exc:
        parser.error(str(exc))
    artifacts = ensure_model_artifacts(
        variant=variant,
        calibration_cache_path=args.int8_calibration_cache,
        calibration_images_dir=args.int8_calibration_images,
    )
    print(f"Base ONNX: {artifacts.base_onnx_path}")
    print(f"DS8 ONNX: {artifacts.ds8_onnx_path}")
    print(f"Engine: {artifacts.engine_path}")
    print(f"Labels: {artifacts.labels_path}")
    print(f"Model info: {artifacts.model_info_path}")
    if artifacts.calibration_cache_path is not None:
        print(f"INT8 calibration cache: {artifacts.calibration_cache_path}")
