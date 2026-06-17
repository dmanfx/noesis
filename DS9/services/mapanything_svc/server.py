"""FastAPI microservice for MapAnything inference."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from utils.rate_limited_logger import RateLimitedLogger
from mapanything_config import ServiceConfig, load_service_config

try:
    from mapanything.models import MapAnything  # type: ignore
    from mapanything.utils.image import preprocess_inputs  # type: ignore
except ImportError as exc:  # pragma: no cover - environment must provide mapanything
    MapAnything = None  # type: ignore
    preprocess_inputs = None  # type: ignore
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


T = TypeVar("T")

WEIGHTS_SHA_PATH = Path(__file__).resolve().parents[2] / 'docs/ma-integration/weights.sha'
MAPANYTHING_HF_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--facebook--map-anything-apache"
)


def _resolve_cached_weight_path(path: Path) -> Path:
    if path.exists():
        return path
    try:
        if not path.is_absolute():
            refs_main = MAPANYTHING_HF_CACHE_ROOT / "refs" / "main"
            if refs_main.exists():
                snapshot = refs_main.read_text(encoding="utf-8").strip()
                candidate = MAPANYTHING_HF_CACHE_ROOT / "snapshots" / snapshot / path.as_posix()
                if candidate.exists():
                    return candidate
                candidate_by_name = MAPANYTHING_HF_CACHE_ROOT / "snapshots" / snapshot / path.name
                if candidate_by_name.exists():
                    return candidate_by_name
        snapshots_dir = path.parent.parent
        if snapshots_dir.name != "snapshots":
            return path
        repo_root = snapshots_dir.parent
        refs_main = repo_root / "refs" / "main"
        if refs_main.exists():
            snapshot = refs_main.read_text(encoding="utf-8").strip()
            candidate = repo_root / "snapshots" / snapshot / path.name
            if candidate.exists():
                return candidate
        matches = sorted((repo_root / "snapshots").glob(f"*/{path.name}"))
        if len(matches) == 1 and matches[0].exists():
            return matches[0]
    except Exception:  # pragma: no cover - best effort resolution
        return path
    return path


def verify_weights_checksum() -> None:
    if not WEIGHTS_SHA_PATH.exists():
        return
    try:
        with open(WEIGHTS_SHA_PATH, 'r', encoding='utf-8') as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]
    except Exception as exc:  # pragma: no cover - best effort
        logging.getLogger("mapanything_svc").warning(f"Unable to read weights SHA file: {exc}")
        return
    for line in lines:
        try:
            expected_hash, path_str = line.split(None, 1)
            path = _resolve_cached_weight_path(Path(path_str).expanduser())
            if not path.exists():
                raise FileNotFoundError(f"Weights file missing: {path}")
            hasher = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1_048_576), b''):
                    hasher.update(chunk)
            actual = hasher.hexdigest()
            if actual != expected_hash:
                raise RuntimeError(f"Checksum mismatch for weights: expected {expected_hash}, got {actual}")
        except ValueError:
            logging.getLogger("mapanything_svc").warning(f"Malformed weights entry: {line}")
        except Exception as exc:
            raise RuntimeError(f"Model weights verification failed: {exc}")

class ViewPayload(BaseModel):
    cam_id: str = Field(..., description="Camera identifier for the view")
    img_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded RGB image (HWC uint8). Overrides img if provided.",
    )
    img: Optional[List[List[List[int]]]] = Field(
        default=None,
        description="Nested list RGB image (HWC uint8) when base64 is not used.",
    )
    shape: Optional[Tuple[int, int, int]] = Field(
        default=None,
        description="Optional (H, W, C) shape metadata used with img_b64.",
    )
    intrinsics: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional 3x3 camera intrinsic matrix.",
    )

    @model_validator(mode="after")
    def _ensure_image(cls, values: "ViewPayload") -> "ViewPayload":
        if values.img_b64 is None and values.img is None:
            raise ValueError("Either img_b64 or img must be provided")
        return values

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, shape: Optional[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        if shape is None:
            return shape
        if len(shape) != 3:
            raise ValueError("shape must be a tuple of (H, W, C)")
        return shape


class MonoRequest(BaseModel):
    view: ViewPayload


class MonoResponse(BaseModel):
    cam_id: str
    depth_b64: str
    conf_b64: str
    mask_b64: str
    shape: Tuple[int, int]
    ts_us: int


class MultiRequest(BaseModel):
    scene_id: str = Field(..., description="Multi-view scene identifier")
    views: List[ViewPayload]

    @field_validator("views")
    @classmethod
    def _require_views(cls, views: List[ViewPayload]) -> List[ViewPayload]:
        if not views:
            raise ValueError("At least one view is required")
        return views


class MultiResponse(BaseModel):
    poses: Dict[str, List[float]]
    intrinsics: Dict[str, List[List[float]]]
    depth_b64: Dict[str, str]
    conf_b64: Dict[str, str]
    mask_b64: Dict[str, str]
    shapes: Dict[str, Tuple[int, int]]
    scale: float
    ts_us: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    started_at: float


class MetricsResponse(BaseModel):
    mono_latency_ms: float
    multi_latency_ms: float
    mono_count: int
    multi_count: int
    vram_gb: float
    queue_depths: Dict[str, int]
    last_error: Optional[str]


# ---------------------------------------------------------------------------
# Internal service state
# ---------------------------------------------------------------------------


@dataclass
class SceneQueue:
    queue: "asyncio.Queue[Tuple[MultiRequest, asyncio.Future[MultiResponse]]]"
    worker: asyncio.Task


@dataclass
class ServiceState:
    config: ServiceConfig
    logger: RateLimitedLogger
    model: Optional["MapAnything"] = None
    model_lock: asyncio.Lock = asyncio.Lock()
    started_at: float = time.time()
    last_mono_latency_ms: float = 0.0
    last_multi_latency_ms: float = 0.0
    mono_count: int = 0
    multi_count: int = 0
    last_error: Optional[str] = None
    scene_queues: Dict[str, SceneQueue] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.scene_queues is None:
            self.scene_queues = {}

    def _memory_efficient_flag(self, *, is_multi: bool) -> bool:
        baseline = self.config.inference.memory_efficient_multi if is_multi else self.config.inference.memory_efficient_mono
        device_str = self.config.inference.device
        if not torch.cuda.is_available() or not device_str.startswith('cuda'):
            return baseline
        try:
            device_index = 0
            if ':' in device_str:
                _, idx = device_str.split(':', 1)
                device_index = int(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            if total_bytes <= 0:
                return baseline
            usage_ratio = 1.0 - (free_bytes / total_bytes)
            if usage_ratio >= 0.8:
                return True
        except Exception:
            return baseline
        return baseline

    async def ensure_model_loaded(self) -> "MapAnything":
        if IMPORT_ERROR is not None:
            raise RuntimeError(
                "MapAnything is not available. Verify installation before starting service."
            )
        if not self.config.inference.model_id.endswith('-apache'):
            raise RuntimeError("Only Apache-licensed MapAnything models are permitted")
        async with self.model_lock:
            if self.model is None:
                self.logger.info("Loading MapAnything model: %s", self.config.inference.model_id)
                cache_dir = os.environ.get("MAPANYTHING_WEIGHTS")
                model = await asyncio.to_thread(
                    MapAnything.from_pretrained,  # type: ignore[arg-type]
                    self.config.inference.model_id,
                    cache_dir=cache_dir,
                )
                device = torch.device(self.config.inference.device)
                model.to(device)
                model.eval()
                self.model = model
                try:
                    verify_weights_checksum()
                except Exception as exc:
                    self.logger.error(f"Model checksum verification failed: {exc}")
                    raise
                self.logger.info("MapAnything weights checksum verified")
                self.logger.info("MapAnything model ready on %s", device)
            return self.model

    def _create_scene_worker(self, scene_id: str) -> SceneQueue:
        async_queue: "asyncio.Queue[Tuple[MultiRequest, asyncio.Future[MultiResponse]]]" = asyncio.Queue(maxsize=1)

        async def worker() -> None:
            while True:
                request, future = await async_queue.get()
                try:
                    response = await self._process_multi_request(request)
                    if not future.done():
                        future.set_result(response)
                except Exception as exc:  # pragma: no cover - logged below
                    self.last_error = str(exc)
                    if not future.done():
                        future.set_exception(exc)
                    self.logger.error(f"Multi-view processing failed for scene {request.scene_id}: {exc}")
                finally:
                    async_queue.task_done()

        task = asyncio.create_task(worker())
        queue = SceneQueue(queue=async_queue, worker=task)
        self.scene_queues[scene_id] = queue
        return queue

    def get_scene_queue(self, scene_id: str) -> SceneQueue:
        queue = self.scene_queues.get(scene_id)
        if queue is None:
            queue = self._create_scene_worker(scene_id)
        return queue

    async def enqueue_multi_request(self, request: MultiRequest) -> MultiResponse:
        queue = self.get_scene_queue(request.scene_id)
        future: "asyncio.Future[MultiResponse]" = asyncio.get_event_loop().create_future()
        try:
            queue.queue.put_nowait((request, future))
        except asyncio.QueueFull:
            try:
                dropped_request, dropped_future = queue.queue.get_nowait()
                if not dropped_future.done():
                    dropped_future.set_exception(
                        HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Previous multi-view request dropped")
                    )
                queue.queue.task_done()
            except asyncio.QueueEmpty:
                pass
            queue.queue.put_nowait((request, future))
        return await future

    async def _process_mono_request(self, request: MonoRequest) -> MonoResponse:
        model = await self.ensure_model_loaded()
        np_view, intrinsics = _decode_view(request.view)
        if intrinsics is None:
            h, w = np_view.shape[:2]
            fx = fy = float(max(w, h))
            cx = float(w) / 2.0
            cy = float(h) / 2.0
            intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        prepared_inputs = preprocess_inputs(  # type: ignore[operator]
            [
                {
                    "img": np_view,
                    "intrinsics": intrinsics,
                }
            ],
            resize_mode="fixed_mapping",
        ) if preprocess_inputs else _fallback_preprocess([
            {
                "img": np_view,
                "intrinsics": intrinsics,
            }
        ])

        memory_flag_mono = self._memory_efficient_flag(is_multi=False)

        def _infer() -> List[Dict[str, Any]]:
            with torch.inference_mode():
                return model.infer(  # type: ignore[call-arg]
                    prepared_inputs,
                    memory_efficient_inference=memory_flag_mono,
                    use_amp=True,
                    amp_dtype=self.config.inference.amp_dtype_name,
                    apply_mask=self.config.inference.apply_mask,
                    mask_edges=self.config.inference.mask_edges,
                    confidence_percentile=self.config.inference.confidence_percentile,
                )

        tic = time.perf_counter()
        predictions = await _run_with_retries(_infer)
        toc = time.perf_counter()
        self.last_mono_latency_ms = (toc - tic) * 1000.0
        self.mono_count += 1

        if not predictions:
            raise RuntimeError("MapAnything returned no predictions")
        pred = predictions[0]
        depth, conf, mask = _extract_prediction_slices(pred)
        shape = depth.shape
        return MonoResponse(
            cam_id=request.view.cam_id,
            depth_b64=_encode_float32(depth),
            conf_b64=_encode_float32(conf),
            mask_b64=_encode_mask(mask),
            shape=(int(shape[0]), int(shape[1])),
            ts_us=int(time.time() * 1_000_000),
        )

    async def _process_multi_request(self, request: MultiRequest) -> MultiResponse:
        model = await self.ensure_model_loaded()
        prepared_entries: List[Dict[str, Any]] = []
        cam_order: List[str] = []
        for view in request.views:
            np_view, intrinsics = _decode_view(view)
            if intrinsics is None:
                h, w = np_view.shape[:2]
                fx = fy = float(max(w, h))
                cx = float(w) / 2.0
                cy = float(h) / 2.0
                intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            prepared_entries.append({"img": np_view, "intrinsics": intrinsics})
            cam_order.append(view.cam_id)
        prepared_inputs = preprocess_inputs(  # type: ignore[operator]
            prepared_entries,
            resize_mode="fixed_mapping",
        ) if preprocess_inputs else _fallback_preprocess(prepared_entries)

        memory_flag_multi = self._memory_efficient_flag(is_multi=True)

        def _infer() -> List[Dict[str, Any]]:
            with torch.inference_mode():
                return model.infer(  # type: ignore[call-arg]
                    prepared_inputs,
                    memory_efficient_inference=memory_flag_multi,
                    use_amp=True,
                    amp_dtype=self.config.inference.amp_dtype_name,
                    apply_mask=self.config.inference.apply_mask,
                    mask_edges=self.config.inference.mask_edges,
                    confidence_percentile=self.config.inference.confidence_percentile,
                )

        tic = time.perf_counter()
        predictions = await _run_with_retries(_infer)
        toc = time.perf_counter()
        self.last_multi_latency_ms = (toc - tic) * 1000.0
        self.multi_count += 1

        if len(predictions) != len(cam_order):
            raise RuntimeError("Mismatch between predictions and requested views")

        depth_b64: Dict[str, str] = {}
        conf_b64: Dict[str, str] = {}
        mask_b64: Dict[str, str] = {}
        shapes: Dict[str, Tuple[int, int]] = {}
        poses: Dict[str, List[float]] = {}
        intrinsics_dict: Dict[str, List[List[float]]] = {}
        scales: List[float] = []

        for cam_id, pred in zip(cam_order, predictions):
            depth, conf, mask = _extract_prediction_slices(pred)
            depth_b64[cam_id] = _encode_float32(depth)
            conf_b64[cam_id] = _encode_float32(conf)
            mask_b64[cam_id] = _encode_mask(mask)
            shapes[cam_id] = (int(depth.shape[0]), int(depth.shape[1]))

            pose_tensor = pred.get("camera_poses")
            if pose_tensor is not None:
                pose_matrix = _squeeze_to_matrix(pose_tensor)
                poses[cam_id] = pose_matrix.reshape(-1).tolist()

            intrinsics_tensor = pred.get("intrinsics")
            if intrinsics_tensor is not None:
                intrinsics_matrix = _squeeze_to_matrix(intrinsics_tensor)
                intrinsics_dict[cam_id] = intrinsics_matrix.tolist()

            scale_tensor = pred.get("metric_scaling_factor")
            if scale_tensor is not None:
                scale_value = float(_squeeze_to_scalar(scale_tensor))
                scales.append(scale_value)

        scale = float(np.mean(scales)) if scales else 1.0
        return MultiResponse(
            poses=poses,
            intrinsics=intrinsics_dict,
            depth_b64=depth_b64,
            conf_b64=conf_b64,
            mask_b64=mask_b64,
            shapes=shapes,
            scale=scale,
            ts_us=int(time.time() * 1_000_000),
        )

    def build_metrics(self) -> MetricsResponse:
        queue_depths: Dict[str, int] = {scene_id: queue.queue.qsize() for scene_id, queue in self.scene_queues.items()}
        vram_gb = 0.0
        if torch.cuda.is_available():
            try:
                device = torch.device(self.config.inference.device)
                vram_gb = float(torch.cuda.memory_allocated(device)) / (1024 ** 3)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.last_error = str(exc)
        return MetricsResponse(
            mono_latency_ms=self.last_mono_latency_ms,
            multi_latency_ms=self.last_multi_latency_ms,
            mono_count=self.mono_count,
            multi_count=self.multi_count,
            vram_gb=vram_gb,
            queue_depths=queue_depths,
            last_error=self.last_error,
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _decode_view(view: ViewPayload) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if view.img_b64 is not None:
        if view.shape is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "shape must be provided with img_b64")
        raw = base64.b64decode(view.img_b64)
        expected_size = int(view.shape[0]) * int(view.shape[1]) * int(view.shape[2])
        if len(raw) != expected_size:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Decoded image size mismatch")
        image = np.frombuffer(raw, dtype=np.uint8).reshape(view.shape)
    else:
        image = np.asarray(view.img, dtype=np.uint8)  # type: ignore[arg-type]
    if image.ndim != 3 or image.shape[2] != 3:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Image must have shape (H, W, 3)")
    intrinsics = np.asarray(view.intrinsics, dtype=np.float32) if view.intrinsics is not None else None
    return image, intrinsics


def _fallback_preprocess(views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    norm = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    for entry in views:
        image: np.ndarray = entry["img"]
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - norm[:, None, None]) / std[:, None, None]
        processed_entry: Dict[str, Any] = {
            "img": tensor[None],
            "data_norm_type": ["dinov2"],
        }
        if entry.get("intrinsics") is not None:
            processed_entry["intrinsics"] = torch.as_tensor(entry["intrinsics"], dtype=torch.float32)[None]
        processed.append(processed_entry)
    return processed


def _encode_float32(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.float32).tobytes()).decode('ascii')


def _encode_mask(mask: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).decode('ascii')


def _extract_prediction_slices(pred: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_tensor = pred.get("depth_z")
    conf_tensor = pred.get("conf")
    mask_tensor = pred.get("mask")
    if depth_tensor is None or conf_tensor is None:
        raise RuntimeError("Prediction missing depth or confidence outputs")

    depth_np = _squeeze_to_array(depth_tensor, channel_last=True)
    conf_np = _squeeze_to_array(conf_tensor, channel_last=False)
    mask_np = _squeeze_to_array(mask_tensor, channel_last=True) if mask_tensor is not None else np.ones_like(depth_np, dtype=bool)
    return depth_np.astype(np.float32), conf_np.astype(np.float32), mask_np.astype(bool)


def _squeeze_to_array(tensor: Any, *, channel_last: bool) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    if array.ndim == 4:  # (B, H, W, C)
        array = array[0]
    if array.ndim == 3:
        if channel_last:
            array = array[..., 0]
        else:
            array = array[0]
    return array


def _squeeze_to_matrix(tensor: Any) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    while array.ndim > 2:
        array = array[0]
    return array


def _squeeze_to_scalar(tensor: Any) -> float:
    if isinstance(tensor, torch.Tensor):
        value = tensor.detach().cpu().reshape(-1)[0]
        return float(value.item()) if hasattr(value, "item") else float(value)
    array = np.asarray(tensor).reshape(-1)[0]
    return float(array)


async def _run_with_retries(callable_: Callable[[], T], *, max_attempts: int = 3, base_delay: float = 0.5) -> T:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.to_thread(callable_)
        except Exception as exc:  # pragma: no cover - propagation handled above
            last_exc = exc
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
    raise last_exc or RuntimeError("Operation failed after retries")


def _create_logger() -> RateLimitedLogger:
    logging.basicConfig(level=logging.INFO)
    base_logger = logging.getLogger("mapanything_svc")
    base_logger.setLevel(logging.INFO)
    return RateLimitedLogger(base_logger, rate_limit_seconds=2.0)


# ---------------------------------------------------------------------------
# FastAPI wiring
# ---------------------------------------------------------------------------


config = load_service_config()
logger = _create_logger()
state = ServiceState(config=config, logger=logger)
app = FastAPI(title="MapAnything Service", version="1.0.0")


class _AccessLogOnceFilter(logging.Filter):
    """Filter that allows a matching access log entry only the first time."""

    def __init__(self, needle: str) -> None:
        super().__init__()
        self._needle = needle
        self._seen = False
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logging behavior
        try:
            message = record.getMessage()
        except Exception:
            return True
        if self._needle not in message:
            return True
        with self._lock:
            if self._seen:
                return False
            self._seen = True
            return True


logging.getLogger("uvicorn.access").addFilter(_AccessLogOnceFilter("POST /infer_mono"))


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    expected = state.config.service.api_key
    if expected and x_api_key != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting MapAnything microservice")
    if IMPORT_ERROR is not None:
        logger.error("MapAnything import failed: %s", IMPORT_ERROR)
        raise RuntimeError("MapAnything not installed")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", state.config.inference.device.split(":")[-1])
    torch.backends.cuda.matmul.allow_tf32 = True
    await state.ensure_model_loaded()


@app.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        device=state.config.inference.device,
        started_at=state.started_at,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    return state.build_metrics()


@app.post("/infer_mono", response_model=MonoResponse, dependencies=[Depends(verify_api_key)])
async def infer_mono(request: MonoRequest) -> MonoResponse:
    try:
        return await state._process_mono_request(request)
    except HTTPException:
        raise
    except Exception as exc:
        state.last_error = str(exc)
        logger.error(f"Mono inference failed: {exc}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Mono inference failed") from exc


@app.post("/infer_multi", response_model=MultiResponse, dependencies=[Depends(verify_api_key)])
async def infer_multi(request: MultiRequest) -> MultiResponse:
    try:
        return await state.enqueue_multi_request(request)
    except HTTPException:
        raise
    except Exception as exc:
        state.last_error = str(exc)
        logger.error(f"Multi inference failed: {exc}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Multi inference failed") from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Any, exc: Exception) -> JSONResponse:  # pragma: no cover - global safety net
    state.last_error = str(exc)
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal server error"})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "services.mapanything_svc.server:app",
        host=config.service.host,
        port=config.service.port,
        reload=False,
    )
