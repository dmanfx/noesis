"""Shared configuration utilities for MapAnything integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import load_ma_config


def _strip_comment(raw: str) -> str:
    return raw.split("#", 1)[0].strip()


def _to_bool(raw: str) -> bool:
    return _strip_comment(raw).lower() in {"1", "true", "yes", "on"}


def _to_int(raw: str) -> int:
    return int(_strip_comment(raw))


def _to_float(raw: str) -> float:
    return float(_strip_comment(raw))


def _to_optional_bytes(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    cleaned = _strip_comment(raw)
    if not cleaned:
        return None
    lower = cleaned.lower()

    suffix_multipliers = {
        "k": 1024,
        "kb": 1024,
        "m": 1024 ** 2,
        "mb": 1024 ** 2,
        "g": 1024 ** 3,
        "gb": 1024 ** 3,
        "t": 1024 ** 4,
        "tb": 1024 ** 4,
    }

    for suffix, multiplier in suffix_multipliers.items():
        if lower.endswith(suffix):
            number = lower[: -len(suffix)].strip()
            if not number:
                raise ValueError(f"Missing numeric component for byte size: {raw}")
            return int(float(number) * multiplier)

    return int(float(cleaned))


@dataclass(frozen=True)
class ServiceSettings:
    host: str
    port: int
    api_key: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class InferenceSettings:
    model_id: str
    device: str
    amp_dtype_name: str
    memory_efficient_mono: bool
    memory_efficient_multi: bool
    apply_mask: bool
    mask_edges: bool
    confidence_percentile: int


@dataclass(frozen=True)
class PerformanceSettings:
    max_res: int
    mono_freq_hz: float
    multi_batch_size: int
    multi_interval_s: float
    min_conf: float


@dataclass(frozen=True)
class StorageSettings:
    depth_base: str
    calib_base: str
    max_snapshots_per_camera: int
    snapshot_retention_minutes: float
    max_total_bytes: Optional[int]


@dataclass(frozen=True)
class ServiceConfig:
    service: ServiceSettings
    inference: InferenceSettings
    performance: PerformanceSettings
    storage: StorageSettings

    @classmethod
    def from_ini(cls, ini_dict: Dict[str, Dict[str, str]]) -> "ServiceConfig":
        service_section = ini_dict.get("service", {})
        inference_section = ini_dict.get("inference", {})
        performance_section = ini_dict.get("performance", {})
        storage_section = ini_dict.get("storage", {})

        service = ServiceSettings(
            host=service_section.get("host", "127.0.0.1"),
            port=_to_int(service_section.get("port", "8001")),
            api_key=service_section.get("api_key", "noesis_secret"),
        )
        inference = InferenceSettings(
            model_id=inference_section.get("model_id", "facebook/map-anything-apache"),
            device=inference_section.get("device", "cuda:0"),
            amp_dtype_name=_normalize_amp(inference_section.get("amp_dtype", "bf16")),
            memory_efficient_mono=_to_bool(inference_section.get("memory_efficient_mono", "false")),
            memory_efficient_multi=_to_bool(inference_section.get("memory_efficient_multi", "true")),
            apply_mask=_to_bool(inference_section.get("apply_mask", "true")),
            mask_edges=_to_bool(inference_section.get("mask_edges", "true")),
            confidence_percentile=_to_int(inference_section.get("confidence_percentile", "10")),
        )
        performance = PerformanceSettings(
            max_res=_to_int(performance_section.get("max_res", "768")),
            mono_freq_hz=_to_float(performance_section.get("mono_freq", "2")),
            multi_batch_size=_to_int(performance_section.get("multi_batch_size", "6")),
            multi_interval_s=_to_float(performance_section.get("multi_interval", "60")),
            min_conf=_to_float(performance_section.get("min_conf", "0.5")),
        )
        storage = StorageSettings(
            depth_base=storage_section.get("depth_base", "data/depth"),
            calib_base=storage_section.get("calib_base", "data/calib"),
            max_snapshots_per_camera=_to_int(storage_section.get("max_snapshots_per_camera", "600")),
            snapshot_retention_minutes=_to_float(storage_section.get("snapshot_retention_minutes", "10")),
            max_total_bytes=_to_optional_bytes(storage_section.get("max_total_bytes")),
        )
        return cls(service=service, inference=inference, performance=performance, storage=storage)


def load_service_config() -> ServiceConfig:
    """Load MapAnything service configuration."""
    ini_dict = load_ma_config()
    return ServiceConfig.from_ini(ini_dict)


__all__ = [
    "ServiceConfig",
    "ServiceSettings",
    "InferenceSettings",
    "PerformanceSettings",
    "StorageSettings",
    "load_service_config",
]
