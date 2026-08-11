"""Validated DS9 MapAnything tensor and candidate-build profiles.

The default profile describes the current production contract. Candidate
profiles are inert until a pipeline model entry explicitly names one.
"""

from __future__ import annotations

import configparser
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
PROFILE_AUTHORITY = DS9_ROOT / "config" / "mapanything_profiles.json"
FLOAT32_BYTES = 4
INPUT_CHANNELS = 3


class MapAnythingProfileError(ValueError):
    """Raised when a MapAnything profile or binding is inconsistent."""


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MapAnythingProfileError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MapAnythingProfileError(f"{label} must be a non-negative integer")
    return value


def _logical_path(value: object, label: str) -> str:
    raw = str(value or "").strip()
    candidate = Path(raw)
    if (
        not raw
        or candidate.is_absolute()
        or ".." in candidate.parts
        or tuple(candidate.parts[:1]) != ("DS9",)
        or candidate.as_posix() != raw
    ):
        raise MapAnythingProfileError(
            f"{label} must be a portable path below DS9/"
        )
    return raw


@dataclass(frozen=True)
class MapAnythingProfile:
    """One fixed-shape, fixed-batch MapAnything engine contract."""

    name: str
    status: str
    batch_size: int
    input_height: int
    input_width: int
    patch_size: int
    opset: int
    precision: str
    interval: int
    onnx: str
    engine: str
    infer_config: str
    functional_fixture: str
    output_layers: tuple[str, ...]
    builder_optimization_level: int
    max_aux_streams: int
    workspace_mib: int

    @property
    def pixels_per_frame(self) -> int:
        return self.input_height * self.input_width

    @property
    def input_batch_bytes(self) -> int:
        return (
            self.batch_size
            * INPUT_CHANNELS
            * self.pixels_per_frame
            * FLOAT32_BYTES
        )

    @property
    def output_bytes_per_frame(self) -> int:
        return len(self.output_layers) * self.pixels_per_frame * FLOAT32_BYTES

    @property
    def output_batch_bytes(self) -> int:
        return self.batch_size * self.output_bytes_per_frame

    @property
    def fixed_shape(self) -> str:
        return (
            f"{self.batch_size}x{INPUT_CHANNELS}x"
            f"{self.input_height}x{self.input_width}"
        )

    @property
    def trtexec_build_args(self) -> tuple[str, ...]:
        shape = f"images:{self.fixed_shape}"
        return (
            f"--minShapes={shape}",
            f"--optShapes={shape}",
            f"--maxShapes={shape}",
            f"--builderOptimizationLevel={self.builder_optimization_level}",
            f"--maxAuxStreams={self.max_aux_streams}",
            f"--memPoolSize=workspace:{self.workspace_mib}",
        )

    def logical_path(self, field: str) -> Path:
        try:
            raw = getattr(self, field)
        except AttributeError as exc:
            raise MapAnythingProfileError(
                f"unknown MapAnything profile path field {field!r}"
            ) from exc
        return Path(str(raw))

    def workspace_path(
        self,
        field: str,
        *,
        artifact_root: Path | None = None,
    ) -> Path:
        logical = self.logical_path(field)
        if artifact_root is None:
            return REPO_ROOT / logical
        if tuple(logical.parts[:2]) != ("DS9", "models"):
            raise MapAnythingProfileError(
                f"{field} is not a DS9 model artifact path"
            )
        return artifact_root / Path(*logical.parts[1:])

    def validate_runtime_binding(self, model_config: Mapping[str, Any]) -> None:
        batch_size = model_config.get("batch_size", model_config.get("batch-size"))
        if batch_size is None or int(batch_size) != self.batch_size:
            raise MapAnythingProfileError(
                f"MapAnything profile {self.name} requires batch_size="
                f"{self.batch_size}"
            )
        explicit_profile = model_config.get("profile") is not None
        engine_raw = str(
            model_config.get("engine")
            or model_config.get("model-engine-file")
            or ""
        ).strip()
        if explicit_profile and not engine_raw:
            raise MapAnythingProfileError(
                f"explicit MapAnything profile {self.name} requires its engine"
            )
        if engine_raw and Path(engine_raw).name != Path(self.engine).name:
            raise MapAnythingProfileError(
                f"MapAnything profile {self.name} requires engine "
                f"{Path(self.engine).name}"
            )
        infer_config_raw = str(
            model_config.get("config-file-path")
            or model_config.get("config_file_path")
            or model_config.get("infer_config")
            or ""
        ).strip()
        if explicit_profile and not infer_config_raw:
            raise MapAnythingProfileError(
                f"explicit MapAnything profile {self.name} requires inference "
                f"config {Path(self.infer_config).name}"
            )
        if (
            infer_config_raw
            and Path(infer_config_raw).name != Path(self.infer_config).name
        ):
            raise MapAnythingProfileError(
                f"MapAnything profile {self.name} requires inference config "
                f"{Path(self.infer_config).name}"
            )

    def validate_infer_config(self) -> None:
        config_path = self.workspace_path("infer_config")
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        try:
            loaded = parser.read(config_path, encoding="utf-8")
        except configparser.Error as exc:
            raise MapAnythingProfileError(
                f"unable to parse {self.name} inference config"
            ) from exc
        if loaded != [str(config_path)] or "property" not in parser:
            raise MapAnythingProfileError(
                f"missing {self.name} inference config {config_path}"
            )
        properties = parser["property"]
        expected = {
            "interval": str(self.interval),
            "model-engine-file": self.engine,
            "batch-size": str(self.batch_size),
            "network-mode": "0",
            "infer-dims": (
                f"{INPUT_CHANNELS};{self.input_height};{self.input_width}"
            ),
            "output-blob-names": ";".join(self.output_layers),
            "maintain-aspect-ratio": "1",
            "symmetric-padding": "1",
            "scaling-compute-hw": "1",
            "scaling-filter": "4",
            "output-tensor-meta": "1",
        }
        mismatches = {
            key: {"expected": value, "observed": properties.get(key)}
            for key, value in expected.items()
            if properties.get(key) != value
        }
        if mismatches:
            raise MapAnythingProfileError(
                f"{self.name} inference config drifted: {mismatches}"
            )

    def source_tensor_contract(self) -> dict[str, Any]:
        batch_dimension: int | str = "batch"
        outputs = [
            {
                "name": name,
                "dtype": "FLOAT",
                "shape": [
                    batch_dimension,
                    1,
                    self.input_height,
                    self.input_width,
                ],
            }
            for name in self.output_layers
        ]
        return {
            "input": {
                "name": "images",
                "dtype": "FLOAT",
                "shape": [
                    batch_dimension,
                    INPUT_CHANNELS,
                    self.input_height,
                    self.input_width,
                ],
            },
            "outputs": outputs,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "batch_size": self.batch_size,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "patch_size": self.patch_size,
            "opset": self.opset,
            "precision": self.precision,
            "interval": self.interval,
            "onnx": self.onnx,
            "engine": self.engine,
            "infer_config": self.infer_config,
            "functional_fixture": self.functional_fixture,
            "output_layers": list(self.output_layers),
            "builder": {
                "builder_optimization_level": self.builder_optimization_level,
                "max_aux_streams": self.max_aux_streams,
                "workspace_mib": self.workspace_mib,
            },
            "derived": {
                "pixels_per_frame": self.pixels_per_frame,
                "input_batch_bytes": self.input_batch_bytes,
                "output_bytes_per_frame": self.output_bytes_per_frame,
                "output_batch_bytes": self.output_batch_bytes,
                "fixed_shape": self.fixed_shape,
            },
        }


def _parse_profile(name: str, payload: object) -> MapAnythingProfile:
    required = {
        "status",
        "batch_size",
        "input_height",
        "input_width",
        "patch_size",
        "opset",
        "precision",
        "interval",
        "onnx",
        "engine",
        "infer_config",
        "functional_fixture",
        "output_layers",
        "builder",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise MapAnythingProfileError(
            f"MapAnything profile {name!r} has unexpected or missing fields"
        )
    if not name or not all(character.isalnum() or character == "_" for character in name):
        raise MapAnythingProfileError(f"invalid MapAnything profile name {name!r}")
    status = str(payload["status"])
    if status not in {"canonical", "candidate"}:
        raise MapAnythingProfileError(
            f"MapAnything profile {name} has invalid status {status!r}"
        )
    height = _positive_int(payload["input_height"], f"{name}.input_height")
    width = _positive_int(payload["input_width"], f"{name}.input_width")
    patch_size = _positive_int(payload["patch_size"], f"{name}.patch_size")
    if height % patch_size or width % patch_size:
        raise MapAnythingProfileError(
            f"MapAnything profile {name} dimensions must be patch aligned"
        )
    if not math.isclose(width / height, 16.0 / 9.0, rel_tol=0.02):
        raise MapAnythingProfileError(
            f"MapAnything profile {name} must remain within 2% of 16:9"
        )
    precision = str(payload["precision"])
    if precision != "fp32":
        raise MapAnythingProfileError(
            f"MapAnything profile {name} must use correctness-first fp32"
        )
    layers = payload["output_layers"]
    if not isinstance(layers, list) or layers != ["depth", "conf", "mask"]:
        raise MapAnythingProfileError(
            f"MapAnything profile {name} outputs must be depth/conf/mask"
        )
    builder = payload["builder"]
    if not isinstance(builder, Mapping) or set(builder) != {
        "builder_optimization_level",
        "max_aux_streams",
        "workspace_mib",
    }:
        raise MapAnythingProfileError(
            f"MapAnything profile {name} builder contract is invalid"
        )
    optimization_level = builder["builder_optimization_level"]
    if (
        isinstance(optimization_level, bool)
        or not isinstance(optimization_level, int)
        or not 0 <= optimization_level <= 5
    ):
        raise MapAnythingProfileError(
            f"{name}.builder_optimization_level must be in [0, 5]"
        )
    max_aux_streams = builder["max_aux_streams"]
    if (
        isinstance(max_aux_streams, bool)
        or not isinstance(max_aux_streams, int)
        or max_aux_streams < 0
    ):
        raise MapAnythingProfileError(
            f"{name}.max_aux_streams must be a non-negative integer"
        )
    return MapAnythingProfile(
        name=name,
        status=status,
        batch_size=_positive_int(payload["batch_size"], f"{name}.batch_size"),
        input_height=height,
        input_width=width,
        patch_size=patch_size,
        opset=_positive_int(payload["opset"], f"{name}.opset"),
        precision=precision,
        interval=_nonnegative_int(payload["interval"], f"{name}.interval"),
        onnx=_logical_path(payload["onnx"], f"{name}.onnx"),
        engine=_logical_path(payload["engine"], f"{name}.engine"),
        infer_config=_logical_path(payload["infer_config"], f"{name}.infer_config"),
        functional_fixture=_logical_path(
            payload["functional_fixture"], f"{name}.functional_fixture"
        ),
        output_layers=tuple(layers),
        builder_optimization_level=optimization_level,
        max_aux_streams=max_aux_streams,
        workspace_mib=_positive_int(
            builder["workspace_mib"], f"{name}.workspace_mib"
        ),
    )


@lru_cache(maxsize=4)
def load_mapanything_profiles(
    authority_path: Path = PROFILE_AUTHORITY,
) -> tuple[str, dict[str, MapAnythingProfile]]:
    try:
        payload = json.loads(authority_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MapAnythingProfileError(
            f"unable to load MapAnything profile authority {authority_path}"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "default_profile", "profiles"}
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("profiles"), Mapping)
    ):
        raise MapAnythingProfileError("invalid MapAnything profile authority")
    profiles = {
        str(name): _parse_profile(str(name), value)
        for name, value in payload["profiles"].items()
    }
    for profile in profiles.values():
        profile.validate_infer_config()
    default_name = str(payload.get("default_profile") or "")
    if default_name not in profiles or profiles[default_name].status != "canonical":
        raise MapAnythingProfileError(
            "default MapAnything profile must name a canonical profile"
        )
    if sum(profile.status == "canonical" for profile in profiles.values()) != 1:
        raise MapAnythingProfileError(
            "MapAnything profile authority must have exactly one canonical profile"
        )
    return default_name, profiles


def get_mapanything_profile(
    name: str | None = None,
    *,
    authority_path: Path = PROFILE_AUTHORITY,
) -> MapAnythingProfile:
    default_name, profiles = load_mapanything_profiles(authority_path)
    selected = str(name or default_name).strip()
    try:
        return profiles[selected]
    except KeyError as exc:
        raise MapAnythingProfileError(
            f"unknown MapAnything profile {selected!r}; "
            f"valid profiles: {', '.join(sorted(profiles))}"
        ) from exc


def resolve_runtime_mapanything_profile(
    model_config: Mapping[str, Any],
) -> MapAnythingProfile:
    raw_name = model_config.get("profile")
    profile = get_mapanything_profile(
        str(raw_name).strip() if raw_name is not None else None
    )
    profile.validate_runtime_binding(model_config)
    return profile


__all__ = [
    "MapAnythingProfile",
    "MapAnythingProfileError",
    "PROFILE_AUTHORITY",
    "get_mapanything_profile",
    "load_mapanything_profiles",
    "resolve_runtime_mapanything_profile",
]
