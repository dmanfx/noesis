"""Strict DS9 ownership and contract validation for V3DT profiles."""

from __future__ import annotations

import hashlib
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from noesis_core.strict_json import strict_json_loads


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
DS9_CONFIG_ROOT = DS9_ROOT / "config"
DS9_V3DT_CONFIG_ROOT = DS9_CONFIG_ROOT / "v3dt"


def _explicit_artifact_root() -> Path | None:
    raw = str(os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", "") or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"NOESIS_DS9_ARTIFACT_ROOT must be absolute: {raw}")
    resolved = candidate.resolve(strict=False)
    if resolved in {Path("/"), REPO_ROOT.resolve(), DS9_ROOT.resolve()}:
        raise ValueError(f"refusing unsafe NOESIS_DS9_ARTIFACT_ROOT: {resolved}")
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(
            f"NOESIS_DS9_ARTIFACT_ROOT must not be inside the checkout: {resolved}"
        )
    return resolved


def _artifact_dir(env_name: str, relative: str, default: Path) -> Path:
    override = str(os.environ.get(env_name, "") or "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{env_name} must be absolute: {override}")
        return path.resolve(strict=False)
    artifact_root = _explicit_artifact_root()
    if artifact_root is not None:
        return (artifact_root / relative).resolve(strict=False)
    return default.resolve(strict=False)


def _subdir(env_name: str, default: Path) -> Path:
    override = str(os.environ.get(env_name, "") or "").strip()
    if not override:
        return default.resolve(strict=False)
    path = Path(override).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{env_name} must be absolute: {override}")
    return path.resolve(strict=False)


DS9_MODEL_ROOT = _artifact_dir("NOESIS_MODEL_DIR", "models", DS9_ROOT / "models")
DS9_ONNX_ROOT = _subdir("NOESIS_ONNX_DIR", DS9_MODEL_ROOT / "onnx")
DS9_ENGINE_ROOT = _subdir("NOESIS_ENGINE_DIR", DS9_MODEL_ROOT / "engines")
V3DT_CAMERAS_CONFIG = DS9_CONFIG_ROOT / "cameras_v3dt.yaml"

TRACKER_REID_SOURCE = DS9_MODEL_ROOT / "tracker_reid" / "resnet50_market1501.etlt"
TRACKER_REID_SOURCE_SHA256 = (
    "0e5b7f702ce7e3734e45f27b819866f15f6b083048481d1afae2089136e20201"
)
TRACKER_REID_PROVENANCE = (
    DS9_ROOT / "models" / "tracker_reid" / "resnet50_market1501.etlt.provenance.json"
)
TRACKER_REID_ENGINE = (
    DS9_ENGINE_ROOT / "tracker_reid_resnet50_market1501_b32_fp16.engine"
)
BODYPOSE_SOURCE = DS9_ONNX_ROOT / "bodypose3dnet_accuracy.onnx"
BODYPOSE_SOURCE_SHA256 = (
    "0452b785a70fcd6bc5bd4069249bdfd85eb139c9e9216bcf81f89df33945d028"
)
BODYPOSE_PROVENANCE = (
    DS9_ROOT / "models" / "onnx" / "bodypose3dnet_accuracy.onnx.provenance.json"
)
BODYPOSE_ENGINE = DS9_ENGINE_ROOT / "bodypose3dnet_accuracy_b1_fp16.engine"

EXPECTED_CAMERA_ORDER = ("living-room", "kitchen", "family-room")
EXPECTED_CAMINFO_NAMES = tuple(f"camInfo_{name}.yml" for name in EXPECTED_CAMERA_ORDER)
MV3DT_PUBLISH_TOPICS = (
    "localhost:1883;ds3d/cam0",
    "localhost:1883;ds3d/cam1",
    "localhost:1883;ds3d/cam2",
)
MV3DT_SUBSCRIBE_TOPICS = (
    (MV3DT_PUBLISH_TOPICS[0],),
    (MV3DT_PUBLISH_TOPICS[2],),
    (MV3DT_PUBLISH_TOPICS[1],),
)
MV3DT_ASSOCIATOR_CONTRACT = {
    "multiViewAssociatorType": 1,
    "enableLatePeerReAssoc": 1,
    "enableIDCorrection": 1,
    "enableSeeThrough": 1,
    "enableMsgSync": 1,
    "maxPeerTrackletSize": 50,
    "recentlyActiveAge": 178,
    "minCommonFrames4MatchScore": 2,
    "maxPeerToPredDistance4Fusion": 1.35,
    "minPeerVisibility4Fusion": 0.15,
    "minPeerTrackletMatchScore": 0.48,
    "maxTrackletMatchingTimeSearchRange": 1,
    "maxPeerFrameDiff4NoDet": 2,
    "communicatorInitSleepTime": 0,
}
MV3DT_KITCHEN_FAMILY_ASSOCIATOR_OVERRIDES = {
    # Doorway occlusion can displace the vendor cylinder-foot estimate by about
    # 4 m. Permit ID adoption after two common frames; appearance remains part
    # of the vendor score, and the multi-person replay is the false-merge gate.
    "minCommonFrames4MatchScore": 2,
    "maxPeerToPredDistance4Fusion": 4.75,
    "minPeerVisibility4Fusion": 0.05,
    "minPeerTrackletMatchScore": 0.18,
}
EXPECTED_STREAM_SIZE = (1920, 1080)
EXPECTED_OBJECT_MODEL_HEIGHT_M = 2.2
MV3DT_KITCHEN_FAMILY_OBJECT_MODEL_HEIGHT_M = 1.7
EXPECTED_OBJECT_MODEL_RADIUS_M = 0.35
EXPECTED_BODYPOSE_INPUTS = {
    "input0": ["batch", 3, 256, 192],
    "k_inv": ["batch", 3, 3],
    "mean_limb_lengths": ["batch", 36],
    "scale_normalized_mean_limb_lengths": ["batch", 36],
    "t_form_inv": ["batch", 3, 3],
}


class V3DTAssetError(ValueError):
    """Raised when the DS9 V3DT profile is incomplete or crosses ownership roots."""


def derive_nvmot_tracker_engine_path(
    staged_etlt: Path,
    *,
    batch_size: int,
    gpu_id: int,
    network_mode: int,
) -> Path:
    """Return the exact source-adjacent engine name emitted by NvMOT."""

    source = Path(staged_etlt).expanduser().resolve(strict=False)
    if source.suffix.lower() != ".etlt":
        raise V3DTAssetError(f"NvMOT tracker source must be ETLT: {source}")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise V3DTAssetError(f"NvMOT tracker batch size must be positive: {batch_size}")
    if isinstance(gpu_id, bool) or int(gpu_id) < 0:
        raise V3DTAssetError(f"NvMOT tracker GPU ID must be nonnegative: {gpu_id}")
    precision = {0: "fp32", 1: "fp16", 2: "int8"}.get(int(network_mode))
    if precision is None:
        raise V3DTAssetError(
            f"unsupported NvMOT tracker network mode: {network_mode}"
        )
    return source.with_name(
        f"{source.name}_b{int(batch_size)}_gpu{int(gpu_id)}_{precision}.engine"
    )


@dataclass(frozen=True)
class V3DTAssetBundle:
    profile: str
    pipeline_config: Path
    cameras_config: Path
    tracker_config: Path
    pub_sub_config: Path | None
    mqtt_config_template: Path | None
    camera_models: tuple[Path, ...]
    tracker_reid_source: Path
    tracker_reid_engine: Path
    bodypose_source: Path
    bodypose_engine: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _reject_symlink_ancestors(path: Path, *, label: str) -> None:
    """Reject a path whose existing lexical ancestry crosses a symlink."""

    absolute = Path(path).expanduser().absolute()
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink():
            raise V3DTAssetError(f"{label} contains a symlink: {ancestor}")


def _resolve_reference(raw: Any, *, owner_file: Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        return Path("")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    virtual_model_prefixes = (
        ("DS9/models/engines/", DS9_ENGINE_ROOT),
        ("DS9/models/onnx/", DS9_ONNX_ROOT),
        ("DS9/models/tracker_reid/", DS9_MODEL_ROOT / "tracker_reid"),
        ("models/engines/", DS9_ENGINE_ROOT),
        ("models/onnx/", DS9_ONNX_ROOT),
        ("models/tracker_reid/", DS9_MODEL_ROOT / "tracker_reid"),
    )
    for prefix, root in virtual_model_prefixes:
        if text.startswith(prefix):
            return (root / text[len(prefix) :]).resolve(strict=False)
    if text.startswith("DS9/"):
        return (REPO_ROOT / candidate).resolve(strict=False)
    if text.startswith(("config/", "pipelines/", "build/")):
        try:
            owner_file.resolve(strict=False).relative_to(DS9_ROOT.resolve())
            scope = DS9_ROOT
        except ValueError:
            scope = REPO_ROOT
        return (scope / candidate).resolve(strict=False)
    return (owner_file.parent / candidate).resolve(strict=False)


def _mapping(value: Any, *, label: str, errors: list[str]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    errors.append(f"{label} must be a mapping")
    return {}


def _require_file(path: Path, *, label: str, errors: list[str]) -> None:
    try:
        valid = not path.is_symlink() and path.is_file() and path.stat().st_size > 0
    except OSError:
        valid = False
    if not valid:
        errors.append(f"{label} is missing or empty: {path}")


def _require_owned_reference(
    raw: Any,
    *,
    owner_file: Path,
    root: Path,
    label: str,
    errors: list[str],
    require_file: bool = True,
) -> Path:
    path = _resolve_reference(raw, owner_file=owner_file)
    if path == Path(""):
        errors.append(f"{label} is required")
        return path
    if not _is_below(path, root):
        errors.append(f"{label} must be DS9-owned below {root}: {path}")
        return path
    if require_file:
        _require_file(path, label=label, errors=errors)
    return path


def _validate_pipeline_paths(
    pipeline: Mapping[str, Any],
    pipeline_path: Path,
    profile: str,
    errors: list[str],
    *,
    activation_state: str = "",
) -> None:
    sources = pipeline.get("sources")
    if not isinstance(sources, list) or len(sources) != len(EXPECTED_CAMERA_ORDER):
        errors.append(
            f"sources must contain exactly {len(EXPECTED_CAMERA_ORDER)} cameras"
        )
        sources = []
    for index, source in enumerate(sources):
        source_cfg = _mapping(source, label=f"sources[{index}]", errors=errors)
        expected_name = EXPECTED_CAMERA_ORDER[index]
        if str(source_cfg.get("uri_secret") or "").strip() != expected_name:
            errors.append(f"sources[{index}].uri_secret must be {expected_name!r}")
        if str(source_cfg.get("uri") or "").strip():
            errors.append(f"sources[{index}] must not embed a materialized camera URI")
        if source_cfg.get("element") != "nvurisrcbin":
            errors.append(f"sources[{index}].element must be nvurisrcbin")
        dewarper = source_cfg.get("dewarper")
        if not isinstance(dewarper, Mapping) or not bool(dewarper.get("enable", False)):
            errors.append(
                f"sources[{index}].dewarper must be enabled for the locked profile"
            )
        else:
            _require_owned_reference(
                dewarper.get("config-file"),
                owner_file=pipeline_path,
                root=DS9_CONFIG_ROOT,
                label=f"sources[{index}].dewarper.config-file",
                errors=errors,
            )

    preprocess = _mapping(pipeline.get("preprocess"), label="preprocess", errors=errors)
    _require_owned_reference(
        preprocess.get("config-file"),
        owner_file=pipeline_path,
        root=DS9_ROOT / "pipelines",
        label="preprocess.config-file",
        errors=errors,
    )

    models = _mapping(pipeline.get("models"), label="models", errors=errors)
    for name, raw_model in models.items():
        model = _mapping(raw_model, label=f"models.{name}", errors=errors)
        if model.get("enable") is False:
            continue
        _require_owned_reference(
            model.get("config-file-path"),
            owner_file=pipeline_path,
            root=DS9_ROOT,
            label=f"models.{name}.config-file-path",
            errors=errors,
        )
        _require_owned_reference(
            model.get("engine"),
            owner_file=pipeline_path,
            root=DS9_ENGINE_ROOT,
            label=f"models.{name}.engine",
            errors=errors,
            require_file=False,
        )

    depth_tracking = models.get("depth_tracking")
    if (
        not isinstance(depth_tracking, Mapping)
        or depth_tracking.get("enable") is not False
    ):
        errors.append(
            "models.depth_tracking.enable must be false for the V3DT profile"
        )

    if pipeline.get("batch_size") != len(EXPECTED_CAMERA_ORDER):
        errors.append(f"batch_size must be {len(EXPECTED_CAMERA_ORDER)}")
    streammux = _mapping(pipeline.get("streammux"), label="streammux", errors=errors)
    if streammux.get("batch-size") != len(EXPECTED_CAMERA_ORDER):
        errors.append(f"streammux.batch-size must be {len(EXPECTED_CAMERA_ORDER)}")
    if (streammux.get("width"), streammux.get("height")) != EXPECTED_STREAM_SIZE:
        errors.append(
            f"streammux width/height must be {EXPECTED_STREAM_SIZE[0]}x{EXPECTED_STREAM_SIZE[1]}"
        )
    if streammux.get("enable-padding") != 0:
        errors.append(
            "streammux.enable-padding must be 0 for the locked V3DT projection"
        )
    if streammux.get("num-surfaces-per-frame") != 1:
        errors.append("streammux.num-surfaces-per-frame must be 1")
    if profile == "mv3dt":
        sync_inputs = streammux.get("sync-inputs")
        batched_push_timeout = streammux.get("batched-push-timeout")
        if batched_push_timeout != -1:
            errors.append(
                "MV3DT streammux.batched-push-timeout must be -1 so the "
                "ordered peer-message synchronizer receives complete batches"
            )
        if activation_state == "evaluation_only":
            if sync_inputs not in {0, 1}:
                errors.append(
                    "evaluation-only MV3DT streammux.sync-inputs must be 0 or 1"
                )
        elif activation_state == "ready_opt_in":
            if sync_inputs != 0:
                errors.append(
                    "ready Kitchen/Family MV3DT streammux.sync-inputs must be 0 "
                    "for the non-PTP RTSP camera set"
                )
        elif sync_inputs != 1:
            errors.append("streammux.sync-inputs must be 1 for the MV3DT profile")

    analytics = _mapping(pipeline.get("analytics"), label="analytics", errors=errors)
    for key in ("config-file", "stages_config"):
        _require_owned_reference(
            analytics.get(key),
            owner_file=pipeline_path,
            root=DS9_CONFIG_ROOT,
            label=f"analytics.{key}",
            errors=errors,
        )
    exclude = _mapping(
        analytics.get("exclude"), label="analytics.exclude", errors=errors
    )
    _require_owned_reference(
        exclude.get("config-file"),
        owner_file=pipeline_path,
        root=DS9_CONFIG_ROOT,
        label="analytics.exclude.config-file",
        errors=errors,
    )


def _validate_caminfo(
    path: Path,
    errors: list[str],
    *,
    expected_height_m: float = EXPECTED_OBJECT_MODEL_HEIGHT_M,
) -> None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        errors.append(f"unable to parse camera model {path}: {exc}")
        return
    if not isinstance(payload, Mapping):
        errors.append(f"camera model must be a mapping: {path}")
        return
    matrix = payload.get("projectionMatrix_3x4_w2p")
    if (
        not isinstance(matrix, Sequence)
        or isinstance(matrix, (str, bytes))
        or len(matrix) != 12
    ):
        errors.append(
            f"camera model projectionMatrix_3x4_w2p must contain 12 values: {path}"
        )
    else:
        try:
            matrix_values = [float(value) for value in matrix]
        except (TypeError, ValueError):
            matrix_values = []
        if len(matrix_values) != 12 or not all(
            math.isfinite(value) for value in matrix_values
        ):
            errors.append(
                f"camera model projection matrix must contain 12 finite numbers: {path}"
            )
    model_info = payload.get("modelInfo")
    if not isinstance(model_info, Mapping):
        errors.append(f"camera model modelInfo is required: {path}")
        return
    expected_model = {
        "height": expected_height_m,
        "radius": EXPECTED_OBJECT_MODEL_RADIUS_M,
    }
    for key, expected in expected_model.items():
        try:
            value = float(model_info.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value) or not math.isclose(value, expected, abs_tol=1e-9):
            errors.append(f"camera model modelInfo.{key} must be {expected}: {path}")


def _validate_cameras_config(path: Path, errors: list[str]) -> None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        errors.append(f"unable to parse V3DT cameras config {path}: {exc}")
        return
    if not isinstance(payload, Mapping):
        errors.append(f"V3DT cameras config must be a mapping: {path}")
        return
    models = _mapping(
        payload.get("intrinsics_models"), label="intrinsics_models", errors=errors
    )
    cameras = _mapping(payload.get("cameras"), label="cameras", errors=errors)
    if len(cameras) != len(EXPECTED_CAMERA_ORDER):
        errors.append(
            f"cameras must contain exactly {len(EXPECTED_CAMERA_ORDER)} entries"
        )
    for index, expected_name in enumerate(EXPECTED_CAMERA_ORDER):
        raw_camera = cameras.get(index, cameras.get(str(index)))
        camera = _mapping(raw_camera, label=f"cameras.{index}", errors=errors)
        if str(camera.get("name") or "") != expected_name:
            errors.append(f"cameras.{index}.name must be {expected_name!r}")
        model_name = str(camera.get("model") or "").strip()
        model = _mapping(
            models.get(model_name),
            label=f"intrinsics_models.{model_name}",
            errors=errors,
        )
        if list(model.get("resolution") or []) != list(EXPECTED_STREAM_SIZE):
            errors.append(
                f"intrinsics model {model_name!r} must use the dewarped "
                f"{EXPECTED_STREAM_SIZE[0]}x{EXPECTED_STREAM_SIZE[1]} resolution"
            )
        intrinsics = _mapping(
            model.get("intrinsics"),
            label=f"intrinsics_models.{model_name}.intrinsics",
            errors=errors,
        )
        for key in ("fx", "fy", "cx", "cy"):
            try:
                value = float(intrinsics.get(key))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or value <= 0.0:
                errors.append(f"intrinsics model {model_name!r} has invalid {key}")
        rectification = _mapping(
            model.get("rectification"),
            label=f"intrinsics_models.{model_name}.rectification",
            errors=errors,
        )
        config_root = (
            DS9_CONFIG_ROOT
            if _is_below(path, DS9_CONFIG_ROOT)
            else REPO_ROOT / "config"
        )
        _require_owned_reference(
            rectification.get("dewarper_config"),
            owner_file=path,
            root=config_root,
            label=f"intrinsics_models.{model_name}.rectification.dewarper_config",
            errors=errors,
        )


def _validate_source_provenance(
    path: Path,
    *,
    source_path: Path,
    expected_sha256: str,
    expected_output: str,
    label: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    _require_file(path, label=f"{label} provenance", errors=errors)
    if not path.is_file():
        return None
    try:
        payload = strict_json_loads(
            path.read_bytes(),
            label=f"{label} provenance",
        )
    except Exception as exc:
        errors.append(f"unable to parse {label} provenance {path}: {exc}")
        return None
    if not isinstance(payload, Mapping):
        errors.append(f"{label} provenance must be an object: {path}")
        return None
    for key in ("source_sha256", "output_sha256"):
        if payload.get(key) != expected_sha256:
            errors.append(f"{label} provenance {key} does not match the locked digest")
    if payload.get("output") != expected_output:
        errors.append(f"{label} provenance output must be {expected_output!r}")
    try:
        output_bytes = int(payload.get("output_bytes"))
    except (TypeError, ValueError):
        output_bytes = -1
    if source_path.is_file() and output_bytes != source_path.stat().st_size:
        errors.append(
            f"{label} provenance output_bytes does not match the staged source"
        )
    return payload


def _atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        temporary.write_text(
            yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8"
        )
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_mv3dt_tracker_contract(
    tracker_config: Mapping[str, Any],
    *,
    tracker_path: Path,
    activation_state: str,
    errors: list[str],
) -> tuple[Path, Path]:
    target_management = _mapping(
        tracker_config.get("TargetManagement"),
        label="TargetManagement",
        errors=errors,
    )
    if (
        activation_state in {"evaluation_only", "ready_opt_in"}
        and target_management.get("probationAge") != 2
    ):
        errors.append("TargetManagement.probationAge must be 2 for Kitchen/Family MV3DT")

    associator = _mapping(
        tracker_config.get("MultiViewAssociator"),
        label="MultiViewAssociator",
        errors=errors,
    )
    unknown_associator_keys = set(associator) - set(MV3DT_ASSOCIATOR_CONTRACT)
    if unknown_associator_keys:
        errors.append(
            "MultiViewAssociator contains unsupported DeepStream 9.1 keys: "
            + ", ".join(sorted(str(key) for key in unknown_associator_keys))
        )
    expected_associator = dict(MV3DT_ASSOCIATOR_CONTRACT)
    if activation_state in {"evaluation_only", "ready_opt_in"}:
        expected_associator.update(MV3DT_KITCHEN_FAMILY_ASSOCIATOR_OVERRIDES)
    for key, expected in expected_associator.items():
        if associator.get(key) != expected:
            errors.append(f"MultiViewAssociator.{key} must be {expected!r}")

    communicator = _mapping(
        tracker_config.get("Communicator"), label="Communicator", errors=errors
    )
    allowed_communicator_keys = {
        "communicatorType",
        "pubSubInfoConfigPath",
        "mqttProtoAdaptorConfigPath",
    }
    unknown_communicator_keys = set(communicator) - allowed_communicator_keys
    if unknown_communicator_keys:
        errors.append(
            "Communicator contains unsupported DeepStream 9.1 keys: "
            + ", ".join(sorted(str(key) for key in unknown_communicator_keys))
        )
    if communicator.get("communicatorType") != 2:
        errors.append("Communicator.communicatorType must be 2 (MQTT)")

    pub_sub_path = _require_owned_reference(
        communicator.get("pubSubInfoConfigPath"),
        owner_file=tracker_path,
        root=DS9_V3DT_CONFIG_ROOT,
        label="Communicator.pubSubInfoConfigPath",
        errors=errors,
    )
    mqtt_template_path = _require_owned_reference(
        communicator.get("mqttProtoAdaptorConfigPath"),
        owner_file=tracker_path,
        root=DS9_V3DT_CONFIG_ROOT,
        label="Communicator.mqttProtoAdaptorConfigPath",
        errors=errors,
    )
    try:
        mqtt_template = mqtt_template_path.read_text(encoding="utf-8")
    except OSError as exc:
        mqtt_template = ""
        errors.append(
            f"unable to read MV3DT MQTT template {mqtt_template_path}: {exc}"
        )
    if "share-connection = 1" not in mqtt_template:
        errors.append("MV3DT MQTT template must set share-connection = 1")
    if "set-threaded = 0" not in mqtt_template:
        errors.append("MV3DT MQTT template must set set-threaded = 0")

    try:
        pub_sub = yaml.safe_load(pub_sub_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        pub_sub = {}
        errors.append(f"unable to parse MV3DT pub/sub config {pub_sub_path}: {exc}")
    if not isinstance(pub_sub, Mapping):
        pub_sub = {}
        errors.append(f"MV3DT pub/sub config must be a mapping: {pub_sub_path}")
    unknown_pub_sub_keys = set(pub_sub) - {
        "pubBrokerTopicStr",
        "subPeerBrokerTopicStrs",
    }
    if unknown_pub_sub_keys:
        errors.append(
            "MV3DT pub/sub config contains unsupported keys: "
            + ", ".join(sorted(str(key) for key in unknown_pub_sub_keys))
        )
    publish_topics = tuple(
        str(value) for value in pub_sub.get("pubBrokerTopicStr", []) or []
    )
    if publish_topics != MV3DT_PUBLISH_TOPICS:
        errors.append(
            "pubBrokerTopicStr must follow living-room, kitchen, family-room order"
        )
    raw_subscriptions = pub_sub.get("subPeerBrokerTopicStrs")
    subscriptions: tuple[tuple[str, ...], ...] = ()
    if isinstance(raw_subscriptions, list):
        try:
            subscriptions = tuple(
                tuple(str(topic) for topic in topics) for topics in raw_subscriptions
            )
        except TypeError:
            subscriptions = ()
    if subscriptions != MV3DT_SUBSCRIBE_TOPICS:
        errors.append(
            "subPeerBrokerTopicStrs must encode only Kitchen <-> Family Room; "
            "Living Room must have no cross-camera MV3DT peer edge"
        )
    return pub_sub_path, mqtt_template_path


def validate_v3dt_assets(
    pipeline_config: Path,
    *,
    cameras_config: Path | None = None,
    require_engines: bool = True,
    require_sources: bool = True,
    expected_profile: str | None = None,
    expected_activation_state: str | None = None,
) -> V3DTAssetBundle:
    """Validate one DS9-owned V3DT pipeline and return its resolved asset graph."""

    errors: list[str] = []
    pipeline_path = Path(pipeline_config).expanduser().resolve(strict=False)
    if not _is_below(pipeline_path, DS9_CONFIG_ROOT):
        errors.append(
            f"V3DT pipeline config must be below {DS9_CONFIG_ROOT}: {pipeline_path}"
        )
    _require_file(pipeline_path, label="V3DT pipeline config", errors=errors)
    try:
        pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise V3DTAssetError(
            f"unable to parse V3DT pipeline config {pipeline_path}: {exc}"
        ) from exc
    if not isinstance(pipeline, Mapping):
        raise V3DTAssetError(f"V3DT pipeline config must be a mapping: {pipeline_path}")

    cameras_path = (
        Path(cameras_config or V3DT_CAMERAS_CONFIG).expanduser().resolve(strict=False)
    )
    allowed_shared_cameras = {
        (REPO_ROOT / "config" / "cameras.yaml").resolve(strict=False),
        (REPO_ROOT / "config" / "cameras_v3dt_baseline.yaml").resolve(strict=False),
    }
    if (
        not _is_below(cameras_path, DS9_CONFIG_ROOT)
        and cameras_path not in allowed_shared_cameras
    ):
        errors.append(
            "V3DT cameras config must be DS9-owned or one of the exact shared camera "
            f"inventories: {cameras_path}"
        )
    _require_file(cameras_path, label="V3DT cameras config", errors=errors)
    if cameras_path.is_file():
        _validate_cameras_config(cameras_path, errors)

    profile_config = _mapping(pipeline.get("v3dt"), label="v3dt", errors=errors)
    profile = str(profile_config.get("profile") or "").strip().lower()
    if profile not in {"sv3dt", "mv3dt"}:
        errors.append("v3dt.profile must be sv3dt or mv3dt")
    expected_profile_normalized = str(expected_profile or "").strip().lower()
    if expected_profile_normalized and profile != expected_profile_normalized:
        errors.append(
            f"v3dt.profile is {profile!r}; expected {expected_profile_normalized}"
        )
    activation_state = str(profile_config.get("activation_state") or "").strip()
    if profile == "mv3dt":
        if activation_state not in {"deferred", "evaluation_only", "ready_opt_in"}:
            errors.append(
                "v3dt.activation_state must be deferred, evaluation_only, or "
                "ready_opt_in for MV3DT"
            )
        if activation_state in {"evaluation_only", "ready_opt_in"}:
            if profile_config.get("evaluation_scope") != "kitchen-family":
                errors.append(
                    "Kitchen/Family MV3DT must set evaluation_scope=kitchen-family"
                )
            expected_authority = (
                "review_only" if activation_state == "evaluation_only" else "accepted"
            )
            if profile_config.get("geometry_authority") != expected_authority:
                errors.append(
                    f"{activation_state} MV3DT must set "
                    f"geometry_authority={expected_authority}"
                )
            geometry_binding_path = _require_owned_reference(
                profile_config.get("geometry_binding"),
                owner_file=pipeline_path,
                root=DS9_V3DT_CONFIG_ROOT,
                label="v3dt.geometry_binding",
                errors=errors,
            )
            if geometry_binding_path.is_file():
                try:
                    geometry_binding = strict_json_loads(
                        geometry_binding_path.read_text(encoding="utf-8"),
                        label="MV3DT geometry binding",
                    )
                except Exception as exc:
                    geometry_binding = {}
                    errors.append(
                        f"unable to parse MV3DT geometry binding "
                        f"{geometry_binding_path}: {exc}"
                    )
                if not isinstance(geometry_binding, Mapping):
                    geometry_binding = {}
                    errors.append(
                        f"MV3DT geometry binding must be a mapping: "
                        f"{geometry_binding_path}"
                    )
                ready = activation_state == "ready_opt_in"
                expected_geometry = {
                    "schema": "noesis.mv3dt.geometry_binding.v1",
                    "status": "accepted" if ready else "review_only",
                    "canonical_use": ready,
                    "fixed_gauge": "family-room",
                    "moving_room": "kitchen",
                    "accepted": ready,
                }
                for key, expected in expected_geometry.items():
                    if geometry_binding.get(key) != expected:
                        errors.append(
                            f"MV3DT geometry binding {key} must be {expected!r}"
                        )
    expected_activation = str(expected_activation_state or "").strip()
    if expected_activation and activation_state != expected_activation:
        errors.append(
            f"v3dt.activation_state is {activation_state!r}; expected "
            f"{expected_activation!r}"
        )
    if profile_config.get("world_frame") != "backend_world_m":
        errors.append(
            "v3dt.world_frame must be backend_world_m after camInfo axis restoration"
        )
    if profile_config.get("caminfo_world_axes") != "xzy":
        errors.append("v3dt.caminfo_world_axes must bind the locked xzy camInfo map")
    camera_order = tuple(
        str(value) for value in profile_config.get("camera_order", []) or []
    )
    if camera_order != EXPECTED_CAMERA_ORDER:
        errors.append(f"v3dt.camera_order must be {list(EXPECTED_CAMERA_ORDER)}")
    _validate_pipeline_paths(
        pipeline,
        pipeline_path,
        profile,
        errors,
        activation_state=activation_state,
    )

    tracker = _mapping(pipeline.get("tracker"), label="tracker", errors=errors)
    tracker_path = _require_owned_reference(
        tracker.get("config-file"),
        owner_file=pipeline_path,
        root=DS9_V3DT_CONFIG_ROOT,
        label="tracker.config-file",
        errors=errors,
    )
    expected_tracker_name = (
        "nvtracker_mv3dt.yaml" if profile == "mv3dt" else "nvtracker_v3dt.yaml"
    )
    if tracker_path.name != expected_tracker_name:
        errors.append(
            f"{profile or 'V3DT'} tracker config must be {expected_tracker_name}, "
            f"got {tracker_path.name}"
        )
    ll_lib = str(tracker.get("ll-lib-file") or "")
    if "/deepstream-9.1/" not in ll_lib:
        errors.append(
            f"tracker.ll-lib-file must bind explicitly to DeepStream 9.1: {ll_lib!r}"
        )
    if (
        tracker.get("tracker-width"),
        tracker.get("tracker-height"),
    ) != EXPECTED_STREAM_SIZE:
        errors.append(
            f"tracker width/height must be {EXPECTED_STREAM_SIZE[0]}x{EXPECTED_STREAM_SIZE[1]}"
        )

    try:
        tracker_config = yaml.safe_load(tracker_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        tracker_config = {}
        errors.append(f"unable to parse V3DT tracker config {tracker_path}: {exc}")
    if not isinstance(tracker_config, Mapping):
        tracker_config = {}
        errors.append(f"V3DT tracker config must be a mapping: {tracker_path}")

    if profile == "mv3dt" and activation_state in {
        "evaluation_only",
        "ready_opt_in",
    }:
        streammux = pipeline.get("streammux")
        sync_inputs = (
            streammux.get("sync-inputs")
            if isinstance(streammux, Mapping)
            else None
        )
        base_config = tracker_config.get("BaseConfig")
        use_batch_number = (
            base_config.get("useBatchNumForFrameId")
            if isinstance(base_config, Mapping)
            else None
        )
        if sync_inputs == 0 and use_batch_number != 1:
            errors.append(
                "Kitchen/Family MV3DT with sync-inputs=0 must set "
                "BaseConfig.useBatchNumForFrameId=1"
            )

    pub_sub_config: Path | None = None
    mqtt_config_template: Path | None = None
    if profile == "mv3dt":
        pub_sub_config, mqtt_config_template = _validate_mv3dt_tracker_contract(
            tracker_config,
            tracker_path=tracker_path,
            activation_state=activation_state,
            errors=errors,
        )
    elif any(
        key in tracker_config for key in ("MultiViewAssociator", "Communicator")
    ):
        errors.append(
            "SV3DT tracker config must not contain MultiViewAssociator or Communicator"
        )

    state_estimator = _mapping(
        tracker_config.get("StateEstimator"), label="StateEstimator", errors=errors
    )
    if state_estimator.get("stateEstimatorType") != 3:
        errors.append("StateEstimator.stateEstimatorType must be 3")
    projection = _mapping(
        tracker_config.get("ObjectModelProjection"),
        label="ObjectModelProjection",
        errors=errors,
    )
    if (
        projection.get("outputFootLocation") != 1
        or projection.get("outputVisibility") != 1
    ):
        errors.append(
            "ObjectModelProjection must enable foot-location and visibility output"
        )
    raw_camera_models = projection.get("cameraModelFilepath")
    if not isinstance(raw_camera_models, list) or len(raw_camera_models) != len(
        EXPECTED_CAMERA_ORDER
    ):
        errors.append(
            f"ObjectModelProjection.cameraModelFilepath must contain {len(EXPECTED_CAMERA_ORDER)} entries"
        )
        raw_camera_models = []
    camera_models: list[Path] = []
    for index, raw in enumerate(raw_camera_models):
        path = _require_owned_reference(
            raw,
            owner_file=tracker_path,
            root=DS9_V3DT_CONFIG_ROOT,
            label=f"ObjectModelProjection.cameraModelFilepath[{index}]",
            errors=errors,
        )
        camera_models.append(path)
        if (
            index < len(EXPECTED_CAMINFO_NAMES)
            and path.name != EXPECTED_CAMINFO_NAMES[index]
        ):
            errors.append(
                f"camera model index {index} must be {EXPECTED_CAMINFO_NAMES[index]}, got {path.name}"
            )
        if path.is_file():
            _validate_caminfo(
                path,
                errors,
                expected_height_m=(
                    MV3DT_KITCHEN_FAMILY_OBJECT_MODEL_HEIGHT_M
                    if profile == "mv3dt"
                    and activation_state in {"evaluation_only", "ready_opt_in"}
                    else EXPECTED_OBJECT_MODEL_HEIGHT_M
                ),
            )

    reid = _mapping(tracker_config.get("ReID"), label="ReID", errors=errors)
    if reid.get("reidType") != 2 or reid.get("batchSize") != 32:
        errors.append("ReID must use reidType=2 and batchSize=32")
    if (
        list(reid.get("inferDims") or []) != [3, 256, 128]
        or reid.get("networkMode") != 1
    ):
        errors.append("ReID must use FP16 inferDims [3, 256, 128]")
    if reid.get("tltModelKey") != "nvidia_tao":
        errors.append("ReID.tltModelKey must be nvidia_tao")
    reid_source = _require_owned_reference(
        reid.get("tltEncodedModel"),
        owner_file=tracker_path,
        root=DS9_MODEL_ROOT / "tracker_reid",
        label="ReID.tltEncodedModel",
        errors=errors,
        require_file=require_sources,
    )
    reid_engine = _require_owned_reference(
        reid.get("modelEngineFile"),
        owner_file=tracker_path,
        root=DS9_ENGINE_ROOT,
        label="ReID.modelEngineFile",
        errors=errors,
        require_file=require_engines,
    )

    pose = _mapping(
        tracker_config.get("PoseEstimator"), label="PoseEstimator", errors=errors
    )
    if pose.get("poseEstimatorType") != 1 or pose.get("batchSize") != 1:
        errors.append("PoseEstimator must use poseEstimatorType=1 and batchSize=1")
    if (
        list(pose.get("inferDims") or []) != [3, 256, 192]
        or pose.get("networkMode") != 1
    ):
        errors.append("PoseEstimator must use FP16 inferDims [3, 256, 192]")
    pose_source = _require_owned_reference(
        pose.get("onnxFile"),
        owner_file=tracker_path,
        root=DS9_MODEL_ROOT / "onnx",
        label="PoseEstimator.onnxFile",
        errors=errors,
        require_file=require_sources,
    )
    pose_engine = _require_owned_reference(
        pose.get("modelEngineFile"),
        owner_file=tracker_path,
        root=DS9_ENGINE_ROOT,
        label="PoseEstimator.modelEngineFile",
        errors=errors,
        require_file=require_engines,
    )

    for label, path, expected in (
        ("tracker ReID source", reid_source, TRACKER_REID_SOURCE_SHA256),
        ("BodyPose3DNet source", pose_source, BODYPOSE_SOURCE_SHA256),
    ):
        if path.is_file() and _sha256(path) != expected:
            errors.append(f"{label} SHA-256 does not match the provenance lock: {path}")

    tracker_provenance = _validate_source_provenance(
        TRACKER_REID_PROVENANCE,
        source_path=TRACKER_REID_SOURCE,
        expected_sha256=TRACKER_REID_SOURCE_SHA256,
        expected_output="DS9/models/tracker_reid/resnet50_market1501.etlt",
        label="tracker ReID source",
        errors=errors,
    )
    bodypose_provenance = _validate_source_provenance(
        BODYPOSE_PROVENANCE,
        source_path=BODYPOSE_SOURCE,
        expected_sha256=BODYPOSE_SOURCE_SHA256,
        expected_output="DS9/models/onnx/bodypose3dnet_accuracy.onnx",
        label="BodyPose3DNet source",
        errors=errors,
    )
    if bodypose_provenance is not None:
        if bodypose_provenance.get("input_tensors") != EXPECTED_BODYPOSE_INPUTS:
            errors.append(
                "BodyPose3DNet provenance input tensor contract does not match NvMOT"
            )
    if tracker_provenance is not None:
        if tracker_provenance.get("tlt_model_key") != "nvidia_tao":
            errors.append("tracker ReID provenance must lock tlt_model_key=nvidia_tao")

    expected_paths = (
        ("ReID.tltEncodedModel", reid_source, TRACKER_REID_SOURCE),
        ("ReID.modelEngineFile", reid_engine, TRACKER_REID_ENGINE),
        ("PoseEstimator.onnxFile", pose_source, BODYPOSE_SOURCE),
        ("PoseEstimator.modelEngineFile", pose_engine, BODYPOSE_ENGINE),
    )
    for label, actual, expected in expected_paths:
        if actual.resolve(strict=False) != expected.resolve(strict=False):
            errors.append(f"{label} must resolve to {expected}, got {actual}")

    if errors:
        raise V3DTAssetError(
            "DS9 V3DT asset contract failed:\n- " + "\n- ".join(errors)
        )
    return V3DTAssetBundle(
        profile=profile,
        pipeline_config=pipeline_path,
        cameras_config=cameras_path,
        tracker_config=tracker_path,
        pub_sub_config=pub_sub_config,
        mqtt_config_template=mqtt_config_template,
        camera_models=tuple(camera_models),
        tracker_reid_source=reid_source,
        tracker_reid_engine=reid_engine,
        bodypose_source=pose_source,
        bodypose_engine=pose_engine,
    )


def materialize_v3dt_tracker_config(
    bundle: V3DTAssetBundle,
    destination: Path,
    *,
    output_root: Path,
    tracker_reid_engine: Path | None = None,
    bodypose_engine: Path | None = None,
    mqtt_runtime_config: Path | None = None,
) -> Path:
    """Write an engine-only NvMOT config with absolute DS9-owned paths.

    NvMOT resolves paths inside its low-level YAML independently of Service
    Maker's pipeline-path normalization. The generated runtime config excludes
    model sources so NvMOT cannot silently rebuild an engine; the committed
    source-rich config remains the explicit offline maintenance input.
    """

    destination = Path(destination).expanduser().resolve(strict=False)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    if not _is_below(destination, output_root):
        raise V3DTAssetError(
            f"generated V3DT tracker config must remain below {output_root}: {destination}"
        )
    if destination.suffix.lower() not in {".yaml", ".yml"}:
        raise V3DTAssetError(
            f"generated V3DT tracker config must be YAML: {destination}"
        )

    payload = yaml.safe_load(bundle.tracker_config.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise V3DTAssetError(
            f"V3DT tracker config must be a mapping: {bundle.tracker_config}"
        )
    projection = payload.get("ObjectModelProjection")
    reid = payload.get("ReID")
    pose = payload.get("PoseEstimator")
    if (
        not isinstance(projection, dict)
        or not isinstance(reid, dict)
        or not isinstance(pose, dict)
    ):
        raise V3DTAssetError(
            "V3DT tracker config is missing projection, ReID, or pose mappings"
        )

    if bundle.profile == "mv3dt":
        communicator = payload.get("Communicator")
        if not isinstance(communicator, dict) or bundle.pub_sub_config is None:
            raise V3DTAssetError(
                "MV3DT tracker materialization requires validated communicator assets"
            )
        if mqtt_runtime_config is None:
            raise V3DTAssetError(
                "MV3DT tracker materialization requires an owner-only MQTT runtime config"
            )
        mqtt_path = Path(mqtt_runtime_config).expanduser()
        if not mqtt_path.is_absolute():
            raise V3DTAssetError(
                f"MV3DT MQTT runtime config must be absolute: {mqtt_path}"
            )
        _reject_symlink_ancestors(mqtt_path, label="MV3DT MQTT runtime config")
        mqtt_path = mqtt_path.resolve(strict=False)
        if _is_below(mqtt_path, REPO_ROOT):
            raise V3DTAssetError(
                f"MV3DT MQTT runtime config must remain outside the checkout: {mqtt_path}"
            )
        if not mqtt_path.is_file() or mqtt_path.stat().st_size <= 0:
            raise V3DTAssetError(
                f"MV3DT MQTT runtime config is missing or empty: {mqtt_path}"
            )
        if stat.S_IMODE(mqtt_path.stat().st_mode) & 0o077:
            raise V3DTAssetError(
                f"MV3DT MQTT runtime config must be owner-only: {mqtt_path}"
            )
        communicator["pubSubInfoConfigPath"] = str(bundle.pub_sub_config.resolve())
        communicator["mqttProtoAdaptorConfigPath"] = str(mqtt_path)

    reid_engine = Path(tracker_reid_engine or bundle.tracker_reid_engine).resolve(
        strict=False
    )
    pose_engine = Path(bodypose_engine or bundle.bodypose_engine).resolve(strict=False)
    for label, path in (
        ("generated ReID engine", reid_engine),
        ("generated BodyPose engine", pose_engine),
    ):
        if not _is_below(path, DS9_ENGINE_ROOT) and not _is_below(path, output_root):
            raise V3DTAssetError(
                f"{label} must be DS9-owned or a bounded build output: {path}"
            )
        if not path.is_file() or path.stat().st_size <= 0:
            raise V3DTAssetError(f"{label} is missing or empty: {path}")

    projection["cameraModelFilepath"] = [
        str(path.resolve()) for path in bundle.camera_models
    ]
    reid["modelEngineFile"] = str(reid_engine)
    pose["modelEngineFile"] = str(pose_engine)
    for source_key in (
        "calibrationTableFile",
        "onnxFile",
        "tltEncodedModel",
        "tltModelKey",
        "uffFile",
    ):
        reid.pop(source_key, None)
        pose.pop(source_key, None)
    _atomic_write_yaml(destination, payload)
    return destination


def materialize_v3dt_tracker_build_config(
    bundle: V3DTAssetBundle,
    destination: Path,
    *,
    output_root: Path,
    tracker_reid_source: Path,
    tracker_reid_generated_engine: Path,
    gpu_id: int,
    bodypose_engine: Path | None = None,
) -> Path:
    """Write the source-rich NvMOT config used only to build tracker ReID.

    The runtime materializer above deliberately requires existing engines and
    removes every model source.  NvMOT needs the opposite contract for this one
    offline transaction: a locked ETLT copy and NvMOT's exact source-adjacent
    engine output must both live inside a private transaction workspace.
    BodyPose3DNet remains engine-only because it is an ordered prerequisite
    rather than an output of this transaction.
    """

    destination = Path(destination).expanduser().resolve(strict=False)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    if not _is_below(destination, output_root):
        raise V3DTAssetError(
            f"generated V3DT tracker build config must remain below {output_root}: {destination}"
        )
    if destination.suffix.lower() not in {".yaml", ".yml"}:
        raise V3DTAssetError(
            f"generated V3DT tracker build config must be YAML: {destination}"
        )

    payload = yaml.safe_load(bundle.tracker_config.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise V3DTAssetError(
            f"V3DT tracker config must be a mapping: {bundle.tracker_config}"
        )
    projection = payload.get("ObjectModelProjection")
    reid = payload.get("ReID")
    pose = payload.get("PoseEstimator")
    if (
        not isinstance(projection, dict)
        or not isinstance(reid, dict)
        or not isinstance(pose, dict)
    ):
        raise V3DTAssetError(
            "V3DT tracker config is missing projection, ReID, or pose mappings"
        )

    reid_source_raw = Path(tracker_reid_source).expanduser()
    generated_raw = Path(tracker_reid_generated_engine).expanduser()
    for label, raw_path in (
        ("staged tracker ReID source", reid_source_raw),
        ("generated tracker ReID engine", generated_raw),
    ):
        if raw_path.is_symlink():
            raise V3DTAssetError(f"{label} must not be a symlink: {raw_path}")
        _reject_symlink_ancestors(raw_path.parent, label=f"{label} parent")

    reid_source = reid_source_raw.resolve(strict=False)
    generated = generated_raw.resolve(strict=False)
    if not _is_below(reid_source, output_root):
        raise V3DTAssetError(
            "staged tracker ReID source must remain below the transaction output "
            f"root {output_root}: {reid_source}"
        )
    if not reid_source.is_file() or reid_source.stat().st_size <= 0:
        raise V3DTAssetError(
            f"staged tracker ReID source is missing or empty: {reid_source}"
        )
    reid_source_info = reid_source.lstat()
    if (
        not stat.S_ISREG(reid_source_info.st_mode)
        or reid_source_info.st_uid != os.getuid()
        or reid_source_info.st_nlink != 1
    ):
        raise V3DTAssetError(
            "staged tracker ReID source must be an owner-owned single-link "
            f"regular file: {reid_source}"
        )
    canonical_source = Path(bundle.tracker_reid_source).resolve(strict=False)
    if reid_source.name != canonical_source.name:
        raise V3DTAssetError(
            "staged tracker ReID source must preserve the locked filename: "
            f"expected={canonical_source.name} observed={reid_source.name}"
        )
    reid_source_sha256 = _sha256(reid_source)
    if reid_source_sha256 != TRACKER_REID_SOURCE_SHA256:
        raise V3DTAssetError(
            "staged tracker ReID source SHA-256 does not match the provenance lock: "
            f"{reid_source}"
        )

    try:
        batch_size = int(reid.get("batchSize"))
        network_mode = int(reid.get("networkMode"))
    except (TypeError, ValueError) as exc:
        raise V3DTAssetError(
            "V3DT tracker build config has invalid ReID batchSize or networkMode"
        ) from exc
    expected_generated = derive_nvmot_tracker_engine_path(
        reid_source,
        batch_size=batch_size,
        gpu_id=gpu_id,
        network_mode=network_mode,
    )
    if generated != expected_generated:
        raise V3DTAssetError(
            "generated tracker ReID engine must use NvMOT's exact derived path: "
            f"expected={expected_generated} observed={generated}"
        )
    if not _is_below(generated, output_root):
        raise V3DTAssetError(
            "generated tracker ReID engine must remain below the transaction output "
            f"root {output_root}: {generated}"
        )
    if generated.exists() or generated.is_symlink():
        raise V3DTAssetError(
            "generated tracker ReID engine must be absent before NvMOT build: "
            f"{generated}"
        )

    pose_engine_raw = Path(bodypose_engine or bundle.bodypose_engine).expanduser()
    for label, raw_path in (("generated BodyPose engine", pose_engine_raw),):
        if raw_path.is_symlink():
            raise V3DTAssetError(f"{label} must not be a symlink: {raw_path}")
        _reject_symlink_ancestors(raw_path.parent, label=f"{label} parent")
        path = raw_path.resolve(strict=False)
        if not path.is_file() or path.stat().st_size <= 0:
            raise V3DTAssetError(f"{label} is missing or empty: {path}")
        if not _is_below(path, DS9_MODEL_ROOT) and not _is_below(path, output_root):
            raise V3DTAssetError(
                f"{label} must be DS9-owned or a bounded build input: {path}"
            )

    pose_engine = pose_engine_raw.resolve(strict=False)

    projection["cameraModelFilepath"] = [
        str(path.resolve()) for path in bundle.camera_models
    ]
    reid["modelEngineFile"] = str(generated)
    reid["tltEncodedModel"] = str(reid_source)
    reid["tltModelKey"] = "nvidia_tao"
    for source_key in ("calibrationTableFile", "onnxFile", "uffFile"):
        reid.pop(source_key, None)

    pose["modelEngineFile"] = str(pose_engine)
    for source_key in (
        "calibrationTableFile",
        "onnxFile",
        "tltEncodedModel",
        "tltModelKey",
        "uffFile",
    ):
        pose.pop(source_key, None)

    _atomic_write_yaml(destination, payload)
    return destination


__all__ = [
    "BODYPOSE_ENGINE",
    "BODYPOSE_PROVENANCE",
    "BODYPOSE_SOURCE",
    "BODYPOSE_SOURCE_SHA256",
    "EXPECTED_BODYPOSE_INPUTS",
    "EXPECTED_CAMERA_ORDER",
    "EXPECTED_OBJECT_MODEL_HEIGHT_M",
    "EXPECTED_OBJECT_MODEL_RADIUS_M",
    "EXPECTED_STREAM_SIZE",
    "derive_nvmot_tracker_engine_path",
    "TRACKER_REID_ENGINE",
    "TRACKER_REID_PROVENANCE",
    "TRACKER_REID_SOURCE",
    "TRACKER_REID_SOURCE_SHA256",
    "V3DT_CAMERAS_CONFIG",
    "V3DTAssetBundle",
    "V3DTAssetError",
    "materialize_v3dt_tracker_build_config",
    "materialize_v3dt_tracker_config",
    "validate_v3dt_assets",
]
