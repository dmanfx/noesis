from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


def _load_module(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class V3DTProfileTests(unittest.TestCase):
    def test_tracking_mode_normalization_fails_closed(self) -> None:
        code = r"""
import os
from types import SimpleNamespace
from unittest import mock

with (
    mock.patch("noesis.native_artifact_provenance.attest_ds9_native_artifacts"),
    mock.patch("noesis.runtime_paths.require_ds9_native_extension_origins"),
):
    from noesis import ds9_runtime_core as runtime

assert runtime._normalize_tracking_mode("") == "baseline"
assert runtime._normalize_tracking_mode("auto") == "baseline"
assert runtime._normalize_tracking_mode("standard") == "baseline"
assert runtime._normalize_tracking_mode("sv3dt") == "v3dt"

try:
    runtime._normalize_tracking_mode("v3dtt")
except SystemExit as exc:
    assert "Unsupported DS9 tracking mode" in str(exc)
else:
    raise AssertionError("invalid tracking mode did not fail closed")

os.environ["NOESIS_TRACKING_MODE"] = "v3dtt"
try:
    runtime._resolve_tracking_mode(
        SimpleNamespace(tracking_mode=None, v3dt=False)
    )
except SystemExit as exc:
    assert "Unsupported DS9 tracking mode" in str(exc)
else:
    raise AssertionError("invalid tracking-mode environment did not fail closed")
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            (str(DS9_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", ""))
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_preflight_accepts_current_595_driver(self) -> None:
        preflight = _load_module(
            "ds9_preflight_driver_595_test", "DS9/scripts/ds9_preflight.py"
        )
        result = SimpleNamespace(returncode=0, stdout="595.71.05\n", stderr="")
        with (
            mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(preflight.subprocess, "run", return_value=result),
        ):
            self.assertTrue(preflight._driver_version_ok())

    def test_preflight_rejects_driver_below_exact_ds9_floor(self) -> None:
        preflight = _load_module(
            "ds9_preflight_driver_floor_test", "DS9/scripts/ds9_preflight.py"
        )
        result = SimpleNamespace(returncode=0, stdout="595.58.02\n", stderr="")
        with (
            mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(preflight.subprocess, "run", return_value=result),
        ):
            self.assertFalse(preflight._driver_version_ok())

    def test_explicit_artifact_root_resolves_models_and_shared_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            artifact_root = Path(raw_root).resolve()
            code = """
from pathlib import Path
from noesis.v3dt_assets import validate_v3dt_assets

expected_root = Path(__import__('os').environ['NOESIS_DS9_ARTIFACT_ROOT']).resolve()
for cameras in (
    Path('DS9/config/cameras_v3dt.yaml'),
    Path('config/cameras.yaml'),
    Path('config/cameras_v3dt_baseline.yaml'),
):
    bundle = validate_v3dt_assets(
        Path('DS9/config/infer_v3dt.yaml'),
        cameras_config=cameras,
        require_engines=False,
        require_sources=False,
    )
    assert bundle.bodypose_source == expected_root / 'models/onnx/bodypose3dnet_accuracy.onnx'
    assert bundle.tracker_reid_source == expected_root / 'models/tracker_reid/resnet50_market1501.etlt'
"""
            env = dict(os.environ)
            env["NOESIS_DS9_ARTIFACT_ROOT"] = str(artifact_root)
            env["PYTHONPATH"] = os.pathsep.join(
                (str(DS9_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", ""))
            )
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_preflight_keeps_pipeline_path_after_reid_inspection(self) -> None:
        preflight = _load_module(
            "ds9_preflight_v3dt_test", "DS9/scripts/ds9_preflight.py"
        )
        defaults = preflight._parse_args([])
        self.assertEqual(defaults.config, DS9_ROOT / "config" / "infer.yaml")
        self.assertEqual(defaults.cameras_config, REPO_ROOT / "config" / "cameras.yaml")
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            pipeline = root / "infer_v3dt.yaml"
            cameras = root / "cameras.yaml"
            pipeline.write_text(
                "version: 1\nmodels: {}\nv3dt: {}\ntracker: {}\n", encoding="utf-8"
            )
            cameras.write_text("version: 1\n", encoding="utf-8")
            bundle = SimpleNamespace(
                tracker_config=Path("tracker.yaml"), camera_models=(1, 2, 3)
            )
            with (
                mock.patch.object(preflight, "materialize_pipeline_config"),
                mock.patch.object(
                    preflight, "validate_v3dt_assets", return_value=bundle
                ) as validate,
            ):
                preflight._config_assets_ok(pipeline, cameras)
            validate.assert_called_once_with(
                pipeline,
                cameras_config=cameras,
                require_engines=True,
            )

    def test_smoke_defaults_to_ds9_v3dt_and_preserves_root_camera_override(
        self,
    ) -> None:
        smoke = _load_module(
            "ds9_v3dt_smoke_test_module", "DS9/scripts/sv3dt_meta_smoke_test.py"
        )
        defaults = smoke._parse_args([])
        self.assertEqual(
            Path(defaults.pipeline_config), DS9_ROOT / "config" / "infer_v3dt.yaml"
        )
        self.assertEqual(
            Path(defaults.cameras_config), DS9_ROOT / "config" / "cameras_v3dt.yaml"
        )
        self.assertEqual(defaults.tracking_mode, "v3dt")
        self.assertIsNone(defaults.auth_token_file)

        root_cameras = REPO_ROOT / "config" / "cameras.yaml"
        explicit = smoke._parse_args(["--cameras-config", str(root_cameras)])
        self.assertEqual(Path(explicit.cameras_config), root_cameras)

    def test_v3dt_smoke_spawns_only_with_required_auth_file_path(self) -> None:
        smoke = _load_module(
            "ds9_v3dt_smoke_auth_test_module", "DS9/scripts/sv3dt_meta_smoke_test.py"
        )
        auth_file = Path("/private/gateway-token")
        auth = smoke.RequiredInternalAuth(token_file=auth_file, _token="secret-sentinel")
        args = SimpleNamespace(
            ws_port=6123,
            tracking_mode="v3dt",
            pipeline_config="DS9/config/infer_v3dt.yaml",
            cameras_config="DS9/config/cameras_v3dt.yaml",
        )
        fake = mock.Mock()
        with mock.patch.object(smoke.subprocess, "Popen", return_value=fake) as popen:
            assert smoke._spawn_runtime(args, auth) is fake
        command = popen.call_args.args[0]
        environment = popen.call_args.kwargs["env"]
        self.assertNotIn("secret-sentinel", " ".join(command))
        self.assertEqual(environment["NOESIS_INTERNAL_AUTH_MODE"], "required")
        self.assertEqual(environment["NOESIS_INTERNAL_AUTH_TOKEN_FILE"], str(auth_file))

    def test_staging_capacity_keeps_configured_residual_headroom(self) -> None:
        staging = _load_module(
            "ds9_stage_capacity_test", "DS9/scripts/stage_canonical_sources.py"
        )
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root) / "artifact-root"
            source = Path(raw_root) / "source.onnx"
            source.write_bytes(b"x" * 100)
            plan = [(source, Path("models/onnx/source.onnx"))]
            with mock.patch.dict(
                os.environ, {"NOESIS_DS9_MIN_FREE_HEADROOM_BYTES": "10"}
            ):
                with mock.patch.object(
                    staging.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=209),
                ):
                    with self.assertRaisesRegex(OSError, "residual_headroom=10"):
                        staging._require_staging_capacity(root, plan)
                with mock.patch.object(
                    staging.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=210),
                ):
                    staging._require_staging_capacity(root, plan)

    def test_artifact_root_cannot_live_inside_checkout(self) -> None:
        staging = _load_module(
            "ds9_stage_checkout_root_test", "DS9/scripts/stage_canonical_sources.py"
        )
        nested = REPO_ROOT / "DS9" / "models" / "forbidden-artifact-root"
        with mock.patch.dict(
            os.environ, {"NOESIS_DS9_ARTIFACT_ROOT": str(nested)}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "must not be inside the checkout"):
                staging._artifact_root()

    def test_staging_rejects_symlinked_destination_components(self) -> None:
        staging = _load_module(
            "ds9_stage_symlink_test", "DS9/scripts/stage_canonical_sources.py"
        )
        with tempfile.TemporaryDirectory() as raw_root:
            base = Path(raw_root)
            artifact_root = base / "artifacts"
            outside = base / "outside"
            artifact_root.mkdir()
            outside.mkdir()
            (artifact_root / "models").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                staging._bounded_destination(
                    artifact_root, Path("models/onnx/source.onnx")
                )

    def test_launcher_maps_explicit_artifact_root_to_runtime_model_dirs(self) -> None:
        launcher = _load_module(
            "ds9_runtime_launcher_v3dt_test", "DS9/noesis/ds9_runtime.py"
        )
        with tempfile.TemporaryDirectory() as raw_root:
            artifact_root = Path(raw_root).resolve()
            with mock.patch.dict(
                os.environ,
                {"NOESIS_DS9_ARTIFACT_ROOT": str(artifact_root)},
                clear=False,
            ), mock.patch.object(
                launcher, "attest_ds9_native_artifacts", return_value={}
            ) as attest_native_artifacts, mock.patch.object(
                launcher, "load_ds9_native_extensions", return_value={}
            ) as load_native_extensions:
                for name in (
                    "NOESIS_MODEL_DIR",
                    "NOESIS_ONNX_DIR",
                    "NOESIS_ENGINE_DIR",
                ):
                    os.environ.pop(name, None)
                launcher._set_ds9_environment()
                self.assertEqual(
                    Path(os.environ["NOESIS_MODEL_DIR"]), artifact_root / "models"
                )
                self.assertEqual(
                    Path(os.environ["NOESIS_ONNX_DIR"]),
                    artifact_root / "models" / "onnx",
                )
                self.assertEqual(
                    Path(os.environ["NOESIS_ENGINE_DIR"]),
                    artifact_root / "models" / "engines",
                )
                load_native_extensions.assert_called_once_with(
                    (DS9_ROOT / "native_extensions").resolve()
                )
                attest_native_artifacts.assert_called_once_with(
                    ds9_root=DS9_ROOT,
                    native_dir=(DS9_ROOT / "native_extensions").resolve(),
                )

    def test_effective_v3dt_pipeline_declares_mode_and_recognizable_tracker_path(
        self,
    ) -> None:
        code = r"""
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

with (
    mock.patch("noesis.native_artifact_provenance.attest_ds9_native_artifacts"),
    mock.patch("noesis.runtime_paths.require_ds9_native_extension_origins"),
):
    from noesis import ds9_runtime_core as runtime

with tempfile.TemporaryDirectory() as raw_root:
    root = Path(raw_root)
    pipeline = root / "infer.yaml"
    pipeline.write_text("version: 1\nmodels: {}\ntracker: {}\n", encoding="utf-8")
    os.environ["NOESIS_BUILD_DIR"] = str(root / "build")
    with mock.patch.object(
        runtime,
        "materialize_v3dt_tracker_config",
        side_effect=lambda _bundle, destination, output_root: Path(destination).resolve(),
    ):
        effective = runtime._materialize_effective_pipeline_yaml(
            pipeline,
            "noop",
            logging.getLogger("v3dt-effective-test"),
            tracking_mode="v3dt",
            v3dt_bundle=SimpleNamespace(),
        )
    payload = yaml.safe_load(effective.read_text(encoding="utf-8"))
    assert payload["tracking_mode"] == "v3dt"
    tracker_path = Path(payload["tracker"]["config-file"])
    assert tuple(tracker_path.parts[-3:]) == (
        "config",
        "v3dt",
        "nvtracker_v3dt_runtime.yaml",
    )
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            (str(DS9_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", ""))
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_preflight_maps_relative_nvinfer_model_path_to_artifact_root(self) -> None:
        preflight = _load_module(
            "ds9_preflight_model_root_test", "DS9/scripts/ds9_preflight.py"
        )
        with tempfile.TemporaryDirectory() as raw_root:
            artifact_root = Path(raw_root).resolve()
            with mock.patch.dict(
                os.environ,
                {"NOESIS_DS9_ARTIFACT_ROOT": str(artifact_root)},
                clear=False,
            ):
                for name in (
                    "NOESIS_MODEL_DIR",
                    "NOESIS_ONNX_DIR",
                    "NOESIS_ENGINE_DIR",
                ):
                    os.environ.pop(name, None)
                resolved = preflight._resolve(
                    "../models/onnx/reid.onnx",
                    base=DS9_ROOT / "pipelines",
                )
            self.assertEqual(resolved, artifact_root / "models" / "onnx" / "reid.onnx")

    def test_manifest_virtual_model_paths_map_to_artifact_root(self) -> None:
        validator = _load_module(
            "ds9_manifest_artifact_root_test",
            "DS9/scripts/validate_asset_manifest.py",
        )
        with tempfile.TemporaryDirectory() as raw_root:
            artifact_root = Path(raw_root).resolve()
            self.assertEqual(
                validator._physical_path(
                    Path("DS9/models/engines/v3dt.engine"), artifact_root
                ),
                artifact_root / "models" / "engines" / "v3dt.engine",
            )
            self.assertEqual(
                validator._physical_path(
                    Path("DS9/native_extensions/v3dt.so"), artifact_root
                ),
                REPO_ROOT / "DS9" / "native_extensions" / "v3dt.so",
            )

    def test_v3dt_builder_uses_documented_nvmot_query_init_lifecycle(self) -> None:
        source = (
            DS9_ROOT
            / "csrc"
            / "v3dt_engine_builder"
            / "v3dt_tracker_engine_builder.cpp"
        ).read_text(encoding="utf-8")
        for token in (
            "NvMOT_Query",
            "NvMOT_Init",
            "NvMOT_DeInit",
            "NvMOTBatchMode_Batch",
            "NvMOTConfigStatus_OK",
            "config.computeConfig = NVMOTCOMP_GPU",
            "config.numTransforms = 1U",
            "transform.bufferType = NVBUF_MEM_CUDA_DEVICE",
            "query.contextHandle = context",
            "context = nullptr",
        ):
            self.assertIn(token, source)
        init_offset = source.index("const NvMOTStatus init_status = init_fn")
        bind_offset = source.index("query.contextHandle = context")
        query_offset = source.index("const NvMOTStatus query_status = query_fn")
        self.assertLess(init_offset, bind_offset)
        self.assertLess(bind_offset, query_offset)

    def test_nvmot_tracker_engine_derivation_is_exact_and_fail_closed(self) -> None:
        assets = _load_module(
            "ds9_v3dt_derived_engine_test", "DS9/noesis/v3dt_assets.py"
        )
        source = Path("/tmp/sdk-source/resnet50_market1501.etlt")
        self.assertEqual(
            assets.derive_nvmot_tracker_engine_path(
                source,
                batch_size=32,
                gpu_id=0,
                network_mode=1,
            ),
            Path(
                "/tmp/sdk-source/"
                "resnet50_market1501.etlt_b32_gpu0_fp16.engine"
            ),
        )
        for kwargs in (
            {"batch_size": 0, "gpu_id": 0, "network_mode": 1},
            {"batch_size": 32, "gpu_id": -1, "network_mode": 1},
            {"batch_size": 32, "gpu_id": 0, "network_mode": 9},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(
                assets.V3DTAssetError
            ):
                assets.derive_nvmot_tracker_engine_path(source, **kwargs)

    def test_sdk_source_inventory_rejects_every_unexpected_sibling(self) -> None:
        builder = _load_module(
            "ds9_v3dt_sdk_inventory_test",
            "DS9/scripts/build_v3dt_tracker_engine.py",
        )
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            root.chmod(0o700)
            staged = root / "resnet50_market1501.etlt"
            staged.write_bytes(b"locked-source")
            inventory = builder._sdk_source_inventory(
                root, expected_names={staged.name}
            )
            self.assertEqual(
                [row["name"] for row in inventory["entries"]], [staged.name]
            )
            unexpected = root / "unexpected.engine"
            unexpected.write_bytes(b"unexpected")
            with self.assertRaisesRegex(
                builder.EngineMaintenanceError, "inventory differs"
            ):
                builder._sdk_source_inventory(
                    root, expected_names={staged.name}
                )

    def test_v3dt_builder_orders_sdk_adoption_before_existing_gates(self) -> None:
        source = (
            DS9_ROOT / "scripts" / "build_v3dt_tracker_engine.py"
        ).read_text(encoding="utf-8")
        real = source[source.index("with engine_maintenance_lock") :]
        tokens = (
            "staged_source_copy = copy_regular_file_exclusive",
            "pre_build_inventory = _sdk_source_inventory",
            '"build-tracker-engine"',
            "post_build_inventory = _sdk_source_inventory",
            "run.adopt_derived_candidate",
            "run.record_candidate",
            '"load-candidate"',
            "run.revalidate_inputs",
            "run.install_candidate",
            '"load-installed"',
        )
        offsets = [real.index(token) for token in tokens]
        self.assertEqual(offsets, sorted(offsets))

    def test_v3dt_builder_executes_context_bound_lifecycle_and_cleans_up(self) -> None:
        cxx = shutil.which("c++")
        if cxx is None:
            self.skipTest("C++ compiler is unavailable")

        fake_header = r"""
#ifndef NOESIS_FAKE_NVDSTRACKER_H
#define NOESIS_FAKE_NVDSTRACKER_H

#include <cstdint>

#define NVMOTCOMP_GPU 0x01
typedef std::uint8_t NvMOTCompute;

typedef enum {
  NVBUF_MEM_DEFAULT,
  NVBUF_MEM_CUDA_PINNED,
  NVBUF_MEM_CUDA_DEVICE,
  NVBUF_MEM_CUDA_UNIFIED
} NvBufSurfaceMemType;

typedef struct _NvMOTPerTransformBatchConfig {
  NvBufSurfaceMemType bufferType;
  std::uint32_t maxWidth;
  std::uint32_t maxHeight;
  std::uint32_t maxPitch;
  std::uint32_t maxSize;
} NvMOTPerTransformBatchConfig;

typedef struct _NvMOTMiscConfig {
  std::uint32_t gpuId;
} NvMOTMiscConfig;

typedef struct _NvMOTConfig {
  NvMOTCompute computeConfig;
  std::uint32_t maxStreams;
  std::uint32_t maxBufSurfAddrSize;
  std::uint8_t numTransforms;
  NvMOTPerTransformBatchConfig* perTransformBatchConfig;
  NvMOTMiscConfig miscConfig;
  std::uint16_t customConfigFilePathSize;
  char* customConfigFilePath;
} NvMOTConfig;

typedef enum {
  NvMOTConfigStatus_OK,
  NvMOTConfigStatus_Error,
  NvMOTConfigStatus_Invalid,
  NvMOTConfigStatus_Unsupported
} NvMOTConfigStatus;

typedef enum {
  NvMOTBatchMode_Error = 0,
  NvMOTBatchMode_Batch = 1 << 0,
  NvMOTBatchMode_NonBatch = 1 << 1
} NvMOTBatchMode;

typedef struct _NvMOTConfigResponse {
  NvMOTConfigStatus summaryStatus;
  NvMOTConfigStatus computeStatus;
  NvMOTConfigStatus transformBatchStatus;
  NvMOTConfigStatus miscConfigStatus;
  NvMOTConfigStatus customConfigStatus;
} NvMOTConfigResponse;

typedef enum {
  NvMOTStatus_OK,
  NvMOTStatus_Error,
  NvMOTStatus_Invalid_Path
} NvMOTStatus;

struct NvMOTContext {
  std::uint32_t marker;
};
typedef struct NvMOTContext* NvMOTContextHandle;

typedef struct _NvMOTQuery {
  NvMOTCompute computeConfig;
  std::uint8_t numTransforms;
  std::uint32_t colorFormats[1];
  NvBufSurfaceMemType memType;
  std::uint32_t maxTargetsPerStream;
  std::uint32_t maxShadowTrackingAge;
  bool outputReidTensor;
  std::uint32_t reidFeatureSize;
  bool outputTrajectory;
  bool outputVisibility;
  bool outputFootLocation;
  bool outputConvexHull;
  std::uint32_t maxConvexHullSize;
  bool supportPastFrame;
  NvMOTBatchMode batchMode;
  bool outputTerminatedTracks;
  std::uint32_t maxTrajectoryBufferLength;
  bool outputShadowTracks;
  NvMOTContextHandle contextHandle;
} NvMOTQuery;

extern "C" {
NvMOTStatus NvMOT_Init(
    NvMOTConfig*, NvMOTContextHandle*, NvMOTConfigResponse*);
void NvMOT_DeInit(NvMOTContextHandle);
NvMOTStatus NvMOT_Query(std::uint16_t, char*, NvMOTQuery*);
}

#endif
"""
        fake_library = r"""
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "nvdstracker.h"

namespace {
NvMOTContext fake_context{0x4e564d4fU};

void record(const char* event) {
  const char* path = std::getenv("NOESIS_FAKE_NVMOT_LOG");
  if (path != nullptr) {
    std::ofstream stream(path, std::ios::app);
    stream << event << '\n';
  }
}

bool mode_is(const char* expected) {
  const char* mode = std::getenv("NOESIS_FAKE_NVMOT_MODE");
  return mode != nullptr && std::strcmp(mode, expected) == 0;
}
}  // namespace

extern "C" NvMOTStatus NvMOT_Init(
    NvMOTConfig* config,
    NvMOTContextHandle* context,
    NvMOTConfigResponse* response) {
  record("init");
  if (config == nullptr || context == nullptr || response == nullptr) {
    return NvMOTStatus_Error;
  }
  *context = &fake_context;
  response->summaryStatus = NvMOTConfigStatus_OK;
  response->computeStatus = NvMOTConfigStatus_OK;
  response->transformBatchStatus = NvMOTConfigStatus_OK;
  response->miscConfigStatus = NvMOTConfigStatus_OK;
  response->customConfigStatus = NvMOTConfigStatus_OK;
  const bool bootstrap_is_valid =
      config->computeConfig == NVMOTCOMP_GPU &&
      config->maxStreams == 3U && config->maxBufSurfAddrSize == 12U &&
      config->numTransforms == 1U &&
      config->perTransformBatchConfig != nullptr &&
      config->perTransformBatchConfig->bufferType == NVBUF_MEM_CUDA_DEVICE &&
      config->miscConfig.gpuId == 0U &&
      config->customConfigFilePathSize > 0U &&
      config->customConfigFilePath != nullptr;
  if (!bootstrap_is_valid || mode_is("init_error")) {
    return NvMOTStatus_Error;
  }
  return NvMOTStatus_OK;
}

extern "C" NvMOTStatus NvMOT_Query(
    std::uint16_t path_size, char* path, NvMOTQuery* query) {
  record("query");
  if (mode_is("query_error")) {
    return NvMOTStatus_Error;
  }
  if (path_size == 0U || path == nullptr || query == nullptr ||
      query->contextHandle != &fake_context) {
    return NvMOTStatus_Error;
  }
  query->computeConfig = NVMOTCOMP_GPU;
  query->numTransforms = mode_is("bad_caps") ? 0U : 1U;
  query->memType = NVBUF_MEM_CUDA_DEVICE;
  query->batchMode = NvMOTBatchMode_Batch;
  return NvMOTStatus_OK;
}

extern "C" void NvMOT_DeInit(NvMOTContextHandle context) {
  record(context == &fake_context ? "deinit" : "bad_deinit");
}
"""

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "nvdstracker.h").write_text(fake_header, encoding="utf-8")
            fake_source = root / "fake_nvmot.cpp"
            fake_source.write_text(fake_library, encoding="utf-8")
            fake_so = root / "libfake_nvmot.so"
            helper = root / "v3dt_tracker_engine_builder"
            for command in (
                [
                    cxx,
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{root}",
                    str(fake_source),
                    "-o",
                    str(fake_so),
                ],
                [
                    cxx,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{root}",
                    str(
                        DS9_ROOT
                        / "csrc"
                        / "v3dt_engine_builder"
                        / "v3dt_tracker_engine_builder.cpp"
                    ),
                    "-ldl",
                    "-o",
                    str(helper),
                ],
            ):
                compiled = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(
                    compiled.returncode, 0, compiled.stdout + compiled.stderr
                )

            tracker_config = root / "tracker.yaml"
            tracker_config.write_text("BaseConfig: {}\n", encoding="utf-8")
            event_log = root / "events.log"
            cases = (
                ("success", 0, ("init", "query", "deinit")),
                ("query_error", 1, ("init", "query", "deinit")),
                ("bad_caps", 1, ("init", "query", "deinit")),
                ("init_error", 1, ("init", "deinit")),
            )
            for mode, expected_returncode, expected_events in cases:
                with self.subTest(mode=mode):
                    event_log.unlink(missing_ok=True)
                    env = dict(os.environ)
                    env["NOESIS_FAKE_NVMOT_LOG"] = str(event_log)
                    env["NOESIS_FAKE_NVMOT_MODE"] = mode
                    proc = subprocess.run(
                        [
                            str(helper),
                            "--tracker-config",
                            str(tracker_config),
                            "--tracker-lib",
                            str(fake_so),
                            "--streams",
                            "3",
                            "--width",
                            "1920",
                            "--height",
                            "1080",
                            "--gpu-id",
                            "0",
                        ],
                        cwd=REPO_ROOT,
                        env=env,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(
                        proc.returncode,
                        expected_returncode,
                        proc.stdout + proc.stderr,
                    )
                    self.assertEqual(
                        tuple(event_log.read_text(encoding="utf-8").splitlines()),
                        expected_events,
                    )

    def test_engine_maintenance_gates_the_installed_ds9_driver_floor(self) -> None:
        source = (
            DS9_ROOT / "scripts" / "run_canonical_engine_maintenance.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("REQUIRED_DRIVER_MAJOR=590", source)
        self.assertIn("HOST_DRIVER_MAJOR < REQUIRED_DRIVER_MAJOR", source)
        self.assertIn("upgrade the driver before any DS9 engine build", source)


if __name__ == "__main__":
    unittest.main()
