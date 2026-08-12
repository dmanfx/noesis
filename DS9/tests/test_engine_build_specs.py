from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load_rebuild_module():
    path = ROOT / "DS9" / "scripts" / "rebuild_engines.py"
    spec = importlib.util.spec_from_file_location("ds9_rebuild_engines", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load DS9 engine rebuild module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EngineBuildSpecTests(unittest.TestCase):
    def test_mapanything_is_correctness_first_fp32_without_empty_precision_arg(self) -> None:
        module = _load_rebuild_module()
        specs = {row.name: row for row in module._specs(include_mapanything=True)}
        mapanything = specs["mapanything"]
        self.assertIsNone(mapanything.precision_arg)
        self.assertEqual(
            mapanything.engine.name,
            "mapanything_images_294x518_b3_fp32.plan",
        )
        self.assertNotIn("--bf16", mapanything.trtexec_args)
        authority = json.loads(
            (ROOT / "DS9/config/engine_source_contracts.json").read_text(
                encoding="utf-8"
            )
        )["contracts"]["mapanything"]
        self.assertEqual(
            module._build_contract(
                mapanything,
                onnx_metadata=authority["onnx"],
                plugin_args=[],
            ),
            {**authority["maintenance_build"], "onnx": authority["onnx"]},
        )

        with (
            mock.patch.object(module, "_stage_onnx"),
            mock.patch.object(
                module,
                "validate_source_contract",
                return_value={"onnx": authority["onnx"]},
            ),
            mock.patch.object(module, "_print_run") as print_run,
        ):
            module._build(
                mapanything,
                trtexec="/usr/bin/trtexec",
                dry_run=True,
            )
        commands = [list(call.args[0]) for call in print_run.call_args_list]
        build = next(command for command in commands if any(str(value).startswith("--onnx=") for value in command))
        quality = next(command for command in commands if "--dumpOutput" in command)
        self.assertNotIn("", build)
        self.assertNotIn("--fp16", build)
        self.assertNotIn("--bf16", build)
        self.assertIn("--dumpOutput", quality)
        self.assertFalse(any("--skipInference" in str(value) for value in quality))

    def test_reviewed_sources_are_ds9_staged_and_ignore_external_model_root(self) -> None:
        with mock.patch.dict(os.environ, {"NOESIS_DS9_SOURCE_MODELS_ROOT": "/source-models"}):
            module = _load_rebuild_module()
        specs = {row.name: row for row in module._specs(include_mapanything=False)}
        self.assertEqual(
            specs["reid_swin"].source_onnx,
            module.DS9_ONNX / "reid_swin_tiny_market1501_aicity156_featuredim256.onnx",
        )
        self.assertEqual(
            specs["reid_swin"].trtexec_args,
            (
                "--minShapes=input:1x3x256x128",
                "--optShapes=input:16x3x256x128",
                "--maxShapes=input:16x3x256x128",
            ),
        )
        self.assertEqual(
            specs["depth_anything_v2_tracking"].source_onnx,
            module.DS9_ONNX / "depth_anything_v2_metric_hypersim_vits_294x518_b3.onnx",
        )
        self.assertEqual(
            specs["wholebody49_s_masks"].source_onnx,
            module.DS9_ONNX / "deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx",
        )
        self.assertEqual(
            specs["wholebody49_x_boxes"].source_onnx,
            module.DS9_ONNX / "deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx",
        )

    def test_wholebody49_specs_are_ds9_owned_fp16_batch_three(self) -> None:
        module = _load_rebuild_module()
        specs = {row.name: row for row in module._specs(include_mapanything=False)}
        expected = {
            "wholebody49_s_masks": "deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine",
            "wholebody49_x_boxes": "deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine",
        }
        variants = {
            "wholebody49_s_masks": "s_masks",
            "wholebody49_x_boxes": "x_boxes",
        }
        for name, engine_name in expected.items():
            spec = specs[name]
            self.assertEqual(spec.engine, module.DS9_ENGINES / engine_name)
            self.assertEqual(spec.staged_onnx.parent, module.DS9_ONNX)
            self.assertEqual(spec.trtexec_args, ())
            self.assertEqual(spec.builder, module.WHOLEBODY_BUILDER_CONTRACT)
            self.assertEqual(spec.builder_variant, variants[name])

    def test_wholebody49_dry_runs_use_the_exact_reviewed_cpp_builder(self) -> None:
        module = _load_rebuild_module()
        specs = {row.name: row for row in module._specs(include_mapanything=False)}
        contracts = json.loads(
            (ROOT / "DS9/config/engine_source_contracts.json").read_text(
                encoding="utf-8"
            )
        )["contracts"]
        for name in ("wholebody49_s_masks", "wholebody49_x_boxes"):
            with self.subTest(name=name):
                spec = specs[name]
                expected_contract = dict(contracts[name]["maintenance_build"])
                expected_contract["onnx"] = contracts[name]["onnx"]
                self.assertEqual(
                    module._build_contract(
                        spec,
                        onnx_metadata=contracts[name]["onnx"],
                        plugin_args=[],
                    ),
                    expected_contract,
                )
                with (
                    mock.patch.object(module, "_stage_onnx"),
                    mock.patch.object(
                        module,
                        "validate_source_contract",
                        return_value={"onnx": contracts[name]["onnx"]},
                    ),
                    mock.patch.object(module, "_print_run") as print_run,
                ):
                    module._build(
                        spec,
                        trtexec="/usr/bin/trtexec",
                        dry_run=True,
                    )
                commands = [tuple(call.args[0]) for call in print_run.call_args_list]
                build = [
                    command
                    for command in commands
                    if command
                    and str(command[0]).startswith(
                        "/tmp/noesis-wholebody49-engine-builder-"
                    )
                    and "--onnx" in command
                ]
                self.assertEqual(len(build), 1)
                self.assertIn("--variant", build[0])
                self.assertIn(spec.builder_variant, build[0])
                self.assertFalse(
                    any(
                        any(str(token).startswith("--saveEngine=") for token in command)
                        for command in commands
                    )
                )
                compile_commands = [
                    command
                    for command in commands
                    if any(
                        str(token).endswith("/wholebody49_engine_builder.cpp")
                        for token in command
                    )
                ]
                self.assertEqual(len(compile_commands), 1)
                self.assertIn("-Werror", compile_commands[0])
                self.assertIn("-lnvonnxparser", compile_commands[0])

    def test_wholebody49_cpp_source_is_fail_closed_and_exact(self) -> None:
        module = _load_rebuild_module()
        source = module.WHOLEBODY_BUILDER_SOURCE.read_text(encoding="utf-8")
        for exact_version_assertion in (
            "NV_TENSORRT_MAJOR == 10",
            "NV_TENSORRT_MINOR == 16",
            "NV_TENSORRT_PATCH == 0",
            "NV_TENSORRT_BUILD == 72",
            "TensorRT 10.16.0.72",
        ):
            self.assertIn(exact_version_assertion, source)
        for token in (
            "createNetworkV2(0U)",
            "parser->parse(onnx.bytes.data(), onnx.bytes.size()",
            "MemoryPoolType::kWORKSPACE",
            "MemoryPoolType::kTACTIC_DRAM",
            "getMemoryPoolLimit",
            "profile->setDimensions",
            "profile->isValid()",
            "config->getNbOptimizationProfiles() != 1",
            "O_EXCL",
            "O_NOFOLLOW",
            "status=PASS",
            "status=FAIL",
            "kSMasksMaximumEngineBytes",
            "kXBoxesMaximumEngineBytes",
            "kSMasksWorkspaceBytes = 6144ULL * 1024ULL * 1024ULL",
            "kXBoxesWorkspaceBytes = 4096ULL * 1024ULL * 1024ULL",
            "workspace_bytes(Variant variant)",
            "selected_workspace_bytes = workspace_bytes(options.variant)",
            "kSMasksBuilderOptimizationLevel = 0",
            "kXBoxesBuilderOptimizationLevel = 3",
            "builder_optimization_level(Variant variant)",
            "selected_builder_optimization_level =",
            "setBuilderOptimizationLevel(selected_builder_optimization_level)",
            "getBuilderOptimizationLevel()",
            "if (severity == Severity::kVERBOSE)",
            "kLoggerMinimumSeverity[] = \"info\"",
            "kLoggerVerbosePolicy[] = \"ignored_before_copy\"",
            "kLoggerCapturedTruncationPolicy[] = \"fatal\"",
            "kLoggerErrorStatePolicy[] = \"sticky_fatal\"",
            "static_assert(is_positive_mib_aligned(kSMasksWorkspaceBytes)",
            "static_assert(is_positive_mib_aligned(kXBoxesWorkspaceBytes)",
            "static_assert(is_positive_power_of_two(kTacticDramBytes)",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertNotIn("parseFromFile", source)
        self.assertNotIn("kSTRONGLY_TYPED", source)
        self.assertNotIn("::unlink", source)
        self.assertEqual(source.count("setBuilderOptimizationLevel("), 1)
        self.assertEqual(source.count("getBuilderOptimizationLevel()"), 2)
        self.assertNotIn("setTacticSources", source)
        self.assertNotIn("setMaxAuxStreams", source)

    def test_wholebody49_logger_ignores_verbose_before_bounded_copy(self) -> None:
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required for the extracted logger contract test")
        source = (
            ROOT
            / "DS9/csrc/wholebody49_engine_builder/wholebody49_engine_builder.cpp"
        ).read_text(encoding="utf-8")
        logger_start = source.index("class Logger final")
        logger_end = source.index("\nclass ErrorRecorder final", logger_start)
        logger_source = source[logger_start:logger_end]
        maximum = re.search(
            r"constexpr std::size_t kMaximumLogMessageBytes = [^;]+;", source
        )
        self.assertIsNotNone(maximum)
        harness = textwrap.dedent(
            f"""
            #include <array>
            #include <atomic>
            #include <cstddef>
            #include <cstdio>
            #include <mutex>
            #include <string>

            namespace nvinfer1 {{
            class ILogger {{
             public:
              enum class Severity {{
                kINTERNAL_ERROR = 0,
                kERROR = 1,
                kWARNING = 2,
                kINFO = 3,
                kVERBOSE = 4,
              }};
              virtual ~ILogger() noexcept = default;
              virtual void log(Severity severity, char const* message) noexcept = 0;
            }};
            }}

            {maximum.group(0)}
            {logger_source}

            int main() {{
              std::string const overlong(kMaximumLogMessageBytes + 32U, 'x');

              Logger verbose;
              verbose.log(nvinfer1::ILogger::Severity::kVERBOSE,
                          overlong.c_str());
              if (verbose.truncated() || verbose.has_error()) return 1;

              Logger info;
              info.log(nvinfer1::ILogger::Severity::kINFO, overlong.c_str());
              if (!info.truncated() || info.has_error()) return 2;

              Logger warning;
              warning.log(nvinfer1::ILogger::Severity::kWARNING,
                          overlong.c_str());
              if (!warning.truncated() || warning.has_error()) return 3;

              Logger error;
              error.log(nvinfer1::ILogger::Severity::kERROR, overlong.c_str());
              if (!error.truncated() || !error.has_error()) return 4;

              Logger internal;
              internal.log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
                           overlong.c_str());
              if (!internal.truncated() || !internal.has_error()) return 5;

              Logger real_error;
              real_error.log(nvinfer1::ILogger::Severity::kERROR,
                             "REAL_ERROR_SENTINEL");
              if (real_error.truncated() || !real_error.has_error()) return 6;
              return 0;
            }}
            """
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            harness_path = root / "logger_contract.cpp"
            executable = root / "logger_contract"
            harness_path.write_text(harness, encoding="utf-8")
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-Werror",
                    str(harness_path),
                    "-o",
                    str(executable),
                    "-pthread",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [str(executable)], check=True, capture_output=True, text=True
            )
        self.assertEqual(completed.stdout, "")
        self.assertIn("REAL_ERROR_SENTINEL", completed.stderr)

    def test_wholebody49_transcript_proof_requires_all_exact_markers(self) -> None:
        common_path = ROOT / "DS9/scripts/engine_maintenance_common.py"
        spec = importlib.util.spec_from_file_location("ds9_maintenance_common_test", common_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        def transcript(variant: str, workspace_bytes: int) -> str:
            return "\n".join(
                (
                    "[NOESIS_TRT_BUILDER] contract=noesis.ds9.wholebody49_builder.v1",
                    "[NOESIS_TRT_BUILDER] network_mode=explicit_batch_trt10_default",
                    f"[NOESIS_TRT_BUILDER] variant={variant}",
                    "[NOESIS_TRT_BUILDER] profile=images:3x3x640x640",
                    f"[NOESIS_TRT_BUILDER] workspace_bytes={workspace_bytes}",
                    "[NOESIS_TRT_BUILDER] tactic_dram_bytes=2147483648",
                    "[NOESIS_TRT_BUILDER] builder_optimization_level="
                    f"{0 if variant == 's_masks' else 3}",
                    "[NOESIS_TRT_BUILDER] logger_minimum_severity=info",
                    "[NOESIS_TRT_BUILDER] logger_verbose_policy=ignored_before_copy",
                    "[NOESIS_TRT_BUILDER] logger_captured_truncation=fatal",
                    "[NOESIS_TRT_BUILDER] logger_error_state=sticky_fatal",
                    "[NOESIS_TRT_BUILDER] engine_bytes=33429356",
                    "[NOESIS_TRT_BUILDER] status=PASS",
                )
            )

        s_transcript = transcript("s_masks", 6442450944)
        x_transcript = transcript("x_boxes", 4294967296)
        common.validate_wholebody_builder_transcript(
            s_transcript, expected_variant="s_masks"
        )
        common.validate_wholebody_builder_transcript(
            x_transcript, expected_variant="x_boxes"
        )
        with self.assertRaisesRegex(common.EngineMaintenanceError, "exact marker"):
            common.validate_wholebody_builder_transcript(
                s_transcript.replace("tactic_dram_bytes=2147483648\n", ""),
                expected_variant="s_masks",
            )
        for bad_transcript in (
            s_transcript.replace(
                "[NOESIS_TRT_BUILDER] workspace_bytes=6442450944\n", ""
            ),
            s_transcript.replace(
                "[NOESIS_TRT_BUILDER] workspace_bytes=6442450944",
                "[NOESIS_TRT_BUILDER] workspace_bytes=4294967296",
            ),
            s_transcript.replace(
                "[NOESIS_TRT_BUILDER] workspace_bytes=6442450944",
                "[NOESIS_TRT_BUILDER] workspace_bytes=6442450944\n"
                "[NOESIS_TRT_BUILDER] workspace_bytes=6442450944",
            ),
        ):
            with self.subTest(bad_transcript=bad_transcript):
                with self.assertRaisesRegex(
                    common.EngineMaintenanceError, "variant workspace marker"
                ):
                    common.validate_wholebody_builder_transcript(
                        bad_transcript,
                        expected_variant="s_masks",
                    )
        for marker in (
            "logger_minimum_severity=info",
            "logger_verbose_policy=ignored_before_copy",
            "logger_captured_truncation=fatal",
            "logger_error_state=sticky_fatal",
        ):
            with self.subTest(logger_marker=marker):
                exact_line = f"[NOESIS_TRT_BUILDER] {marker}"
                with self.assertRaisesRegex(
                    common.EngineMaintenanceError, "exact marker"
                ):
                    common.validate_wholebody_builder_transcript(
                        s_transcript.replace(exact_line + "\n", ""),
                        expected_variant="s_masks",
                    )
                with self.assertRaisesRegex(
                    common.EngineMaintenanceError, "exact marker"
                ):
                    common.validate_wholebody_builder_transcript(
                        s_transcript.replace(
                            exact_line, exact_line + "\n" + exact_line
                        ),
                        expected_variant="s_masks",
                    )
        for bad_transcript in (
            s_transcript.replace(
                "[NOESIS_TRT_BUILDER] builder_optimization_level=0\n", ""
            ),
            s_transcript.replace(
                "[NOESIS_TRT_BUILDER] builder_optimization_level=0",
                "[NOESIS_TRT_BUILDER] builder_optimization_level=3",
            ),
            s_transcript.replace(
                "[NOESIS_TRT_BUILDER] builder_optimization_level=0",
                "[NOESIS_TRT_BUILDER] builder_optimization_level=0\n"
                "[NOESIS_TRT_BUILDER] builder_optimization_level=0",
            ),
        ):
            with self.subTest(bad_optimization_transcript=bad_transcript):
                with self.assertRaisesRegex(
                    common.EngineMaintenanceError,
                    "variant optimization marker",
                ):
                    common.validate_wholebody_builder_transcript(
                        bad_transcript,
                        expected_variant="s_masks",
                    )
        with self.assertRaisesRegex(common.EngineMaintenanceError, "fail-closed"):
            common.validate_wholebody_builder_transcript(
                s_transcript + "\n[NOESIS_TRT_BUILDER] status=FAIL reason=test",
                expected_variant="s_masks",
            )
        with self.assertRaisesRegex(common.EngineMaintenanceError, "selected variant"):
            common.validate_wholebody_builder_transcript(
                s_transcript,
                expected_variant="x_boxes",
            )

    def test_wholebody49_memory_pools_apply_pool_specific_alignment(self) -> None:
        common_path = ROOT / "DS9/scripts/engine_maintenance_common.py"
        spec = importlib.util.spec_from_file_location(
            "ds9_maintenance_common_pool_test", common_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        for workspace_bytes in (6442450944, 4294967296):
            with self.subTest(workspace_bytes=workspace_bytes):
                self.assertEqual(
                    common.validate_wholebody_memory_pool_limits(
                        {
                            "workspace": workspace_bytes,
                            "tactic_dram": 2147483648,
                        }
                    ),
                    {
                        "workspace": workspace_bytes,
                        "tactic_dram": 2147483648,
                    },
                )
        with self.assertRaisesRegex(
            common.EngineMaintenanceError, "positive power of two"
        ):
            common.validate_wholebody_memory_pool_limits(
                {"workspace": 4294967296, "tactic_dram": 3221225472}
            )
        with self.assertRaisesRegex(
            common.EngineMaintenanceError, "positive and MiB-aligned"
        ):
            common.validate_wholebody_memory_pool_limits(
                {"workspace": 6442450945, "tactic_dram": 2147483648}
            )

    def test_wholebody49_builder_optimization_level_is_an_exact_integer(self) -> None:
        common_path = ROOT / "DS9/scripts/engine_maintenance_common.py"
        spec = importlib.util.spec_from_file_location(
            "ds9_maintenance_common_optimization_test", common_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        for value in (0, 3, 5):
            with self.subTest(value=value):
                self.assertEqual(
                    common.validate_wholebody_builder_optimization_level(value),
                    value,
                )
        for value in (False, -1, 6, "0"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    common.EngineMaintenanceError,
                    "integer from 0 to 5",
                ):
                    common.validate_wholebody_builder_optimization_level(value)

    def test_wholebody49_logger_policy_is_exact(self) -> None:
        common_path = ROOT / "DS9/scripts/engine_maintenance_common.py"
        spec = importlib.util.spec_from_file_location(
            "ds9_maintenance_common_logger_test", common_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        expected = {
            "minimum_severity": "info",
            "verbose": "ignored_before_copy",
            "captured_message_truncation": "fatal",
            "error_state": "sticky_fatal",
        }
        self.assertEqual(common.validate_wholebody_logger_policy(expected), expected)
        for value in (
            None,
            {},
            {**expected, "minimum_severity": "warning"},
            {**expected, "verbose": "captured"},
            {**expected, "captured_message_truncation": "ignored"},
            {**expected, "error_state": "non_sticky"},
            {**expected, "extra": "unreviewed"},
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    common.EngineMaintenanceError, "logger policy"
                ):
                    common.validate_wholebody_logger_policy(value)

    def test_real_build_always_validates_candidate_and_installed_paths(self) -> None:
        source = (ROOT / "DS9" / "scripts" / "rebuild_engines.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"load-candidate"', source)
        self.assertIn('"load-installed"', source)
        self.assertIn("rollback_after_install_failure", source)

    def test_unscoped_rebuild_is_forbidden(self) -> None:
        module = _load_rebuild_module()
        with (
            mock.patch.object(sys, "argv", ["rebuild_engines.py"]),
            mock.patch.object(module.shutil, "which", return_value="/fake/trtexec"),
            self.assertRaisesRegex(SystemExit, "--only is required"),
        ):
            module.main()

    def test_scoped_rebuild_requires_explicit_external_model_root(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            module = _load_rebuild_module()
        with (
            mock.patch.object(
                sys,
                "argv",
                ["rebuild_engines.py", "--only", "yolo26_m", "--dry-run"],
            ),
            mock.patch.object(module.shutil, "which", return_value="/fake/trtexec"),
            self.assertRaisesRegex(
                SystemExit,
                "must explicitly select the external DS9 model artifact root",
            ),
        ):
            module.main()


if __name__ == "__main__":
    unittest.main()
