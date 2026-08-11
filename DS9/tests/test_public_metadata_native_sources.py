from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"
REID_SOURCE = DS9_ROOT / "native" / "noesis_reid_meta_ext.cpp"
POSE_SOURCE = DS9_ROOT / "native" / "noesis_pose_meta_ext.cpp"
DEPTH_META_SOURCE = DS9_ROOT / "native" / "noesis_depth_meta_ext.cpp"
DEPTH_TENSOR_SOURCE = (
    DS9_ROOT / "native" / "noesis_depth_tracking_tensor_ext.cpp"
)


@pytest.mark.parametrize("source_path", (REID_SOURCE, POSE_SOURCE))
def test_reid_and_pose_native_sources_do_not_bypass_service_maker_metadata(
    source_path: Path,
) -> None:
    source = source_path.read_text(encoding="utf-8")
    forbidden = (
        "Metadata::data_",
        "GList",
        "obj_user_meta_list",
        "frame_user_meta_list",
        "NvDsInferTensorMeta",
        "NvDsInferLayerInfo",
        "NvDsObjectMeta",
        "NvDsFrameMeta",
        "NvDsUserMeta",
        "unwrap_object_meta",
        "unwrap_frame_meta",
        '"gstnvdsinfer.h"',
    )

    for token in forbidden:
        assert token not in source, f"{source_path.name} still uses {token}"

    assert "TensorOutputUserMetadata" in source
    assert ".getLayers()" in source
    assert "tensor.hpp" in source
    assert "NVDSINFER_TENSOR_OUTPUT_META" in source


def test_reid_uses_public_object_iteration_and_typed_tensor_access() -> None:
    source = REID_SOURCE.read_text(encoding="utf-8")

    assert "obj_meta.iterate(" in source
    assert "tensor_meta.uniqueId()" in source
    assert "tensor.shape()" in source
    assert "tensor.dtype()" in source
    assert "tensor.deviceType()" in source
    assert "cudaMemcpyDeviceToHost" in source


def test_reid_requires_the_exact_requested_layer_and_embedding_contract() -> None:
    source = REID_SOURCE.read_text(encoding="utf-8")
    selector = source[
        source.index("deepstream::Tensor* select_reid_layer") :
        source.index("std::optional<std::vector<float>> extract_reid_embedding_impl")
    ]

    assert "layers.find(layer_name)" in selector
    assert "usable_layers" not in selector
    assert "tensor_num_elements" not in selector
    assert "if (gie_id < 0 || layer_name.empty() || expected_dim <= 0)" in source
    assert "values.size() != static_cast<size_t>(expected_dim)" in source
    assert 'py::arg("layer_name") = "fc_pred"' in source
    assert 'py::arg("expected_dim") = 256' in source
    assert source.index("for (float value : values)") < source.index("if (normalize) {")
    assert "matching ReID tensor contains a non-finite embedding" in source


def test_pose_uses_ds9_public_batch_user_metadata_lifecycle() -> None:
    source = POSE_SOURCE.read_text(encoding="utf-8")

    assert "deepstream::BatchMetadata& batch_meta" in source
    assert "batch_meta.acquire(user_meta)" in source
    assert "user_meta.setMetaType(meta_type)" in source
    assert "user_meta.setUserData(" in source
    assert "owner_meta.append(user_meta)" in source
    assert "owner_meta.iterate(" in source
    assert "obj_meta.rectParams()" in source
    assert 'py::arg("batch_meta")' in source


def test_pose_requires_output0_and_transfers_payload_before_append() -> None:
    source = POSE_SOURCE.read_text(encoding="utf-8")
    selector = source[
        source.index("std::pair<std::string, deepstream::Tensor*> select_pose_layer") :
        source.index("bool extract_best_pose_row")
    ]

    assert 'layers.find("output0")' in selector
    assert "usable" not in selector
    set_user_data = source.index("user_meta.setUserData(")
    release_payload = source.index("payload.release();", set_user_data)
    append_metadata = source.index("owner_meta.append(user_meta);", set_user_data)
    assert set_user_data < release_payload < append_metadata


@pytest.mark.parametrize("source_path", (DEPTH_META_SOURCE, DEPTH_TENSOR_SOURCE))
def test_depth_native_sources_do_not_bypass_service_maker_metadata(
    source_path: Path,
) -> None:
    source = source_path.read_text(encoding="utf-8")
    forbidden = (
        "Metadata::data_",
        "GList",
        "obj_user_meta_list",
        "frame_user_meta_list",
        "NvDsInferTensorMeta",
        "NvDsInferLayerInfo",
        "NvDsObjectMeta",
        "NvDsFrameMeta",
        "NvDsUserMeta",
        "unwrap_object_meta",
        "unwrap_frame_meta",
        "out_buf_ptrs_host",
        "out_buf_ptrs_dev",
        '"gstnvdsinfer.h"',
    )

    for token in forbidden:
        assert token not in source, f"{source_path.name} still uses {token}"


def test_depth_meta_uses_public_object_mask_and_batch_user_metadata_lifecycle() -> None:
    source = DEPTH_META_SOURCE.read_text(encoding="utf-8")

    assert "deepstream::BatchMetadata& batch_meta" in source
    assert "batch_meta.acquire(user_meta)" in source
    assert "user_meta.setMetaType(meta_type)" in source
    assert "user_meta.setUserData(" in source
    assert "obj_meta.append(user_meta)" in source
    assert "obj_meta.iterate(" in source
    assert "obj_meta.maskParams()" in source
    assert "raw_size < expected_bytes" in source
    assert source.index('py::arg("batch_meta")') < source.index('py::arg("obj_meta")')
    assert source.index("payload.release();") < source.index("obj_meta.append(user_meta);")


def test_depth_tensor_uses_filtered_public_frame_iteration_and_typed_tensors() -> None:
    source = DEPTH_TENSOR_SOURCE.read_text(encoding="utf-8")

    assert "frame_meta.iterate(" in source
    assert "NVDSINFER_TENSOR_OUTPUT_META" in source
    assert "TensorOutputUserMetadata" in source
    assert "tensor_meta.uniqueId()" in source
    assert "tensor_meta.getLayers()" in source
    assert "tensor.shape()" in source
    assert "tensor.dtype()" in source
    assert "tensor.bits()" in source
    assert "tensor.data()" in source
    assert "tensor.deviceType()" in source
    assert "tensor.deviceId()" in source
    assert "tensor.size()" in source
    assert "delete entry.second" in source
    assert "released.insert(entry.second).second" in source
    assert '"tensor.hpp"' in source
    assert "nppiResize_32f_C1R_Ctx" in source
    assert "cudaMemcpyDeviceToHost" in source
    assert 'layers.find("depth")' in source
    assert "preferred_names" not in source
    assert '"pred"' not in source
    assert '"output"' not in source
    assert "depth_hw(*entry.second)" not in source
    assert "unsigned int device_id_" in source
    assert "aligned.release(), device_id" in source
    assert source.count("CudaDeviceGuard device_guard(device_id_);") == 5
    assert source.count("ensure_capacity(") == 6
    assert source.count("ensure_capacity(sampled_count, device_id_)") == 3
    assert source.count("ensure_capacity(mask_count, device_id_)") == 2
    assert "allocate(count, device_id);" in source
    assert "device_id_ == static_cast<int>(device_id)" in source
    assert "CudaDeviceGuard device_guard(static_cast<unsigned int>(device_id_));" in source
    assert '.def_property_readonly("device_id"' in source


def test_depth_tensor_exposes_one_exact_mapanything_capture_contract() -> None:
    source = DEPTH_TENSOR_SOURCE.read_text(encoding="utf-8")

    assert "capture_mapanything_tensor_layers_exact" in source
    assert '"capture_tensor_layers",' not in source
    assert "matching_meta_count == 0U" in source
    assert "matching_meta_count != 1U" in source
    assert "ambiguous duplicate tensor metadata" in source
    assert '"depth", "conf", "mask"' in source
    assert "owned.size() != required_layers.size()" in source
    assert "dims.size() != 3U" in source
    assert "dims[0] != 1" in source
    assert "dims[1] != expected_height" in source
    assert "dims[2] != expected_width" in source
    assert "copy_mapanything_frame_layer_to_numpy(" in source
    assert "gst-nvinfer attach_tensor_output_meta already advances" in source
    assert "py::gil_scoped_release release;" in source
    assert source.count("py::gil_scoped_release release;") >= 2
    selector = source[
        source.index("py::object capture_mapanything_tensor_layers_exact") :
        source.index("PYBIND11_MODULE")
    ]
    assert "batch_index" not in selector
    assert "expected_batch_size" not in selector
    assert "byte_offset" not in selector
    assert "copy_tensor_slice_to_host" not in source


def test_canonical_ds9_build_routes_link_depth_metadata_and_cuda_npp() -> None:
    build_one = (DS9_ROOT / "scripts" / "build_native_ext_ds9.sh").read_text(
        encoding="utf-8"
    )
    build_all = (DS9_ROOT / "scripts" / "build_all_native_ds9.sh").read_text(
        encoding="utf-8"
    )

    assert "plain|cuda_runtime|cuda_npp" in build_one
    assert 'LDFLAGS+=(-lnppig -lnppidei -lnppc)' in build_one
    assert (
        'noesis_depth_meta_ext "${ROOT}/native/noesis_depth_meta_ext.cpp"'
        in build_all
    )
    assert (
        'noesis_depth_tracking_tensor_ext "${ROOT}/native/noesis_depth_tracking_tensor_ext.cpp" cuda_npp'
        in build_all
    )


def test_canonical_ds9_build_routes_link_reid_and_pose_as_cuda_runtime() -> None:
    build_one = (DS9_ROOT / "scripts" / "build_native_ext_ds9.sh").read_text(
        encoding="utf-8"
    )
    build_all = (DS9_ROOT / "scripts" / "build_all_native_ds9.sh").read_text(
        encoding="utf-8"
    )
    build_pose = (DS9_ROOT / "scripts" / "build_noesis_pose_meta_ext.sh").read_text(
        encoding="utf-8"
    )
    build_reid = (DS9_ROOT / "scripts" / "build_noesis_reid_meta_ext.sh").read_text(
        encoding="utf-8"
    )
    compatibility_builder = (
        DS9_ROOT / "scripts" / "build_native_extensions.sh"
    ).read_text(encoding="utf-8")

    assert "plain|cuda_runtime|cuda_npp" in build_one
    assert 'LDFLAGS+=(-lcudart)' in build_one
    assert (
        'noesis_pose_meta_ext "${ROOT}/native/noesis_pose_meta_ext.cpp" cuda_runtime'
        in build_all
    )
    assert (
        'noesis_reid_meta_ext "${ROOT}/native/noesis_reid_meta_ext.cpp" cuda_runtime'
        in build_all
    )
    assert build_pose.rstrip().endswith("cuda_runtime")
    assert build_reid.rstrip().endswith("cuda_runtime")
    assert 'exec "${DS9_ROOT}/scripts/build_all_native_ds9.sh"' in compatibility_builder
    assert "c++ " not in compatibility_builder
