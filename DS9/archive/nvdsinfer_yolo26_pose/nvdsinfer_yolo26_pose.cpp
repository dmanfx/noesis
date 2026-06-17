// SPDX-FileCopyrightText: 2026 Noesis
// SPDX-License-Identifier: MIT
//
// Minimal DeepStream 8 parser for YOLO26 Pose.
// We rely on output-tensor-meta in Python, so this parser simply returns
// an empty detection list to satisfy nvinfer's bbox parsing requirements.

#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolo26Pose(
    std::vector<NvDsInferLayerInfo> const& /*outputLayersInfo*/,
    NvDsInferNetworkInfo const& /*networkInfo*/,
    NvDsInferParseDetectionParams const& /*detectionParams*/,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  objectList.clear();
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26Pose);
