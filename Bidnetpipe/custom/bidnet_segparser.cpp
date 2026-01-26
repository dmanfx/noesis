/*
 * Bidnetpipe custom semantic segmentation parser for BiSeNetV2 ADE20K.
 *
 * Converts the segmentation output into a binary class_map:
 *   floor_class_id -> floor_class_id
 *   everything else -> background_class_id
 *
 * The floor/background class IDs are provided via environment variables:
 *   BIDNET_FLOOR_CLASS_ID (required)
 *   BIDNET_BG_CLASS_ID (optional, default 0)
 */

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_fp16.h>

#include "nvdsinfer_custom_impl.h"

namespace {

bool get_class_ids(int &floor_id, int &bg_id, unsigned int num_classes) {
    const char *floor_env = std::getenv("BIDNET_FLOOR_CLASS_ID");
    if (!floor_env) {
        std::cerr << "BIDNET: BIDNET_FLOOR_CLASS_ID is not set." << std::endl;
        return false;
    }
    floor_id = std::atoi(floor_env);
    if (floor_id < 0 || static_cast<unsigned int>(floor_id) >= num_classes) {
        std::cerr << "BIDNET: floor class id out of range: " << floor_id << std::endl;
        return false;
    }

    const char *bg_env = std::getenv("BIDNET_BG_CLASS_ID");
    bg_id = bg_env ? std::atoi(bg_env) : 0;
    if (bg_id < 0 || static_cast<unsigned int>(bg_id) >= num_classes) {
        std::cerr << "BIDNET: background class id out of range: " << bg_id << std::endl;
        return false;
    }
    return true;
}

const NvDsInferLayerInfo *select_output_layer(
    const std::vector<NvDsInferLayerInfo> &outputLayersInfo) {
    if (outputLayersInfo.empty()) {
        return nullptr;
    }
    if (outputLayersInfo.size() == 1) {
        return &outputLayersInfo[0];
    }
    for (const auto &layer : outputLayersInfo) {
        if (layer.layerName && std::strcmp(layer.layerName, "segmentation") == 0) {
            return &layer;
        }
    }
    return nullptr;
}

bool resolve_output_layout(const NvDsInferDims &dims, unsigned int num_classes,
                           int &height, int &width, bool &is_nchw) {
    if (dims.numDims == 4U) {
        if (dims.d[1] == static_cast<int>(num_classes)) {
            is_nchw = true;
            height = dims.d[2];
            width = dims.d[3];
            return true;
        }
        if (dims.d[3] == static_cast<int>(num_classes)) {
            is_nchw = false;
            height = dims.d[1];
            width = dims.d[2];
            return true;
        }
    } else if (dims.numDims == 3U) {
        if (dims.d[0] == static_cast<int>(num_classes)) {
            is_nchw = true;
            height = dims.d[1];
            width = dims.d[2];
            return true;
        }
        if (dims.d[2] == static_cast<int>(num_classes)) {
            is_nchw = false;
            height = dims.d[0];
            width = dims.d[1];
            return true;
        }
    }
    return false;
}

}  // namespace

extern "C"
bool NvDsInferParseCustomBiSeNetFloor(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float segmentationThreshold,
    unsigned int numClasses, int *classificationMap,
    float *&classProbabilityMap) {
    if (!classificationMap) {
        std::cerr << "BIDNET: classificationMap is null." << std::endl;
        return false;
    }

    const NvDsInferLayerInfo *layer = select_output_layer(outputLayersInfo);
    if (!layer) {
        std::cerr << "BIDNET: output layer 'segmentation' not found." << std::endl;
        return false;
    }
    if (!layer->buffer) {
        std::cerr << "BIDNET: output layer buffer is null." << std::endl;
        return false;
    }

    int floor_id = 0;
    int bg_id = 0;
    if (!get_class_ids(floor_id, bg_id, numClasses)) {
        return false;
    }

    int height = 0;
    int width = 0;
    bool is_nchw = true;
    if (!resolve_output_layout(layer->inferDims, numClasses, height, width, is_nchw)) {
        std::cerr << "BIDNET: unable to resolve output layout from dims." << std::endl;
        return false;
    }

    if (height != static_cast<int>(networkInfo.height) ||
        width != static_cast<int>(networkInfo.width)) {
        std::cerr << "BIDNET: output dims mismatch. got " << height << "x" << width
                  << " expected " << networkInfo.height << "x" << networkInfo.width
                  << std::endl;
    }

    auto read_prob = [&](size_t idx) -> float {
        if (layer->dataType == FLOAT) {
            return static_cast<const float *>(layer->buffer)[idx];
        }
        if (layer->dataType == HALF) {
            const __half *data = static_cast<const __half *>(layer->buffer);
            return __half2float(data[idx]);
        }
        return 0.0f;
    };

    const size_t spatial = static_cast<size_t>(height) * static_cast<size_t>(width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float max_prob = -1.0f;
            unsigned int max_cls = 0;
            for (unsigned int c = 0; c < numClasses; ++c) {
                size_t idx = 0;
                if (is_nchw) {
                    idx = static_cast<size_t>(c) * spatial +
                          static_cast<size_t>(y) * static_cast<size_t>(width) +
                          static_cast<size_t>(x);
                } else {
                    idx = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                           static_cast<size_t>(x)) *
                              static_cast<size_t>(numClasses) +
                          static_cast<size_t>(c);
                }
                float prob = read_prob(idx);
                if (prob > max_prob) {
                    max_prob = prob;
                    max_cls = c;
                }
            }

            int out_id = bg_id;
            if (max_prob >= segmentationThreshold && static_cast<int>(max_cls) == floor_id) {
                out_id = floor_id;
            }
            classificationMap[y * width + x] = out_id;
        }
    }

    classProbabilityMap = nullptr;
    return true;
}

CHECK_CUSTOM_SEM_SEGMENTATION_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBiSeNetFloor);
