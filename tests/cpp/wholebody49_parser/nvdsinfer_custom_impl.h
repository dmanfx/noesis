#pragma once

#include <vector>

constexpr unsigned int NVDSINFER_MAX_DIMS = 8;

typedef struct {
  unsigned int numDims;
  unsigned int d[NVDSINFER_MAX_DIMS];
  unsigned int numElements;
} NvDsInferDims;

typedef enum {
  FLOAT = 0,
  HALF = 1,
  INT8 = 2,
  INT32 = 3,
  INT64 = 4,
  UINT8 = 5,
} NvDsInferDataType;

typedef struct {
  NvDsInferDataType dataType;
  NvDsInferDims inferDims;
  int bindingIndex;
  const char* layerName;
  void* buffer;
  int isInput;
} NvDsInferLayerInfo;

typedef struct {
  unsigned int width;
  unsigned int height;
  unsigned int channels;
} NvDsInferNetworkInfo;

struct NvDsInferParseDetectionParams {
  unsigned int numClassesConfigured = 0;
  std::vector<float> perClassPreclusterThreshold;
};

typedef struct {
  unsigned int classId;
  float left;
  float top;
  float width;
  float height;
  float detectionConfidence;
#ifdef NOESIS_DEEPSTREAM_9_ABI
  // DeepStream 9 appends OBB rotation to NvDsInferObjectDetectionInfo.
  float rotation_angle;
#endif
} NvDsInferObjectDetectionInfo;

typedef struct {
  unsigned int classId;
  float left;
  float top;
  float width;
  float height;
  float detectionConfidence;
  float* mask;
  unsigned int mask_width;
  unsigned int mask_height;
  unsigned int mask_size;
} NvDsInferInstanceMaskInfo;

#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(custom_parse_func)
#define CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(custom_parse_func)
