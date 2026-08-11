#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseDeimv2Wholebody49(
    std::vector<NvDsInferLayerInfo> const& output_layers,
    NvDsInferNetworkInfo const& network_info,
    NvDsInferParseDetectionParams const& detection_params,
    std::vector<NvDsInferInstanceMaskInfo>& object_list);

extern "C" bool NvDsInferParseDeimv2Wholebody49Boxes(
    std::vector<NvDsInferLayerInfo> const& output_layers,
    NvDsInferNetworkInfo const& network_info,
    NvDsInferParseDetectionParams const& detection_params,
    std::vector<NvDsInferObjectDetectionInfo>& object_list);

namespace {

constexpr std::size_t kQueries = 1240;
constexpr std::size_t kLabelChannels = 6;
constexpr std::size_t kMaskHeight = 80;
constexpr std::size_t kMaskWidth = 80;

std::size_t g_array_allocations = 0;
std::size_t g_array_frees = 0;

bool close_enough(float actual, float expected) {
  return std::fabs(actual - expected) < 0.001F;
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[HARNESS-FAIL] " << message << std::endl;
    std::exit(1);
  }
}

NvDsInferDims dims(std::initializer_list<unsigned int> values) {
  NvDsInferDims result{};
  result.numDims = static_cast<unsigned int>(values.size());
  result.numElements = 1;
  std::size_t index = 0;
  for (const unsigned int value : values) {
    result.d[index++] = value;
    result.numElements *= value;
  }
  return result;
}

struct Fixture {
  std::vector<float> labels;
  std::vector<float> masks;
  NvDsInferLayerInfo label_layer{};
  NvDsInferLayerInfo mask_layer{};
  NvDsInferNetworkInfo network{640, 640, 3};
  NvDsInferParseDetectionParams detection;

  Fixture()
      : labels(kQueries * kLabelChannels, 0.0F),
        masks(kQueries * kMaskHeight * kMaskWidth, 0.75F) {
    for (std::size_t query = 0; query < kQueries; ++query) {
      labels[query * kLabelChannels] = -1.0F;
    }
    set_query(0, 0.0F, 0.25F, 0.10F, 0.75F, 0.90F, 0.90F);
    set_query(1, 1.0F, 0.10F, 0.10F, 0.20F, 0.20F, 0.95F);
    label_layer = {FLOAT, dims({1240, 6}), 0, "label_xyxy_score",
                   labels.data(), 0};
    mask_layer = {FLOAT, dims({1240, 80, 80}), 1, "masks", masks.data(), 0};
    detection.numClassesConfigured = 49;
    detection.perClassPreclusterThreshold.assign(49, 0.99F);
    detection.perClassPreclusterThreshold[0] = 0.50F;
  }

  void set_query(std::size_t query, float class_id, float x1, float y1,
                 float x2, float y2, float score) {
    const std::size_t base = query * kLabelChannels;
    labels[base] = class_id;
    labels[base + 1] = x1;
    labels[base + 2] = y1;
    labels[base + 3] = x2;
    labels[base + 4] = y2;
    labels[base + 5] = score;
  }
};

void release_masks(std::vector<NvDsInferInstanceMaskInfo>& objects) {
  for (auto& object : objects) {
    delete[] object.mask;
    object.mask = nullptr;
  }
  objects.clear();
}

void test_normalized_body_outputs() {
  Fixture fixture;
  std::vector<NvDsInferObjectDetectionInfo> boxes;
  require(NvDsInferParseDeimv2Wholebody49Boxes(
              {fixture.label_layer}, fixture.network, fixture.detection, boxes),
          "bbox parser rejected the exact contract");
  require(boxes.size() == 1, "bbox parser did not enforce body class only");
  require(boxes[0].classId == 0, "bbox parser emitted a non-body class");
  require(close_enough(boxes[0].left, 160.0F) &&
              close_enough(boxes[0].top, 64.0F) &&
              close_enough(boxes[0].width, 320.0F) &&
              close_enough(boxes[0].height, 512.0F),
          "bbox parser did not apply pinned normalized-xyxy semantics");

  const std::size_t allocations_before = g_array_allocations;
  const std::size_t frees_before = g_array_frees;
  std::vector<NvDsInferInstanceMaskInfo> masks;
  require(NvDsInferParseDeimv2Wholebody49(
              {fixture.mask_layer, fixture.label_layer}, fixture.network,
              fixture.detection, masks),
          "mask parser rejected exact named layers in reversed order");
  require(masks.size() == 1, "mask parser did not enforce body class only");
  require(masks[0].mask != nullptr && masks[0].mask_width == 40 &&
              masks[0].mask_height == 64 &&
              masks[0].mask_size == 40U * 64U * sizeof(float),
          "mask parser emitted the wrong normalized ROI");
  require(close_enough(masks[0].mask[0], 0.75F),
          "mask parser emitted the wrong ROI value");
  release_masks(masks);
  require(g_array_allocations - allocations_before == 1 &&
              g_array_frees - frees_before == 1,
          "successful mask ownership was not balanced");
}

void test_exact_layer_names_and_sets() {
  Fixture fixture;
  std::vector<NvDsInferObjectDetectionInfo> boxes;
  std::vector<NvDsInferInstanceMaskInfo> masks;

  auto wrong_label = fixture.label_layer;
  wrong_label.layerName = "renamed_label";
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {wrong_label}, fixture.network, fixture.detection, boxes),
          "bbox parser accepted positional label fallback");

  auto null_name = fixture.label_layer;
  null_name.layerName = nullptr;
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {null_name}, fixture.network, fixture.detection, boxes),
          "bbox parser accepted a null layer name");
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {fixture.label_layer, fixture.mask_layer}, fixture.network,
              fixture.detection, boxes),
          "bbox parser accepted an extra output layer");
  require(!NvDsInferParseDeimv2Wholebody49(
              {fixture.label_layer, fixture.label_layer}, fixture.network,
              fixture.detection, masks),
          "mask parser accepted duplicate label layers");
  require(!NvDsInferParseDeimv2Wholebody49(
              {fixture.mask_layer, fixture.mask_layer}, fixture.network,
              fixture.detection, masks),
          "mask parser accepted duplicate mask layers");
  auto unknown = fixture.mask_layer;
  unknown.layerName = "unknown";
  require(!NvDsInferParseDeimv2Wholebody49(
              {fixture.label_layer, unknown}, fixture.network,
              fixture.detection, masks),
          "mask parser accepted an unknown output layer");
  require(!NvDsInferParseDeimv2Wholebody49(
              {fixture.label_layer, fixture.mask_layer, unknown},
              fixture.network, fixture.detection, masks),
          "mask parser accepted an extra output layer");
}

void test_exact_shapes_types_and_buffers() {
  Fixture fixture;
  std::vector<NvDsInferObjectDetectionInfo> boxes;
  std::vector<NvDsInferInstanceMaskInfo> masks;

  for (const NvDsInferDims invalid : {
           dims({1241, 6}), dims({1240, 7}), dims({3, 1240, 6})}) {
    auto layer = fixture.label_layer;
    layer.inferDims = invalid;
    require(!NvDsInferParseDeimv2Wholebody49Boxes(
                {layer}, fixture.network, fixture.detection, boxes),
            "bbox parser accepted a non-promoted label shape");
  }
  auto wrong_elements = fixture.label_layer;
  --wrong_elements.inferDims.numElements;
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {wrong_elements}, fixture.network, fixture.detection, boxes),
          "bbox parser accepted inconsistent label elements");

  for (const NvDsInferDims invalid : {dims({1241, 80, 80}),
                                      dims({1240, 79, 80}),
                                      dims({3, 1240, 80, 80})}) {
    auto layer = fixture.mask_layer;
    layer.inferDims = invalid;
    require(!NvDsInferParseDeimv2Wholebody49(
                {fixture.label_layer, layer}, fixture.network,
                fixture.detection, masks),
            "mask parser accepted a non-promoted mask shape");
  }
  wrong_elements = fixture.mask_layer;
  --wrong_elements.inferDims.numElements;
  require(!NvDsInferParseDeimv2Wholebody49(
              {fixture.label_layer, wrong_elements}, fixture.network,
              fixture.detection, masks),
          "mask parser accepted inconsistent mask elements");

  auto invalid_label = fixture.label_layer;
  invalid_label.dataType = HALF;
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {invalid_label}, fixture.network, fixture.detection, boxes),
          "bbox parser accepted a non-FLOAT label tensor");
  invalid_label = fixture.label_layer;
  invalid_label.buffer = nullptr;
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {invalid_label}, fixture.network, fixture.detection, boxes),
          "bbox parser accepted a null label buffer");

  auto invalid_mask = fixture.mask_layer;
  invalid_mask.dataType = HALF;
  require(!NvDsInferParseDeimv2Wholebody49(
              {fixture.label_layer, invalid_mask}, fixture.network,
              fixture.detection, masks),
          "mask parser accepted a non-FLOAT mask tensor");
  invalid_mask = fixture.mask_layer;
  invalid_mask.buffer = nullptr;
  require(!NvDsInferParseDeimv2Wholebody49(
              {fixture.label_layer, invalid_mask}, fixture.network,
              fixture.detection, masks),
          "mask parser accepted a null mask buffer");

  auto invalid_network = fixture.network;
  invalid_network.width = 0;
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {fixture.label_layer}, invalid_network, fixture.detection, boxes),
          "bbox parser accepted an invalid network shape");
}

void test_nonfinite_inputs_and_mask_cleanup() {
  Fixture fixture;
  std::vector<NvDsInferObjectDetectionInfo> boxes;
  const float original_class = fixture.labels[0];
  const float original_x1 = fixture.labels[1];
  const float original_score = fixture.labels[5];

  fixture.labels[0] = std::numeric_limits<float>::quiet_NaN();
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {fixture.label_layer}, fixture.network, fixture.detection, boxes),
          "bbox parser accepted a non-finite class ID");
  fixture.labels[0] = original_class;
  fixture.labels[1] = std::numeric_limits<float>::infinity();
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {fixture.label_layer}, fixture.network, fixture.detection, boxes),
          "bbox parser accepted a non-finite coordinate");
  fixture.labels[1] = original_x1;
  fixture.labels[5] = std::numeric_limits<float>::quiet_NaN();
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {fixture.label_layer}, fixture.network, fixture.detection, boxes),
          "bbox parser accepted a non-finite score");
  fixture.labels[5] = original_score;
  fixture.labels[0] = 0.5F;
  require(!NvDsInferParseDeimv2Wholebody49Boxes(
              {fixture.label_layer}, fixture.network, fixture.detection, boxes),
          "bbox parser accepted a non-integral class ID");
  fixture.labels[0] = original_class;

  const std::size_t first_roi_pixel = 8 * kMaskWidth + 20;
  fixture.masks[first_roi_pixel] = std::numeric_limits<float>::quiet_NaN();
  std::vector<NvDsInferInstanceMaskInfo> mask_objects;
  const std::size_t allocations_before = g_array_allocations;
  const std::size_t frees_before = g_array_frees;
  require(!NvDsInferParseDeimv2Wholebody49(
              {fixture.label_layer, fixture.mask_layer}, fixture.network,
              fixture.detection, mask_objects),
          "mask parser accepted a non-finite first ROI");
  require(mask_objects.empty() && g_array_allocations == allocations_before &&
              g_array_frees == frees_before,
          "mask parser allocated before validating the first ROI");

  fixture.masks[first_roi_pixel] = 0.75F;
  fixture.set_query(2, 0.0F, 0.10F, 0.10F, 0.20F, 0.20F, 0.80F);
  const std::size_t second_roi_pixel =
      2 * kMaskHeight * kMaskWidth + 8 * kMaskWidth + 8;
  fixture.masks[second_roi_pixel] =
      std::numeric_limits<float>::quiet_NaN();
  require(!NvDsInferParseDeimv2Wholebody49(
              {fixture.label_layer, fixture.mask_layer}, fixture.network,
              fixture.detection, mask_objects),
          "mask parser accepted a later non-finite ROI");
  require(mask_objects.empty() &&
              g_array_allocations - allocations_before == 1 &&
              g_array_frees - frees_before == 1,
          "mask parser leaked an earlier ROI after a later failure");
}

}  // namespace

void* operator new[](std::size_t size) {
  if (void* allocation = std::malloc(size)) {
    ++g_array_allocations;
    return allocation;
  }
  throw std::bad_alloc();
}

void operator delete[](void* allocation) noexcept {
  if (allocation != nullptr) {
    ++g_array_frees;
    std::free(allocation);
  }
}

void operator delete[](void* allocation, std::size_t) noexcept {
  operator delete[](allocation);
}

int main() {
  test_normalized_body_outputs();
  test_exact_layer_names_and_sets();
  test_exact_shapes_types_and_buffers();
  test_nonfinite_inputs_and_mask_cleanup();
  require(g_array_allocations == g_array_frees,
          "array allocations remained live after the harness");
  std::cout << "[OK] strict Wholebody49 parser contract" << std::endl;
  return 0;
}
