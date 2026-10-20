// Pure CPU fixture. Root owns compilation/execution; no SDK or inference needed.
#define NOESIS_POSE_CROP_GEOMETRY_TEST 1
#include "../../native/noesis_pose_meta_ext.cpp"

#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using Box = std::array<double, 4>;
using noesis::pose::make_crop_geometry;
using noesis::pose::map_keypoint;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, const char* message) {
  require(std::isfinite(actual) && std::abs(actual - expected) < 1e-9, message);
}
double clamp(double value, double maximum) {
  return std::fmin(std::fmax(value, 0.0), maximum);
}

// Independent transcription of installed gstnvinfer.cpp1369..1508. Deliberately
// use floor/parity rather than the production helper's integer bit operations.
struct SdkRecipe {
  double left, top, width, height, destination_width, destination_height;
  double pad_x, pad_y;
};
SdkRecipe sdk_recipe(Box box, int model_width, int model_height, bool letterbox) {
  const double left = std::floor(box[0]) + std::fmod(std::floor(box[0]), 2.0);
  const double top = std::floor(box[1]) + std::fmod(std::floor(box[1]), 2.0);
  const double width = std::floor(box[2]) - std::fmod(std::floor(box[2]), 2.0);
  const double height = std::floor(box[3]) - std::fmod(std::floor(box[3]), 2.0);
  double dw = model_width, dh = model_height;
  if (letterbox) {
    if (model_width * height / width <= model_height) {
      dh = std::floor(model_width * height / width);
    } else {
      dw = std::floor(model_height * width / height);
    }
  }
  return {left, top, width, height, dw, dh,
          letterbox ? std::floor((model_width - dw) / 2.0) : 0.0,
          letterbox ? std::floor((model_height - dh) / 2.0) : 0.0};
}

void check_box(Box box, int model_width = 640, int model_height = 640,
               bool letterbox = true) {
  auto geometry = make_crop_geometry(box[0], box[1], box[2], box[3],
                                     model_width, model_height, letterbox);
  require(geometry.has_value(), "valid geometry refused");
  const auto sdk = sdk_recipe(box, model_width, model_height, letterbox);
  near(geometry->source_left, sdk.left, "source left");
  near(geometry->source_top, sdk.top, "source top");
  near(geometry->source_width, sdk.width, "source width");
  near(geometry->source_height, sdk.height, "source height");
  near(geometry->destination_width, sdk.destination_width, "destination width");
  near(geometry->destination_height, sdk.destination_height, "destination height");
  near(geometry->offset_left, sdk.pad_x, "left padding");
  near(geometry->offset_top, sdk.pad_y, "top padding");
  // Include model padding and outside-model points: output clipping is to the
  // ORIGINAL fractional box, not to the even crop or a guessed source image.
  for (double fx : {-0.5, 0.0, 0.1, 0.5, 0.95, 1.0, 1.5}) {
    for (double fy : {-0.5, 0.0, 0.1, 0.5, 0.95, 1.0, 1.5}) {
      const double source_x = sdk.left + fx * sdk.width;
      const double source_y = sdk.top + fy * sdk.height;
      const double model_x = sdk.pad_x + fx * sdk.destination_width;
      const double model_y = sdk.pad_y + fy * sdk.destination_height;
      constexpr float confidence = 0.8125f;
      auto pixel = map_keypoint(*geometry, model_x, model_y, confidence, false);
      auto normalized = map_keypoint(*geometry, model_x / model_width,
                                     model_y / model_height, confidence, true);
      require(pixel.has_value() && normalized.has_value(), "finite keypoint refused");
      near(pixel->roi_x, clamp(source_x - box[0], box[2]), "original ROI x");
      near(pixel->roi_y, clamp(source_y - box[1], box[3]), "original ROI y");
      near(pixel->absolute_x, box[0] + pixel->roi_x, "absolute x relation");
      near(pixel->absolute_y, box[1] + pixel->roi_y, "absolute y relation");
      near(normalized->absolute_x, pixel->absolute_x, "normalized x");
      near(normalized->absolute_y, pixel->absolute_y, "normalized y");
      require(pixel->confidence == confidence && normalized->confidence == confidence,
              "confidence changed");
    }
  }
}

void causal_k313() {
  const Box box{799.5, 422.5, 31.5, 92.0};
  const auto sdk = sdk_recipe(box, 640, 640, true);
  const double mx = sdk.pad_x + sdk.destination_width / 2.0;
  const double my = sdk.pad_y + sdk.destination_height / 2.0;
  const double wanted_x = sdk.left + sdk.width / 2.0;
  const double wanted_y = sdk.top + sdk.height / 2.0;
  // Old production formula must fail this independent SDK-forward witness.
  const double gain = std::fmin(640.0 / box[2], 640.0 / box[3]);
  const double old_x = box[0] + (mx - (640.0 - box[2] * gain) / 2.0) / gain;
  const double old_y = box[1] + (my - (640.0 - box[3] * gain) / 2.0) / gain;
  near(old_x - wanted_x, 0.25, "old causal x error");
  near(old_y - wanted_y, 0.5, "old causal y error");
  require(std::hypot(old_x - wanted_x, old_y - wanted_y) > 0.55,
          "old geometry unexpectedly agrees");
  auto geometry = make_crop_geometry(box[0], box[1], box[2], box[3], 640, 640, true);
  auto corrected = map_keypoint(*geometry, mx, my, 0.75f, false);
  require(corrected.has_value(), "causal keypoint refused");
  near(corrected->absolute_x, wanted_x, "corrected causal x");
  near(corrected->absolute_y, wanted_y, "corrected causal y");
}

void malformed_and_payload_contract() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const double umax = std::numeric_limits<std::uint32_t>::max();
  for (Box box : {Box{nan, 0, 32, 92}, Box{0, inf, 32, 92},
                  Box{-1, 0, 32, 92}, Box{0, -1, 32, 92},
                  Box{0, 0, 0, 92}, Box{0, 0, 32, 1.99},
                  Box{0, 0, -2, 92}, Box{0, 0, inf, 92},
                  Box{0, 0, 32, nan}, Box{umax, 0, 32, 92},
                  Box{0, 0, umax + 1.0, 92}, Box{umax - 1, 0, 32, 92},
                  Box{0, 0, 2, 10000000}}) {
    require(!make_crop_geometry(box[0], box[1], box[2], box[3], 640, 640, true),
            "malformed/overflow/degenerate box accepted");
  }
  require(!make_crop_geometry(0, 0, 32, 92, 0, 640, true), "zero model accepted");
  require(!make_crop_geometry(0, 0, 32, 92, 640, -1, false), "negative model accepted");
  require(!make_crop_geometry(0, 0, 32, 92, std::numeric_limits<int>::max(),
                              640, true), "SDK guint product overflow accepted");
  // No overflowing product is needed: integer aspect resize itself collapses
  // this valid positive source rectangle to a zero-width destination.
  require(!make_crop_geometry(0, 0, 2, 2000000, 640, 640, true),
          "zero integer destination accepted");
  auto geometry = make_crop_geometry(799.5, 422.5, 31.5, 92, 640, 640, true);
  require(!map_keypoint(*geometry, nan, 20, 0.75f, false), "NaN joint accepted");
  require(!map_keypoint(*geometry, 20, inf, 0.75f, false), "infinite joint accepted");
  require(!map_keypoint(*geometry, 20, 20,
                        std::numeric_limits<float>::quiet_NaN(), false),
          "nonfinite confidence accepted");
  require(!map_keypoint(*geometry, std::numeric_limits<double>::max(), 20,
                        0.75f, true), "normalized overflow accepted");
  // Actual output row has 57 columns: score at4, an unrelated column5, and
  // final51 joint values. The production row-selection/indexing is unchanged.
  std::array<float, 57> row{};
  row[4] = 0.875f;
  row[5] = -1234.0f;
  const std::size_t key_start = row.size() - 17U * 3U;
  require(key_start == 6, "57-column tail index");
  for (std::size_t joint = 0; joint < 17; ++joint) {
    row[key_start + joint * 3] = 320;
    row[key_start + joint * 3 + 1] = 320;
    row[key_start + joint * 3 + 2] = 0.625f;
    auto point = map_keypoint(*geometry, row[key_start + joint * 3],
        row[key_start + joint * 3 + 1], row[key_start + joint * 3 + 2], false);
    require(point && point->confidence == 0.625f && row[4] == 0.875f,
            "row score or joint confidence changed");
  }
}
void context_cutoff_contract() {
  for (double short_side : {30.0, 64.0, 90.0, 127.99}) {
    for (bool transpose : {false, true}) {
      const double w=transpose?260.0:short_side, h=transpose?short_side:260.0;
      auto crop=noesis::pose::context_crop(673,421,w,h,1920,1080);
      if(transpose) {
        require(!crop,"broad/landscape context admitted");
        continue;
      }
      require(crop.has_value(), "sub128 short-side context refused");
      near(crop->original_left,673,"original left changed");
      near(crop->original_top,421,"original top changed");
      near(crop->original_width,w,"original width changed");
      near(crop->original_height,h,"original height changed");
      const auto expected=make_crop_geometry(673-.30*w,421-.30*h,1.60*w,1.60*h,640,640,true);
      require(expected.has_value(),"context expected geometry missing");
      near(crop->source_left,expected->source_left,".30 margin changed");
      near(crop->source_width,expected->source_width,"context SDK alignment changed");
    }
  }
  for(double side:{128.0,129.0,160.0})
    require(!noesis::pose::context_crop(673,421,side,260,1920,1080),"128+ short-side context admitted");
  require(!noesis::pose::context_crop(673,421,15.99,160,1920,1080),"minimum16 changed");
  require(noesis::pose::context_crop(100,200,46.5,98.5,1920,1080).has_value(),"R46 K287 narrow original refused");
  require(!noesis::pose::context_crop(100,200,94.5,125,1920,1080),"R46 F44 broad original expanded");
  require(!noesis::pose::context_crop(100,200,103.5,123,1920,1080),"R46 F50 broad original expanded");
  require(noesis::pose::context_crop(100,200,65,100,1920,1080).has_value(),"exact .65 boundary refused");
  require(!noesis::pose::context_crop(100,200,65.01,100,1920,1080),"above .65 expanded");
  require(noesis::pose::context_crop(100,200,64.99,100,1920,1080).has_value(),"below .65 refused");
}
}  // namespace

int main() {
  causal_k313();
  const std::array<Box, 16> boxes{{
      {688.5, 417.5, 28, 75}, {673, 421, 30.5, 72},
      {655, 427.5000305175781, 31.5, 92.5}, {652, 435, 42.5, 92},
      {764.5, 428.5, 43, 92.5}, {774, 426.5, 32, 91.5},
      {791.5, 423, 33.5, 91}, {799.5, 422.5, 31.5, 92},
      {810, 418.5, 34, 87.5}, {822, 415.5, 28.5, 87},
      {827, 414, 27.5, 84}, {100.75, 200.25, 17.9, 93.9},
      {100, 200, 32, 92}, {100.25, 200.75, 93.9, 17.9},
      {101.9, 201.9, 31.9, 301.9}, {0, 0, 640, 640}}};
  for (const auto& box : boxes) {
    check_box(box);
    check_box(box, 640, 640, false);
  }
  check_box({100.75, 200.25, 17.9, 93.9}, 641, 639, true);
  check_box({100.25, 200.75, 93.9, 17.9}, 641, 639, true);
  malformed_and_payload_contract();
  context_cutoff_contract();
  std::cout << "pose crop geometry: causal old RED witness and SDK roundtrips PASS\n";
}
