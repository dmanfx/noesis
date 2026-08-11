#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kMaxFileTokenLength = 128U;
constexpr std::size_t kWriteChunkBytes = 1U << 30U;

class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, char const* message) noexcept override {
    if (severity <= Severity::kWARNING && message != nullptr) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    fail(
        std::string(operation) + ": " + cudaGetErrorName(status) + " (" +
        cudaGetErrorString(status) + ")");
  }
}

class CudaStream final {
 public:
  CudaStream() {
    check_cuda(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags failed");
  }

  ~CudaStream() {
    if (stream_ != nullptr) {
      const cudaError_t status = cudaStreamDestroy(stream_);
      if (status != cudaSuccess) {
        std::cerr << "CUDA cleanup error: cudaStreamDestroy: "
                  << cudaGetErrorString(status) << '\n';
      }
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class CudaEvent final {
 public:
  CudaEvent() {
    check_cuda(cudaEventCreate(&event_), "cudaEventCreate failed");
  }

  ~CudaEvent() {
    if (event_ != nullptr) {
      const cudaError_t status = cudaEventDestroy(event_);
      if (status != cudaSuccess) {
        std::cerr << "CUDA cleanup error: cudaEventDestroy: "
                  << cudaGetErrorString(status) << '\n';
      }
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class DeviceAllocations final {
 public:
  ~DeviceAllocations() {
    for (auto it = pointers_.rbegin(); it != pointers_.rend(); ++it) {
      if (*it == nullptr) {
        continue;
      }
      const cudaError_t status = cudaFree(*it);
      if (status != cudaSuccess) {
        std::cerr << "CUDA cleanup error: cudaFree: "
                  << cudaGetErrorString(status) << '\n';
      }
    }
  }

  DeviceAllocations(const DeviceAllocations&) = delete;
  DeviceAllocations& operator=(const DeviceAllocations&) = delete;

  DeviceAllocations() = default;

  void* allocate(std::size_t byte_count) {
    if (byte_count == 0U) {
      fail("refusing to allocate an empty CUDA tensor");
    }
    void* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, byte_count), "cudaMalloc failed");
    if (pointer == nullptr) {
      fail("cudaMalloc returned a null pointer");
    }
    pointers_.push_back(pointer);
    return pointer;
  }

 private:
  std::vector<void*> pointers_;
};

struct InputCase {
  std::string name;
  fs::path path;
};

struct Options {
  fs::path engine_path;
  fs::path output_dir;
  std::vector<InputCase> inputs;
};

struct TensorSpec {
  std::string name;
  nvinfer1::TensorIOMode mode = nvinfer1::TensorIOMode::kNONE;
  nvinfer1::TensorLocation location = nvinfer1::TensorLocation::kDEVICE;
  std::vector<std::int64_t> shape;
  std::size_t element_count = 0U;
  std::size_t byte_count = 0U;
};

struct RuntimeTensor {
  TensorSpec spec;
  void* device_storage = nullptr;
  std::vector<float> host_storage;

  void* address() noexcept {
    return spec.location == nvinfer1::TensorLocation::kDEVICE
               ? device_storage
               : static_cast<void*>(host_storage.data());
  }
};

struct OutputStats {
  float minimum = 0.0F;
  float maximum = 0.0F;
};

struct InferenceResult {
  float elapsed_ms = 0.0F;
  std::vector<std::vector<float>> outputs;
};

bool is_ascii_alnum(char value) {
  return (value >= 'a' && value <= 'z') ||
         (value >= 'A' && value <= 'Z') ||
         (value >= '0' && value <= '9');
}

bool is_safe_file_token(const std::string& value) {
  if (value.empty() || value.size() > kMaxFileTokenLength ||
      !is_ascii_alnum(value.front())) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    return is_ascii_alnum(character) || character == '_' || character == '-';
  });
}

std::string usage(const char* program) {
  const std::string executable =
      program != nullptr ? fs::path(program).filename().string()
                         : "rfdetr_1_8_3_media_runner";
  return "Usage: " + executable +
         " --engine PATH --input CASE=PATH [--input CASE=PATH ...] "
         "--output-dir PATH";
}

Options parse_options(int argc, char** argv) {
  if (argc <= 1) {
    fail(usage(argc > 0 ? argv[0] : nullptr));
  }

  Options options;
  bool engine_seen = false;
  bool output_dir_seen = false;
  std::unordered_set<std::string> cases_seen;

  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      std::cout << usage(argv[0]) << '\n';
      std::exit(0);
    }
    if (argument != "--engine" && argument != "--input" &&
        argument != "--output-dir") {
      fail("unknown argument: " + argument + "\n" + usage(argv[0]));
    }
    if (index + 1 >= argc) {
      fail("missing value after " + argument);
    }
    const std::string value(argv[++index]);
    if (value.empty()) {
      fail("empty value after " + argument);
    }

    if (argument == "--engine") {
      if (engine_seen) {
        fail("--engine may be specified only once");
      }
      engine_seen = true;
      options.engine_path = fs::path(value);
      continue;
    }
    if (argument == "--output-dir") {
      if (output_dir_seen) {
        fail("--output-dir may be specified only once");
      }
      output_dir_seen = true;
      options.output_dir = fs::path(value);
      continue;
    }

    const std::size_t separator = value.find('=');
    if (separator == std::string::npos || separator == 0U ||
        separator + 1U >= value.size()) {
      fail("--input must be formatted as CASE=PATH");
    }
    const std::string case_name = value.substr(0U, separator);
    if (!is_safe_file_token(case_name)) {
      fail(
          "unsafe input case token '" + case_name +
          "'; use 1-128 ASCII letters, digits, '_' or '-', beginning with an "
          "ASCII letter or digit");
    }
    if (!cases_seen.insert(case_name).second) {
      fail("duplicate input case: " + case_name);
    }
    options.inputs.push_back(
        InputCase{case_name, fs::path(value.substr(separator + 1U))});
  }

  if (!engine_seen) {
    fail("--engine is required");
  }
  if (!output_dir_seen) {
    fail("--output-dir is required");
  }
  if (options.inputs.empty()) {
    fail("at least one --input CASE=PATH is required");
  }
  return options;
}

std::uintmax_t validated_regular_file_size(
    const fs::path& path, const std::string& description) {
  std::error_code error;
  const fs::file_status status = fs::status(path, error);
  if (error) {
    fail(
        "cannot inspect " + description + " '" + path.string() +
        "': " + error.message());
  }
  if (!fs::is_regular_file(status)) {
    fail(description + " is not a regular file: " + path.string());
  }
  const std::uintmax_t size = fs::file_size(path, error);
  if (error) {
    fail(
        "cannot determine size of " + description + " '" + path.string() +
        "': " + error.message());
  }
  return size;
}

void validate_output_directory(const fs::path& path) {
  std::error_code error;
  const fs::file_status status = fs::status(path, error);
  if (error) {
    fail(
        "cannot inspect output directory '" + path.string() +
        "': " + error.message());
  }
  if (!fs::is_directory(status)) {
    fail("output directory is not a directory: " + path.string());
  }
}

std::vector<std::uint8_t> read_engine_file(const fs::path& path) {
  const std::uintmax_t file_size =
      validated_regular_file_size(path, "TensorRT engine");
  if (file_size == 0U) {
    fail("TensorRT engine is empty: " + path.string());
  }
  if (file_size > std::numeric_limits<std::size_t>::max() ||
      file_size >
          static_cast<std::uintmax_t>(
              std::numeric_limits<std::streamsize>::max())) {
    fail("TensorRT engine is too large to read safely: " + path.string());
  }

  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    fail("cannot open TensorRT engine: " + path.string());
  }
  input.read(
      reinterpret_cast<char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size()) ||
      !input) {
    fail("short read from TensorRT engine: " + path.string());
  }
  if (input.peek() != std::char_traits<char>::eof()) {
    fail("TensorRT engine changed size while it was read: " + path.string());
  }
  return bytes;
}

std::vector<float> read_exact_float_input(
    const fs::path& path, std::size_t element_count, std::size_t byte_count) {
  const std::uintmax_t file_size =
      validated_regular_file_size(path, "media input");
  if (file_size != byte_count) {
    fail(
        "media input byte count mismatch for '" + path.string() +
        "': expected " + std::to_string(byte_count) + ", found " +
        std::to_string(file_size));
  }
  if (byte_count >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    fail("media input is too large to read safely: " + path.string());
  }

  std::vector<float> values(element_count);
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    fail("cannot open media input: " + path.string());
  }
  input.read(
      reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(byte_count));
  if (input.gcount() != static_cast<std::streamsize>(byte_count) || !input) {
    fail("short read from media input: " + path.string());
  }
  if (input.peek() != std::char_traits<char>::eof()) {
    fail("media input changed size while it was read: " + path.string());
  }
  return values;
}

std::vector<std::int64_t> validated_static_shape(
    const nvinfer1::Dims& dimensions, const std::string& tensor_name) {
  if (dimensions.nbDims <= 0) {
    fail("tensor '" + tensor_name + "' has an empty or invalid shape");
  }

  std::vector<std::int64_t> shape;
  shape.reserve(static_cast<std::size_t>(dimensions.nbDims));
  for (int axis = 0; axis < dimensions.nbDims; ++axis) {
    const std::int64_t dimension =
        static_cast<std::int64_t>(dimensions.d[axis]);
    if (dimension <= 0) {
      fail(
          "tensor '" + tensor_name +
          "' has a dynamic or non-positive dimension at axis " +
          std::to_string(axis));
    }
    shape.push_back(dimension);
  }
  return shape;
}

std::pair<std::size_t, std::size_t> validated_tensor_size(
    const std::vector<std::int64_t>& shape, const std::string& tensor_name) {
  std::size_t element_count = 1U;
  for (const std::int64_t dimension : shape) {
    const std::size_t positive_dimension =
        static_cast<std::size_t>(dimension);
    if (positive_dimension >
        std::numeric_limits<std::size_t>::max() / element_count) {
      fail("element count overflow for tensor '" + tensor_name + "'");
    }
    element_count *= positive_dimension;
  }
  if (element_count >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    fail("byte count overflow for tensor '" + tensor_name + "'");
  }
  const std::size_t byte_count = element_count * sizeof(float);
  if (byte_count >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    fail("tensor '" + tensor_name + "' is too large for safe file I/O");
  }
  return std::make_pair(element_count, byte_count);
}

std::string tensor_mode_name(nvinfer1::TensorIOMode mode) {
  if (mode == nvinfer1::TensorIOMode::kINPUT) {
    return "input";
  }
  if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
    return "output";
  }
  fail("TensorRT returned an unsupported tensor I/O mode");
}

std::string tensor_location_name(nvinfer1::TensorLocation location) {
  if (location == nvinfer1::TensorLocation::kDEVICE) {
    return "device";
  }
  if (location == nvinfer1::TensorLocation::kHOST) {
    return "host";
  }
  fail("TensorRT returned an unsupported tensor location");
}

std::vector<TensorSpec> inspect_engine(const nvinfer1::ICudaEngine& engine) {
  const int tensor_count = engine.getNbIOTensors();
  if (tensor_count <= 0) {
    fail("TensorRT engine exposes no I/O tensors");
  }

  std::vector<TensorSpec> tensors;
  tensors.reserve(static_cast<std::size_t>(tensor_count));
  std::unordered_set<std::string> tensor_names;
  std::size_t input_count = 0U;
  std::size_t output_count = 0U;

  for (int index = 0; index < tensor_count; ++index) {
    const char* raw_name = engine.getIOTensorName(index);
    if (raw_name == nullptr || raw_name[0] == '\0') {
      fail("TensorRT engine contains an unnamed I/O tensor");
    }
    const std::string name(raw_name);
    if (!is_safe_file_token(name)) {
      fail(
          "unsafe TensorRT tensor file token '" + name +
          "'; use 1-128 ASCII letters, digits, '_' or '-', beginning with an "
          "ASCII letter or digit");
    }
    if (!tensor_names.insert(name).second) {
      fail("TensorRT engine repeats tensor name '" + name + "'");
    }
    if (engine.getTensorDataType(raw_name) != nvinfer1::DataType::kFLOAT) {
      fail("tensor '" + name + "' is not exact FP32 I/O");
    }

    const nvinfer1::TensorIOMode mode = engine.getTensorIOMode(raw_name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      ++input_count;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      ++output_count;
    } else {
      fail("tensor '" + name + "' has no supported I/O mode");
    }

    const nvinfer1::TensorLocation location =
        engine.getTensorLocation(raw_name);
    if (location != nvinfer1::TensorLocation::kDEVICE &&
        location != nvinfer1::TensorLocation::kHOST) {
      fail("tensor '" + name + "' has no supported TensorRT location");
    }

    const std::vector<std::int64_t> shape =
        validated_static_shape(engine.getTensorShape(raw_name), name);
    const auto [element_count, byte_count] =
        validated_tensor_size(shape, name);
    tensors.push_back(
        TensorSpec{name, mode, location, shape, element_count, byte_count});
  }

  if (input_count != 1U) {
    fail(
        "media runner requires exactly one FP32 input tensor; engine exposes " +
        std::to_string(input_count));
  }
  if (output_count == 0U) {
    fail("media runner requires at least one FP32 output tensor");
  }
  return tensors;
}

std::vector<RuntimeTensor> allocate_tensors(
    const std::vector<TensorSpec>& specs, DeviceAllocations& allocations) {
  std::vector<RuntimeTensor> tensors;
  tensors.reserve(specs.size());
  for (const TensorSpec& spec : specs) {
    RuntimeTensor runtime_tensor;
    runtime_tensor.spec = spec;
    if (spec.location == nvinfer1::TensorLocation::kDEVICE) {
      runtime_tensor.device_storage = allocations.allocate(spec.byte_count);
    } else {
      runtime_tensor.host_storage.resize(spec.element_count);
    }
    tensors.push_back(std::move(runtime_tensor));
  }
  return tensors;
}

void verify_context_shapes(
    const nvinfer1::IExecutionContext& context,
    const std::vector<RuntimeTensor>& tensors) {
  for (const RuntimeTensor& tensor : tensors) {
    const std::vector<std::int64_t> context_shape = validated_static_shape(
        context.getTensorShape(tensor.spec.name.c_str()), tensor.spec.name);
    if (context_shape != tensor.spec.shape) {
      fail(
          "execution-context shape differs from engine shape for tensor '" +
          tensor.spec.name + "'");
    }
  }
}

std::size_t bind_tensors(
    nvinfer1::IExecutionContext& context,
    std::vector<RuntimeTensor>& tensors) {
  std::size_t input_index = tensors.size();
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    RuntimeTensor& tensor = tensors[index];
    if (tensor.spec.mode == nvinfer1::TensorIOMode::kINPUT) {
      input_index = index;
    }
    if (!context.setTensorAddress(
            tensor.spec.name.c_str(), tensor.address())) {
      fail("failed to bind tensor address for '" + tensor.spec.name + "'");
    }
  }
  if (input_index == tensors.size()) {
    fail("internal error: no input tensor was bound");
  }
  return input_index;
}

OutputStats validated_finite_stats(
    const std::vector<float>& values, const std::string& description) {
  if (values.empty()) {
    fail(description + " is empty");
  }
  float minimum = values.front();
  float maximum = values.front();
  if (!std::isfinite(minimum)) {
    fail(description + " contains a non-finite value at element 0");
  }
  for (std::size_t index = 1U; index < values.size(); ++index) {
    const float value = values[index];
    if (!std::isfinite(value)) {
      fail(
          description + " contains a non-finite value at element " +
          std::to_string(index));
    }
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  return OutputStats{minimum, maximum};
}

InferenceResult run_once(
    nvinfer1::IExecutionContext& context,
    std::vector<RuntimeTensor>& tensors,
    std::size_t input_index,
    const std::vector<float>& input_values,
    CudaStream& stream,
    CudaEvent& start_event,
    CudaEvent& end_event) {
  RuntimeTensor& input_tensor = tensors.at(input_index);
  if (input_values.size() != input_tensor.spec.element_count) {
    fail("internal error: input element count changed before inference");
  }

  for (RuntimeTensor& tensor : tensors) {
    if (tensor.spec.mode != nvinfer1::TensorIOMode::kOUTPUT) {
      continue;
    }
    if (tensor.spec.location == nvinfer1::TensorLocation::kDEVICE) {
      check_cuda(
          cudaMemsetAsync(
              tensor.device_storage, 0xFF, tensor.spec.byte_count,
              stream.get()),
          "cudaMemsetAsync output initialization failed");
    } else {
      std::memset(
          tensor.host_storage.data(), 0xFF, tensor.spec.byte_count);
    }
  }

  if (input_tensor.spec.location == nvinfer1::TensorLocation::kDEVICE) {
    check_cuda(
        cudaMemcpyAsync(
            input_tensor.device_storage, input_values.data(),
            input_tensor.spec.byte_count, cudaMemcpyHostToDevice, stream.get()),
        "cudaMemcpyAsync input upload failed");
  } else {
    std::memcpy(
        input_tensor.host_storage.data(), input_values.data(),
        input_tensor.spec.byte_count);
  }

  check_cuda(
      cudaEventRecord(start_event.get(), stream.get()),
      "cudaEventRecord start failed");
  if (!context.enqueueV3(stream.get())) {
    fail("TensorRT enqueueV3 returned false");
  }
  check_cuda(cudaGetLastError(), "CUDA error after TensorRT enqueueV3");
  check_cuda(
      cudaEventRecord(end_event.get(), stream.get()),
      "cudaEventRecord end failed");

  InferenceResult result;
  result.outputs.reserve(tensors.size() - 1U);
  for (RuntimeTensor& tensor : tensors) {
    if (tensor.spec.mode != nvinfer1::TensorIOMode::kOUTPUT) {
      continue;
    }
    result.outputs.emplace_back(tensor.spec.element_count);
    if (tensor.spec.location == nvinfer1::TensorLocation::kDEVICE) {
      check_cuda(
          cudaMemcpyAsync(
              result.outputs.back().data(), tensor.device_storage,
              tensor.spec.byte_count, cudaMemcpyDeviceToHost, stream.get()),
          "cudaMemcpyAsync output download failed");
    }
  }

  check_cuda(
      cudaStreamSynchronize(stream.get()),
      "cudaStreamSynchronize after inference failed");

  std::size_t output_index = 0U;
  for (RuntimeTensor& tensor : tensors) {
    if (tensor.spec.mode != nvinfer1::TensorIOMode::kOUTPUT) {
      continue;
    }
    if (tensor.spec.location == nvinfer1::TensorLocation::kHOST) {
      std::memcpy(
          result.outputs.at(output_index).data(),
          tensor.host_storage.data(), tensor.spec.byte_count);
    }
    ++output_index;
  }

  check_cuda(
      cudaEventElapsedTime(
          &result.elapsed_ms, start_event.get(), end_event.get()),
      "cudaEventElapsedTime failed");
  if (!std::isfinite(result.elapsed_ms) || result.elapsed_ms < 0.0F) {
    fail("CUDA returned a non-finite or negative inference elapsed time");
  }
  return result;
}

std::string json_escape(const std::string& value) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U || character >= 0x7FU) {
          output << "\\u00" << std::hex << std::setw(2)
                 << std::setfill('0') << static_cast<unsigned int>(character)
                 << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

std::string shape_json(const std::vector<std::int64_t>& shape) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << '[';
  for (std::size_t index = 0U; index < shape.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << shape[index];
  }
  output << ']';
  return output.str();
}

std::string float_json(float value) {
  if (!std::isfinite(value)) {
    fail("refusing to serialize a non-finite JSON number");
  }
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<float>::max_digits10)
         << value;
  return output.str();
}

std::string build_manifest(
    const std::string& case_name,
    const std::vector<RuntimeTensor>& tensors,
    std::size_t input_index,
    const OutputStats& input_stats,
    const std::vector<OutputStats>& output_stats,
    float first_elapsed_ms,
    float repeat_elapsed_ms) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "{\n"
         << "  \"schema\":\"noesis.rfdetr_1_8_3.media_runner.v1\",\n"
         << "  \"case\":\"" << json_escape(case_name) << "\",\n"
         << "  \"repeat_stable\":true,\n"
         << "  \"inference_elapsed_ms\":{\"first\":"
         << float_json(first_elapsed_ms)
         << ",\"repeat\":" << float_json(repeat_elapsed_ms) << "},\n"
         << "  \"tensors\":[\n";

  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    const TensorSpec& tensor = tensors[index].spec;
    output << "    {\"name\":\"" << json_escape(tensor.name)
           << "\",\"mode\":\"" << tensor_mode_name(tensor.mode)
           << "\",\"location\":\""
           << tensor_location_name(tensor.location)
           << "\",\"data_type\":\"fp32\",\"shape\":"
           << shape_json(tensor.shape) << ",\"element_count\":"
           << tensor.element_count << ",\"byte_count\":"
           << tensor.byte_count << '}';
    if (index + 1U != tensors.size()) {
      output << ',';
    }
    output << '\n';
  }
  output << "  ],\n";

  const TensorSpec& input = tensors.at(input_index).spec;
  output << "  \"input\":{\"name\":\"" << json_escape(input.name)
         << "\",\"shape\":" << shape_json(input.shape)
         << ",\"byte_count\":" << input.byte_count
         << ",\"finite\":true,\"min\":" << float_json(input_stats.minimum)
         << ",\"max\":" << float_json(input_stats.maximum) << "},\n"
         << "  \"outputs\":[\n";

  std::size_t output_index = 0U;
  for (const RuntimeTensor& runtime_tensor : tensors) {
    const TensorSpec& tensor = runtime_tensor.spec;
    if (tensor.mode != nvinfer1::TensorIOMode::kOUTPUT) {
      continue;
    }
    const OutputStats& stats = output_stats.at(output_index);
    const std::string file_name =
        case_name + "__" + tensor.name + ".bin";
    output << "    {\"name\":\"" << json_escape(tensor.name)
           << "\",\"shape\":" << shape_json(tensor.shape)
           << ",\"byte_count\":" << tensor.byte_count
           << ",\"file\":\"" << json_escape(file_name)
           << "\",\"repeat_stable\":true,\"finite\":true,\"min\":"
           << float_json(stats.minimum) << ",\"max\":"
           << float_json(stats.maximum) << '}';
    ++output_index;
    if (output_index != output_stats.size()) {
      output << ',';
    }
    output << '\n';
  }
  output << "  ]\n"
         << "}\n";
  return output.str();
}

void write_exclusive(
    const fs::path& path, const void* data, std::size_t byte_count) {
  const int descriptor = ::open(
      path.c_str(),
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
      S_IRUSR | S_IWUSR);
  if (descriptor < 0) {
    fail(
        "cannot create output file '" + path.string() +
        "': " + std::strerror(errno));
  }

  bool complete = false;
  try {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t written = 0U;
    while (written < byte_count) {
      const std::size_t remaining = byte_count - written;
      const std::size_t chunk = std::min(remaining, kWriteChunkBytes);
      const ssize_t result =
          ::write(descriptor, bytes + written, chunk);
      if (result < 0) {
        if (errno == EINTR) {
          continue;
        }
        fail(
            "write failed for output file '" + path.string() +
            "': " + std::strerror(errno));
      }
      if (result == 0) {
        fail("zero-byte write for output file: " + path.string());
      }
      written += static_cast<std::size_t>(result);
    }
    if (::fsync(descriptor) != 0) {
      fail(
          "fsync failed for output file '" + path.string() +
          "': " + std::strerror(errno));
    }
    if (::close(descriptor) != 0) {
      fail(
          "close failed for output file '" + path.string() +
          "': " + std::strerror(errno));
    }
    complete = true;
  } catch (...) {
    (void)::close(descriptor);
    (void)::unlink(path.c_str());
    throw;
  }
  if (!complete) {
    fail("internal error while writing output file: " + path.string());
  }
}

void validate_output_destinations(
    const fs::path& output_dir,
    const std::string& case_name,
    const std::vector<RuntimeTensor>& tensors) {
  std::vector<fs::path> paths;
  paths.reserve(tensors.size());
  for (const RuntimeTensor& tensor : tensors) {
    if (tensor.spec.mode == nvinfer1::TensorIOMode::kOUTPUT) {
      paths.push_back(
          output_dir /
          (case_name + "__" + tensor.spec.name + ".bin"));
    }
  }
  paths.push_back(output_dir / (case_name + ".runner.json"));

  for (const fs::path& path : paths) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error &&
        error != std::errc::no_such_file_or_directory) {
      fail(
          "cannot inspect output destination '" + path.string() +
          "': " + error.message());
    }
    if (!error && status.type() != fs::file_type::not_found) {
      fail("refusing to overwrite output destination: " + path.string());
    }
  }
}

void validate_case(
    const InputCase& input_case,
    const fs::path& output_dir,
    nvinfer1::IExecutionContext& context,
    std::vector<RuntimeTensor>& tensors,
    std::size_t input_index,
    CudaStream& stream,
    CudaEvent& start_event,
    CudaEvent& end_event) {
  validate_output_destinations(output_dir, input_case.name, tensors);

  const TensorSpec& input_spec = tensors.at(input_index).spec;
  const std::vector<float> input_values = read_exact_float_input(
      input_case.path, input_spec.element_count, input_spec.byte_count);
  const OutputStats input_stats =
      validated_finite_stats(input_values, "input case '" + input_case.name + "'");

  const InferenceResult first = run_once(
      context, tensors, input_index, input_values, stream, start_event,
      end_event);
  const InferenceResult repeat = run_once(
      context, tensors, input_index, input_values, stream, start_event,
      end_event);
  if (first.outputs.size() != repeat.outputs.size()) {
    fail("internal error: TensorRT output count changed between repeat runs");
  }

  std::vector<OutputStats> output_stats;
  output_stats.reserve(first.outputs.size());
  std::size_t output_index = 0U;
  for (const RuntimeTensor& tensor : tensors) {
    if (tensor.spec.mode != nvinfer1::TensorIOMode::kOUTPUT) {
      continue;
    }
    const std::vector<float>& first_values = first.outputs.at(output_index);
    const std::vector<float>& repeat_values = repeat.outputs.at(output_index);
    const std::string description =
        "output tensor '" + tensor.spec.name + "' for case '" +
        input_case.name + "'";
    output_stats.push_back(
        validated_finite_stats(first_values, description + " first run"));
    (void)validated_finite_stats(
        repeat_values, description + " repeat run");
    if (first_values.size() != repeat_values.size() ||
        std::memcmp(
            first_values.data(), repeat_values.data(),
            tensor.spec.byte_count) != 0) {
      fail(description + " is not bitwise identical across repeat runs");
    }
    ++output_index;
  }

  output_index = 0U;
  for (const RuntimeTensor& tensor : tensors) {
    if (tensor.spec.mode != nvinfer1::TensorIOMode::kOUTPUT) {
      continue;
    }
    const fs::path output_path =
        output_dir /
        (input_case.name + "__" + tensor.spec.name + ".bin");
    write_exclusive(
        output_path, first.outputs.at(output_index).data(),
        tensor.spec.byte_count);
    ++output_index;
  }

  const std::string manifest = build_manifest(
      input_case.name, tensors, input_index, input_stats, output_stats,
      first.elapsed_ms, repeat.elapsed_ms);
  const fs::path manifest_path =
      output_dir / (input_case.name + ".runner.json");
  write_exclusive(manifest_path, manifest.data(), manifest.size());

  std::cout << "validated case " << input_case.name << " ("
            << first.outputs.size() << " output tensors, "
            << first.elapsed_ms << " ms first, " << repeat.elapsed_ms
            << " ms repeat)\n";
}

int run(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  (void)validated_regular_file_size(options.engine_path, "TensorRT engine");
  validate_output_directory(options.output_dir);
  for (const InputCase& input_case : options.inputs) {
    (void)validated_regular_file_size(input_case.path, "media input");
  }

  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
  if (device_count <= 0) {
    fail("no CUDA device is visible");
  }
  int active_device = -1;
  check_cuda(cudaGetDevice(&active_device), "cudaGetDevice failed");
  if (active_device < 0 || active_device >= device_count) {
    fail("CUDA reported an invalid active device");
  }

  TrtLogger logger;
  if (!initLibNvInferPlugins(&logger, "")) {
    fail("initLibNvInferPlugins returned false");
  }

  const std::vector<std::uint8_t> engine_bytes =
      read_engine_file(options.engine_path);
  std::unique_ptr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(logger)};
  if (!runtime) {
    fail("createInferRuntime returned null");
  }
  std::unique_ptr<nvinfer1::ICudaEngine> engine{
      runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size())};
  if (!engine) {
    fail("failed to deserialize TensorRT engine");
  }

  const std::vector<TensorSpec> specs = inspect_engine(*engine);
  std::unique_ptr<nvinfer1::IExecutionContext> context{
      engine->createExecutionContext()};
  if (!context) {
    fail("createExecutionContext returned null");
  }

  DeviceAllocations allocations;
  std::vector<RuntimeTensor> tensors =
      allocate_tensors(specs, allocations);
  verify_context_shapes(*context, tensors);
  const std::size_t input_index = bind_tensors(*context, tensors);

  CudaStream stream;
  CudaEvent start_event;
  CudaEvent end_event;
  for (const InputCase& input_case : options.inputs) {
    validate_case(
        input_case, options.output_dir, *context, tensors, input_index, stream,
        start_event, end_event);
  }
  check_cuda(
      cudaDeviceSynchronize(), "cudaDeviceSynchronize at runner exit failed");
  std::cout << "validated " << options.inputs.size()
            << " media case(s) successfully\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
