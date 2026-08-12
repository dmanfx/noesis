#include <NvInfer.h>
#include <NvInferRuntimeBase.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <array>
#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <vector>

static_assert(NV_TENSORRT_MAJOR == 10, "Wholebody49 builder requires TensorRT 10");
static_assert(NV_TENSORRT_MINOR == 16, "Wholebody49 builder requires TensorRT 10.16");
static_assert(NV_TENSORRT_PATCH == 0, "Wholebody49 builder requires TensorRT 10.16.0");
static_assert(NV_TENSORRT_BUILD == 72, "Wholebody49 builder requires TensorRT 10.16.0.72");

namespace {

constexpr char kContract[] = "noesis.ds9.wholebody49_builder.v1";
constexpr char kInputName[] = "images";
constexpr std::size_t kSMasksWorkspaceBytes = 6144ULL * 1024ULL * 1024ULL;
constexpr std::size_t kXBoxesWorkspaceBytes = 4096ULL * 1024ULL * 1024ULL;
constexpr std::size_t kTacticDramBytes = 2048ULL * 1024ULL * 1024ULL;
constexpr int32_t kSMasksBuilderOptimizationLevel = 0;
constexpr int32_t kXBoxesBuilderOptimizationLevel = 3;
constexpr char kLoggerMinimumSeverity[] = "info";
constexpr char kLoggerVerbosePolicy[] = "ignored_before_copy";
constexpr char kLoggerCapturedTruncationPolicy[] = "fatal";
constexpr char kLoggerErrorStatePolicy[] = "sticky_fatal";
constexpr std::size_t kErrorCapacity = 64U;
constexpr std::size_t kErrorDescriptionBytes = 1024U;
constexpr std::size_t kMaximumOnnxBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumLogMessageBytes = 2048U;
constexpr std::size_t kWriteChunkBytes = 16ULL * 1024ULL * 1024ULL;
constexpr std::size_t kSMasksMaximumEngineBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::size_t kXBoxesMaximumEngineBytes = 768ULL * 1024ULL * 1024ULL;

constexpr bool is_positive_power_of_two(std::size_t value) {
  return value > 0U && (value & (value - 1U)) == 0U;
}

constexpr bool is_positive_mib_aligned(std::size_t value) {
  constexpr std::size_t kMiB = 1024ULL * 1024ULL;
  return value > 0U && value % kMiB == 0U;
}

static_assert(is_positive_mib_aligned(kSMasksWorkspaceBytes),
              "TensorRT S-mask WORKSPACE must be positive and MiB-aligned");
static_assert(is_positive_mib_aligned(kXBoxesWorkspaceBytes),
              "TensorRT X-box WORKSPACE must be positive and MiB-aligned");
static_assert(is_positive_power_of_two(kTacticDramBytes),
              "TensorRT TACTIC_DRAM must be a positive power of two");
static_assert(kSMasksBuilderOptimizationLevel >= 0 &&
                  kSMasksBuilderOptimizationLevel <= 5,
              "TensorRT S-mask builder optimization level is invalid");
static_assert(kXBoxesBuilderOptimizationLevel >= 0 &&
                  kXBoxesBuilderOptimizationLevel <= 5,
              "TensorRT X-box builder optimization level is invalid");

enum class Variant {
  kSMasks,
  kXBoxes,
};

constexpr std::size_t workspace_bytes(Variant variant) {
  return variant == Variant::kSMasks ? kSMasksWorkspaceBytes
                                     : kXBoxesWorkspaceBytes;
}

constexpr int32_t builder_optimization_level(Variant variant) {
  return variant == Variant::kSMasks ? kSMasksBuilderOptimizationLevel
                                     : kXBoxesBuilderOptimizationLevel;
}

struct Options {
  std::filesystem::path onnx;
  std::filesystem::path output;
  Variant variant{Variant::kSMasks};
  bool variant_set{false};
};

struct LoadedOnnx {
  std::vector<std::uint8_t> bytes;
};

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, char const* message) noexcept override {
    if (severity == Severity::kVERBOSE) {
      return;
    }
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      has_error_.store(true, std::memory_order_release);
    }
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      char const* source = message == nullptr ? "<null>" : message;
      std::array<char, kMaximumLogMessageBytes + 1U> bounded{};
      std::size_t length = 0U;
      while (length < kMaximumLogMessageBytes && source[length] != '\0') {
        unsigned char const value = static_cast<unsigned char>(source[length]);
        bounded[length] = value >= 0x20U && value <= 0x7eU
                              ? static_cast<char>(value)
                              : '?';
        ++length;
      }
      if (length == kMaximumLogMessageBytes && source[length] != '\0') {
        truncated_.store(true, std::memory_order_release);
      }
      bounded[length] = '\0';
      std::fprintf(stderr, "[TensorRT][%d] %s\n", static_cast<int>(severity),
                   bounded.data());
    } catch (...) {
      has_error_.store(true, std::memory_order_release);
    }
  }

  bool has_error() const noexcept {
    return has_error_.load(std::memory_order_acquire);
  }

  bool truncated() const noexcept {
    return truncated_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<bool> has_error_{false};
  std::atomic<bool> truncated_{false};
  std::mutex mutex_;
};

class ErrorRecorder final : public nvinfer1::IErrorRecorder {
 public:
  struct Entry {
    nvinfer1::ErrorCode code{nvinfer1::ErrorCode::kUNSPECIFIED_ERROR};
    std::array<char, kErrorDescriptionBytes> description{};
  };

  int32_t getNbErrors() const noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(count_);
  }

  nvinfer1::ErrorCode getErrorCode(int32_t error_index) const noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_index < 0 || static_cast<std::size_t>(error_index) >= count_) {
      return nvinfer1::ErrorCode::kUNSPECIFIED_ERROR;
    }
    return entries_[static_cast<std::size_t>(error_index)].code;
  }

  ErrorDesc getErrorDesc(int32_t error_index) const noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_index < 0 || static_cast<std::size_t>(error_index) >= count_) {
      return "";
    }
    return entries_[static_cast<std::size_t>(error_index)].description.data();
  }

  bool hasOverflowed() const noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    return overflowed_;
  }

  void clear() noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    count_ = 0U;
    overflowed_ = false;
    truncated_ = false;
  }

  bool reportError(nvinfer1::ErrorCode code, ErrorDesc description) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ >= entries_.size()) {
      overflowed_ = true;
      return true;
    }
    Entry& entry = entries_[count_++];
    entry.code = code;
    char const* source = description == nullptr ? "<null>" : description;
    std::size_t length = 0U;
    while (length + 1U < entry.description.size() && source[length] != '\0') {
      unsigned char const value = static_cast<unsigned char>(source[length]);
      entry.description[length] = value >= 0x20U && value <= 0x7eU
                                      ? static_cast<char>(value)
                                      : '?';
      ++length;
    }
    if (source[length] != '\0') {
      truncated_ = true;
    }
    entry.description[length] = '\0';
    return true;
  }

  RefCount incRefCount() noexcept override {
    return references_.fetch_add(1, std::memory_order_acq_rel) + 1;
  }

  RefCount decRefCount() noexcept override {
    RefCount current = references_.load(std::memory_order_acquire);
    while (current > 0) {
      if (references_.compare_exchange_weak(current, current - 1,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return current - 1;
      }
    }
    refcount_underflow_.store(true, std::memory_order_release);
    return 0;
  }

  RefCount ref_count() const noexcept {
    return references_.load(std::memory_order_acquire);
  }

  bool refcount_underflowed() const noexcept {
    return refcount_underflow_.load(std::memory_order_acquire);
  }

  bool truncated() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return truncated_;
  }

  std::string summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string value;
    for (std::size_t index = 0; index < count_; ++index) {
      if (!value.empty()) {
        value += "; ";
      }
      value += "code=" + std::to_string(static_cast<int>(entries_[index].code));
      value += " desc=";
      value += entries_[index].description.data();
    }
    if (overflowed_) {
      value += value.empty() ? "overflowed" : "; overflowed";
    }
    if (truncated_) {
      value += value.empty() ? "description_truncated" : "; description_truncated";
    }
    return value;
  }

 private:
  mutable std::mutex mutex_;
  std::array<Entry, kErrorCapacity> entries_{};
  std::size_t count_{0U};
  bool overflowed_{false};
  bool truncated_{false};
  std::atomic<RefCount> references_{0};
  std::atomic<bool> refcount_underflow_{false};
};

Options parse_options(int argc, char** argv) {
  Options options;
  bool onnx_set = false;
  bool output_set = false;
  for (int index = 1; index < argc; ++index) {
    std::string const argument(argv[index]);
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value after " + argument);
    }
    std::filesystem::path const value(argv[++index]);
    if (argument == "--onnx") {
      if (onnx_set) {
        throw std::runtime_error("--onnx may be specified only once");
      }
      options.onnx = value;
      onnx_set = true;
    } else if (argument == "--output") {
      if (output_set) {
        throw std::runtime_error("--output may be specified only once");
      }
      options.output = value;
      output_set = true;
    } else if (argument == "--variant") {
      if (options.variant_set) {
        throw std::runtime_error("--variant may be specified only once");
      }
      if (value == "s_masks") {
        options.variant = Variant::kSMasks;
      } else if (value == "x_boxes") {
        options.variant = Variant::kXBoxes;
      } else {
        throw std::runtime_error("variant must be s_masks or x_boxes");
      }
      options.variant_set = true;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.onnx.empty() || options.output.empty() || !options.variant_set) {
    throw std::runtime_error("--onnx, --output, and --variant are required");
  }
  if (!options.onnx.is_absolute() || !options.output.is_absolute()) {
    throw std::runtime_error("ONNX and output paths must be absolute");
  }
  return options;
}

void require_absent_output(std::filesystem::path const& output) {
  struct stat info {};
  if (::lstat(output.c_str(), &info) == 0) {
    throw std::runtime_error("refusing to replace an existing candidate: " +
                             output.string());
  }
  if (errno != ENOENT) {
    throw std::system_error(errno, std::generic_category(),
                            "unable to inspect candidate path");
  }
}

bool same_identity(struct stat const& left, struct stat const& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
         left.st_mode == right.st_mode && left.st_nlink == right.st_nlink &&
         left.st_size == right.st_size &&
         left.st_mtim.tv_sec == right.st_mtim.tv_sec &&
         left.st_mtim.tv_nsec == right.st_mtim.tv_nsec &&
         left.st_ctim.tv_sec == right.st_ctim.tv_sec &&
         left.st_ctim.tv_nsec == right.st_ctim.tv_nsec;
}

LoadedOnnx read_verified_onnx(std::filesystem::path const& path) {
  int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  int const descriptor = ::open(path.c_str(), flags);
  if (descriptor < 0) {
    throw std::system_error(errno, std::generic_category(),
                            "unable to open ONNX source");
  }
  struct stat info {};
  if (::fstat(descriptor, &info) != 0) {
    int const error = errno;
    ::close(descriptor);
    throw std::system_error(error, std::generic_category(),
                            "unable to inspect ONNX source");
  }
  if (!S_ISREG(info.st_mode) || info.st_nlink != 1 || info.st_size <= 0) {
    ::close(descriptor);
    throw std::runtime_error(
        "ONNX source must be a nonempty single-link regular file");
  }
  if (static_cast<std::uintmax_t>(info.st_size) > kMaximumOnnxBytes) {
    ::close(descriptor);
    throw std::runtime_error("ONNX source exceeds the reviewed 256 MiB bound");
  }

  LoadedOnnx loaded;
  loaded.bytes.resize(static_cast<std::size_t>(info.st_size));
  std::size_t offset = 0U;
  while (offset < loaded.bytes.size()) {
    ssize_t const count = ::read(descriptor, loaded.bytes.data() + offset,
                                 loaded.bytes.size() - offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      int const error = errno;
      ::close(descriptor);
      throw std::system_error(error, std::generic_category(),
                              "unable to read ONNX source");
    }
    if (count == 0) {
      ::close(descriptor);
      throw std::runtime_error("ONNX source ended before its verified size");
    }
    offset += static_cast<std::size_t>(count);
  }
  struct stat after {};
  if (::fstat(descriptor, &after) != 0 || !same_identity(info, after)) {
    ::close(descriptor);
    throw std::runtime_error("ONNX source changed while its exact bytes were read");
  }
  if (::close(descriptor) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "unable to close ONNX source");
  }
  return loaded;
}

void require_dimensions(nvinfer1::Dims const& observed,
                        std::initializer_list<int32_t> expected,
                        std::string_view label) {
  if (observed.nbDims != static_cast<int32_t>(expected.size())) {
    throw std::runtime_error(std::string(label) + " rank differs from authority");
  }
  std::size_t index = 0U;
  for (int32_t dimension : expected) {
    if (observed.d[index] != dimension) {
      throw std::runtime_error(std::string(label) +
                               " dimensions differ from authority");
    }
    ++index;
  }
}

void validate_network_contract(nvinfer1::INetworkDefinition const& network,
                               Variant variant) {
  if (network.getNbInputs() != 1) {
    throw std::runtime_error("Wholebody49 network must expose exactly one input");
  }
  nvinfer1::ITensor const* input = network.getInput(0);
  if (input == nullptr || input->getName() == nullptr ||
      std::string_view(input->getName()) != kInputName ||
      input->getType() != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error("Wholebody49 input name/type differs from authority");
  }
  require_dimensions(input->getDimensions(), {-1, 3, 640, 640}, "images");

  int32_t const expected_output_count =
      variant == Variant::kSMasks ? 2 : 1;
  int32_t const output_count = network.getNbOutputs();
  if (output_count != expected_output_count) {
    throw std::runtime_error(
        "Wholebody49 network output count differs from the selected variant");
  }
  std::unordered_map<std::string, nvinfer1::ITensor const*> outputs;
  for (int32_t index = 0; index < output_count; ++index) {
    nvinfer1::ITensor const* output = network.getOutput(index);
    if (output == nullptr || output->getName() == nullptr ||
        output->getType() != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("Wholebody49 output name/type differs from authority");
    }
    if (!outputs.emplace(output->getName(), output).second) {
      throw std::runtime_error("Wholebody49 output names must be unique");
    }
  }
  auto const labels = outputs.find("label_xyxy_score");
  if (labels == outputs.end()) {
    throw std::runtime_error("Wholebody49 label_xyxy_score output is missing");
  }
  require_dimensions(labels->second->getDimensions(), {-1, 1240, 6},
                     "label_xyxy_score");
  if (variant == Variant::kXBoxes) {
    if (outputs.size() != 1U) {
      throw std::runtime_error("Wholebody49 box output set differs from authority");
    }
    return;
  }
  auto const masks = outputs.find("masks");
  if (outputs.size() != 2U || masks == outputs.end()) {
    throw std::runtime_error("Wholebody49 mask output set differs from authority");
  }
  require_dimensions(masks->second->getDimensions(), {-1, 1240, 80, 80},
                     "masks");
}

void require_clean(ErrorRecorder const& recorder, Logger const& logger,
                   std::string_view phase) {
  if (recorder.getNbErrors() != 0 || recorder.hasOverflowed() ||
      recorder.truncated() || logger.has_error() || logger.truncated()) {
    throw std::runtime_error(std::string(phase) +
                             " recorded a TensorRT error: " + recorder.summary());
  }
}

void set_exact_memory_limit(nvinfer1::IBuilderConfig& config,
                            nvinfer1::MemoryPoolType pool,
                            std::size_t bytes, ErrorRecorder& recorder,
                            Logger const& logger, std::string_view label) {
  require_clean(recorder, logger, std::string("before ") + std::string(label));
  recorder.clear();
  config.setMemoryPoolLimit(pool, bytes);
  require_clean(recorder, logger, std::string("setting ") + std::string(label));
  if (config.getMemoryPoolLimit(pool) != bytes) {
    throw std::runtime_error(std::string(label) +
                             " memory limit was not accepted exactly");
  }
}

void write_candidate_exclusive(std::filesystem::path const& output,
                               void const* data, std::size_t bytes,
                               std::size_t maximum_bytes) {
  if (data == nullptr || bytes == 0U) {
    throw std::runtime_error("TensorRT returned an empty serialized network");
  }
  if (bytes > maximum_bytes) {
    throw std::runtime_error(
        "serialized network exceeds the selected variant's reviewed size bound");
  }
  int flags = O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC;
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  int descriptor = ::open(output.c_str(), flags, 0600);
  if (descriptor < 0) {
    throw std::system_error(errno, std::generic_category(),
                            "unable to create engine candidate exclusively");
  }
  if (::fchmod(descriptor, 0600) != 0) {
    int const error = errno;
    ::close(descriptor);
    throw std::system_error(error, std::generic_category(),
                            "unable to set private engine-candidate mode");
  }
  try {
    auto const* current = static_cast<std::uint8_t const*>(data);
    std::size_t remaining = bytes;
    while (remaining > 0U) {
      std::size_t const request = std::min(remaining, kWriteChunkBytes);
      ssize_t const written = ::write(descriptor, current, request);
      if (written < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::system_error(errno, std::generic_category(),
                                "unable to write engine candidate");
      }
      if (written == 0) {
        throw std::runtime_error("zero-byte write while serializing engine candidate");
      }
      current += written;
      remaining -= static_cast<std::size_t>(written);
    }
    if (::fsync(descriptor) != 0) {
      throw std::system_error(errno, std::generic_category(),
                              "unable to fsync engine candidate");
    }
    struct stat info {};
    if (::fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) ||
        info.st_uid != ::geteuid() || info.st_nlink != 1 ||
        (info.st_mode & 0777) != 0600 ||
        static_cast<std::size_t>(info.st_size) != bytes) {
      throw std::runtime_error("serialized engine candidate failed file validation");
    }
  } catch (...) {
    ::close(descriptor);
    throw;
  }
  if (::close(descriptor) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "unable to close engine candidate");
  }

  std::filesystem::path const parent = output.parent_path();
  int const directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) {
    throw std::system_error(errno, std::generic_category(),
                            "unable to open candidate directory for fsync");
  }
  int const sync_result = ::fsync(directory);
  int const sync_error = errno;
  ::close(directory);
  if (sync_result != 0) {
    throw std::system_error(sync_error, std::generic_category(),
                            "unable to fsync candidate directory");
  }
}

std::string parser_errors(nvonnxparser::IParser const& parser) {
  std::string result;
  for (int index = 0; index < parser.getNbErrors(); ++index) {
    nvonnxparser::IParserError const* error = parser.getError(index);
    if (!result.empty()) {
      result += "; ";
    }
    result += error == nullptr || error->desc() == nullptr ? "<unknown>" : error->desc();
  }
  return result;
}

std::size_t build(Options const& options, ErrorRecorder& recorder, Logger& logger) {
  require_absent_output(options.output);
  LoadedOnnx const onnx = read_verified_onnx(options.onnx);

  std::size_t engine_bytes = 0U;
  {
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
      throw std::runtime_error("createInferBuilder returned null");
    }
    builder->setErrorRecorder(&recorder);
    if (builder->getErrorRecorder() != &recorder) {
      throw std::runtime_error("builder rejected the error recorder");
    }
    require_clean(recorder, logger, "builder creation");
    if (!builder->platformHasFastFp16()) {
      throw std::runtime_error("selected GPU does not advertise fast native FP16");
    }
    require_clean(recorder, logger, "FP16 capability check");

    // TensorRT 10 networks are always explicit-batch. A zero flag word avoids
    // the deprecated kEXPLICIT_BATCH spelling while retaining that exact mode.
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(0U));
    if (!network) {
      throw std::runtime_error("createNetworkV2 returned null");
    }
    network->setErrorRecorder(&recorder);
    if (network->getErrorRecorder() != &recorder) {
      throw std::runtime_error("network rejected the error recorder");
    }

    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger));
    if (!parser) {
      throw std::runtime_error("createParser returned null");
    }
    bool const parsed = parser->parse(onnx.bytes.data(), onnx.bytes.size(),
                                      options.onnx.c_str());
    if (!parsed || parser->getNbErrors() != 0) {
      throw std::runtime_error("ONNX parser rejected Wholebody49: " +
                               parser_errors(*parser));
    }
    require_clean(recorder, logger, "ONNX parse");
    validate_network_contract(*network, options.variant);

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
      throw std::runtime_error("createBuilderConfig returned null");
    }
    require_clean(recorder, logger, "builder config creation");
    int32_t const selected_builder_optimization_level =
        builder_optimization_level(options.variant);
    recorder.clear();
    config->setBuilderOptimizationLevel(selected_builder_optimization_level);
    require_clean(recorder, logger, "setting builder optimization level");
    if (config->getBuilderOptimizationLevel() !=
        selected_builder_optimization_level) {
      throw std::runtime_error(
          "TensorRT builder optimization level was not accepted exactly");
    }
    recorder.clear();
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    require_clean(recorder, logger, "enabling FP16");
    if (!config->getFlag(nvinfer1::BuilderFlag::kFP16)) {
      throw std::runtime_error("TensorRT did not retain the FP16 builder flag");
    }

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (profile == nullptr) {
      throw std::runtime_error("createOptimizationProfile returned null");
    }
    nvinfer1::Dims4 const shape(3, 3, 640, 640);
    for (nvinfer1::OptProfileSelector selector : {
             nvinfer1::OptProfileSelector::kMIN,
             nvinfer1::OptProfileSelector::kOPT,
             nvinfer1::OptProfileSelector::kMAX,
         }) {
      if (!profile->setDimensions(kInputName, selector, shape)) {
        throw std::runtime_error("TensorRT rejected the Wholebody49 profile dimensions");
      }
    }
    if (!profile->isValid()) {
      throw std::runtime_error("Wholebody49 optimization profile is invalid");
    }
    if (config->addOptimizationProfile(profile) != 0 ||
        config->getNbOptimizationProfiles() != 1) {
      throw std::runtime_error("TensorRT rejected the sole Wholebody49 profile");
    }

    std::size_t const selected_workspace_bytes = workspace_bytes(options.variant);
    set_exact_memory_limit(*config, nvinfer1::MemoryPoolType::kWORKSPACE,
                           selected_workspace_bytes, recorder, logger, "WORKSPACE");
    set_exact_memory_limit(*config, nvinfer1::MemoryPoolType::kTACTIC_DRAM,
                           kTacticDramBytes, recorder, logger, "TACTIC_DRAM");
    if (config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE) !=
            selected_workspace_bytes ||
        config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACTIC_DRAM) !=
            kTacticDramBytes) {
      throw std::runtime_error("TensorRT memory-pool limits drifted before build");
    }
    if (config->getBuilderOptimizationLevel() !=
        selected_builder_optimization_level) {
      throw std::runtime_error(
          "TensorRT builder optimization level drifted before build");
    }

    recorder.clear();
    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
      throw std::runtime_error("buildSerializedNetwork returned null: " +
                               recorder.summary());
    }
    require_clean(recorder, logger, "serialized engine build");
    engine_bytes = serialized->size();
    std::size_t const maximum_engine_bytes =
        options.variant == Variant::kSMasks ? kSMasksMaximumEngineBytes
                                            : kXBoxesMaximumEngineBytes;
    write_candidate_exclusive(options.output, serialized->data(), engine_bytes,
                              maximum_engine_bytes);

    network->setErrorRecorder(nullptr);
    builder->setErrorRecorder(nullptr);
  }
  if (recorder.ref_count() != 0 || recorder.refcount_underflowed()) {
    throw std::runtime_error("TensorRT error-recorder reference contract failed");
  }
  return engine_bytes;
}

}  // namespace

int main(int argc, char** argv) {
  ErrorRecorder recorder;
  Logger logger;
  try {
    Options const options = parse_options(argc, argv);
    std::cout << "[NOESIS_TRT_BUILDER] contract=" << kContract << '\n';
    std::cout << "[NOESIS_TRT_BUILDER] network_mode=explicit_batch_trt10_default\n";
    std::cout << "[NOESIS_TRT_BUILDER] variant="
              << (options.variant == Variant::kSMasks ? "s_masks" : "x_boxes")
              << '\n';
    std::cout << "[NOESIS_TRT_BUILDER] profile=images:3x3x640x640\n";
    std::cout << "[NOESIS_TRT_BUILDER] workspace_bytes="
              << workspace_bytes(options.variant) << '\n';
    std::cout << "[NOESIS_TRT_BUILDER] tactic_dram_bytes=" << kTacticDramBytes << '\n';
    std::cout << "[NOESIS_TRT_BUILDER] builder_optimization_level="
              << builder_optimization_level(options.variant) << '\n';
    std::cout << "[NOESIS_TRT_BUILDER] logger_minimum_severity="
              << kLoggerMinimumSeverity << '\n';
    std::cout << "[NOESIS_TRT_BUILDER] logger_verbose_policy="
              << kLoggerVerbosePolicy << '\n';
    std::cout << "[NOESIS_TRT_BUILDER] logger_captured_truncation="
              << kLoggerCapturedTruncationPolicy << '\n';
    std::cout << "[NOESIS_TRT_BUILDER] logger_error_state="
              << kLoggerErrorStatePolicy << '\n';
    std::size_t const engine_bytes = build(options, recorder, logger);
    std::cout << "[NOESIS_TRT_BUILDER] engine_bytes=" << engine_bytes << '\n';
    std::cout << "[NOESIS_TRT_BUILDER] status=PASS\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "[NOESIS_TRT_BUILDER] status=FAIL reason=" << error.what();
    if (recorder.ref_count() != 0 || recorder.refcount_underflowed()) {
      std::cerr << "; error-recorder lifecycle contract failed"
                << " ref_count=" << recorder.ref_count()
                << " underflow=" << (recorder.refcount_underflowed() ? 1 : 0);
    }
    std::cerr << '\n';
    return 1;
  }
}
