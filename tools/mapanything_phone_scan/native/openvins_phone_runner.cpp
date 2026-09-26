#include "core/VioManager.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/sensor_data.h"
#include "openvins_pose_math.h"

#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct ImageRecord {
  std::int64_t timestamp_ns;
  fs::path path;
};

struct ImuRecord {
  std::int64_t timestamp_ns;
  Eigen::Vector3d gyro;
  Eigen::Vector3d accel;
};

struct Cli {
  fs::path capture_dir;
  fs::path config;
  fs::path output;
  fs::path camera_mask;
  fs::path frame_mapping;
  std::string short_session_consumer;
  std::string profile_id;
  std::string capture_id = "euroc_mh_01_easy_mono_subset";
  std::string camera_sensor_id = "cam0";
  std::string time_domain = "euroc_ns";
  std::string mode = "metric_vio";
  double rotation_std_rad = std::numeric_limits<double>::quiet_NaN();
  double translation_std_m = std::numeric_limits<double>::quiet_NaN();
  double time_offset_std_s = std::numeric_limits<double>::quiet_NaN();
};

struct Calibration {
  Eigen::Matrix4d T_imu_camera;
  double camera_to_imu_offset_s;
  std::vector<std::vector<double>> covariance;
};

struct Snapshot {
  std::int64_t timestamp_ns;
  std::int64_t capture_time_ns;
  std::int64_t source_frame_index;
  std::string prepared_frame_id;
  Eigen::Matrix4d T_world_camera;
  Eigen::Vector3d velocity_imu;
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d accel_bias;
  std::vector<std::vector<double>> covariance;
  Calibration calibration;
};

struct FrameIdentity {
  std::int64_t source_frame_index;
  std::string prepared_frame_id;
  std::int64_t capture_time_ns;
};

constexpr const char *SHORT_CONSUMER = "roomwalk.openvins.short_walk.v1";
constexpr const char *POSE_TIME_REFERENCE = "camera2_encoded_viewport_centre_exposure_midpoint";

std::map<std::int64_t, FrameIdentity> read_frame_identities(const fs::path &capture_dir,
                                                             const std::string &capture_id) {
  std::ifstream input(capture_dir / "openvins_input.json");
  if (!input) return {};
  const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  // The materializer writes this compact identity sequence in stable JSON
  // order.  The native bridge consumes it so output rows retain the imported
  // camera source index and capture identity instead of inventing EuRoC IDs.
  const std::regex pattern(R"ov("source_frame_index"\s*:\s*(-?\d+)\s*,\s*"capture_time_ns"\s*:\s*(-?\d+))ov");
  std::map<std::int64_t, FrameIdentity> identities;
  for (auto match = std::sregex_iterator(text.begin(), text.end(), pattern);
       match != std::sregex_iterator(); ++match) {
    const std::int64_t source_index = std::stoll((*match)[1].str());
    const std::int64_t timestamp_ns = std::stoll((*match)[2].str());
    identities.emplace(timestamp_ns,
                       FrameIdentity{source_index, capture_id + ":source:" + std::to_string(source_index), timestamp_ns});
  }
  return identities;
}

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> values;
  std::stringstream stream(line);
  std::string value;
  while (std::getline(stream, value, ',')) {
    while (!value.empty() && (value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) value.pop_back();
    std::size_t first = 0;
    while (first < value.size() && (value[first] == ' ' || value[first] == '\t')) ++first;
    if (first) value.erase(0, first);
    values.push_back(value);
  }
  return values;
}

std::int64_t parse_i64(const std::string &value) {
  std::size_t used = 0;
  const auto parsed = std::stoll(value, &used);
  if (used != value.size()) {
    throw std::runtime_error("invalid integer timestamp: " + value);
  }
  return parsed;
}

std::map<std::int64_t, FrameIdentity> read_short_frame_mapping(
    const Cli &cli, const std::vector<ImageRecord> &images) {
  std::ifstream input(cli.frame_mapping);
  std::string line;
  if (!input || !std::getline(input, line) || line != "source_frame_index,capture_time_ns,pose_time_ns")
    throw std::runtime_error("short-profile frame mapping header is missing or invalid");
  std::map<std::int64_t, FrameIdentity> result;
  std::int64_t previous_source = -1, previous_pose = -1;
  std::size_t index = 0;
  while (std::getline(input, line)) {
    if (line.size() > 128 || index >= images.size() || index >= 12000)
      throw std::runtime_error("short-profile frame mapping exceeds its bound");
    const auto values = split_csv(line);
    if (values.size() != 3) throw std::runtime_error("invalid short-profile frame mapping row");
    const auto source_index = parse_i64(values[0]);
    const auto source_time = parse_i64(values[1]);
    const auto pose_time = parse_i64(values[2]);
    if (source_index != static_cast<std::int64_t>(index) || source_time <= 0 || pose_time < source_time ||
        pose_time - source_time > 150000000 || source_time <= previous_source || pose_time <= previous_pose ||
        pose_time != images[index].timestamp_ns)
      throw std::runtime_error("short-profile frame mapping disagrees with dense camera timestamps");
    result.emplace(pose_time, FrameIdentity{source_index,
        cli.capture_id + ":source:" + std::to_string(source_index), source_time});
    previous_source = source_time;
    previous_pose = pose_time;
    ++index;
  }
  if (index != images.size()) throw std::runtime_error("short-profile frame mapping is incomplete");
  return result;
}

double parse_double(const std::string &value) {
  std::size_t used = 0;
  const auto parsed = std::stod(value, &used);
  if (used != value.size() || !std::isfinite(parsed)) {
    throw std::runtime_error("invalid numeric CSV value: " + value);
  }
  return parsed;
}

std::vector<ImageRecord> read_images(const fs::path &capture_dir) {
  const fs::path csv_path = capture_dir / "cam0" / "data.csv";
  const fs::path image_dir = capture_dir / "cam0" / "data";
  std::ifstream input(csv_path);
  if (!input) {
    throw std::runtime_error("missing EuRoC camera data.csv: " + csv_path.string());
  }
  std::vector<ImageRecord> records;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto values = split_csv(line);
    if (values.size() < 2) {
      continue;
    }
    ImageRecord record{parse_i64(values[0]), image_dir / values[1]};
    if (!fs::is_regular_file(record.path)) {
      throw std::runtime_error("missing camera image: " + record.path.string());
    }
    records.push_back(std::move(record));
  }
  if (records.size() < 2) {
    throw std::runtime_error("camera data.csv has fewer than two images");
  }
  return records;
}

std::vector<ImuRecord> read_imu(const fs::path &capture_dir) {
  const fs::path csv_path = capture_dir / "imu0" / "data.csv";
  std::ifstream input(csv_path);
  if (!input) {
    throw std::runtime_error("missing EuRoC IMU data.csv: " + csv_path.string());
  }
  std::vector<ImuRecord> records;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto values = split_csv(line);
    if (values.size() < 7) {
      continue;
    }
    // EuRoC stores timestamp, wx, wy, wz, ax, ay, az in SI units.
    records.push_back({parse_i64(values[0]),
                       Eigen::Vector3d(parse_double(values[1]), parse_double(values[2]), parse_double(values[3])),
                       Eigen::Vector3d(parse_double(values[4]), parse_double(values[5]), parse_double(values[6]))});
  }
  if (records.size() < 2) {
    throw std::runtime_error("IMU data.csv has fewer than two samples");
  }
  return records;
}

cv::Mat read_png_gray(const fs::path &path) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, path.c_str())) {
    throw std::runtime_error("libpng could not read image: " + path.string());
  }
  image.format = PNG_FORMAT_GRAY;
  cv::Mat output(static_cast<int>(image.height), static_cast<int>(image.width), CV_8UC1);
  if (!png_image_finish_read(&image, nullptr, output.data, static_cast<png_int_32>(output.step), nullptr)) {
    const std::string message = image.message;
    png_image_free(&image);
    throw std::runtime_error("libpng could not decode image: " + message);
  }
  png_image_free(&image);
  return output;
}

std::vector<double> as_vector(const Eigen::Vector3d &value) {
  return {value(0), value(1), value(2)};
}

std::vector<std::vector<double>> as_matrix(const Eigen::Matrix4d &value) {
  std::vector<std::vector<double>> result(4, std::vector<double>(4));
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      result[row][col] = value(row, col);
    }
  }
  return result;
}

std::vector<std::vector<double>> checked_covariance(Eigen::MatrixXd result) {
  if (!result.allFinite()) {
    throw std::runtime_error("OpenVINS pose covariance is nonfinite");
  }
  result = (result + result.transpose()) * 0.5;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(result);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("OpenVINS covariance eigensolver failed");
  }
  const auto eigenvalues = solver.eigenvalues();
  if (!eigenvalues.allFinite() || eigenvalues.minCoeff() < -1e-7) {
    throw std::runtime_error("OpenVINS pose covariance is materially non-PSD");
  }
  // Only remove roundoff-scale negative eigenvalues.  A materially invalid
  // estimator covariance must fail closed instead of being hidden by clipping.
  result = solver.eigenvectors() * eigenvalues.cwiseMax(0.0).asDiagonal() * solver.eigenvectors().transpose();

  std::vector<std::vector<double>> output(result.rows(), std::vector<double>(result.cols()));
  for (int row = 0; row < result.rows(); ++row) {
    for (int col = 0; col < result.cols(); ++col) {
      output[row][col] = result(row, col);
    }
  }
  return output;
}

std::vector<std::vector<double>> camera_covariance(const std::shared_ptr<ov_msckf::State> &state) {
  const auto calib = state->_calib_IMUtoCAM.at(0);
  std::vector<std::shared_ptr<ov_type::Type>> variables = {state->_imu->q(), state->_imu->p()};
  if (calib->id() >= 0) variables.push_back(calib);
  const Eigen::MatrixXd joint = ov_msckf::StateHelper::get_marginal_covariance(state, variables);
  const auto J = noesis_phone::camera_pose_jacobian(state->_imu->Rot(), calib->Rot(), calib->pos());
  const Eigen::MatrixXd active_J = J.leftCols(joint.cols());
  // Includes every IMU/extrinsic cross covariance when online calibration is enabled.
  return checked_covariance(active_J * joint * active_J.transpose());
}

Calibration read_calibration(const std::shared_ptr<ov_msckf::State> &state, bool online) {
  const auto calib = state->_calib_IMUtoCAM.at(0);
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = calib->Rot().transpose();
  transform.block<3, 1>(0, 3) = -calib->Rot().transpose() * calib->pos();
  std::vector<std::vector<double>> covariance;
  if (online) covariance = checked_covariance(ov_msckf::StateHelper::get_marginal_covariance(
      state, {calib, state->_calib_dt_CAMtoIMU}));
  return {transform, state->_calib_dt_CAMtoIMU->value()(0), covariance};
}

void check_calibration_bounds(const Calibration &current, const Calibration &seed) {
  const double angle = Eigen::AngleAxisd(current.T_imu_camera.block<3, 3>(0, 0) *
                                        seed.T_imu_camera.block<3, 3>(0, 0).transpose()).angle();
  if (!current.T_imu_camera.allFinite() || !std::isfinite(current.camera_to_imu_offset_s) ||
      current.T_imu_camera.block<3, 1>(0, 3).norm() > 0.50 ||
      std::abs(current.camera_to_imu_offset_s) > 0.100 ||
      angle > 0.50 ||
      (current.T_imu_camera.block<3, 1>(0, 3) - seed.T_imu_camera.block<3, 1>(0, 3)).norm() > 0.20 ||
      std::abs(current.camera_to_imu_offset_s - seed.camera_to_imu_offset_s) > 0.050) {
    throw std::runtime_error("online calibration exceeded a provisional experiment bound");
  }
}

void write_json_string(std::ostream &out, const std::string &value) {
  out << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (character < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character)
              << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(character);
        }
    }
  }
  out << '"';
}

void write_json_value(std::ostream &out, const std::vector<double> &values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) out << ',';
    out << std::setprecision(17) << values[index];
  }
  out << ']';
}

void write_json_value(std::ostream &out, const std::vector<std::vector<double>> &values) {
  out << '[';
  for (std::size_t row = 0; row < values.size(); ++row) {
    if (row) out << ',';
    write_json_value(out, values[row]);
  }
  out << ']';
}

void write_calibration(std::ostream &out, const Calibration &calib) {
  out << "{\"T_imu_camera\":";
  write_json_value(out, as_matrix(calib.T_imu_camera));
  out << ",\"camera_to_imu_offset_s\":" << std::setprecision(17) << calib.camera_to_imu_offset_s
      << ",\"imu_to_camera_offset_ns\":" << -calib.camera_to_imu_offset_s * 1e9
      << ",\"covariance\":";
  write_json_value(out, calib.covariance);
  out << ",\"covariance_convention\":\"openvins_jpl_left_rotation_I_to_C_position_I_in_C_camera_to_imu_offset\","
         "\"covariance_order\":[\"theta_x_rad\",\"theta_y_rad\",\"theta_z_rad\","
         "\"p_I_in_C_x_m\",\"p_I_in_C_y_m\",\"p_I_in_C_z_m\",\"camera_to_imu_offset_s\"]}";
}

Cli parse_cli(int argc, char **argv) {
  Cli cli;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value for " + argument);
    }
    if (argument == "--capture-dir") cli.capture_dir = argv[++index];
    else if (argument == "--config") cli.config = argv[++index];
    else if (argument == "--output") cli.output = argv[++index];
    else if (argument == "--camera-mask") cli.camera_mask = argv[++index];
    else if (argument == "--frame-mapping") cli.frame_mapping = argv[++index];
    else if (argument == "--short-session-consumer") cli.short_session_consumer = argv[++index];
    else if (argument == "--profile-id") cli.profile_id = argv[++index];
    else if (argument == "--capture-id") cli.capture_id = argv[++index];
    else if (argument == "--camera-sensor-id") cli.camera_sensor_id = argv[++index];
    else if (argument == "--time-domain") cli.time_domain = argv[++index];
    else if (argument == "--mode") cli.mode = argv[++index];
    else if (argument == "--calibration-rotation-std-rad") cli.rotation_std_rad = parse_double(argv[++index]);
    else if (argument == "--calibration-translation-std-m") cli.translation_std_m = parse_double(argv[++index]);
    else if (argument == "--calibration-time-offset-std-s") cli.time_offset_std_s = parse_double(argv[++index]);
    else throw std::runtime_error("unknown argument: " + argument);
  }
  if (cli.capture_dir.empty() || cli.config.empty() || cli.output.empty()) {
    throw std::runtime_error("usage: openvins_phone_runner --capture-dir DIR --config FILE --output FILE");
  }
  if (cli.mode != "metric_vio" && cli.mode != "calibration" && cli.mode != "profile_validation")
    throw std::runtime_error("unsupported estimator mode");
  if (cli.mode == "profile_validation" || !cli.short_session_consumer.empty() || !cli.frame_mapping.empty() || !cli.profile_id.empty()) {
    if (cli.short_session_consumer != SHORT_CONSUMER || cli.frame_mapping.empty() || cli.profile_id.empty() ||
        cli.profile_id.size() > 128 || cli.mode == "calibration")
      throw std::runtime_error("short-profile mode requires the exact consumer version, profile ID and frame mapping");
  }
  if (cli.mode == "calibration" &&
      (!std::isfinite(cli.rotation_std_rad) || cli.rotation_std_rad <= 0 || cli.rotation_std_rad > 0.50 ||
       !std::isfinite(cli.translation_std_m) || cli.translation_std_m <= 0 || cli.translation_std_m > 0.20 ||
       !std::isfinite(cli.time_offset_std_s) || cli.time_offset_std_s <= 0 || cli.time_offset_std_s > 0.050)) {
    throw std::runtime_error("calibration mode requires explicit bounded prior standard deviations");
  }
  return cli;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Cli cli = parse_cli(argc, argv);
    const bool online = cli.mode == "calibration";
    const bool profile_validation = cli.mode == "profile_validation";
    const bool short_profile = !cli.short_session_consumer.empty();
    const auto images = read_images(cli.capture_dir);
    const auto imu = read_imu(cli.capture_dir);
    const double time_origin = static_cast<double>(images.front().timestamp_ns) * 1e-9;
    const std::int64_t time_origin_ns = images.front().timestamp_ns;
    const auto relative_time = [&](std::int64_t stamp) {
      // Preserve hardware integer nanoseconds before converting the bounded
      // relative interval. The legacy modes retain their existing convention.
      return short_profile ? static_cast<double>(stamp - time_origin_ns) * 1e-9
                           : static_cast<double>(stamp) * 1e-9 - time_origin;
    };
    const auto frame_identities = short_profile ? read_short_frame_mapping(cli, images)
                                               : read_frame_identities(cli.capture_dir, cli.capture_id);

    auto parser = std::make_shared<ov_core::YamlParser>(cli.config.string());
    ov_msckf::VioManagerOptions options;
    options.print_and_load(parser);
    if (options.state_options.num_cameras != 1 || options.use_stereo) {
      throw std::runtime_error("accepted phone baseline requires monocular OpenVINS mode");
    }
    if (options.state_options.do_calib_camera_pose != online ||
        options.state_options.do_calib_camera_timeoffset != online ||
        options.state_options.do_calib_camera_intrinsics || options.init_options.init_dyn_mle_opt_calib) {
      throw std::runtime_error("estimator calibration flags do not match the explicit runner mode");
    }
    if (short_profile &&
        (options.state_options.do_calib_imu_intrinsics || options.state_options.do_calib_imu_g_sensitivity ||
         options.downsample_cameras || options.num_opencv_threads < 1 || options.num_opencv_threads > 4 ||
         options.use_multi_threading_subs || options.init_options.init_dyn_mle_max_threads < 1 ||
         options.init_options.init_dyn_mle_max_threads > 2 ||
         !ov_msckf::State::Dm(options.state_options.imu_model, options.vec_dw).isIdentity(1e-12) ||
         !ov_msckf::State::Dm(options.state_options.imu_model, options.vec_da).isIdentity(1e-12) ||
         !options.vec_tg.isZero(1e-12) ||
         !ov_core::quat_2_Rot(options.q_GYROtoIMU).isIdentity(1e-12) ||
         !ov_core::quat_2_Rot(options.q_ACCtoIMU).isIdentity(1e-12))) {
      throw std::runtime_error("short-profile consumer requires fixed identity native IMU correction and bounded threads/images");
    }
    const cv::Size camera_size(options.camera_intrinsics.at(0)->w(), options.camera_intrinsics.at(0)->h());
    if (short_profile && std::max(camera_size.width, camera_size.height) > 1280)
      throw std::runtime_error("short-profile analysis images exceed 1280 pixels");
    cv::Mat camera_mask = cli.camera_mask.empty() ? cv::Mat::zeros(camera_size, CV_8UC1) : read_png_gray(cli.camera_mask);
    if (camera_mask.size() != camera_size) {
      throw std::runtime_error("static camera mask dimensions do not match calibrated camera resolution");
    }
    std::size_t masked_pixels = 0;
    for (int row = 0; row < camera_mask.rows; ++row) {
      for (int col = 0; col < camera_mask.cols; ++col) {
        const auto value = camera_mask.at<std::uint8_t>(row, col);
        if (value != 0 && value != 255) {
          throw std::runtime_error("static camera mask must contain only 0 valid and 255 invalid pixels");
        }
        if (value == 255) ++masked_pixels;
      }
    }
    if (!cli.camera_mask.empty()) {
      std::cerr << "Loaded static camera mask " << camera_size.width << 'x' << camera_size.height
                << ": " << masked_pixels << " invalid pixels\n";
    }
    ov_msckf::VioManager vio(options);
    const auto initial_state = vio.get_state();
    if (online) {
      Eigen::Matrix<double, 7, 7> prior = Eigen::Matrix<double, 7, 7>::Zero();
      prior.diagonal().head<3>().setConstant(cli.rotation_std_rad * cli.rotation_std_rad);
      prior.diagonal().segment<3>(3).setConstant(cli.translation_std_m * cli.translation_std_m);
      prior(6, 6) = cli.time_offset_std_s * cli.time_offset_std_s;
      ov_msckf::StateHelper::set_initial_covariance(initial_state, prior,
          {initial_state->_calib_IMUtoCAM.at(0), initial_state->_calib_dt_CAMtoIMU});
    }
    const Calibration seed = read_calibration(initial_state, online);
    if (online) check_calibration_bounds(seed, seed);
    // OpenVINS names this calibrated quantity CAM-to-IMU: the IMU sample
    // needed for a camera message at t is at t + dt.  Feed through that
    // target and retain one following sample so the propagator has a genuine
    // bracket at the camera interval boundary.
    std::size_t imu_index = 0;
    std::size_t emitted = 0;
    std::vector<Snapshot> states;
    bool previously_initialized = false;
    double previous_state_time = -std::numeric_limits<double>::infinity();
    std::size_t reset_count = 0;
    double previous_imu_target = -std::numeric_limits<double>::infinity();
    for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
      const auto &image = images[image_index];
      const double camera_time = relative_time(image.timestamp_ns);
      // Re-read the learned offset before every camera propagation. A cached
      // initial offset would feed the wrong IMU interval after calibration updates.
      const double imu_target_time = camera_time + vio.get_state()->_calib_dt_CAMtoIMU->value()(0);
      if (imu_target_time <= previous_imu_target) throw std::runtime_error("learned offset reverses the IMU propagation interval");
      previous_imu_target = imu_target_time;
      while (imu_index < imu.size() && relative_time(imu[imu_index].timestamp_ns) <= imu_target_time) {
        ov_core::ImuData message;
        message.timestamp = relative_time(imu[imu_index].timestamp_ns);
        message.wm = imu[imu_index].gyro;
        message.am = imu[imu_index].accel;
        vio.feed_measurement_imu(message);
        ++imu_index;
      }
      if (imu_index < imu.size()) {
        ov_core::ImuData bracket_message;
        bracket_message.timestamp = relative_time(imu[imu_index].timestamp_ns);
        bracket_message.wm = imu[imu_index].gyro;
        bracket_message.am = imu[imu_index].accel;
        vio.feed_measurement_imu(bracket_message);
        ++imu_index;
      }
      ov_core::CameraData message;
      message.timestamp = camera_time;
      message.sensor_ids = {0};
      message.images = {read_png_gray(image.path)};
      if (message.images.front().size() != camera_size) {
        throw std::runtime_error("decoded camera image dimensions do not match calibrated camera resolution");
      }
      // Pinned OpenVINS TrackKLT excludes values >127.  Reuse this immutable
      // static mask for every rectified frame; zero means the original path.
      message.masks = {camera_mask};
      vio.feed_measurement_camera(message);
      const auto current = vio.get_state();
      if (short_profile && previously_initialized &&
          (!vio.initialized() || current->_timestamp < previous_state_time)) ++reset_count;
      if (vio.initialized()) {
        previously_initialized = true;
        previous_state_time = current->_timestamp;
      }
      const auto current_calibration = read_calibration(current, online);
      if (online) check_calibration_bounds(current_calibration, seed);
      if (short_profile && (!current_calibration.T_imu_camera.isApprox(seed.T_imu_camera, 1e-12) ||
                            std::abs(current_calibration.camera_to_imu_offset_s - seed.camera_to_imu_offset_s) > 1e-12))
        throw std::runtime_error("short-profile fixed calibration changed during estimation");
      // Keep a pose only when the state update belongs to this camera frame.
      // A delayed or reset update must not be emitted with a stale identity.
      if (vio.initialized() && std::abs(current->_timestamp - camera_time) <= 1e-9) {
        const auto calib = current->_calib_IMUtoCAM.at(0);
        const Eigen::Matrix4d T_world_camera = noesis_phone::world_camera_transform(
            current->_imu->Rot(), current->_imu->pos(), calib->Rot(), calib->pos());
        auto identity = frame_identities.find(image.timestamp_ns);
        const std::int64_t source_frame_index = identity == frame_identities.end()
                                                     ? static_cast<std::int64_t>(image_index)
                                                     : identity->second.source_frame_index;
        const std::string prepared_frame_id = identity == frame_identities.end()
                                                  ? cli.capture_id + ":source:" + std::to_string(source_frame_index)
                                                  : identity->second.prepared_frame_id;
        states.push_back({image.timestamp_ns,
                          identity == frame_identities.end() ? image.timestamp_ns : identity->second.capture_time_ns,
                          source_frame_index,
                          prepared_frame_id,
                          T_world_camera,
                          current->_imu->vel(),
                          current->_imu->bias_g(),
                          current->_imu->bias_a(),
                          camera_covariance(current),
                          current_calibration});
        ++emitted;
      }
    }
    if (!vio.initialized() || states.size() < 2) {
      throw std::runtime_error("OpenVINS did not initialize on the supplied monocular sequence");
    }
    double maximum_pose_gap_s = 0.0;
    for (std::size_t i = 1; i < states.size(); ++i)
      maximum_pose_gap_s = std::max(maximum_pose_gap_s,
          static_cast<double>(states[i].timestamp_ns - states[i-1].timestamp_ns) * 1e-9);
    const double end_gap_s = static_cast<double>(images.back().timestamp_ns - states.back().timestamp_ns) * 1e-9;
    const double coverage = static_cast<double>(states.size()) / static_cast<double>(images.size());
    // Versioned practical tracking gate, not evidence of metric accuracy.
    // Do not apply it to legacy runs or to evidence-producing profile checks.
    if (short_profile && !profile_validation &&
        (reset_count != 0 || coverage < 0.8 || maximum_pose_gap_s > 0.2 || end_gap_s > 0.2))
      throw std::runtime_error("short-profile runtime quality failed: reset, coverage below 0.8, or pose/end gap over 0.2s");

    fs::create_directories(cli.output.parent_path());
    std::ofstream output(cli.output);
    if (!output) throw std::runtime_error("cannot open output: " + cli.output.string());
    output << "{\"schema\":";
    write_json_string(output, profile_validation ? "noesis.phone_capture.vio_profile_validation_result.v1" :
                              online ? "noesis.phone_capture.vio_calibration_result.v1" : "noesis.phone_capture.vio_result.v1");
    // Short metric runs are admitted only by the wrapper's separate runtime
    // quality callback. Validation output can never become a metric result.
    output << ",\"estimator\":\"openvins\",\"accepted_for_metric_vio\":" << ((online || short_profile) ? "false" : "true") << ",\"frame\":{";
    output << "\"source\":\"camera\",\"target\":\"vio_world\",\"pose_convention\":\"T_vio_world_camera\","
           << "\"units\":\"meters\",\"capture_id\":";
    write_json_string(output, cli.capture_id);
    output << ",\"camera_sensor_id\":";
    write_json_string(output, cli.camera_sensor_id);
    output << ",\"time_domain\":";
    write_json_string(output, cli.time_domain);
    if (short_profile) {
      output << ",\"pose_time_reference\":";
      write_json_string(output, POSE_TIME_REFERENCE);
      output << ",\"capture_time_reference\":\"original_camera_sensor_timestamp\"";
    }
    output << ",\"camera_axes\":\"x_right_y_down_z_forward\",\"world_axes\":\"z_up_gravity_up\","
           << "\"pose_origin\":\"camera_optical_center\",\"velocity_origin\":\"imu_center\","
           << "\"gravity_frame\":\"vio_world\",\"gravity_semantics\":\"physical_world_acceleration\"},"
           << "\"scale\":{\"mode\":";
    write_json_string(output, (online || profile_validation) ? "provisional_metric" : "metric");
    output << ",\"source\":";
    write_json_string(output, profile_validation ? "short_session_profile_validation" :
                              online ? "provisional_imu_camera_prior_online_calibration" : "imu_camera_calibration");
    output << "},\"camera_mask\":{\"applied\":" << (cli.camera_mask.empty() ? "false" : "true")
           << ",\"invalid_pixel_count\":" << masked_pixels
           << ",\"resolution_px\":[" << camera_size.width << ',' << camera_size.height << "]},"
           << "\"covariance_frame\":\"camera_pose_tangent_se3_row_major\",\"covariance_tangent_frame\":\"vio_world_rotation_additive_position\",\"poses\":[";
    for (std::size_t index = 0; index < states.size(); ++index) {
      if (index) output << ',';
      const auto &entry = states[index];
      output << "{\"capture_time_ns\":" << entry.capture_time_ns;
      if (short_profile) output << ",\"pose_time_ns\":" << entry.timestamp_ns;
      output << ",\"prepared_frame_id\":";
      write_json_string(output, entry.prepared_frame_id);
      output << ",\"source_frame_index\":" << entry.source_frame_index << ","
             << "\"T_vio_world_camera\":";
      write_json_value(output, as_matrix(entry.T_world_camera));
      output << ",\"velocity_mps\":";
      write_json_value(output, as_vector(entry.velocity_imu));
      output << ",\"gravity_mps2\":[0,0,-9.81],\"estimator_gravity_mps2\":[0,0,9.81],\"gyro_bias_rads\":";
      write_json_value(output, as_vector(entry.gyro_bias));
      output << ",\"accel_bias_mps2\":";
      write_json_value(output, as_vector(entry.accel_bias));
      output << ",\"covariance\":";
      write_json_value(output, entry.covariance);
      if (online) {
        output << ",\"calibration\":";
        write_calibration(output, entry.calibration);
      }
      output << "}";
    }
    output << ']';
    if (online) {
      output << ",\"status\":\"provisional\",\"camera_intrinsics_optimized\":false,\"initial_calibration\":";
      write_calibration(output, seed);
      output << ",\"calibration\":";
      write_calibration(output, states.back().calibration);
      output << ",\"bounds\":{\"rotation_change_rad\":0.50,\"translation_change_m\":0.20,"
                "\"time_offset_change_s\":0.050,\"translation_norm_m\":0.50,\"absolute_time_offset_s\":0.100}";
    }
    if (short_profile) {
      output << ",\"status\":";
      write_json_string(output, profile_validation ? "provisional" : "pending_runtime_quality_control");
      output << ",\"short_session_consumer\":{\"consumer_version\":";
      write_json_string(output, SHORT_CONSUMER);
      output << ",\"profile_id\":";
      write_json_string(output, cli.profile_id);
      output << ",\"validation_run\":" << (profile_validation ? "true" : "false")
             << ",\"fixed_camera_imu_calibration\":true,\"native_imu_corrections\":\"identity\","
                "\"dense_frame_mapping_verified\":true,\"rolling_shutter_compensated\":false,"
                "\"image_motion_model\":\"centre_timed_global_shutter_approximation\"},"
                "\"fixed_calibration\":{\"T_imu_camera\":";
      write_json_value(output, as_matrix(seed.T_imu_camera));
      output << ",\"imu_to_camera_offset_ns\":" << std::llround(-seed.camera_to_imu_offset_s * 1e9) << "}";
    }
    output << ",\"quality\":{\"initialized\":true,\"tracking_ratio\":"
           << static_cast<double>(emitted) / static_cast<double>(images.size())
           << ",\"initialization_delay_s\":"
           << static_cast<double>(states.front().timestamp_ns - images.front().timestamp_ns) * 1e-9
           << ",\"retained_interval_s\":"
           << static_cast<double>(states.back().timestamp_ns - states.front().timestamp_ns) * 1e-9;
    if (short_profile) output << ",\"maximum_pose_gap_s\":" << maximum_pose_gap_s
                              << ",\"end_gap_s\":" << end_gap_s << ",\"reset_count\":" << reset_count;
    output << "},\"segments\":[{\"id\":\"openvins-0\",\"reset\":"
           << ((short_profile && reset_count) ? "true" : "false") << ",\"start_capture_time_ns\":"
           << states.front().capture_time_ns << ",\"end_capture_time_ns\":" << states.back().capture_time_ns;
    if (short_profile) output << ",\"start_pose_time_ns\":" << states.front().timestamp_ns
                              << ",\"end_pose_time_ns\":" << states.back().timestamp_ns;
    output << "}]}";
    output << std::endl;
    std::cerr << "OpenVINS processed " << images.size() << " images and emitted " << states.size() << " initialized camera states\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "openvins_phone_runner: " << error.what() << '\n';
    return 2;
  }
}
