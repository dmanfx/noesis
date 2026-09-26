cmake_minimum_required(VERSION 3.20)
project(roomwalk_calibration LANGUAGES CXX)

# Standalone CPU-only consumer of pinned Basalt templates. No upstream library,
# ROS, GUI, image decoder, or global installation is required by this target.
set(DEPENDENCY_ROOT "" CACHE PATH "Directory populated by build.py")
set(EIGEN_ROOT "" CACHE PATH "Eigen 5.0.1 source/include root")
foreach(header basalt/include/basalt/optimization/spline_optimize.h
    basalt-headers/include/basalt/calibration/calibration.hpp
    sophus/sophus/se3.hpp cereal/include/cereal/cereal.hpp
    json/single_include/nlohmann/json.hpp fmt/include/fmt/format.h)
  if(NOT EXISTS "${DEPENDENCY_ROOT}/${header}")
    message(FATAL_ERROR "Missing pinned dependency: ${header}")
  endif()
endforeach()
if(NOT EXISTS "${EIGEN_ROOT}/Eigen/Core")
  message(FATAL_ERROR "EIGEN_ROOT must contain Eigen/Core")
endif()
find_package(TBB REQUIRED)
add_executable(roomwalk_calibrate_imu runner.cpp)
target_compile_features(roomwalk_calibrate_imu PRIVATE cxx_std_17)
target_compile_definitions(roomwalk_calibrate_imu PRIVATE FMT_HEADER_ONLY)
target_include_directories(roomwalk_calibrate_imu SYSTEM PRIVATE
  "${EIGEN_ROOT}" "${DEPENDENCY_ROOT}/basalt/include"
  "${DEPENDENCY_ROOT}/basalt-headers/include" "${DEPENDENCY_ROOT}/sophus"
  "${DEPENDENCY_ROOT}/cereal/include" "${DEPENDENCY_ROOT}/json/single_include"
  "${DEPENDENCY_ROOT}/fmt/include")
target_link_libraries(roomwalk_calibrate_imu PRIVATE TBB::tbb)
# These templates are large; one translation unit at modest optimization keeps
# compilation bounded. Never link the separately published libbasalt ABI.
target_compile_options(roomwalk_calibrate_imu PRIVATE -O1 -g0)
