cmake_minimum_required(VERSION 3.16)
project(noesis_openvins_phone_runner LANGUAGES CXX)

# The bridge is kept outside the upstream checkout. OPENVINS_BUILD must
# already contain the selected upstream ov_msckf_lib; this target never
# substitutes a local estimator or silently rebuilds a different checkout.
set(OPENVINS_ROOT "" CACHE PATH "Pinned OpenVINS source checkout")
set(OPENVINS_BUILD "" CACHE PATH "OpenVINS CMake build directory")
set(DEPENDENCY_ROOT "" CACHE PATH "Optional prefix containing Eigen/OpenCV/Ceres/glog/Boost")
if(NOT IS_DIRECTORY "${OPENVINS_ROOT}/ov_msckf/src" OR
   NOT IS_DIRECTORY "${OPENVINS_ROOT}/ov_core/src" OR
   NOT IS_DIRECTORY "${OPENVINS_ROOT}/ov_init/src")
  message(FATAL_ERROR "OPENVINS_ROOT must contain ov_msckf/src, ov_core/src, and ov_init/src")
endif()
if(NOT IS_DIRECTORY "${OPENVINS_BUILD}")
  message(FATAL_ERROR "OPENVINS_BUILD must point to the built OpenVINS directory")
endif()

if(DEPENDENCY_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH
    "${DEPENDENCY_ROOT}/usr"
    "${DEPENDENCY_ROOT}/usr/lib/x86_64-linux-gnu")
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc video features2d calib3d aruco)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(Ceres REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(PNG REQUIRED)

set(OPENVINS_LIBRARY "${OPENVINS_BUILD}/libov_msckf_lib.so")
if(NOT EXISTS "${OPENVINS_LIBRARY}")
  message(FATAL_ERROR "missing built OpenVINS library: ${OPENVINS_LIBRARY}")
endif()
add_library(OpenVINS::ov_msckf_lib SHARED IMPORTED GLOBAL)
set_target_properties(OpenVINS::ov_msckf_lib PROPERTIES
  IMPORTED_LOCATION "${OPENVINS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES
    "${OPENVINS_ROOT}/ov_msckf/src;${OPENVINS_ROOT}/ov_core/src;${OPENVINS_ROOT}/ov_init/src"
)

add_executable(openvins_phone_runner openvins_phone_runner.cpp)
target_compile_features(openvins_phone_runner PRIVATE cxx_std_17)
target_compile_definitions(openvins_phone_runner PRIVATE ROS_AVAILABLE=0 ENABLE_ARUCO_TAGS=0)
target_include_directories(openvins_phone_runner PRIVATE
  "${OPENVINS_ROOT}/ov_msckf/src"
  "${OPENVINS_ROOT}/ov_core/src"
  "${OPENVINS_ROOT}/ov_init/src"
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(openvins_phone_runner PRIVATE
  OpenVINS::ov_msckf_lib
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  Ceres::ceres
  glog::glog
  PNG::PNG
)
set_target_properties(openvins_phone_runner PROPERTIES
  BUILD_RPATH "${OPENVINS_BUILD};${DEPENDENCY_ROOT}/usr/lib/x86_64-linux-gnu"
)
# DT_RUNPATH is not searched transitively for libov_msckf_lib's dependencies;
# use the legacy DT_RPATH so a service can launch this child without globally
# changing the Python process's Torch/OpenCV library search path.
target_link_options(openvins_phone_runner PRIVATE -Wl,--disable-new-dtags)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_openvins_pose_math test_openvins_pose_math.cpp)
  target_compile_features(test_openvins_pose_math PRIVATE cxx_std_17)
  target_include_directories(test_openvins_pose_math PRIVATE "${OPENVINS_ROOT}/ov_core/src")
  target_link_libraries(test_openvins_pose_math PRIVATE Eigen3::Eigen)
  add_test(NAME openvins_pose_math COMMAND test_openvins_pose_math)
endif()
