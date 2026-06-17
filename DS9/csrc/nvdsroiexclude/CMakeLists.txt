cmake_minimum_required(VERSION 3.16)
project(noesis_nvdsroiexclude LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0)

set(DEEPSTREAM_HOME "$ENV{NOESIS_DEEPSTREAM_HOME}")
if(NOT DEEPSTREAM_HOME)
  set(DEEPSTREAM_HOME "/opt/nvidia/deepstream/deepstream-9.0")
endif()

add_library(gstnvdsroiexclude SHARED gstnvdsroiexclude.cpp)
target_include_directories(gstnvdsroiexclude PRIVATE
  ${GST_INCLUDE_DIRS}
  "${DEEPSTREAM_HOME}/sources/includes"
)
target_compile_options(gstnvdsroiexclude PRIVATE ${GST_CFLAGS_OTHER} -Wall -Wextra -Wpedantic)
target_link_directories(gstnvdsroiexclude PRIVATE
  ${GST_LIBRARY_DIRS}
  "${DEEPSTREAM_HOME}/lib"
)
target_link_libraries(gstnvdsroiexclude PRIVATE
  ${GST_LIBRARIES}
  nvds_meta
  nvdsgst_meta
)
set_target_properties(gstnvdsroiexclude PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME "gstnvdsroiexclude"
  BUILD_RPATH "${DEEPSTREAM_HOME}/lib"
  INSTALL_RPATH "${DEEPSTREAM_HOME}/lib"
)
