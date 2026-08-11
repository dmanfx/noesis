cmake_minimum_required(VERSION 3.16)
project(noesis_nvdsroiexclude_ds8 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0)

set(DEEPSTREAM_HOME "$ENV{NOESIS_DEEPSTREAM_HOME}" CACHE PATH "DeepStream 8 SDK root")
if(NOT DEEPSTREAM_HOME)
  set(DEEPSTREAM_HOME "/opt/nvidia/deepstream/deepstream-8.0")
endif()
if(NOT IS_DIRECTORY "${DEEPSTREAM_HOME}")
  message(FATAL_ERROR "DeepStream 8 SDK root does not exist: ${DEEPSTREAM_HOME}")
endif()
get_filename_component(DEEPSTREAM_HOME_REAL "${DEEPSTREAM_HOME}" REALPATH)

set(NVDS_VERSION_HEADER "${DEEPSTREAM_HOME_REAL}/sources/includes/nvds_version.h")
set(NVDS_META_HEADER "${DEEPSTREAM_HOME_REAL}/sources/includes/nvdsmeta.h")
set(NVDS_GST_META_HEADER "${DEEPSTREAM_HOME_REAL}/sources/includes/gstnvdsmeta.h")
set(NVDS_META_LIBRARY "${DEEPSTREAM_HOME_REAL}/lib/libnvds_meta.so")
set(NVDS_GST_META_LIBRARY "${DEEPSTREAM_HOME_REAL}/lib/libnvdsgst_meta.so")

foreach(REQUIRED_PATH IN ITEMS
    "${NVDS_VERSION_HEADER}"
    "${NVDS_META_HEADER}"
    "${NVDS_GST_META_HEADER}"
    "${NVDS_META_LIBRARY}"
    "${NVDS_GST_META_LIBRARY}")
  if(NOT EXISTS "${REQUIRED_PATH}")
    message(FATAL_ERROR "Missing required DeepStream 8 build input: ${REQUIRED_PATH}")
  endif()
endforeach()

file(READ "${NVDS_VERSION_HEADER}" NVDS_VERSION_CONTENTS)
if(NOT NVDS_VERSION_CONTENTS MATCHES "#[ \t]*define[ \t]+NVDS_VERSION_MAJOR[ \t]+8([^0-9]|$)")
  message(FATAL_ERROR
    "DS8 nvdsroiexclude must be built against DeepStream major 8: ${DEEPSTREAM_HOME_REAL}")
endif()

add_library(gstnvdsroiexclude SHARED gstnvdsroiexclude.cpp)
target_include_directories(gstnvdsroiexclude PRIVATE
  ${GST_INCLUDE_DIRS}
  "${DEEPSTREAM_HOME_REAL}/sources/includes"
)
target_compile_options(gstnvdsroiexclude PRIVATE
  ${GST_CFLAGS_OTHER}
  -Wall
  -Wextra
  -Wpedantic
  -Werror
)
target_compile_definitions(gstnvdsroiexclude PRIVATE
  PACKAGE="noesis"
  PACKAGE_VERSION="1.0.0"
  NOESIS_DEEPSTREAM_MAJOR="8"
)
target_link_directories(gstnvdsroiexclude PRIVATE ${GST_LIBRARY_DIRS})
target_link_libraries(gstnvdsroiexclude PRIVATE
  ${GST_LIBRARIES}
  "${NVDS_META_LIBRARY}"
  "${NVDS_GST_META_LIBRARY}"
)
set_target_properties(gstnvdsroiexclude PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME "gstnvdsroiexclude"
  BUILD_RPATH "${DEEPSTREAM_HOME_REAL}/lib"
  INSTALL_RPATH "${DEEPSTREAM_HOME_REAL}/lib"
)
