cmake_minimum_required(VERSION 3.16)
project(noesis_eos_ds9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0)

add_library(gstnoesiseos SHARED gstnoesiseos.cpp)
target_include_directories(gstnoesiseos PRIVATE ${GST_INCLUDE_DIRS})
target_compile_options(gstnoesiseos PRIVATE
  ${GST_CFLAGS_OTHER}
  -Wall
  -Wextra
  -Wpedantic
  -Werror
)
target_compile_definitions(gstnoesiseos PRIVATE
  PACKAGE="noesis"
  PACKAGE_VERSION="1.0.0"
  NOESIS_DEEPSTREAM_MAJOR="9"
)
target_link_directories(gstnoesiseos PRIVATE ${GST_LIBRARY_DIRS})
target_link_libraries(gstnoesiseos PRIVATE ${GST_LIBRARIES})
set_target_properties(gstnoesiseos PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME "gstnoesiseos"
)
