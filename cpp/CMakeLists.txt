# Build of the C++ port of the particle filter, and its Python module pf_cpp.
# The easiest way to build is `python cpp/build_cpp.py` (from the project root), which checks the requirements and runs:
#
#   cmake -S cpp -B cpp/build -DPython_EXECUTABLE=<python> [-G Ninja]
#   cmake --build cpp/build --config Release
#
# The module is placed in the project root, so that it is imported like the Python modules (including by spawned worker processes).
# Eigen is taken from the system if it is installed (version 3.4), and downloaded otherwise.
cmake_minimum_required(VERSION 3.18...3.31)
project(pf_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Python and nanobind (installed with pip)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Eigen (header only): the system's, or a download of the 3.4.0 release
option(PF_FETCH_EIGEN "Always download Eigen, rather than using an installed one" OFF)
if(NOT PF_FETCH_EIGEN)
  find_package(Eigen3 3.4 QUIET NO_MODULE)
endif()
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Eigen 3.4 not found on the system, downloading it")
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  include(FetchContent)
  FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    SOURCE_SUBDIR do_not_build)  # only the headers are needed, so Eigen's own CMake project is not added
  FetchContent_MakeAvailable(eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}")
endif()

# The particle filter, independent of Python
add_library(pf_core STATIC
  utils_sample.cpp
  models_obs_rew.cpp
  models_context.cpp
  particle.cpp
  idealobserver.cpp)
target_include_directories(pf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pf_core PUBLIC Eigen3::Eigen)
if(MSVC)
  target_compile_options(pf_core PRIVATE /W4)
else()
  target_compile_options(pf_core PRIVATE -Wall -Wextra)
endif()
set_target_properties(pf_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The Python module
nanobind_add_module(pf_cpp bindings.cpp)
target_link_libraries(pf_cpp PRIVATE pf_core)
if(MSVC)
  target_compile_options(pf_cpp PRIVATE /bigobj)
endif()
# The generator expression stops multi-configuration generators (Visual Studio, Xcode) from adding a Release/ subdirectory
set_target_properties(pf_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_SOURCE_DIR}/..>")
