# Build of the C++ port of the particle filter, and its Python module pf_cpp.
#
#   cmake -S cpp -B cpp/build -G Ninja -DPython_EXECUTABLE=$(which python)
#   cmake --build cpp/build
#
# The module is placed in the project root, so that it is imported like the Python modules (including by spawned worker processes).
cmake_minimum_required(VERSION 3.18...3.31)
project(pf_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# The particle filter, independent of Python
add_library(pf_core STATIC
  utils_sample.cpp
  models_obs_rew.cpp
  models_context.cpp
  particle.cpp
  idealobserver.cpp)
target_include_directories(pf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pf_core PUBLIC Eigen3::Eigen)
target_compile_options(pf_core PRIVATE -Wall -Wextra)
set_target_properties(pf_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The Python module
nanobind_add_module(pf_cpp bindings.cpp)
target_link_libraries(pf_cpp PRIVATE pf_core)
set_target_properties(pf_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
