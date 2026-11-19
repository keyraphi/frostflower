cmake_minimum_required(VERSION 3.25)
project(frostflower VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

enable_language(CUDA)
# TEMPORARY: match the Winder default for fast iteration; expand for releases.
set(CUDA_ARCHITECTURES 120 89)
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v3.1.0
    SYSTEM
)
FetchContent_MakeAvailable(nanobind)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src/cpp)
