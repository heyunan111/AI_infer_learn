cmake_minimum_required(VERSION 3.18)
project(onnx_gpu_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# 指定 triplet
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(VCPKG_TARGET_TRIPLET "x64-mingw-dynamic" CACHE STRING "")
else()
    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "")
endif()

# 指定 vcpkg 工具链
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "C:/onnxruntime-win-x64-gpu-1.22.1/onnxruntime-win-x64-gpu-1.22.1")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

add_executable(onnx_gpu_test main.cpp)

target_include_directories(onnx_gpu_test PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(onnx_gpu_test PRIVATE
    ${ONNXRUNTIME_LIB_DIR}
)

target_link_libraries(onnx_gpu_test PRIVATE
    onnxruntime
    onnxruntime_providers_shared
    onnxruntime_providers_cuda
    ${OpenCV_LIBS}
)
