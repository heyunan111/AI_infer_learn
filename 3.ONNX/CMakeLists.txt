cmake_minimum_required(VERSION 3.18)

# 必须在 project() 之前设置 vcpkg
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-mingw-dynamic" CACHE STRING "")
endif()

project(onnx_gpu_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成 compile_commands.json 供 clangd 使用

# 查找 OpenCV (vcpkg 安装的是 opencv4)
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
