cmake_minimum_required(VERSION 3.18)
project(onnx_gpu_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(VCPKG_TARGET_TRIPLET "x64-windows")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(VCPKG_TARGET_TRIPLET "x64-mingw-dynamic")
endif()

set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_TARGET_TRIPLET ${VCPKG_TARGET_TRIPLET} CACHE STRING "")


# 找 OpenCV
find_package(OpenCV REQUIRED)

# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_ROOT "C:/onnxruntime-win-x64-gpu-1.22.1/onnxruntime-win-x64-gpu-1.22.1") 
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

add_executable(onnx_gpu_test main.cpp)

# 包含头文件
target_include_directories(onnx_gpu_test PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR} 
)

# 链接库
target_link_directories(onnx_gpu_test PRIVATE
    ${ONNXRUNTIME_LIB_DIR}
)

target_link_libraries(onnx_gpu_test PRIVATE
    onnxruntime
    onnxruntime_providers_shared
    onnxruntime_providers_cuda
    ${OpenCV_LIBS}
)
