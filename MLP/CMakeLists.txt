cmake_minimum_required(VERSION 3.21)
project(MLP LANGUAGES CXX)

# 设置 C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 找 OpenCV 和 Eigen3
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# 自动查找所有源文件
file(GLOB_RECURSE SOURCE_FILES 
    "*.cpp"
    "*.cxx"
    "*.cc"
    "*.c"
)

file(GLOB_RECURSE HEADER_FILES 
    "*.h"
    "*.hpp"
    "*.hxx"
    "*.hh"
)

# 过滤掉不需要的文件（可选）
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/test/.*")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/build/.*")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/third_party/.*")
list(FILTER SOURCE_FILES EXCLUDE REGEX "cnpy/*")

list(FILTER HEADER_FILES EXCLUDE REGEX ".*/test/.*")
list(FILTER HEADER_FILES EXCLUDE REGEX ".*/build/.*")
list(FILTER HEADER_FILES EXCLUDE REGEX ".*/third_party/.*")
list(FILTER HEADER_FILES EXCLUDE REGEX "cnpy/*")

# 打印找到的文件（用于调试）
message(STATUS "找到的源文件: ${SOURCE_FILES}")
message(STATUS "找到的头文件: ${HEADER_FILES}")

# 主程序
add_executable(MLP
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

# 链接库
target_link_libraries(MLP
    PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

# 包含头文件目录
target_include_directories(MLP
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}  # 添加当前目录到包含路径
)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 添加编译选项（可选）
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(MLP PRIVATE -Wall -Wextra -O2)
endif()