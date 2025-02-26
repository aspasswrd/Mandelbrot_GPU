cmake_minimum_required(VERSION 3.12)
project(MandelbrotGPU)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(
        ${SDL2_INCLUDE_DIRS}
        ${OpenCL_INCLUDE_DIRS}
)

add_executable(mandelbrot mandelbrot.cpp)

# Добавляем флаг для OpenCL
target_compile_definitions(mandelbrot PRIVATE
        -DCL_HPP_TARGET_OPENCL_VERSION=300
)

target_link_libraries(mandelbrot
        ${SDL2_LIBRARIES}
        ${OpenCL_LIBRARIES}
)

if(UNIX AND NOT APPLE)
    target_link_libraries(mandelbrot pthread)
endif()