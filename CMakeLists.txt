cmake_minimum_required(VERSION 3.20)
project(AI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -pthread")

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/usr/share/libtorch")
add_executable(AI src/main.cpp)

find_package(Torch REQUIRED)
if (TORCH_FOUND)
    target_link_libraries(AI PUBLIC "${TORCH_LIBRARIES}")
    target_include_directories(AI PUBLIC "${TORCH_INCLUDE_DIRS}")
    target_compile_definitions(AI PUBLIC "${TORCH_DEFINITIONS}")
endif ()
