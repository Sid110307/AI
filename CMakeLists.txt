cmake_minimum_required(VERSION 3.2)
project(AI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread -lre2")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/usr/share/libtorch")
add_executable(AI
        ${PROJECT_SOURCE_DIR}/AI.cpp
        ${PROJECT_SOURCE_DIR}/threadPool.cpp
        ${PROJECT_SOURCE_DIR}/include/threadPool.h
        ${PROJECT_SOURCE_DIR}/include/datasets.h
        ${PROJECT_SOURCE_DIR}/include/utils.h)

find_package(Torch REQUIRED)
if (TORCH_FOUND)
    target_link_libraries(AI PUBLIC "${TORCH_LIBRARIES}")
    target_include_directories(AI PUBLIC "${TORCH_INCLUDE_DIRS}")
    target_compile_definitions(AI PUBLIC "${TORCH_DEFINITIONS}")
endif ()

find_package(CURL REQUIRED)
if (CURL_FOUND)
    target_link_libraries(AI PUBLIC "${CURL_LIBRARIES}")
    target_include_directories(AI PUBLIC "${CURL_INCLUDE_DIRS}")
endif ()

find_package(libzip REQUIRED)
if (libzip_FOUND)
    target_link_libraries(AI PUBLIC "/usr/share/libzip/lib/libzip.so")
    target_include_directories(AI PUBLIC "/usr/share/libzip/include")
endif ()
