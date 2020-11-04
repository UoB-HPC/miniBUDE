cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(makedeck LANGUAGES CXX)

#SET(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ENABLE_ASAN OFF)

find_package(OpenMP)

add_executable(makedeck
        input-utils.cpp
        ref-kernel.cpp
        main.cpp
        )

target_compile_options(makedeck
        PUBLIC
        -Wall
        -Wextra
        -Wcast-align
        -Wfatal-errors
        -Werror=return-type
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-ignored-attributes
        )

set(DEBUG_OPTIONS -O1 -fno-omit-frame-pointer)
set(RELEASE_OPTIONS -O3 -ffast-math -march=native -mtune=native)

if (ENABLE_ASAN)
    list(APPEND DEBUG_OPTIONS -fsanitize=address)
    target_link_libraries(makedeck PUBLIC -Wl,-lasan)
endif ()

target_compile_options(makedeck PUBLIC "$<$<CONFIG:RelWithDebInfo>:${RELEASE_OPTIONS}>")
target_compile_options(makedeck PUBLIC "$<$<CONFIG:Release>:${RELEASE_OPTIONS}>")
target_compile_options(makedeck PUBLIC "$<$<CONFIG:Debug>:${DEBUG_OPTIONS}>")

if (OpenMP_CXX_FOUND)
    target_link_libraries(makedeck PUBLIC OpenMP::OpenMP_CXX)
endif ()


Include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.2)
FetchContent_MakeAvailable(Catch2)

add_executable(tests
        input-utils.cpp
        ref-kernel.cpp
        test-bude.cpp
        test-main.cpp)
target_link_libraries(tests Catch2::Catch2)

include(CTest)
#include(Catch)
#catch_discover_tests(tests)
