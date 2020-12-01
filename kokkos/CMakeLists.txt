cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(bude_sycl)
SET(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0074 NEW) #see https://github.com/kokkos/kokkos/blob/master/BUILD.md


if (KOKKOS_IN_TREE)
    message(STATUS "Building using in-tree kokkos source at `${KOKKOS_IN_TREE}`")
    add_subdirectory(${KOKKOS_IN_TREE} ${CMAKE_BINARY_DIR}/kokkos)
    add_definitions(-DKOKKOS_IN_TREE=${KOKKOS_IN_TREE})
else ()
    find_package(Kokkos REQUIRED)
endif ()


set(SOURCES
        src/bude.cpp
        src/vec-pose-inner.cpp)

include_directories(src)
add_executable(bude ${SOURCES})

separate_arguments(CXX_EXTRA_FLAGS)
separate_arguments(CXX_EXTRA_LINKER_FLAGS)

target_compile_options(bude
        PUBLIC
        -Wall
        -Wextra
        -Wcast-align
        -Wfatal-errors
        -Werror=return-type
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-ignored-attributes

        ${EXTRA_FLAGS}
        )


if (NOT DEFINED WG_SIZE)
    message(STATUS "WG_SIZE unspecified, defaulting to 64")
    set(WG_SIZE 64)
endif ()


if (NOT WG_SIZE MATCHES "^[0-9]+$")
    message(FATAL_ERROR "WG_SIZE must be an integer, got ${WG_SIZE}")
endif ()

add_definitions(-DWG_SIZE=${WG_SIZE})

set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer ${CXX_EXTRA_FLAGS})
set(RELEASE_OPTIONS -Ofast -ffast-math ${CXX_EXTRA_FLAGS})

target_compile_options(bude PUBLIC "$<$<CONFIG:RelWithDebInfo>:${RELEASE_OPTIONS}>")
target_compile_options(bude PUBLIC "$<$<CONFIG:Release>:${RELEASE_OPTIONS}>")
target_compile_options(bude PUBLIC "$<$<CONFIG:Debug>:${DEBUG_OPTIONS}>")

target_link_libraries(bude Kokkos::kokkos)

if (${CMAKE_VERSION} VERSION_LESS "3.13.0")
    message(WARNING "target_link_options is only available in CMake >= 3.13.0, using fallback target_link_libraries, this may cause issues with some compilers")
    message(WARNING "whitespaces are not supported for CXX_EXTRA_LINKER_FLAGS/CXX_EXTRA_FLAGS in this mode as they are treated as libraries arguments (CMake splits them)")
    if (DEFINED CXX_EXTRA_LINKER_FLAGS)
        list(APPEND EXTRA_LINK_FLAGS "-Wl,${CXX_EXTRA_LINKER_FLAGS}")
    endif ()

    target_link_libraries(bude PUBLIC ${EXTRA_LINK_FLAGS})
    target_link_libraries(bude PUBLIC ${CXX_EXTRA_FLAGS})

else ()
    target_link_options(bude PUBLIC LINKER:${CXX_EXTRA_LINKER_FLAGS})
    target_link_options(bude PUBLIC ${CXX_EXTRA_FLAGS})
endif ()


 
