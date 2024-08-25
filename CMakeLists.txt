cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(miniBUDE VERSION 2.0 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

# some nicer defaults for standard C++
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)


## Flags for debugging only, enable for development (ASan only works on few models)
set(SANITIZE OFF)
if (SANITIZE)
    set(DEBUG_FLAGS ${DEBUG_FLAGS} -fsanitize=address)
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address)
endif ()

# the final executable name
set(EXE_NAME bude)

# for chrono and some basic CXX features, models can overwrite this if required
set(CMAKE_CXX_STANDARD 17)

## select default build type if not given
if (NOT CMAKE_BUILD_TYPE)
    message("No CMAKE_BUILD_TYPE specified, defaulting to 'Release'")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ((NOT BUILD_TYPE STREQUAL RELEASE) AND (NOT BUILD_TYPE STREQUAL DEBUG))
    message(FATAL_ERROR "Only Release or Debug is supported, got `${CMAKE_BUILD_TYPE}`")
endif ()

# setup some defaults flags for everything
set(DEFAULT_DEBUG_FLAGS -O2)
set(DEFAULT_RELEASE_FLAGS -O3
        $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:PGI>>:-fast>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:NVHPC>>,$<NOT:$<CXX_COMPILER_ID:PGI>>>:-ffast-math>
)

macro(hint_flag FLAG DESCRIPTION)
    if (NOT DEFINED ${FLAG})
        message(STATUS "${FLAG}: ${DESCRIPTION}")
    else ()
        # i.e. `-DFOO="-a -b"` becomes CMake's semicolon separated list `FOO=`-a;-b`
        separate_arguments(${FLAG})
    endif ()
endmacro()

# hint common extra flag options for all models if they are not set
hint_flag(CXX_EXTRA_FLAGS "
        Appends to common compile flags. These will be appended at link phase as well.
        To use separate flags at link phase, set `CXX_EXTRA_LINK_FLAGS`")
hint_flag(CXX_EXTRA_LINK_FLAGS "
        Appends to link flags which appear *before* the objects.
        Do not use this for linking libraries, as the link line is order-dependent")
hint_flag(CXX_EXTRA_LIBRARIES "
        Append to link flags which appear *after* the objects.
        Use this for linking extra libraries (e.g `-lmylib`, or simply `mylib`)")
hint_flag(CXX_EXTRA_LINKER_FLAGS "
        Append to linker flags (i.e GCC's `-Wl` or equivalent)")

# copy CXX_EXTRA_FLAGS <- CXX_EXTRA_LINK_FLAGS
if ((DEFINED CXX_EXTRA_FLAGS) AND (NOT DEFINED CXX_EXTRA_LINK_FLAGS))
    set(CXX_EXTRA_LINK_FLAGS ${CXX_EXTRA_FLAGS})
endif ()

option(USE_TBB "Enable oneTBB library for *supported* models. Enabling this on models that
                don't explicitly link against TBB is a no-op, see description of your selected
                model on how this is used." OFF)

option(FETCH_TBB "Fetch (download) the oneTBB library for *supported* models. This uses CMake's
                  FetchContent feature. Specify version by setting FETCH_TBB_VERSION" OFF)
set(FETCH_TBB_VERSION "v2021.10.0" CACHE STRING "Specify version of oneTBB to use if FETCH_TBB is ON")

if (FETCH_TBB)
    FetchContent_Declare(
            TBB
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG "${FETCH_TBB_VERSION}"
    )
    # Don't fail builds on waring (TBB has -Wall while not being free of warnings from unused symbols...)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(TBB_STRICT OFF)
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(TBB)
    if (NOT TBB_POPULATED)
        FetchContent_Populate(TBB)
        add_subdirectory(${tbb_SOURCE_DIR} ${tbb_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()

option(USE_ONEDPL "Enable oneDPL library for *supported* models. Enabling this on models that
                   don't explicitly link against DPL is a no-op, see description of your selected
                   model on how this is used." OFF)

option(FETCH_ONEDPL "Fetch (download) the oneDPL library for *supported* models. This uses CMake's
                  FetchContent feature. Specify version by setting FETCH_ONEDPL_VERSION" OFF)
set(FETCH_ONEDPL_VERSION "oneDPL-2022.2.0-rc1" CACHE STRING "Specify version of oneTBB to use if FETCH_ONEDPL is ON")

if (FETCH_ONEDPL)
    FetchContent_Declare(
            oneDPL
            GIT_REPOSITORY https://github.com/oneapi-src/oneDPL.git
            GIT_TAG "${FETCH_ONEDPL_VERSION}"
    )
    string(TOLOWER ${USE_ONEDPL} ONEDPL_BACKEND)
    # XXX oneDPL looks for omp instead of openmp, which mismatches(!) with ONEDPL_PAR_BACKEND if using find_package
    if (ONEDPL_BACKEND STREQUAL "openmp")
        set(ONEDPL_BACKEND omp)
    endif ()
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(oneDPL)
    if (NOT oneDPL_POPULATED)
        FetchContent_Populate(oneDPL)
        if (USE_TBB)
            macro(find_package NAME)
                if ("${NAME}" STREQUAL "TBB")
                    message(STATUS "Discarding oneDPL's call to find_package(${NAME} ${ARGN})")
                else ()
                    _find_package(${NAME} ${ARGN})
                endif ()
            endmacro()
        endif ()
        add_subdirectory(${onedpl_SOURCE_DIR} ${onedpl_BINARY_DIR} EXCLUDE_FROM_ALL)

        # Fixup oneDPL's omission on setting DPCPP definitions.
        # We do this after the creation of the oneDPL target.
        if (ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
            target_compile_definitions(oneDPL INTERFACE ONEDPL_USE_DPCPP_BACKEND=1)
        endif ()

    endif ()
endif ()

option(USE_CPU_FEATURES "Enable the cpu_feature library for host CPU detection" OFF)


if (USE_CPU_FEATURES)
    FetchContent_Declare(
            cpu_features
            URL https://github.com/google/cpu_features/archive/refs/tags/v0.6.0.tar.gz
    )

    FetchContent_MakeAvailable(cpu_features)
    list(APPEND LINK_LIBRARIES cpu_features)
    list(APPEND IMPL_DEFINITIONS USE_CPU_FEATURES)
endif ()

# include our macros
include(cmake/register_models.cmake)

# register out models <model_name> <preprocessor_def_name> <source files...>
register_model(serial SERIAL fasten.hpp)
register_model(omp OMP fasten.hpp)
register_model(ocl OCL fasten.hpp)
register_model(std-indices STD_INDICES fasten.hpp)
register_model(std-ranges STD_RANGES fasten.hpp) # TODO
register_model(hip HIP fasten.hpp)
register_model(cuda CUDA fasten.hpp)
register_model(kokkos KOKKOS fasten.hpp)
register_model(sycl SYCL fasten.hpp)
register_model(acc ACC fasten.hpp)
# defining RAJA collides with the RAJA namespace so USE_RAJA
register_model(raja USE_RAJA fasten.hpp)
register_model(tbb TBB fasten.hpp)
register_model(thrust THRUST fasten.hpp) # TODO

set(USAGE ON CACHE BOOL "Whether to print all custom flags for the selected model")

message(STATUS "Available models:  ${REGISTERED_MODELS}")
if (NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL is unspecified, pick one from the available models")
else ()
    string(TOLOWER ${MODEL} MODEL)
    message(STATUS "Selected model  :  ${MODEL}")
endif ()

# load the $MODEL.cmake file and setup the correct IMPL_* based on $MODEL
load_model(${MODEL})

if (USAGE)
    # print the usage of the registered flag options
    registered_flags_action(print RESULT)
    message(STATUS "${RESULT}")
endif ()
# check required/set default for all registered flag options
registered_flags_action(check RESULT)
message(STATUS "${RESULT}")

list(APPEND IMPL_SOURCES src/main.cpp)

# run model specific setup, i.e append build flags, etc
setup()

# CMake insists that -O2 (or equivalent) is the universally accepted optimisation level
# we remove that here and use our own <BUILD_TYPE>_FLAGS
if (CMAKE_CXX_FLAGS_${BUILD_TYPE})
    wipe_gcc_style_optimisation_flags(CMAKE_CXX_FLAGS_${BUILD_TYPE})
endif ()

message(STATUS "Default ${CMAKE_BUILD_TYPE} flags are `${DEFAULT_${BUILD_TYPE}_FLAGS}`, set ${BUILD_TYPE}_FLAGS to override (CXX_EXTRA_* flags are not affected)")


# setup common build flag defaults if there are no overrides
if (NOT DEFINED ${BUILD_TYPE}_FLAGS)
    set(ACTUAL_${BUILD_TYPE}_FLAGS ${DEFAULT_${BUILD_TYPE}_FLAGS})
else ()
    set(ACTUAL_${BUILD_TYPE}_FLAGS ${${BUILD_TYPE}_FLAGS})
endif ()

set(DEFAULT_PPWI "1,2,4,8,16,32,64,128")

if (NOT PPWI)
    message(STATUS "PPWI not set, defaulting to ${DEFAULT_PPWI}")
    set(PPWI ${DEFAULT_PPWI})
endif ()


message(STATUS "CXX vendor  : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "Platform    : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "PPWI        : ${PPWI}")
message(STATUS "Sources     : ${IMPL_SOURCES}")
message(STATUS "Libraries   : ${LINK_LIBRARIES} ${CXX_EXTRA_LIBRARIES}")
message(STATUS "CXX Flags   : ${CMAKE_CXX_FLAGS_${BUILD_TYPE}} ${ACTUAL_${BUILD_TYPE}_FLAGS} ${CXX_EXTRA_FLAGS}
    CXX flags derived from (CMake + (Override ? Override : Default) + Extras), where:
        CMake                    = `${CMAKE_CXX_FLAGS_${BUILD_TYPE}}`
        Default                  = `${DEFAULT_${BUILD_TYPE}_FLAGS}`
        Override (${BUILD_TYPE}_FLAGS) = `${${BUILD_TYPE}_FLAGS}`
        Extras (CXX_EXTRA_FLAGS) = `${CXX_EXTRA_FLAGS}`")
message(STATUS "Link Flags  : ${LINK_FLAGS} ${CXX_EXTRA_LINK_FLAGS}")
message(STATUS "Linker Flags: ${CMAKE_EXE_LINKER_FLAGS} ${CXX_EXTRA_LINKER_FLAGS} ")
message(STATUS "Defs        : ${IMPL_DEFINITIONS}")
message(STATUS "Executable  : ${BIN_NAME}")

# below we have all the usual CMake target setup steps


include_directories(${CMAKE_BINARY_DIR}/generated)
add_executable(${EXE_NAME} ${IMPL_SOURCES})
target_link_libraries(${EXE_NAME} PUBLIC ${LINK_LIBRARIES})
target_compile_definitions(${EXE_NAME} PUBLIC ${IMPL_DEFINITIONS})

if (CXX_EXTRA_LIBRARIES)
    target_link_libraries(${EXE_NAME} PUBLIC ${CXX_EXTRA_LIBRARIES})
endif ()

target_compile_options(${EXE_NAME} PUBLIC "$<$<CONFIG:Release>:${ACTUAL_RELEASE_FLAGS};${CXX_EXTRA_FLAGS}>")
target_compile_options(${EXE_NAME} PUBLIC "$<$<CONFIG:Debug>:${ACTUAL_DEBUG_FLAGS};${CXX_EXTRA_FLAGS}>")

target_link_options(${EXE_NAME} PUBLIC LINKER:${CXX_EXTRA_LINKER_FLAGS})
target_link_options(${EXE_NAME} PUBLIC ${LINK_FLAGS} ${CXX_EXTRA_LINK_FLAGS})

# setup git_watcher...
set(PRE_CONFIGURE_FILE "${CMAKE_SOURCE_DIR}/src/meta_vcs.h.in")
set(POST_CONFIGURE_FILE "${CMAKE_BINARY_DIR}/generated/meta_vcs.h")

set(GIT_FAIL_IF_NONZERO_EXIT FALSE) # Dont' fail the build because of VCS; use FALSE here because git_watcher says so
include("${CMAKE_SOURCE_DIR}/cmake/git_watcher.cmake")
add_dependencies(${EXE_NAME} check_git)


# some models require the target to be already specified so they can finish their setup here
# this only happens if the model.cmake definition contains the `setup_target` macro
if (COMMAND setup_target)
    setup_target(${EXE_NAME})
endif ()

# see https://gitlab.kitware.com/cmake/cmake/-/issues/17379
if (CMAKE_CUDA_COMPILER MATCHES "nvcc$")
    string(REGEX REPLACE "\\," "\\\\," PPWI "${PPWI}")
endif ()

target_compile_definitions(${EXE_NAME} PRIVATE USE_PPWI=${PPWI})

if ((CMAKE_GENERATOR MATCHES "Unix Makefiles") OR (CMAKE_GENERATOR MATCHES "Ninja"))
    add_custom_target(extract_compile_commands ALL
            COMMAND ${CMAKE_COMMAND}
            -DCOMMAND_FILE=${CMAKE_BINARY_DIR}/compile_commands.json
            -DPRE_CONFIGURE_FILE="${CMAKE_SOURCE_DIR}/src/meta_build.h.in"
            -DPOST_CONFIGURE_FILE="${CMAKE_BINARY_DIR}/generated/meta_build.h"
            -DSRC_DIR_PREFIX="${CMAKE_SOURCE_DIR}/src"
            -DOUT_DIR_PREFIX="CMakeFiles/${EXE_NAME}.dir"
            -P ${CMAKE_SOURCE_DIR}/cmake/extract_compile_command.cmake
    )
    add_dependencies(${EXE_NAME} extract_compile_commands)
else ()
    set(COMPILE_COMMANDS COMPILE_FLAGS)
endif ()

set_target_properties(${EXE_NAME} PROPERTIES OUTPUT_NAME "${BIN_NAME}")

install(TARGETS ${EXE_NAME} DESTINATION bin)
