
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection, this is used for host compilation"
        "c++")

register_flag_optional(MEM "Device memory mode:
        DEFAULT   - allocate host and device memory pointers.
        MANAGED   - use CUDA Managed Memory.
        PAGEFAULT - shared memory, only host pointers allocated."
        "DEFAULT")

register_flag_required(CMAKE_CUDA_COMPILER
        "Path to the CUDA nvcc compiler")

# XXX we may want to drop this eventually and use CMAKE_CUDA_ARCHITECTURES directly
register_flag_required(CUDA_ARCH
        "Nvidia architecture, will be passed in via `-arch=` (e.g `sm_70`) for nvcc")

register_flag_optional(CUDA_EXTRA_FLAGS
        "Additional CUDA flags passed to nvcc, this is appended after `CUDA_ARCH`"
        "")


macro(setup)

    # XXX CMake 3.18 supports CMAKE_CUDA_ARCHITECTURES/CUDA_ARCHITECTURES but we support older CMakes
    if (POLICY CMP0104)
        cmake_policy(SET CMP0104 OLD)
    endif ()

    enable_language(CUDA)
    register_definitions(MEM=${MEM})

    # add -forward-unknown-to-host-compiler for compatibility reasons
    # add -std=c++17 manually as older CMake seems to omit this (source gets treated as C otherwise)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -forward-unknown-to-host-compiler -arch=${CUDA_ARCH} -use_fast_math -restrict -keep ${CUDA_EXTRA_FLAGS}")

    # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG
    # appended later
    wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})

    # since we're doing a single-source w/ templated kernel build
    # everything is a CUDA file

    set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)


    message(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
endmacro()

