register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that supports OpenMP as per CMake detection"
        "c++")

macro(setup)
    find_package(OpenMP REQUIRED)
    register_link_library(OpenMP::OpenMP_CXX)

    # propagate flags to linker so that it links with the offload stuff as well
    register_append_cxx_flags(ANY ${OMP_FLAGS})
    if (OFFLOAD_APPEND_LINK_FLAG)
        register_append_link_flags(${OMP_FLAGS})
    endif ()
endmacro()

