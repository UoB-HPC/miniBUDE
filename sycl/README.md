# miniBUDE SYCL

This is a CPU implementation of miniBUDE using SYCL.

## Building

This implementation uses CMake.
First, generate a build:

    cmake3 -Bbuild -H. -DCMAKE_BUILD_TYPE=Release -DSYCL_RUNTIME=COMPUTECPP -DComputeCpp_DIR=<path_to_computecpp> -DOpenCL_INCLUDE_DIR=include/

Flags:

* `SYCL_RUNTIME`: one of `HIPSYCL|HIPSYCL-NEXT|COMPUTECPP|DPCPP`
  * For `SYCL_RUNTIME=HIPSYCL`, supply hipSYCL (versions before 38bc08d) install path with `HIPSYCL_INSTALL_DIR`
  * For `SYCL_RUNTIME=HIPSYCL-NEXT`, supply hipSYCL (versions after 38bc08d) install path with `HIPSYCL_INSTALL_DIR`
  * For `SYCL_RUNTIME=COMPUTECPP`, supply ComputeCpp install path with `ComputeCpp_DIR`
  * For `SYCL_RUNTIME=DPCPP`, make sure the DPC++ compiler (`dpcpp`) is available in `PATH`
* `CXX_EXTRA_FLAGS`: `STRING`, appends extra flags that will be passed on to the compiler, applies to all configs
* `CXX_EXTRA_LINKER_FLAGS`: `STRING`, appends extra linker flags (the comma separated list after the `-Wl` flag) to the linker; applies to all configs

**IMPORTANT:** you must delete your CMake `build` directory if you change `SYCL_RUNTIME` otherwise changes may not be picked up.

If parts of your toolchain are installed at different places, you'll have to specify it manually, for example:

    cmake3 -Bbuild -H.  \
    -DSYCL_RUNTIME=COMPUTECPP \
    -DComputeCpp_DIR=/nfs/software/x86_64/computecpp/1.1.3 \
    -DCMAKE_C_COMPILER=/nfs/software/x86_64/gcc/9.1.0/bin/gcc \
    -DCMAKE_CXX_COMPILER=/nfs/software/x86_64/gcc/9.1.0/bin/g++ \
    -DCMAKE_BUILD_TYPE=Release


Proceed with compiling:

    cmake3 --build build --target bude --config Release -j $(nproc)

The binary can be found at `build/bude`.

### Target Architecture

By default, the native architecture is targeted via `-march=native`, but this can be changed by setting the appropriate flags in `CXX_EXTRA_FLAGS`.
For example, the following are both valid:

    -DCXX_EXTRA_FLAGS="-march=znver2 ... "
    -DCXX_EXTRA_FLAGS="-march=skylake-avx512 ... "

### Block Sizes

This implementation includes a tunable block size similar to OpenCL workgroups.
The default value is `64`, which is suitable for 512-bit vectors, e.g. in Skylake or A64FX, but higher values may sometimes be beneficial.
For 128-bit vectors, `16` is a good choice.

This parameter can be set at runtime, use `--help` to find out how the block size can be changed.

Using block size of `0` will invoke the simple `parallel_for` with `range<1>` instead of `nd_range<1>`.

## Running

This implementation has no special run-time options.
The `-n` and `-i` parameters are available.
Run `bude -h` for a help message.
