# miniBUDE OpenMP

This is a CPU implementation of miniBUDE using OpenMP 3.

## Building

Select the compiler you want to use and pass it to Make:

    make COMPILER=GNU

The supported compilers names are: `ARM`, `CLANG`, `CRAY`, `GNU`, `INTEL`.

### Target Architecture

By default, the native architecture is targetted, but this can be changed by setting the `ARCH` parameter to the name of the target processor.
For example, the following are both valid:

    make COMPILER=GNU ARCH=thunderx2t99
    make COMPILER=GNU ARCH=skylake-avx512

### Block Sizes

This implementation includes a tunable block size similar to OpenCL workgroups.
The default value is `64`, which is suitable for 512-bit vectors.
Intel platforms benefit from more unrolling, so the default on x86 is `256`.
For 128-bit vectors, e.g. in ThunderX2, `16` is a good choice.

This parameter can be set using the `WGSIZE` parameter, as follows:

    make COMPILER=GNU ARCH=thunderx2t99 WGSIZE=16

For AVX-512 targets, the 512-bit registers (`zmm`) are used by default, because this increases performance.
To disable this and fall back to the compiler's default, which is 256-bit vectors as of Cascade Lake, set `AVX512` to the empty string:

    make COMPILER=GNU ARCH=skylake-avx512 AVX512=''


## Running

This implementation has no special run-time options.
The `-n` and `-i` parameters are available, and the number of threads should be set through the `OMP_NUM_THREADS` environment variable.
Run `bude -h` for a help message.
