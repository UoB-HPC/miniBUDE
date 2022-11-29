# miniBUDE Chapel

This is an implementation of miniBUDE using Chapel.

## Building

Prerequisites

 * Chapel >= 1.28

### Block Sizes

This implementation includes a tunable block size similar to OpenCL workgroups.
The default value is `64`, which is suitable for 512-bit vectors.
Intel platforms benefit from more unrolling, so the default on x86 is `256`.

This parameter can be set using the `WGSIZE` parameter, as follows:

    make WGSIZE=16

For AVX-512 targets, the 512-bit registers (`zmm`) are used by default, because this increases performance.
To disable this and fall back to the compiler's default, which is 256-bit vectors as of Cascade Lake, set `AVX512` to the empty string:

    make COMPILER=GNU ARCH=skylake-avx512 AVX512=''


## Running

This implementation has no special run-time options.
The `-n` and `-i` parameters are available, and the number of threads should be set through the `CHPL_RT_NUM_THREADS_PER_LOCALE` environment variable.
Run `bude -h` for a help message.
