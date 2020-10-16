# BUDE OpenMP

This is a CPU implementation of BUDE using OpenMP 3.

## Building

Select the compiler you want to use and pass it to Make:

```
make COMPILER=GNU
```

The supported compilers names are: `ARM`, `CLANG`, `CRAY`, `GNU`, `INTEL`.

### Target Architecture

By default, the native architecture is targetted, but this can be changed by setting the `ARCH` parameter to the name of the target processor.
For example, the following are both valid:

```
make COMPILER=GNU ARCH=thunderx2t99
make COMPILER=GNU ARCH=skylake-avx512
```

### Block Sizes

This implementation includes a tunable block size similar to OpenCL workgroups.
The default value is `64`, which is suitable for 512-bit vectors, e.g. in Skylake or A64FX, but higher values may sometimes be beneficial.
For 128-bit vectors, `16` is a good choice.

This parameter can be set using the `WGSIZE` parameter, as follows:

```
make COMPILER=GNU ARCH=thunderx2t99 WGSIZE=16
```

## Running

This implementation has no special run-time options.
The `-n` and `-i` parameters are available, and the number of threads should be set through the `OMP_NUM_THREADS` environment variable.
Run `bude-openmp -h` for a help message.
