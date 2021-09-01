# miniBUDE Kokkos

This is an implementation of miniBUDE using Kokkos.

## Building

This implementation uses CMake.
First, generate a build:

```shell
> cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=Release
```

Flags:

* `CXX_EXTRA_FLAGS`: `STRING`, appends extra flags that will be passed on to the compiler, applies to all configs
* `CXX_EXTRA_LINKER_FLAGS`: `STRING`, appends extra linker flags (the comma separated list after the `-Wl` flag) to the linker; applies to all configs
* `KOKKOS_IN_TREE`: `STRING`, use a specific Kokkos **source** directory for an in-tree build where Kokkos and the project is compiled together.
  * `FORWARD_CXX_EXTRA_FLAGS_TO_KOKKOS` : `ON|OFF`, whether to forward `CXX_EXTRA_FLAGS` when building Kokkos. This is `OFF` by default as Kokkos has a set of tested flags for each compiler. 
* `Kokkos_ROOT`: `STRING`, path to the local Kokkos installation, this is optional and mutually exclusive with `KOKKOS_IN_TREE`.  
* `DEFAULT_WGSIZE`: `INGEGER`, sets the [block size](#block-size). Defaults to 64.
* `CUSTOM_SYSTEM_INCLUDE_FLAG`, sets the prefix flag for including system libraries. For example, setting this flag to `-I` replaces all `-isystem <headers...>` with `-I <headers...>`. 

Compilers can be specified via the usual CMake options, for example:

```shell
> cmake -Bbuild -H.  \
    -DCMAKE_C_COMPILER=/nfs/software/x86_64/gcc/9.1.0/bin/gcc \
    -DCMAKE_CXX_COMPILER=/nfs/software/x86_64/gcc/9.1.0/bin/g++ \
    -DCMAKE_BUILD_TYPE=Release
```

**IMPORTANT:** If need to specify [Kokkos flags](https://github.com/kokkos/kokkos/blob/master/BUILD.md#kokkos-keyword-listing) at build-time, you must use the `KOKKOS_IN_TREE` option. For example:

```shell
> cmake -Bbuild -H.  \
    -DKOKKOS_IN_TREE=<path_to_kokkos_src> \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ARCH_ZEN2=ON \
    -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -DCMAKE_BUILD_TYPE=Release 
```

Proceed with compiling:

```shell
> cmake --build build --target bude --config Release -j $(nproc)
```

The binary can be found at `build/bude`.


### Block Sizes

This implementation includes a tunable block size similar to OpenCL workgroups.
The default value is `64`, which is suitable for 512-bit vectors, e.g. in Skylake or A64FX, but higher values may sometimes be beneficial.
For 128-bit vectors, `16` is a good choice.


## Running

This implementation has no special run-time options.
The `-n` and `-i` parameters are available.
Run `bude -h` for a help message.
