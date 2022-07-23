# miniBUDE

This mini-app is an implementation of the core computation of the Bristol University Docking Engine (BUDE) in different HPC programming models.
The benchmark is a virtual screening run of the NDM-1 protein and runs the energy evaluation for a single generation of poses repeatedly, for a configurable number of iterations.
Increasing the iteration count has similar performance effects to docking multiple ligands back-to-back in a production BUDE docking run.

## Structure

The top-level `data` directory contains the input common to implementations.
The top-level `makedeck` directory contains an input deck generation program and a set of mol2/bhff input files.
Each other subdirectory in `src` contains a separate C/C++ implementation.

## Building

Drivers, compiler and software applicable to whichever implementation you would like to build against is required.

### CMake

The project supports building with CMake >= 3.14.0, which can be installed without root via the [official script](https://cmake.org/download/).

Each miniBUDE implementation (programming model) is built as follows:

```shell
$ cd miniBUDE

# configure the build, build type defaults to Release
# The -DMODEL flag is required
$ cmake -Bbuild -H. -DMODEL=<model> <model specific flags prefixed with -D...>

# compile
$ cmake --build build

# run executables in ./build
$ ./build/<model>-bude
```

The `MODEL` option selects one implementation of miniBUDE to build.
The source for each model's implementations are located in `./src/<model>`.

Currently available models are:
```
omp;ocl;std-indices;std-ranges;hip;cuda;kokkos;sycl;acc;raja;tbb;thrust
```

#### Overriding default flags
By default, we have defined a set of optimal flags for known HPC compilers.
There are assigned those to `RELEASE_FLAGS`, and you can override them if required.

To find out what flag each model supports or requires, simply configure while only specifying the model.
For example:
```shell
> cd miniBUDE
> cmake -Bbuild -H. -DMODEL=omp 
No CMAKE_BUILD_TYPE specified, defaulting to 'Release'
-- CXX_EXTRA_FLAGS: 
        Appends to common compile flags. These will be appended at link phase as well.
        To use separate flags at link phase, set `CXX_EXTRA_LINK_FLAGS`
-- CXX_EXTRA_LINK_FLAGS: 
        Appends to link flags which appear *before* the objects.
        Do not use this for linking libraries, as the link line is order-dependent
-- CXX_EXTRA_LIBRARIES: 
        Append to link flags which appear *after* the objects.
        Use this for linking extra libraries (e.g `-lmylib`, or simply `mylib`)
-- CXX_EXTRA_LINKER_FLAGS: 
        Append to linker flags (i.e GCC's `-Wl` or equivalent)
-- Available models:  omp;ocl;std-indices;std-ranges;hip;cuda;kokkos;sycl;acc;raja;tbb;thrust
-- Selected model  :  omp
-- Supported flags:

   CMAKE_CXX_COMPILER (optional, default=c++): Any CXX compiler that supports OpenMP as per CMake detection (and offloading if enabled with `OFFLOAD`)
   ARCH (optional, default=): This overrides CMake's CMAKE_SYSTEM_PROCESSOR detection which uses (uname -p), this is mainly for use with
         specialised accelerators only and not to be confused with offload which is is mutually exclusive with this.
         Supported values are:
          - NEC
   OFFLOAD (optional, default=OFF): Whether to use OpenMP offload, the format is <VENDOR:ARCH?>|ON|OFF.
        We support a small set of known offload flags for clang, gcc, and icpx.
        However, as offload support is rapidly evolving, we recommend you directly supply them via OFFLOAD_FLAGS.
        For example:
          * OFFLOAD=NVIDIA:sm_60
          * OFFLOAD=AMD:gfx906
          * OFFLOAD=INTEL
          * OFFLOAD=ON OFFLOAD_FLAGS=...
   OFFLOAD_FLAGS (optional, default=): If OFFLOAD is enabled, this *overrides* the default offload flags
   OFFLOAD_APPEND_LINK_FLAG (optional, default=ON): If enabled, this appends all resolved offload flags (OFFLOAD=<vendor:arch> or directly from OFFLOAD_FLAGS) to the link flags.
        This is required for most offload implementations so that offload libraries can linked correctly.


```

### Benchmarks

Two input decks are included in this repository:

* `bm1` is a short benchmark (~100 ms/iteration on a 64-core ThunderX2 node) based on a small ligand (26 atoms)
* `bm2` is a long benchmark (~25 s/iteration on a 64-core ThunderX2 node) based on a big ligand (2672 atoms)* `bm2` is a long benchmark (~25 s/iteration on a 64-core ThunderX2 node) based on a big ligand (2672 atoms)
* `bm2_long` is a very long benchmark based on `bm2` but with 1048576 poses instead of 65536

They are located in the [`data`](data/) directory, and `bm1` is run by default.
All implementations accept a `--deck` parameter to specify an input deck directory.
See [`makedeck`](makedeck/) for how to generate additional input decks.

## Citing

Please cite miniBUDE using the following reference:

> Andrei Poenaru, Wei-Chen Lin and Simon McIntosh-Smith. ‘A Performance Analysis of Modern Parallel Programming Models Using a Compute-Bound Application’. In: 36th International Conference, ISC High Performance 2021. Frankfurt, Germany, 2021. In press.
