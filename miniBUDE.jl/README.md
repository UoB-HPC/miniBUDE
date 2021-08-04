miniBUDE Julia
==============

This is an implementation of miniBUDE in Julia which contains the following variants:

 * `Threaded.jl` - Threaded implementation with `Threads.@threads` macros
 * `CUDA.jl` - Direct port of miniBUDE native CUDA implementation using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
 * `AMDGPU.jl` - Direct port of miniBUDE's native HIP(via CUDA) implementation using [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)
 * `oneAPi.jl` - Direct port of miniBUDE's native SYCL implementation using [oneAPi.jl](https://github.com/JuliaGPU/oneAPI.jl)
 * `KernelAbstractions.jl` - Direct port of miniBUDE's native CUDA implementation using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)

### Build & Run

Prerequisites

 * Julia >= 1.6+

With Julia on path, run the benchmark with:

```shell
> cd JuliaStream.jl
> julia --project -e 'import Pkg; Pkg.instantiate()' # only required on first run
> julia --project src/<IMPL>.jl # e.g `julia --project src/KernelAbstractions.jl`, see variants listed above.
```

**Important:**
 * Julia is 1-indexed, so N >= 1 in `--device N` (alternatively, use device name substring).
 * Thread count for `Threaded.jl` must be set via the `JULIA_NUM_THREADS` environment variable (e.g `export JULIA_NUM_THREADS=$(nproc)`) otherwise it defaults to 1.
 * Certain implementations such as CUDA and AMDGPU will do hardware detection at runtime and may download and/or compile further software packages for the platform.
