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

A set of reduced dependency projects are available for the following backend and implementations:

 * `AMDGPU` supports:
   - `AMDGPU.jl`
 * `CUDA` supports:
   - `CUDA.jl`
 * `oneAPI` supports:
   - `oneAPIStream.jl`  
 * `KernelAbstractions` supports:
   - `KernelAbstractions.jl`
 * `Threaded` supports:
   - `ThreadedStream.jl`
    
With Julia on path, run the benchmark with:

```shell
> cd JuliaStream.jl
> julia --project=<BACKEND> -e 'import Pkg; Pkg.instantiate()' # only required on first run
> julia --project=<BACKEND> src/<IMPL>.jl
```

For example. to run the CUDA implementation:

```shell
> cd JuliaStream.jl
> julia --project=CUDA -e 'import Pkg; Pkg.instantiate()' 
> julia --project=CUDA src/CUDA.jl
```

**Important:**
 * Julia is 1-indexed, so N >= 1 in `--device N` (alternatively, use device name substring).
 * Thread count for `Threaded.jl` must be set via the `JULIA_NUM_THREADS` environment variable (e.g `export JULIA_NUM_THREADS=$(nproc)`) otherwise it defaults to 1.
 * Certain implementations such as CUDA and AMDGPU will do hardware detection at runtime and may download and/or compile further software packages for the platform.
 * If Julia is launched behind some sort of launcher (e.g `aprun`), you may need to specify the `-H` option pointing to Julia's bin directory, so that Julia can find the correct libraries at runtime, for example: `aprun -n 1 -d 64 julia -H "$(dirname "$(which julia)")" ... `
***

Alternatively, the top-level project `Project.toml` contains all dependencies needed to run all implementations in `src`.
There may be instances where some packages are locked to an older version because of transitive dependency requirements.

To run the benchmark using the top-level project, run the benchmark with:
```shell
> cd JuliaStream.jl
> julia --project -e 'import Pkg; Pkg.instantiate()'  
> julia --project src/<IMPL>.jl
```
