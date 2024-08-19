# Changelog
All notable changes to this project will be documented in this file.

## [v2.0] - 2024-08-19

### Added
- CI via GitHub Actions
- Support for decks with arbitrary pose count (was fixed to 65536 before)
- Kernel runtime summary statistics (min, max, sum, avg, stdDev)
- Partial auto-tune to find the best PPWI/WGSIZE combination
- PPWI/WGSIZE heatmap script 
- Flag for machine-readable CSV output
- Flag for toggling optional energy output to file
- Executable now embeds compile commands used at build-time
- New models: C++ std, C++20 std, Intel TBB, RAJA, Thrust, serial
- Context allocation and transfer (if required) is now measured
- Added optional `cpu_feature` library, can be disabled at build-time

### Changed
- Human-readable output now uses YAML format
- Renamed parameter `NUM_TD_PER_THREAD` to `PPWI` for all implementations
- Consolidated builds to use a shared CMake script, Makefiles removed
- All implementation now share a common C++ driver with device selection based on index or name substrings
- More robust input deck/parameter validation (bad params now gives reason at launch instead of SEGV mid-benchmark)
- Optimised deck IO for faster poses to memory
- OpenCL implementation now embeds kernel directly in executable
- OpenCL implementation now use the official [OpenCL C++ Bindings](https://github.khronos.org/OpenCL-CLHPP/)
- Kokkos implementation now supports team policies
- SYCL implementation now uses `discard_write` instead of `read_write` for storing energy results
- HIP implementation is now standalone and not derived from the CUDA via hipify
- Fused OpenMP and OpenMP target implementation via macros
- OpenMP/OpenMP target now supports the team directive as wgsize


### Fixed
- SYCL and Kokkos impl. not respecting runtime pose count flag

## [v1.0] - 2021-06-24

Initial public release.
