# BUDE Benchmark

This repository contains implementation of the core computation of the Bristol University Docking Engine (BUDE) in different HPC programming models.
The benchmark is a virtual screening run of the NDM-1 protein and runs the energy evaluation for a single generation of poses repeatedly, for a configurable number of iterations.
Increasing the iteration count has similar performance effects to docking multiple ligands back-to-back in a production BUDE docking run.

## Structure

The top-level `data` directory contains the input common to implementations.
The top-level `makedeck` directory contains an input deck generation program and a set of mol2/bhff input files.
Each other subdirectory contains a separate implementation:

- [OpenMP](openmp/) for CPUs
- [OpenMP target](openmp-target/) for GPUs
- [OpenCL](opencl/) for GPUs
- [CUDA](cuda/) for GPUs
- [OpenACC](openacc/) for GPUs

## Building

To build with the default options, type `make` in an implementation directory.
There are options to choose the compiler used and the architecture targetted.

Refer to each implementation's README for further build instructions.

## Running

To run with the default options, run the binary without any flags.
To adjust the run time, use `-i` to set the number of iterations.
For very short runs, e.g. for simulation, use set `-n 1024` to reduce the number of poses.

Refer to each implementation's README for further run instructions.
