# Contributing

## Commit Messages

When committing to this repository, prefix your commit messages with the implementation(s) affected.
When changing top-level files, a prefix is not necessary The following are examples of acceptable
commit messaged:

    openmp: Code change
    openmp, opencl: Code change
    all: Switch to CMake
    data: Data change
    Update README

## Compiler and Platform Support

Each implementation's `model.cmake` should detail the compilers supported and any platform
restrictions. For non C/C++ based projects (i.e. not managed by CMake, such as Julia), a `README.md`
detailing steps to compile and run the implementation should be included. 

