# miniBUDE OpenMP Target

This is a GPU implementation of miniBUDE using OpenMP target.

## Building

    make

This implementations supports the following compilers:

* Cray
* LLVM 10+ (built with offloading support)


The default compiler and flags can be overridden using `CC` and `CFLAGS`, respectively.

### Target Architecture

By default, the NVIDIA V100 is targetted, but this can be changed by setting the `ARCH` parameter to the name of the target GPU architecture.

For example, to target a P100:

    make ARCH=sm_60

This implementation has a tuning parameter that can be used to change the amount of work given to each GPU thread: `TD_PER_THREAD`.
Optimal values are automatically set for Pascal (`2`) and Volta (`4`), but on other platforms this may need setting manually:

    make ARCH=... TD_PER_THREAD=8

## Running

This implementation has no special run-time options.
The `-n` and `-i` parameters are available.
Run `bude -h` for a help message.
