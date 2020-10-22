# BUDE OpenMP Target

This is a GPU implementation of BUDE using OpenMP target.

## Building

This implementations supports the following compilers:

* CCE (Cray)
* LLVM 10+ (built with offloading support)

```
make
```

The default compiler and flags can be overridden using `CC` and `CFLAGS`, respectively.

### Target Architecture

By default, the NVIDIA V100 is targetted, but this can be changed by setting the `ARCH` parameter to the name of the target GPU architecture.

For example, to target a P100:

```
make ARCH=sm_60
```

## Running

This implementation has no special run-time options.
The `-n` and `-i` parameters are available.
Run `bude -h` for a help message.
