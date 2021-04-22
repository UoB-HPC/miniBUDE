# miniBUDE OpenACC

This is a GPU implementation of miniBUDE using OpenACC.

## Building

Select the compiler you want to use and pass it to Make:

```
make COMPILER=GNU
```

The supported compilers names are: `CRAY`, `GNU`, `PGI`.

## Running

This implementation has no special run-time options.
The `-n` and `-i` parameters are available.
Run `bude -h` for a help message.
