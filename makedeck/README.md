# miniBUDE makedeck

This program generates input decks for miniBUDE from a set of mol2 and bhff files.
Sample mol2 and bhff files can be found in `raw/*`.

## Building

The program uses CMake and requires a C++17-conformant compiler with OpenMP support and the appropriate C++17 standard library.
For GCC, the first version supported is 9.3.0.

First, generate a build:

    cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=Release

And then proceed with compilation:

    cmake --build build --target makedeck --config Release

The binary will be located at `build/makedeck`.


## Running

To generate an input deck, simply specify all the required parameters as described in `--help`:

```
This program generates input decks for the bude benchmark program.
Usage: ./makedeck [OPTIONS]

Options:
  -h  --help         Print this message
  -f  --forcefield   Path to a forcefield bhff file
  -p  --protein      Path to a protein mol2 file
  -l  --ligand       Path to a ligand mol2 file
  -s  --pose-seed    The random seed used to generate pose combinations (default: 42)
  -n  --pose-length  The amount of poses to generate (default: 65536)
  -o  --out          The output directory (containing {protein,ligand,forcefield,poses}.dat,{input,energies}.txt) name of the decks
      --force        If specified, any file/directory that matches the output dir name will be deleted/overwritten

```


For example:

    ./makedeck -f heavy_by-residue_2016-v1.bhff -p scan_receptorSurfPoint00000099.mol2 -l scan_receptorSurfPoint00000099.mol2 -o output --force

The deck will be created in a directory named `output`.
It should contain the following files:

```
forcefield.in
ligand.in
poses.in
protein.in
params.txt # contains original program arguments used to generate this deck
ref_energies.out # contains reference energies used for verification
```

The input deck directory can then be specified in all miniBUDE implementations:

    ./bude --deck output

## Testing

This project contains unit tests.
First, create a debug build:

    cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=Debug

And then proceed with compilation:

    cmake --build build --target tests --config Debug

And then run the test binary `build/tests`.
