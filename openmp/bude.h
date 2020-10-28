#ifndef PPWI
#define PPWI 1
#endif
#ifndef WGSIZE
#define WGSIZE 4
#endif

#define DEFAULT_ITERS  8
#define DEFAULT_NPOSES 65536
#define REF_NPOSES     65536

#define DATA_DIR          "../data"
#define FILE_LIGAND       "/ligand.dat"
#define FILE_PROTEIN      "/protein.dat"
#define FILE_FORCEFIELD   "/forcefield.dat"
#define FILE_POSES        "/poses.dat"
#define FILE_REF_ENERGIES "/ref_energies.txt"

typedef struct
{
  float x, y, z;
  int   type;
} Atom;

typedef struct
{
  int   hbtype;
  float radius;
  float hphb;
  float elsc;
} FFParams;
