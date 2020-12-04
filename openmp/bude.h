#ifndef WGSIZE
#define WGSIZE 4
#endif

#define DEFAULT_ITERS  8
#define DEFAULT_NPOSES 65536
#define REF_NPOSES     65536

#define DATA_DIR          "../data/bm1"
#define FILE_LIGAND       "/ligand.in"
#define FILE_PROTEIN      "/protein.in"
#define FILE_FORCEFIELD   "/forcefield.in"
#define FILE_POSES        "/poses.in"
#define FILE_REF_ENERGIES "/ref_energies.out"

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
