#ifndef PPWI
#define PPWI 1
#endif
#ifndef WGSIZE
#define WGSIZE 4
#endif

#define DEFAULT_ITERS  8
#define DEFAULT_NPOSES 65536

#define DATA_DIR        "../data"
#define FILE_LIGAND     DATA_DIR "/ligand.dat"
#define FILE_PROTEIN    DATA_DIR "/protein.dat"
#define FILE_FORCEFIELD DATA_DIR "/forcefield.dat"
#define FILE_POSES      DATA_DIR "/poses.dat"

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
