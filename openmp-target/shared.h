#ifndef __SHARED
#define __SHARED

#define NUM_TD_PER_THREAD 4

typedef struct
{
    float x, y, z;
    int   index;
} Atom;

typedef struct
{
    int   hbtype;
    float radius;
    float hphb;
    float elsc;
} FFParams;

typedef struct 
{
    Atom* d_protein;
    Atom* d_ligand;
    FFParams* d_forcefield;
    float* d_poses[6];
    float* d_results;

    int deviceIndex;
    int wgsize;
    int posesPerWI;
} Cuda;

typedef struct
{
    int    natlig;
    int    natpro;
    int    ntypes;
    int    nposes;
    Atom     * protein;
    Atom     * ligand;
    FFParams * forcefield;
    float    * poses[6];
    char     * deckDir;
    int iterations;
} Params;

extern Cuda _cuda;
extern Params params;

#ifdef __cplusplus
extern "C"
#endif 
double getTimestamp();

#ifdef __cplusplus
extern "C"
#endif
void printTimings(double start, double end, double poses_per_wi);

#endif
