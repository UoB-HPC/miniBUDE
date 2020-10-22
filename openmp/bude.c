#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "bude.h"

struct
{
  int    natlig;
  int    natpro;
  int    ntypes;
  int    nposes;
  Atom     *restrict protein;
  Atom     *restrict ligand;
  FFParams *restrict forcefield;
  float    *restrict poses[6];

  int iterations;
} params = {0};

double   getTimestamp();
void     loadParameters(int argc, char *argv[]);
void     freeParameters();
void     printTimings(double start, double end, double poses_per_wi);

void     runOpenMP(float *energies);

void fasten_main(const int natlig,
                 const int natpro,
                 const Atom *restrict protein,
                 const Atom *restrict ligand,
                 const float *restrict transforms_0,
                 const float *restrict transforms_1,
                 const float *restrict transforms_2,
                 const float *restrict transforms_3,
                 const float *restrict transforms_4,
                 const float *restrict transforms_5,
                 float *restrict results,
                 const FFParams *restrict forcefield,
                 const int group);

int main(int argc, char *argv[])
{
  loadParameters(argc, argv);
  printf("\nPoses:      %d\n", params.nposes);
  printf("Iterations: %d\n", params.iterations);

  float *energiesOMP = calloc(params.nposes, sizeof(float));

  runOpenMP(energiesOMP);


  FILE *output = fopen("energies.dat", "w");

  // Print some energies
  printf("\nEnergies\n");
  for (int i = 0; i < params.nposes; i++)
  {
    fprintf(output, "%f\n", energiesOMP[i]);
    if (i < 16)
      printf("%7.2f\n", energiesOMP[i]);
  }
  printf("\n");

  fclose(output);

  // Validate energies
  FILE* ref_energies = fopen(FILE_REF_ENERGIES, "r");
  float e, diff, maxdiff = -100.0f;
  size_t n_ref_poses = params.nposes;
  if (params.nposes > REF_NPOSES) {
    printf("Only validating the first %d poses.\n", REF_NPOSES);
    n_ref_poses = REF_NPOSES;
  }

  for (size_t i = 0; i < n_ref_poses; i++)
  {
    fscanf(ref_energies, "%f", &e);
    if (fabs(e) < 1.f && fabs(energiesOMP[i])< 1.f) continue;

    diff = fabs(e - energiesOMP[i]) / e;
    if (diff > maxdiff) maxdiff = diff;
  }
  printf("Largest difference was %.3f%%.\n\n", 100*maxdiff); // Expect numbers to be accurate to 2 decimal places
  fclose(ref_energies);

  free(energiesOMP);
  freeParameters();
}

void runOpenMP(float *restrict results)
{
  printf("\nRunning C/OpenMP\n");

  double start = getTimestamp();

#pragma omp parallel
  for (int itr = 0; itr < params.iterations; itr++)
  {
#pragma omp for
    for (unsigned group = 0; group < (params.nposes/WGSIZE/PPWI); group++)
    {
      fasten_main(params.natlig, params.natpro, params.protein, params.ligand,
                  params.poses[0], params.poses[1], params.poses[2],
                  params.poses[3], params.poses[4], params.poses[5],
                  results, params.forcefield, group);
    }
  }

  double end = getTimestamp();

  printTimings(start, end, PPWI);
}

FILE* openFile(const char *name, long *length)
{
  FILE *file = NULL;
  if (!(file = fopen(name, "rb")))
  {
    fprintf(stderr, "Failed to open '%s'\n", name);
    exit(1);
  }

  fseek(file, 0, SEEK_END);
  *length = ftell(file);
  rewind(file);

  return file;
}

int parseInt(const char *str)
{
  char *next;
  int value = strtoul(str, &next, 10);
  return strlen(next) ? -1 : value;
}

void loadParameters(int argc, char *argv[])
{
  // Defaults
  params.iterations = DEFAULT_ITERS;
  int nposes        = DEFAULT_NPOSES;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || (params.iterations = parseInt(argv[i])) < 0)
      {
        printf("Invalid number of iterations\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--numposes") || !strcmp(argv[i], "-n"))
    {
      if (++i >= argc || (nposes = parseInt(argv[i])) < 0)
      {
        printf("Invalid number of poses\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./bude [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print this message\n");
      printf("  -i  --iterations I       Repeat kernel I times (default: %d)\n", DEFAULT_ITERS);
      printf("  -n  --numposes   N       Compute energies for N poses (default: %d)\n", DEFAULT_NPOSES);
      printf("\n");
      exit(0);
    }
    else
    {
      printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
      exit(1);
    }
  }

  FILE *file = NULL;
  long length;

  file = openFile(FILE_LIGAND, &length);
  params.natlig = length / sizeof(Atom);
  params.ligand = malloc(params.natlig*sizeof(Atom));
  fread(params.ligand, sizeof(Atom), params.natlig, file);
  fclose(file);

  file = openFile(FILE_PROTEIN, &length);
  params.natpro = length / sizeof(Atom);
  params.protein = malloc(params.natpro*sizeof(Atom));
  fread(params.protein, sizeof(Atom), params.natpro, file);
  fclose(file);

  file = openFile(FILE_FORCEFIELD, &length);
  params.ntypes = length / sizeof(FFParams);
  params.forcefield = malloc(params.ntypes*sizeof(FFParams));
  fread(params.forcefield, sizeof(FFParams), params.ntypes, file);
  fclose(file);

  file = openFile(FILE_POSES, &length);
  for (int i = 0; i < 6; i++)
    params.poses[i] = malloc(nposes*sizeof(float));

  long available = length / 6 / sizeof(float);
  params.nposes = 0;
  while (params.nposes < nposes)
  {
    long fetch = nposes - params.nposes;
    if (fetch > available)
      fetch = available;

    for (int i = 0; i < 6; i++)
    {
      fseek(file, i*available*sizeof(float), SEEK_SET);
      fread(params.poses[i] + params.nposes, sizeof(float), fetch, file);
    }
    rewind(file);

    params.nposes += fetch;
  }
  fclose(file);
}

void freeParameters()
{
  free(params.ligand);
  free(params.protein);
  free(params.forcefield);
  for (int i = 0; i < 6; i++)
    free(params.poses[i]);
}

void printTimings(double start, double end, double poses_per_wi)
{
  double ms = ((end-start)/params.iterations)*1e-3;

  // Compute FLOP/s
  double runtime   = ms*1e-3;
  double ops_per_wi = 27*poses_per_wi
    + params.natlig*(3 + 18*poses_per_wi + params.natpro*(11 + 30*poses_per_wi))
    + poses_per_wi;
  double total_ops     = ops_per_wi * (params.nposes/poses_per_wi);
  double flops      = total_ops / runtime;
  double gflops     = flops / 1e9;

  double total_finsts = 25.0 * params.natpro * params.natlig * params.nposes;
  double finsts = total_finsts / runtime;
  double gfinsts = finsts / 1e9;

  double interactions         =
      (double)params.nposes
    * (double)params.natlig
    * (double)params.natpro;
  double interactions_per_sec = interactions / runtime;

  // Print stats
  printf("- Total time:     %7.3lf ms\n", (end-start)*1e-3);
  printf("- Average time:   %7.3lf ms\n", ms);
  printf("- Interactions/s: %7.3lf billion\n", (interactions_per_sec / 1e9));
  printf("- GFLOP/s:        %7.3lf\n", gflops);
  printf("- GFInst/s:       %7.3lf\n", gfinsts);
}

double getTimestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec*1e6;
}
