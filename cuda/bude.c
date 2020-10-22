#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "shared.h"

#define MAX_PLATFORMS     8
#define MAX_DEVICES      32
#define MAX_INFO_STRING 256

#define DATA_DIR          "../data"
#define FILE_LIGAND       DATA_DIR "/ligand.dat"
#define FILE_PROTEIN      DATA_DIR "/protein.dat"
#define FILE_FORCEFIELD   DATA_DIR "/forcefield.dat"
#define FILE_POSES        DATA_DIR "/poses.dat"
#define FILE_REF_ENERGIES DATA_DIR "/ref_energies.dat"

#define REF_NPOSES 65536

// Energy evaluation parameters
#define CNSTNT   45.0f
#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST  5.5f
#define NPPDIST   1.0f

void loadParameters(int argc, char *argv[]);
void freeParameters();
void printTimings(double start, double end, double poses_per_wi);
void checkError(int err, const char *op);
void runCUDA(float* results);

int main(int argc, char *argv[])
{
  loadParameters(argc, argv);
  printf("\nPoses:      %d\n", params.nposes);
  printf("Iterations: %d\n", params.iterations);

  float *resultsCUDA = malloc(params.nposes*sizeof(float));
  float *resultsRef = malloc(params.nposes*sizeof(float));

  runCUDA(resultsCUDA);


  // Load reference results from file
  FILE* ref_energies = fopen(FILE_REF_ENERGIES, "r");
  size_t n_ref_poses = params.nposes;
  if (params.nposes > REF_NPOSES) {
    printf("Only validating the first %d poses.\n", REF_NPOSES);
    n_ref_poses = REF_NPOSES;
  }

  for (size_t i = 0; i < n_ref_poses; i++)
    fscanf(ref_energies, "%f", &resultsRef[i]);

  fclose(ref_energies);

  float maxdiff = -100.0f;
  printf("\n Reference        CUDA   (diff)\n");
  for (int i = 0; i < n_ref_poses; i++)
  {
    if (fabs(resultsRef[i]) < 1.f && fabs(resultsCUDA[i]) < 1.f) continue;

    float diff = fabs(resultsRef[i] - resultsCUDA[i]) / resultsCUDA[i];
    if (diff > maxdiff) {
      maxdiff = diff;
      // printf ("Maxdiff: %.2f (%.3f vs %.3f)\n", maxdiff, resultsRef[i], resultsCUDA[i]);
    }

    if (i < 8)
      printf("%7.2f    vs   %7.2f  (%5.2f%%)\n", resultsRef[i], resultsCUDA[i], 100*diff);
  }
  printf("\nLargest difference was %.3f%%\n\n", maxdiff);

  free(resultsCUDA);
  free(resultsRef);

  freeParameters();
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
  params.iterations = 8;
  _cuda.wgsize      = 64;
  _cuda.posesPerWI  = 4;
  int nposes        = 65536;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--device") || !strcmp(argv[i], "-d"))
    {
      if (++i >= argc || (_cuda.deviceIndex = parseInt(argv[i])) < 0)
      {
        printf("Invalid device index\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
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
    else if (!strcmp(argv[i], "--posesperwi") || !strcmp(argv[i], "-p"))
    {
      if (++i >= argc || (_cuda.posesPerWI = parseInt(argv[i])) < 0)
      {
        printf("Invalid poses-per-workitem value\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--wgsize") || !strcmp(argv[i], "-w"))
    {
      if (++i >= argc || (_cuda.wgsize = parseInt(argv[i])) < 0)
      {
        printf("Invalid work-group size\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./bude [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print this message\n");
      printf("      --list               List available devices\n");
      printf("      --device     INDEX   Select device at INDEX\n");
      printf("  -i  --iterations I       Repeat kernel I times\n");
      printf("  -n  --numposes   N       Compute results for N poses\n");
      printf("  -p  --poserperwi PPWI    Compute PPWI poses per work-item\n");
      printf("  -w  --wgsize     WGSIZE  Run with work-group size WGSIZE\n");
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

