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

#define DATA_DIR          "../data/bm1"
#define FILE_LIGAND       "/ligand.in"
#define FILE_PROTEIN      "/protein.in"
#define FILE_FORCEFIELD   "/forcefield.in"
#define FILE_POSES        "/poses.in"
#define FILE_REF_ENERGIES "/ref_energies.out"

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
void runOpenMPTarget(float *results);

FILE* openFile(const char *parent, const char *child,
               const char* mode, long *length)
{
  char name[strlen(parent) + strlen(child) + 1];
  strcpy(name, parent);
  strcat(name, child);

  FILE *file = NULL;
  if (!(file = fopen(name, mode)))
  {
    fprintf(stderr, "Failed to open '%s'\n", name);
    exit(1);
  }
  if(length){
    fseek(file, 0, SEEK_END);
    *length = ftell(file);
    rewind(file);
  }
  return file;
}

int main(int argc, char *argv[])
{
  loadParameters(argc, argv);
  printf("\n");
  printf("Poses     : %d\n", params.nposes);
  printf("Iterations: %d\n", params.iterations);
  printf("Ligands   : %d\n", params.natlig);
  printf("Proteins  : %d\n", params.natpro);
  printf("Deck      : %s\n", params.deckDir);
  float *resultsOMP = malloc(params.nposes*sizeof(float));
  float *resultsRef = malloc(params.nposes*sizeof(float));

  runOpenMPTarget(resultsOMP);

  // Load reference results from file
  FILE* ref_energies = openFile(params.deckDir, FILE_REF_ENERGIES, "r", NULL);
  size_t n_ref_poses = params.nposes;
  if (params.nposes > REF_NPOSES) {
    printf("Only validating the first %d poses.\n", REF_NPOSES);
    n_ref_poses = REF_NPOSES;
  }

  for (size_t i = 0; i < n_ref_poses; i++)
    fscanf(ref_energies, "%f", &resultsRef[i]);

  fclose(ref_energies);

  float maxdiff = -100.0f;
  printf("\n Reference        OMP4   (diff)\n");
  for (int i = 0; i < n_ref_poses; i++)
  {
    if (fabs(resultsRef[i]) < 1.f && fabs(resultsOMP[i]) < 1.f) continue;

    float diff = fabs(resultsRef[i] - resultsOMP[i]) / resultsOMP[i];
    if (diff > maxdiff)
      maxdiff = diff;

    if (i < 8)
      printf("%7.2f    vs   %7.2f  (%5.2f%%)\n", resultsRef[i], resultsOMP[i], 100*diff);
  }
  printf("\nLargest difference was %.3f%%\n\n", maxdiff*100);

  free(resultsOMP);
  free(resultsRef);

  freeParameters();
}

void runOpenMPTarget(float *results)
{
  int natlig_s = params.natlig;
  int natpro_s = params.natpro;
  int ntypes_s = params.ntypes;
  int nposes_s = params.nposes;
  int iterations_s = params.iterations;
  Atom *restrict protein_s = params.protein;
  Atom *restrict ligand_s = params.ligand;
  FFParams *restrict forcefield_s = params.forcefield;
  float *restrict poses_0 = params.poses[0];
  float *restrict poses_1 = params.poses[1];
  float *restrict poses_2 = params.poses[2];
  float *restrict poses_3 = params.poses[3];
  float *restrict poses_4 = params.poses[4];
  float *restrict poses_5 = params.poses[5];

  printf("\nRunning C/OpenMP4\n");

  double start;
  double end;

#pragma omp target data \
  map(to: protein_s[:params.natpro], ligand_s[:params.natlig], \
      forcefield_s[:params.ntypes], poses_0[:params.nposes], \
      poses_1[:params.nposes], poses_2[:params.nposes], \
      poses_3[:params.nposes], poses_4[:params.nposes], \
      poses_5[:params.nposes])\
  map(from: results[:params.nposes])
  {
    start = getTimestamp();

    for (int itr = 0; itr < iterations_s; itr++)
    {
#pragma omp target teams distribute parallel for
      for (int i = 0; i < nposes_s/NUM_TD_PER_THREAD; ++i)
      {
        const int ind = i*NUM_TD_PER_THREAD;
        float etot[NUM_TD_PER_THREAD];
        float transform[NUM_TD_PER_THREAD][3][4];

#pragma unroll(NUM_TD_PER_THREAD)
        for(int jj = 0; jj < NUM_TD_PER_THREAD; ++jj)
        {
          const int index = ind+jj;

          // Compute transformation matrix
          const float sx = sin(poses_0[index]);
          const float cx = cos(poses_0[index]);
          const float sy = sin(poses_1[index]);
          const float cy = cos(poses_1[index]);
          const float sz = sin(poses_2[index]);
          const float cz = cos(poses_2[index]);

          transform[jj][0][0] = cy*cz;
          transform[jj][0][1] = sx*sy*cz - cx*sz;
          transform[jj][0][2] = cx*sy*cz + sx*sz;
          transform[jj][0][3] = poses_3[index];
          transform[jj][1][0] = cy*sz;
          transform[jj][1][1] = sx*sy*sz + cx*cz;
          transform[jj][1][2] = cx*sy*sz - sx*cz;
          transform[jj][1][3] = poses_4[index];
          transform[jj][2][0] = -sy;
          transform[jj][2][1] = sx*cy;
          transform[jj][2][2] = cx*cy;
          transform[jj][2][3] = poses_5[index];
          etot[jj] = 0.0f;
        }

        // Loop over ligand atoms
        for (int il = 0; il < natlig_s; ++il) // loop over ligand atoms
        {
          // Load ligand atom data
          const Atom l_atom = ligand_s[il];
          const FFParams l_params = forcefield_s[l_atom.index];
          const int lhphb_ltz = l_params.hphb<0.f;
          const int lhphb_gtz = l_params.hphb>0.f;

          float lpos_x[3];
          float lpos_y[3];
          float lpos_z[3];

          for(int jj = 0; jj < NUM_TD_PER_THREAD; ++jj)
          {
            // Transform ligand atom
            lpos_x[jj] = transform[jj][0][3]
              + l_atom.x * transform[jj][0][0]
              + l_atom.y * transform[jj][0][1]
              + l_atom.z * transform[jj][0][2];
            lpos_y[jj] = transform[jj][1][3]
              + l_atom.x * transform[jj][1][0]
              + l_atom.y * transform[jj][1][1]
              + l_atom.z * transform[jj][1][2];
            lpos_z[jj] = transform[jj][2][3]
              + l_atom.x * transform[jj][2][0]
              + l_atom.y * transform[jj][2][1]
              + l_atom.z * transform[jj][2][2];
          }

          // Loop over protein atoms
          for (int ip = 0; ip < natpro_s; ++ip) // loop over protein atoms
          {
            // Load protein atom data
            const Atom p_atom = protein_s[ip];
            const FFParams p_params = forcefield_s[p_atom.index];

            const float radij   = p_params.radius + l_params.radius;
            const float r_radij = 1.f / radij;

            const float elcdst  =
              (p_params.hbtype==HBTYPE_F && l_params.hbtype==HBTYPE_F)
              ? 4.f : 2.f;
            const float elcdst1 =
              (p_params.hbtype==HBTYPE_F && l_params.hbtype==HBTYPE_F)
              ? 0.25f : 0.5f;
            const int   type_E  =
              ((p_params.hbtype==HBTYPE_E || l_params.hbtype==HBTYPE_E));

            const int  phphb_ltz = p_params.hphb <  0.f;
            const int  phphb_gtz = p_params.hphb >  0.f;
            const int  phphb_nz  = p_params.hphb != 0.f;
            const float p_hphb   =
              p_params.hphb * (phphb_ltz && lhphb_gtz ? -1.f : 1.f);
            const float l_hphb   =
              l_params.hphb * (phphb_gtz && lhphb_ltz ? -1.f : 1.f);
            const float distdslv =
              (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST)
               : (lhphb_ltz ? NPPDIST : -FLT_MAX));
            const float r_distdslv = 1.f / distdslv;

            const float chrg_init = l_params.elsc * p_params.elsc;
            const float dslv_init = p_hphb + l_hphb;

            for(int jj = 0; jj < NUM_TD_PER_THREAD; ++jj)
            {
              // Calculate distance between atoms
              const float x = lpos_x[jj] - p_atom.x;
              const float y = lpos_y[jj] - p_atom.y;
              const float z = lpos_z[jj] - p_atom.z;
              const float distij = sqrt(x*x + y*y + z*z);

              // Calculate the sum of the sphere radii
              const float distbb = distij - radij;
              const int  zone1   = (distbb < 0.f);

              // Calculate steric energy
              etot[jj] += (1.f - (distij*r_radij))
                * (zone1 ? 2.f*HARDNESS : 0.f);

              // Calculate formal and dipole charge interactions
              float chrg_e = chrg_init
                * ((zone1 ? 1.f : (1.f - distbb*elcdst1))
                    * (distbb<elcdst ? 1.f : 0.f));
              float neg_chrg_e = -fabs(chrg_e);
              chrg_e = type_E ? neg_chrg_e : chrg_e;
              etot[jj] += chrg_e*CNSTNT;

              // Calculate the two cases for Nonpolar-Polar
              // repulsive interactions
              float coeff  = (1.f - (distbb*r_distdslv));
              float dslv_e = dslv_init *
                ((distbb<distdslv && phphb_nz) ? 1.f : 0.f);
              dslv_e *= (zone1 ? 1.f : coeff);
              etot[jj] += dslv_e;
            }
          }
        }

        // Write result
        for(int jj = 0; jj < NUM_TD_PER_THREAD; ++jj)
        {
          results[ind+jj] = etot[jj]*0.5f;
        }
      }
    }

    end = getTimestamp();
  }


  printTimings(start, end, 1);
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
  params.deckDir    = DATA_DIR;
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
    else if (!strcmp(argv[i], "--deck"))
    {
      if (++i >= argc)
      {
        printf("Invalid deck\n");
        exit(1);
      }
      params.deckDir = argv[i];
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
      printf("      --deck       DECK    Use the DECK directory as input deck\n");
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

  file = openFile(params.deckDir, FILE_LIGAND, "rb", &length);
  params.natlig = length / sizeof(Atom);
  params.ligand = malloc(params.natlig*sizeof(Atom));
  fread(params.ligand, sizeof(Atom), params.natlig, file);
  fclose(file);

  file = openFile(params.deckDir, FILE_PROTEIN, "rb", &length);
  params.natpro = length / sizeof(Atom);
  params.protein = malloc(params.natpro*sizeof(Atom));
  fread(params.protein, sizeof(Atom), params.natpro, file);
  fclose(file);

  file = openFile(params.deckDir, FILE_FORCEFIELD, "rb", &length);
  params.ntypes = length / sizeof(FFParams);
  params.forcefield = malloc(params.ntypes*sizeof(FFParams));
  fread(params.forcefield, sizeof(FFParams), params.ntypes, file);
  fclose(file);

  file = openFile(params.deckDir, FILE_POSES, "rb", &length);
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

