#include <float.h>
#include <math.h>
#include <stddef.h>

#include "bude.h"

// Energy evaluation parameters
#define CNSTNT   45.0f
#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST  5.5f
#define NPPDIST   1.0f

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
                 const int group)
{
  float transform[3][4][WGSIZE];
  float etot[WGSIZE];

#pragma omp simd
  for (int l = 0; l < WGSIZE; l++)
  {
    int ix = group*WGSIZE + l;

    // Compute transformation matrix
    const float sx = sinf(transforms_0[ix]);
    const float cx = cosf(transforms_0[ix]);
    const float sy = sinf(transforms_1[ix]);
    const float cy = cosf(transforms_1[ix]);
    const float sz = sinf(transforms_2[ix]);
    const float cz = cosf(transforms_2[ix]);

    transform[0][0][l] = cy*cz;
    transform[0][1][l] = sx*sy*cz - cx*sz;
    transform[0][2][l] = cx*sy*cz + sx*sz;
    transform[0][3][l] = transforms_3[ix];
    transform[1][0][l] = cy*sz;
    transform[1][1][l] = sx*sy*sz + cx*cz;
    transform[1][2][l] = cx*sy*sz - sx*cz;
    transform[1][3][l] = transforms_4[ix];
    transform[2][0][l] = -sy;
    transform[2][1][l] = sx*cy;
    transform[2][2][l] = cx*cy;
    transform[2][3][l] = transforms_5[ix];

    etot[l] = 0.f;
  }

  {
    // Loop over ligand atoms
    int il = 0;
    do
    {
      // Load ligand atom data
      const Atom l_atom = ligand[il];
      const FFParams l_params = forcefield[l_atom.type];
      const int lhphb_ltz = l_params.hphb<0.f;
      const int lhphb_gtz = l_params.hphb>0.f;

      // Transform ligand atom
      float lpos_x[WGSIZE], lpos_y[WGSIZE], lpos_z[WGSIZE];

#pragma omp simd
      for (int l = 0; l < WGSIZE; l++)
      {
        lpos_x[l] = transform[0][3][l]
          + l_atom.x * transform[0][0][l]
          + l_atom.y * transform[0][1][l]
          + l_atom.z * transform[0][2][l];
        lpos_y[l] = transform[1][3][l]
          + l_atom.x * transform[1][0][l]
          + l_atom.y * transform[1][1][l]
          + l_atom.z * transform[1][2][l];
        lpos_z[l] = transform[2][3][l]
          + l_atom.x * transform[2][0][l]
          + l_atom.y * transform[2][1][l]
          + l_atom.z * transform[2][2][l];
      }

      // Loop over protein atoms
      int ip = 0;
      do
      {
        // Load protein atom data
        const Atom p_atom = protein[ip];
        const FFParams p_params = forcefield[p_atom.type];

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

#pragma omp simd
        for (int l = 0; l < WGSIZE; l++)
        {
          // Calculate distance between atoms
          const float x      = lpos_x[l] - p_atom.x;
          const float y      = lpos_y[l] - p_atom.y;
          const float z      = lpos_z[l] - p_atom.z;
          const float distij = sqrtf(x*x + y*y + z*z);

          // Calculate the sum of the sphere radii
          const float distbb = distij - radij;

          const int  zone1   = (distbb < 0.f);

          // Calculate steric energy
          etot[l] += (1.f - (distij*r_radij)) * (zone1 ? 2.f*HARDNESS : 0.f);

          // Calculate formal and dipole charge interactions
          float chrg_e = chrg_init
            * ((zone1 ? 1.f : (1.f - distbb*elcdst1))
               * (distbb<elcdst ? 1.f : 0.f));
          float neg_chrg_e = -fabsf(chrg_e);
          chrg_e = type_E ? neg_chrg_e : chrg_e;
          etot[l]  += chrg_e*CNSTNT;

          // Calculate the two cases for Nonpolar-Polar repulsive interactions
          float coeff  = (1.f - (distbb*r_distdslv));
          float dslv_e = dslv_init * ((distbb<distdslv && phphb_nz) ? 1.f : 0.f);
          dslv_e *= (zone1 ? 1.f : coeff);
          etot[l]   += dslv_e;
        }
      } while (++ip < natpro); // loop over protein atoms
    } while (++il < natlig); // loop over ligand atoms
  }

#pragma omp simd
  for (int l = 0; l < WGSIZE; l++)
  {
    // Write result
    results[group*WGSIZE + l] = etot[l]*0.5f;
  }
}
