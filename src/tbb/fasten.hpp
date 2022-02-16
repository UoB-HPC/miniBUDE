#pragma once

#include "../bude.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "tbb/tbb.h"

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS TbbBude

#if defined(PARTITIONER_AUTO)
using tbb_partitioner = tbb::auto_partitioner;
  #define PARTITIONER_NAME "auto_partitioner"
#elif defined(PARTITIONER_AFFINITY)
using tbb_partitioner = tbb::affinity_partitioner;
  #define PARTITIONER_NAME "affinity_partitioner"
#elif defined(PARTITIONER_STATIC)
using tbb_partitioner = tbb::static_partitioner;
  #define PARTITIONER_NAME "static_partitioner"
#elif defined(PARTITIONER_SIMPLE)
using tbb_partitioner = tbb::simple_partitioner;
  #define PARTITIONER_NAME "simple_partitioner"
#else
// default to auto
using tbb_partitioner = tbb::auto_partitioner;
  #define PARTITIONER_NAME "auto_partitioner"
#endif

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

  static void fasten_main(const Params &p, std::vector<float> &results) {

    tbb_partitioner partitioner;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, p.nposes() / PPWI),
        [&](const tbb::blocked_range<size_t> &r) {
          for (size_t group = r.begin(); group < r.end(); ++group) {

            std::array<std::array<Vec4<float>, 3>, PPWI> transform = {};
            std::array<float, PPWI> etot = {};

#pragma omp simd
            for (int l = 0; l < PPWI; l++) {
              int ix = group * PPWI + l;

              const float sx = std::sin(p.poses[0][ix]);
              const float cx = std::cos(p.poses[0][ix]);
              const float sy = std::sin(p.poses[1][ix]);
              const float cy = std::cos(p.poses[1][ix]);
              const float sz = std::sin(p.poses[2][ix]);
              const float cz = std::cos(p.poses[2][ix]);

              transform[l][0].x = cy * cz;
              transform[l][0].y = sx * sy * cz - cx * sz;
              transform[l][0].z = cx * sy * cz + sx * sz;
              transform[l][0].w = p.poses[3][ix];
              transform[l][1].x = cy * sz;
              transform[l][1].y = sx * sy * sz + cx * cz;
              transform[l][1].z = cx * sy * sz - sx * cz;
              transform[l][1].w = p.poses[4][ix];
              transform[l][2].x = -sy;
              transform[l][2].y = sx * cy;
              transform[l][2].z = cx * cy;
              transform[l][2].w = p.poses[5][ix];
            }

            // Loop over ligand atoms
            for (const Atom &l_atom : p.ligand) {
              const FFParams l_params = p.forcefield[l_atom.type];
              const int lhphb_ltz = l_params.hphb < ZERO;
              const int lhphb_gtz = l_params.hphb > ZERO;

              // Transform ligand atom
              std::array<Vec3<float>, PPWI> lpos = {};
#pragma omp simd
              for (int l = 0; l < PPWI; l++) {
                lpos[l].x = transform[l][0].w + l_atom.x * transform[l][0].x + l_atom.y * transform[l][0].y +
                            l_atom.z * transform[l][0].z;
                lpos[l].y = transform[l][1].w + l_atom.x * transform[l][1].x + l_atom.y * transform[l][1].y +
                            l_atom.z * transform[l][1].z;
                lpos[l].z = transform[l][2].w + l_atom.x * transform[l][2].x + l_atom.y * transform[l][2].y +
                            l_atom.z * transform[l][2].z;
              }

              // Loop over protein atoms
              for (const Atom &p_atom : p.protein) {
                //          // Load protein atom data
                const FFParams p_params = p.forcefield[p_atom.type];

                const float radij = p_params.radius + l_params.radius;
                const float r_radij = ONE / radij;

                const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
                const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
                const int type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

                const int phphb_ltz = p_params.hphb < ZERO;
                const int phphb_gtz = p_params.hphb > ZERO;
                const int phphb_nz = p_params.hphb != ZERO;
                const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
                const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
                const float distdslv =
                    (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
                const float r_distdslv = ONE / distdslv;

                const float chrg_init = l_params.elsc * p_params.elsc;
                const float dslv_init = p_hphb + l_hphb;

#pragma omp simd
                for (int l = 0; l < PPWI; l++) {

                  // Calculate distance between atoms
                  const float x = lpos[l].x - p_atom.x;
                  const float y = lpos[l].y - p_atom.y;
                  const float z = lpos[l].z - p_atom.z;
                  const float distij = std::sqrt(x * x + y * y + z * z);

                  // Calculate the sum of the sphere radii
                  const float distbb = distij - radij;

                  const int zone1 = (distbb < ZERO);

                  //  Calculate steric energy
                  etot[l] += (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : ZERO);

                  // Calculate formal and dipole charge interactions
                  float chrg_e =
                      chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * (distbb < elcdst ? ONE : ZERO));
                  float neg_chrg_e = -std::abs(chrg_e);
                  chrg_e = type_E ? neg_chrg_e : chrg_e;
                  etot[l] += chrg_e * CNSTNT;

                  // Calculate the two cases for Nonpolar-Polar repulsive interactions
                  float coeff = (ONE - (distbb * r_distdslv));
                  float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? ONE : ZERO);
                  dslv_e *= (zone1 ? ONE : coeff);
                  etot[l] += dslv_e;
                }
              }
            }

#pragma omp simd
            for (int l = 0; l < PPWI; l++) {
              results[group * PPWI + l] = etot[l] * HALF;
            }
          }
        },
        partitioner);
  }

public:
  IMPL_CLS() = default;

  [[nodiscard]] std::string name() override { return "tbb"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override { return {{0, "(not exposed)"}}; };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t) const override {

    if (wgsize != 1 && wgsize != 0) {
      throw std::invalid_argument("Only wgsize = {1|0} (i.e no workgroup) are supported for OpenMP, got " +
                                  std::to_string((wgsize)));
    }

    Sample sample(PPWI, wgsize, p.nposes());

    for (size_t i = 0; i < p.totalIterations(); ++i) {
      auto kernelStart = now();
      fasten_main(p, sample.energies);
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    return sample;
  };
};
