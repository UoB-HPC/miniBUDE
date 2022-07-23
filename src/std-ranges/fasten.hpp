#pragma once

#include "../bude.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <ranges>
#include <string>

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS StdRangesBude

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

  static void fasten_main(size_t group,                                        //
                          const std::vector<Atom> &protein,                    //
                          const std::vector<Atom> &ligand,                     //
                          const std::array<std::vector<float>, 6> &transforms, //
                          const std::vector<FFParams> &forcefield,             //
                          std::vector<float> &energies                         //
  ) {

    float transform[3][4][PPWI];
    float etot[PPWI];

#pragma omp simd
    for (int l = 0; l < PPWI; l++) {
      int ix = group * PPWI + l;

      // Compute transformation matrix
      const float sx = std::sin(transforms[0][ix]);
      const float cx = std::cos(transforms[0][ix]);
      const float sy = std::sin(transforms[1][ix]);
      const float cy = std::cos(transforms[1][ix]);
      const float sz = std::sin(transforms[2][ix]);
      const float cz = std::cos(transforms[2][ix]);

      transform[0][0][l] = cy * cz;
      transform[0][1][l] = sx * sy * cz - cx * sz;
      transform[0][2][l] = cx * sy * cz + sx * sz;
      transform[0][3][l] = transforms[3][ix];
      transform[1][0][l] = cy * sz;
      transform[1][1][l] = sx * sy * sz + cx * cz;
      transform[1][2][l] = cx * sy * sz - sx * cz;
      transform[1][3][l] = transforms[4][ix];
      transform[2][0][l] = -sy;
      transform[2][1][l] = sx * cy;
      transform[2][2][l] = cx * cy;
      transform[2][3][l] = transforms[5][ix];

      etot[l] = 0.f;
    }

    // Loop over ligand atoms

    for (const Atom &l_atom : ligand) {
      // Load ligand atom data
      const FFParams l_params = forcefield[l_atom.type];
      const int lhphb_ltz = l_params.hphb < 0.f;
      const int lhphb_gtz = l_params.hphb > 0.f;

      // Transform ligand atom
      float lpos_x[PPWI], lpos_y[PPWI], lpos_z[PPWI];

#pragma omp simd
      for (int l = 0; l < PPWI; l++) {
        lpos_x[l] = transform[0][3][l] + l_atom.x * transform[0][0][l] + l_atom.y * transform[0][1][l] +
                    l_atom.z * transform[0][2][l];
        lpos_y[l] = transform[1][3][l] + l_atom.x * transform[1][0][l] + l_atom.y * transform[1][1][l] +
                    l_atom.z * transform[1][2][l];
        lpos_z[l] = transform[2][3][l] + l_atom.x * transform[2][0][l] + l_atom.y * transform[2][1][l] +
                    l_atom.z * transform[2][2][l];
      }
      // Loop over protein atoms

      for (const Atom &p_atom : protein) {
        // Load protein atom data
        const FFParams p_params = forcefield[p_atom.type];

        const float radij = p_params.radius + l_params.radius;
        const float r_radij = ONE / radij;

        const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
        const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
        const int type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

        const int phphb_ltz = p_params.hphb < 0.f;
        const int phphb_gtz = p_params.hphb > 0.f;
        const int phphb_nz = p_params.hphb != 0.f;
        const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
        const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
        const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
        const float r_distdslv = ONE / distdslv;

        const float chrg_init = l_params.elsc * p_params.elsc;
        const float dslv_init = p_hphb + l_hphb;

#pragma omp simd
        for (int l = 0; l < PPWI; l++) {
          // Calculate distance between atoms
          const float x = lpos_x[l] - p_atom.x;
          const float y = lpos_y[l] - p_atom.y;
          const float z = lpos_z[l] - p_atom.z;
          const float distij = std::sqrt(x * x + y * y + z * z);

          // Calculate the sum of the sphere radii
          const float distbb = distij - radij;

          const int zone1 = (distbb < ZERO);

          // Calculate steric energy
          etot[l] += (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : 0.f);

          // Calculate formal and dipole charge interactions
          float chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * (distbb < elcdst ? ONE : ZERO));
          float neg_chrg_e = -std::abs(chrg_e);
          chrg_e = type_E ? neg_chrg_e : chrg_e;
          etot[l] += chrg_e * CNSTNT;

          // Calculate the two cases for Nonpolar-Polar repulsive interactions
          float coeff = (ONE - (distbb * r_distdslv));
          float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? ONE : 0.f);
          dslv_e *= (zone1 ? ONE : coeff);
          etot[l] += dslv_e;
        }
      }
    }

    // Write result
#pragma omp simd
    for (int l = 0; l < PPWI; l++) {
      energies[group * PPWI + l] = etot[l] * HALF;
    }
  }

public:
  IMPL_CLS() = default;

  [[nodiscard]] std::string name() override { return "std20"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override { return {{0, "CPU"}}; };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t) const override {

    if (wgsize != 1 && wgsize != 0) {
      throw std::invalid_argument("Only wgsize = {1|0} (i.e no workgroup) are supported for OpenMP, got " +
                                  std::to_string((wgsize)));
    }

    Sample sample(PPWI, wgsize, p.nposes());

    for (size_t i = 0; i < p.totalIterations(); ++i) {
      auto kernelStart = now();

      std::for_each_n(std::execution::par_unseq, std::views::iota(0).begin(), p.nposes() / PPWI, [&](int group) {
        fasten_main(group,               //
                    p.protein, p.ligand, //
                    p.poses, p.forcefield, sample.energies);
      });

      auto kernelEnd = now();

      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    return sample;
  };
};
