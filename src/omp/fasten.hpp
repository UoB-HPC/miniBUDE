#pragma once

#include "../bude.h"
#include "omp.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS OmpBude

static volatile float discard;

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

  static inline void fasten_main(size_t group, size_t ntypes, size_t nposes, size_t natlig, size_t natpro,        //
                                 const Atom *protein, const Atom *ligand,                                         //
                                 const float *transforms_0, const float *transforms_1, const float *transforms_2, //
                                 const float *transforms_3, const float *transforms_4, const float *transforms_5, //
                                 const FFParams *forcefield, float *energies                                      //
  ) {

    float transform[3][4][PPWI];
    float etot[PPWI];

#pragma omp simd
    for (int l = 0; l < PPWI; l++) {
      int ix = group * PPWI + l;

      // Compute transformation matrix
      const float sx = std::sin(transforms_0[ix]);
      const float cx = std::cos(transforms_0[ix]);
      const float sy = std::sin(transforms_1[ix]);
      const float cy = std::cos(transforms_1[ix]);
      const float sz = std::sin(transforms_2[ix]);
      const float cz = std::cos(transforms_2[ix]);

      transform[0][0][l] = cy * cz;
      transform[0][1][l] = sx * sy * cz - cx * sz;
      transform[0][2][l] = cx * sy * cz + sx * sz;
      transform[0][3][l] = transforms_3[ix];
      transform[1][0][l] = cy * sz;
      transform[1][1][l] = sx * sy * sz + cx * cz;
      transform[1][2][l] = cx * sy * sz - sx * cz;
      transform[1][3][l] = transforms_4[ix];
      transform[2][0][l] = -sy;
      transform[2][1][l] = sx * cy;
      transform[2][2][l] = cx * cy;
      transform[2][3][l] = transforms_5[ix];

      etot[l] = 0.f;
    }

    // Loop over ligand atoms
    for (int il = 0; il < natlig; il++) {
      // Load ligand atom data
      const Atom l_atom = ligand[il];
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
      for (int ip = 0; ip < natpro; ip++) {
        // Load protein atom data
        const Atom p_atom = protein[ip];
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

  [[nodiscard]] std::string name() override { return "omp"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override {
    std::vector<Device> devices;
#ifdef OMP_TARGET
    for (int i = 0; i < omp_get_num_devices(); ++i) {
      devices.template emplace_back(i, "OMP target device #" + std::to_string(i) + " " +
                                           (omp_get_initial_device() == i ? "(host)" : "") +
                                           (omp_get_default_device() == i ? "(default)" : ""));
    }

    devices.template emplace_back(devices.size(), "OMP host device #" + std::to_string(omp_get_initial_device()));
#else
    devices.template emplace_back(0, "OMP CPU");
#endif
    return devices;
  };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t device) const override {

#ifdef OMP_TARGET
    omp_set_default_device(int(device));
    if (int actual = omp_get_default_device(); actual != int(device)) {
      throw std::runtime_error("Unable to set omp default device, need " + std::to_string(device) + ", got " +
                               std::to_string(actual));
    }
#else
    if (wgsize != 1 && wgsize != 0) {
      throw std::invalid_argument("Only wgsize = {1|0} (i.e no workgroup) are supported for OpenMP, got " +
                                  std::to_string((wgsize)));
    }
#endif

    Sample sample(PPWI, wgsize, p.nposes());

    auto contextStart = now();

    const auto ntypes = p.ntypes();
    const auto nposes = p.nposes();
    const auto natlig = p.natlig();
    const auto natpro = p.natpro();

    std::array<float *, 6> poses{};
    auto protein = static_cast<Atom *>(std::malloc(sizeof(Atom) * natpro));
    auto ligand = static_cast<Atom *>(std::malloc(sizeof(Atom) * natlig));
    auto forcefield = static_cast<FFParams *>(std::malloc(sizeof(FFParams) * ntypes));
    auto energies = static_cast<float *>(std::malloc(sizeof(float) * nposes));

    for (auto i = 0; i < 6; i++)
      poses[i] = static_cast<float *>(std::malloc(sizeof(float) * nposes));

#pragma omp parallel
    {
      for (auto i = 0; i < 6; i++) {
#pragma omp for nowait
        for (auto j = 0; j < nposes; j++)
          poses[i][j] = p.poses[i][j];
      }
#pragma omp for nowait
      for (auto i = 0; i < nposes; i++)
        energies[i] = 0.f;

#pragma omp for nowait
      for (auto i = 0; i < natpro; i++)
        protein[i] = p.protein[i];

#pragma omp for nowait
      for (auto i = 0; i < natlig; i++)
        ligand[i] = p.ligand[i];

#pragma omp for nowait
      for (auto i = 0; i < ntypes; i++)
        forcefield[i] = p.forcefield[i];
    }
    auto contextEnd = now();
    sample.contextTime = {contextStart, contextEnd};

    auto poses_0 = poses[0];
    auto poses_1 = poses[1];
    auto poses_2 = poses[2];
    auto poses_3 = poses[3];
    auto poses_4 = poses[4];
    auto poses_5 = poses[5];

#ifdef OMP_TARGET // clang-format off
  #pragma omp target data                                      \
    map(from: energies[:nposes])                               \
    map(to:                                                    \
      protein[:natpro], ligand[:natlig], forcefield[:ntypes],  \
      poses_0[:nposes], poses_1[:nposes], poses_2[:nposes],    \
      poses_3[:nposes], poses_4[:nposes], poses_5[:nposes]     \
    )
#endif // OMP_TARGET clang-format on
    for (size_t i = 0; i < p.totalIterations(); ++i) {
      auto kernelStart = now();

#ifdef OMP_TARGET // clang-format off
  #pragma omp target teams num_teams(wgsize)
  #pragma omp distribute parallel for
#else
  #pragma omp parallel for // note: orphaned 'omp teams' directives are prohibited
#endif // OMP_TARGET clang-format on
      for (size_t group = 0; group < (nposes / PPWI); group++) {
        fasten_main(group, ntypes, nposes, natlig, natpro,                //
                    protein, ligand,                                      //
                    poses_0, poses_1, poses_2, poses_3, poses_4, poses_5, //
                    forcefield, energies);
      }
      #pragma omp target update from(energies[:1])
      discard = energies[0];
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    std::copy(energies, energies + p.nposes(), sample.energies.begin());
    std::free(protein);
    std::free(ligand);
    std::free(forcefield);
    std::free(energies);
    for (auto &pose : poses)
      std::free(pose);

    return sample;
  };
};
