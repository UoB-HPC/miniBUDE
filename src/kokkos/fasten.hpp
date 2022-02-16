#pragma once

#include "../bude.h"
#include <string>

#include <Kokkos_Core.hpp>

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS KokkosBude

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

public:
  static void fasten_main(size_t wgsize,                                              //
                          size_t ntypes, size_t nposes, size_t natlig, size_t natpro, //
                          const Kokkos::View<const Atom *> &proteins, const Kokkos::View<const Atom *> &ligands,
                          const Kokkos::View<const FFParams *> &forcefields, //
                          const Kokkos::View<const float *> &transforms_0,
                          const Kokkos::View<const float *> &transforms_1,
                          const Kokkos::View<const float *> &transforms_2,
                          const Kokkos::View<const float *> &transforms_3,
                          const Kokkos::View<const float *> &transforms_4,
                          const Kokkos::View<const float *> &transforms_5, const Kokkos::View<float *> &etotals) {

    size_t global = std::ceil(double(nposes) / PPWI);
    global = size_t(std::ceil(double(global) / double(wgsize)));

    // blockIdx  = league_rank = get_group(_id)
    // blockDim  = team_size   = get_local_range
    // threadIdx = team_rank   = get_local_id

    Kokkos::TeamPolicy<> policy((int(global)), (int(wgsize)));
    policy.set_scratch_size(0, Kokkos::PerTeam(int(ntypes * sizeof(FFParams))));

    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type group) {
          const int lid = group.team_rank();
          const int gid = group.league_rank();
          const int lrange = group.team_size();

          float etot[PPWI];
          float transform[3][4][PPWI];

          size_t ix = gid * lrange * PPWI + lid;
          ix = ix < nposes ? ix : nposes - PPWI;

          Kokkos::View<FFParams *,                                          //
                       Kokkos::DefaultExecutionSpace::scratch_memory_space, //
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>             //
              local_forcefield(group.team_scratch(0), group.team_size());

          // TODO async copy
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(group, ntypes),
                               [&](const int i) { local_forcefield[i] = forcefields[i]; });

          // Compute transformation matrix to private memory
          const size_t lsz = lrange;
          for (size_t i = 0; i < PPWI; i++) {
            size_t index = ix + i * lsz;

            const float sx = std::sin(transforms_0[index]);
            const float cx = std::cos(transforms_0[index]);
            const float sy = std::sin(transforms_1[index]);
            const float cy = std::cos(transforms_1[index]);
            const float sz = std::sin(transforms_2[index]);
            const float cz = std::cos(transforms_2[index]);

            transform[0][0][i] = cy * cz;
            transform[0][1][i] = sx * sy * cz - cx * sz;
            transform[0][2][i] = cx * sy * cz + sx * sz;
            transform[0][3][i] = transforms_3[index];
            transform[1][0][i] = cy * sz;
            transform[1][1][i] = sx * sy * sz + cx * cz;
            transform[1][2][i] = cx * sy * sz - sx * cz;
            transform[1][3][i] = transforms_4[index];
            transform[2][0][i] = -sy;
            transform[2][1][i] = sx * cy;
            transform[2][2][i] = cx * cy;
            transform[2][3][i] = transforms_5[index];

            etot[i] = ZERO;
          }

          group.team_barrier();

          // Loop over ligand atoms
          size_t il = 0;
          do {
            // Load ligand atom data
            const Atom l_atom = ligands[il];
            const FFParams l_params = local_forcefield[l_atom.type];
            const bool lhphb_ltz = l_params.hphb < ZERO;
            const bool lhphb_gtz = l_params.hphb > ZERO;

            float lpos_x[PPWI], lpos_y[PPWI], lpos_z[PPWI];
            for (size_t i = 0; i < PPWI; i++) {
              // Transform ligand atom
              lpos_x[i] = transform[0][3][i] + l_atom.x * transform[0][0][i] + l_atom.y * transform[0][1][i] +
                          l_atom.z * transform[0][2][i];
              lpos_y[i] = transform[1][3][i] + l_atom.x * transform[1][0][i] + l_atom.y * transform[1][1][i] +
                          l_atom.z * transform[1][2][i];
              lpos_z[i] = transform[2][3][i] + l_atom.x * transform[2][0][i] + l_atom.y * transform[2][1][i] +
                          l_atom.z * transform[2][2][i];
            }

            // Loop over protein atoms
            size_t ip = 0;
            do {
              // Load protein atom data
              const Atom p_atom = proteins[ip];
              const FFParams p_params = local_forcefield[p_atom.type];

              const float radij = p_params.radius + l_params.radius;
              const float r_radij = 1.f / (radij);

              const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
              const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
              const bool type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

              const bool phphb_ltz = p_params.hphb < ZERO;
              const bool phphb_gtz = p_params.hphb > ZERO;
              const bool phphb_nz = p_params.hphb != ZERO;
              const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
              const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
              const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
              const float r_distdslv = 1.f / (distdslv);

              const float chrg_init = l_params.elsc * p_params.elsc;
              const float dslv_init = p_hphb + l_hphb;

              for (size_t i = 0; i < PPWI; i++) {
                // Calculate distance between atoms
                const float x = lpos_x[i] - p_atom.x;
                const float y = lpos_y[i] - p_atom.y;
                const float z = lpos_z[i] - p_atom.z;

                const float distij = std::sqrt(x * x + y * y + z * z);

                // Calculate the sum of the sphere radii
                const float distbb = distij - radij;
                const bool zone1 = (distbb < ZERO);

                // Calculate steric energy
                etot[i] += (ONE - (distij * r_radij)) * (zone1 ? 2 * HARDNESS : ZERO);

                // Calculate formal and dipole charge interactions
                float chrg_e = chrg_init * ((zone1 ? 1 : (ONE - distbb * elcdst1)) * (distbb < elcdst ? 1 : ZERO));
                const float neg_chrg_e = -std::fabs(chrg_e);
                chrg_e = type_E ? neg_chrg_e : chrg_e;
                etot[i] += chrg_e * CNSTNT;

                // Calculate the two cases for Nonpolar-Polar repulsive interactions
                const float coeff = (ONE - (distbb * r_distdslv));
                float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? 1 : ZERO);
                dslv_e *= (zone1 ? 1 : coeff);
                etot[i] += dslv_e;
              }
            } while (++ip < natpro); // loop over protein atoms
          } while (++il < natlig);   // loop over ligand atoms

          // Write results
          const size_t td_base = gid * lrange * PPWI + lid;

          if (td_base < nposes) {
            for (size_t i = 0; i < PPWI; i++) {
              etotals[td_base + i * lrange] = etot[i] * HALF;
            }
          }
        });
  }

  template <typename T> static Kokkos::View<T *> mkView(const std::string &name, const std::vector<T> &xs) {
    Kokkos::View<T *> view(name, xs.size());
    auto mirror = Kokkos::create_mirror_view(view);
    for (size_t i = 0; i < xs.size(); i++)
      mirror[i] = xs[i];
    Kokkos::deep_copy(view, mirror);
    return view;
  }

public:
  IMPL_CLS() = default;

  [[nodiscard]] std::string name() { return "kokkos"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override {
    return {{0, std::string(typeid(Kokkos::DefaultExecutionSpace).name()) + " (specify backend at compile-time)"}};
  };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t) const override {

    // initialising twice is an error
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }

    Sample sample(PPWI, wgsize, p.nposes());

    auto contextStart = now();

    auto protein = mkView("protein", p.protein);
    auto ligand = mkView("ligand", p.ligand);
    auto transforms_0 = mkView("transforms_0", p.poses[0]);
    auto transforms_1 = mkView("transforms_1", p.poses[1]);
    auto transforms_2 = mkView("transforms_2", p.poses[2]);
    auto transforms_3 = mkView("transforms_3", p.poses[3]);
    auto transforms_4 = mkView("transforms_4", p.poses[4]);
    auto transforms_5 = mkView("transforms_5", p.poses[5]);
    auto forcefield = mkView("forcefield", p.forcefield);
    Kokkos::View<float *> results("results", sample.energies.size());
    Kokkos::fence();
    auto contextEnd = now();
    sample.contextTime = {contextStart, contextEnd};

    for (size_t i = 0; i < p.iterations + p.warmupIterations; ++i) {
      auto kernelStart = now();
      fasten_main(wgsize, p.ntypes(), p.nposes(), p.natlig(), p.natpro(), //
                  protein, ligand, forcefield,                            //
                  transforms_0, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5, results);
      Kokkos::fence();
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    auto result_mirror = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(result_mirror, results);
    for (size_t i = 0; i < results.size(); i++) {
      sample.energies[i] = result_mirror[i];
    }

    return sample;
  };
};
