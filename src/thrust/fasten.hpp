#pragma once

#include "../bude.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS ThrustBude

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

public:
  static void fasten_main(const Params &p, //
                          thrust::device_vector<Atom> &protein, thrust::device_vector<Atom> &ligand,
                          thrust::device_vector<float> &transforms_0, thrust::device_vector<float> &transforms_1,
                          thrust::device_vector<float> &transforms_2, thrust::device_vector<float> &transforms_3,
                          thrust::device_vector<float> &transforms_4, thrust::device_vector<float> &transforms_5,
                          thrust::device_vector<FFParams> &forcefield, thrust::device_vector<float> &energies) {

    thrust::counting_iterator<int> groups(0);
    thrust::for_each(
        groups, groups + (p.nposes() / PPWI),
        [natlig = p.natlig(), natpro = p.natpro(),                                                                   //
         protein = protein.data(), ligand = ligand.data(),                                                           //
         transforms_0 = transforms_0.data(), transforms_1 = transforms_1.data(), transforms_2 = transforms_2.data(), //
         transforms_3 = transforms_3.data(), transforms_4 = transforms_4.data(), transforms_5 = transforms_5.data(), //
         forcefield = forcefield.data(),                                                                             //
         energies = energies.data()] __device__
#ifndef __NVCC__
            // otherwise, we get "init-captures are not allowed for extended __host__ __device__"
            __host__
#endif
        (const int group) {
          std::array<std::array<Vec4<float>, 3>, PPWI> transform = {};
          std::array<float, PPWI> etot = {};

#pragma omp simd
          for (int l = 0; l < PPWI; l++) {
            int ix = group * PPWI + l;

            const float sx = std::sin(transforms_0[ix]);
            const float cx = std::cos(transforms_0[ix]);
            const float sy = std::sin(transforms_1[ix]);
            const float cy = std::cos(transforms_1[ix]);
            const float sz = std::sin(transforms_2[ix]);
            const float cz = std::cos(transforms_2[ix]);

            transform[l][0].x = cy * cz;
            transform[l][0].y = sx * sy * cz - cx * sz;
            transform[l][0].z = cx * sy * cz + sx * sz;
            transform[l][0].w = transforms_3[ix];
            transform[l][1].x = cy * sz;
            transform[l][1].y = sx * sy * sz + cx * cz;
            transform[l][1].z = cx * sy * sz - sx * cz;
            transform[l][1].w = transforms_4[ix];
            transform[l][2].x = -sy;
            transform[l][2].y = sx * cy;
            transform[l][2].z = cx * cy;
            transform[l][2].w = transforms_5[ix];
          }

          // Loop over ligand atoms
          for (size_t il = 0; il < natlig; il++) {
            const Atom &l_atom = ligand[il];
            const FFParams &l_params = forcefield[l_atom.type];
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
            for (size_t ip = 0; ip < natpro; ip++) {
              // Load protein atom data
              const Atom &p_atom = protein[ip];
              const FFParams &p_params = forcefield[p_atom.type];

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
              const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
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
                float chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * (distbb < elcdst ? ONE : ZERO));
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

// Write result
#pragma omp simd
          for (int l = 0; l < PPWI; l++) {
            energies[group * PPWI + l] = etot[l] *= HALF;
          }
        });
  }

public:
  IMPL_CLS() = default;

  [[nodiscard]] std::string name() override { return "thrust"; };

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA ||                                                               \
    (defined(THRUST_DEVICE_SYSTEM_HIP) && THRUST_DEVICE_SYSTEM_HIP == THRUST_DEVICE_SYSTEM)

  #ifdef __NVCC__
    #define IMPL_FN__(fn) cuda##fn
    #define IMPL_TYPE__(tpe) cuda##tpe
  #elif defined(__HIP_PLATFORM_HCC__)
    #define IMPL_FN__(fn) hip##fn
    #define IMPL_TYPE__(tpe) hip##tpe##_t
  #else
    #error Unsupported compiler for Thrust
  #endif

  static inline void checkError(const IMPL_FN__(Error_t) err = IMPL_FN__(GetLastError())) {
    if (err != IMPL_FN__(Success)) {
      throw std::runtime_error(std::string(IMPL_FN__(GetErrorName(err))) + ": " +
                               std::string(IMPL_FN__(GetErrorString(err))));
    }
  };

  [[nodiscard]] std::vector<Device> enumerateDevices() override {
    int count = 0;
    checkError(IMPL_FN__(GetDeviceCount(&count)));
    std::vector<Device> devices(count);
    for (int i = 0; i < count; ++i) {
  #if defined(__HIP_PLATFORM_HCC__) //  can't use IMPL_TYPE__ here because of the extra _t suffix, thanks AMD
      hipDeviceProp_t props{};
  #else
      cudaDeviceProp props{};
  #endif
      checkError(IMPL_FN__(GetDeviceProperties(&props, i)));
      devices[i] = {i, std::string(props.name) + " (" +                                        //
                           std::to_string(props.totalGlobalMem / 1024 / 1024) + "MB;" +        //
                           "sm_" + std::to_string(props.major) + std::to_string(props.minor) + //
                           ")"};
    }
    return devices;
  };

  #undef IMPL_FN__
  #undef IMPL_TPE__

#else

  [[nodiscard]] std::vector<Device> enumerateDevices() override { return {{0, "(not exposed)"}}; };

#endif

  static inline void synchronise() {
// rocThrust doesn't synchronise between thrust calls
#if defined(THRUST_DEVICE_SYSTEM_HIP) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
    hipDeviceSynchronize();
#endif
  }

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t) const override {

    if (wgsize != 1 && wgsize != 0) {
      throw std::invalid_argument("Only wgsize = {1|0} (i.e no workgroup) are supported for OpenMP, got " +
                                  std::to_string((wgsize)));
    }

    Sample sample(PPWI, wgsize, p.nposes());

    thrust::device_vector<Atom> protein(p.protein);
    thrust::device_vector<Atom> ligand(p.ligand);
    thrust::device_vector<float> transforms_0(p.poses[0]);
    thrust::device_vector<float> transforms_1(p.poses[1]);
    thrust::device_vector<float> transforms_2(p.poses[2]);
    thrust::device_vector<float> transforms_3(p.poses[3]);
    thrust::device_vector<float> transforms_4(p.poses[4]);
    thrust::device_vector<float> transforms_5(p.poses[5]);
    thrust::device_vector<FFParams> forcefield(p.forcefield);
    thrust::device_vector<float> energies(sample.energies.size());

    for (size_t i = 0; i < p.totalIterations(); ++i) {
      auto kernelStart = now();
      fasten_main(p, protein, ligand,                       //
                  transforms_0, transforms_1, transforms_2, //
                  transforms_3, transforms_4, transforms_5, //
                  forcefield, energies                      //
      );
      synchronise();
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }
    thrust::copy(energies.begin(), energies.end(), sample.energies.begin());

    return sample;
  };
};
