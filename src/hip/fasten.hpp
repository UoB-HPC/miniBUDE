#pragma once

#include "../bude.h"
#include "hip/hip_runtime.h"
#include <iostream>
#include <string>

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS HipBude

template <size_t PPWI>
static __global__ void fasten_main(int natlig, int natpro,
                                   const Atom *protein_molecule, //
                                   const Atom *ligand_molecule,  //
                                   const float *transforms_0, const float *transforms_1, const float *transforms_2,
                                   const float *transforms_3, const float *transforms_4, const float *transforms_5,
                                   float *etotals, const FFParams *global_forcefield, int numTransforms) {
  // Get index of first TD
  int ix = blockIdx.x * blockDim.x * PPWI + threadIdx.x;

  // Have extra threads do the last member intead of return.
  // A return would disable use of barriers, so not using return is better
  ix = ix < numTransforms ? ix : numTransforms - PPWI;

#ifdef USE_SHARED
  extern __shared__ FFParams forcefield[];
  if (ix < num_atom_types) {
    forcefield[ix] = global_forcefield[ix];
  }
#else
  const FFParams *forcefield = global_forcefield;
#endif

  // Compute transformation matrix to private memory
  float etot[PPWI];
  float4 transform[PPWI][3];
  const size_t lsz = blockDim.x;
  for (int i = 0; i < PPWI; i++) {
    size_t index = ix + i * lsz;

    const float sx = sin(transforms_0[index]);
    const float cx = cos(transforms_0[index]);
    const float sy = sin(transforms_1[index]);
    const float cy = cos(transforms_1[index]);
    const float sz = sin(transforms_2[index]);
    const float cz = cos(transforms_2[index]);

    transform[i][0].x = cy * cz;
    transform[i][0].y = sx * sy * cz - cx * sz;
    transform[i][0].z = cx * sy * cz + sx * sz;
    transform[i][0].w = transforms_3[index];
    transform[i][1].x = cy * sz;
    transform[i][1].y = sx * sy * sz + cx * cz;
    transform[i][1].z = cx * sy * sz - sx * cz;
    transform[i][1].w = transforms_4[index];
    transform[i][2].x = -sy;
    transform[i][2].y = sx * cy;
    transform[i][2].z = cx * cy;
    transform[i][2].w = transforms_5[index];

    etot[i] = ZERO;
  }

#ifdef USE_SHARED
  __syncthreads();
#endif

  // Loop over ligand atoms
  int il = 0;
  do {
    // Load ligand atom data
    const Atom l_atom = ligand_molecule[il];

    const FFParams l_params = forcefield[l_atom.type];
    const bool lhphb_ltz = l_params.hphb < ZERO;
    const bool lhphb_gtz = l_params.hphb > ZERO;

    float3 lpos[PPWI];
    const float4 linitpos = make_float4(l_atom.x, l_atom.y, l_atom.z, ONE);
    for (int i = 0; i < PPWI; i++) {
      // Transform ligand atom
      lpos[i].x = transform[i][0].w + linitpos.x * transform[i][0].x + linitpos.y * transform[i][0].y +
                  linitpos.z * transform[i][0].z;
      lpos[i].y = transform[i][1].w + linitpos.x * transform[i][1].x + linitpos.y * transform[i][1].y +
                  linitpos.z * transform[i][1].z;
      lpos[i].z = transform[i][2].w + linitpos.x * transform[i][2].x + linitpos.y * transform[i][2].y +
                  linitpos.z * transform[i][2].z;
    }

    // Loop over protein atoms
    int ip = 0;
    do {
      // Load protein atom data
      const Atom p_atom = protein_molecule[ip];

      const FFParams p_params = forcefield[p_atom.type];

      const float radij = p_params.radius + l_params.radius;
      const float r_radij = 1.0f / radij;

      const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
      const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
      const bool type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

      const bool phphb_ltz = p_params.hphb < ZERO;
      const bool phphb_gtz = p_params.hphb > ZERO;
      const bool phphb_nz = p_params.hphb != ZERO;
      const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
      const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
      const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
      const float r_distdslv = 1.0f / distdslv;

      const float chrg_init = l_params.elsc * p_params.elsc;
      const float dslv_init = p_hphb + l_hphb;

      for (int i = 0; i < PPWI; i++) {
        // Calculate distance between atoms
        const float x = lpos[i].x - p_atom.x;
        const float y = lpos[i].y - p_atom.y;
        const float z = lpos[i].z - p_atom.z;
        const float distij = sqrt(x * x + y * y + z * z);

        // Calculate the sum of the sphere radii
        const float distbb = distij - radij;
        const bool zone1 = (distbb < ZERO);

        // Calculate steric energy
        etot[i] += (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : ZERO);

        // Calculate formal and dipole charge interactions
        float chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * (distbb < elcdst ? ONE : ZERO));
        const float neg_chrg_e = -fabs(chrg_e);
        chrg_e = type_E ? neg_chrg_e : chrg_e;
        etot[i] += chrg_e * CNSTNT;

        // Calculate the two cases for Nonpolar-Polar repulsive interactions
        const float coeff = (ONE - (distbb * r_distdslv));
        float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? ONE : ZERO);
        dslv_e *= (zone1 ? ONE : coeff);
        etot[i] += dslv_e;
      }
    } while (++ip < natpro); // loop over protein atoms
  } while (++il < natlig);   // loop over ligand atoms

  // Write results
  const int td_base = blockIdx.x * blockDim.x * PPWI + threadIdx.x;
  if (td_base < numTransforms) {
    for (int i = 0; i < PPWI; i++) {
      etotals[td_base + i * blockDim.x] = etot[i] * HALF;
    }
  }
}

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

public:
  IMPL_CLS() = default;

  static inline void checkError(const hipError_t err = hipGetLastError()) {
    if (err != hipSuccess) {
      throw std::runtime_error(std::string(hipGetErrorName(err)) + ": " + std::string(hipGetErrorString(err)));
    }
  }

  [[nodiscard]] std::string name() override { return "hip"; };


  [[nodiscard]] std::vector<Device> enumerateDevices() override {
    int count = 0;
    checkError(hipGetDeviceCount(&count));
    std::vector<Device> devices(count);
    for (int i = 0; i < count; ++i) {
      hipDeviceProp_t props{};
      checkError(hipGetDeviceProperties(&props, i));
      devices[i] = {i, std::string(props.name) + " (" +                                        //
                           std::to_string(props.totalGlobalMem / 1024 / 1024) + "MB;" +        //
                           "sm_" + std::to_string(props.major) + std::to_string(props.minor) + //
                           ")"};
    }
    return devices;
  };
  //  #define MANAGED

  template <typename T> [[nodiscard]] static T *allocate(size_t size) {
    T *data = nullptr;
#if defined(MANAGED)
    checkError(hipMallocManaged(&data, sizeof(T) * size));
#elif defined(PAGEFAULT)
    data = (T *)std::malloc(sizeof(T) * size);
#else
    checkError(hipMalloc(&data, sizeof(T) * size));
#endif
    return data;
  }

  template <typename T> [[nodiscard]] static T *allocate(const std::vector<T> &xs) {
    T *data = allocate<T>(xs.size());
    checkError(hipMemcpy(data, xs.data(), xs.size() * sizeof(T), hipMemcpyHostToDevice));
    return data;
  }

  template <typename T> static void free(T *data) {
#if defined(PAGEFAULT)
    std::free(data);
#else
    checkError(hipFree(data));
#endif
  }

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t device) const override {

    checkError(hipSetDevice(int(device)));

    Sample sample(PPWI, wgsize, p.nposes());

    auto contextStart = now();
    auto protein = allocate(p.protein);
    auto ligand = allocate(p.ligand);
    auto transforms_0 = allocate(p.poses[0]);
    auto transforms_1 = allocate(p.poses[1]);
    auto transforms_2 = allocate(p.poses[2]);
    auto transforms_3 = allocate(p.poses[3]);
    auto transforms_4 = allocate(p.poses[4]);
    auto transforms_5 = allocate(p.poses[5]);
    auto forcefield = allocate(p.forcefield);
    auto results = allocate<float>(sample.energies.size());
    checkError(hipDeviceSynchronize());
    auto contextEnd = now();

    sample.contextTime = {contextStart, contextEnd};

    size_t global = std::ceil(double(p.nposes()) / PPWI);
    global = std::ceil(double(global) / double(wgsize));
    size_t local = wgsize;
    size_t shared = p.ntypes() * sizeof(FFParams);

    for (size_t i = 0; i < p.totalIterations(); ++i) {
      auto kernelStart = now();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(fasten_main<PPWI>), dim3(global), dim3(local), shared, 0,           //
                         p.natlig(), p.natpro(), protein, ligand,                                            //
                         transforms_0, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5, //
                         results, forcefield, p.nposes());
      checkError(hipDeviceSynchronize());
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    checkError(
        hipMemcpy(sample.energies.data(), results, sample.energies.size() * sizeof(float), hipMemcpyDeviceToHost));

    free(protein);
    free(ligand);
    free(transforms_0);
    free(transforms_1);
    free(transforms_2);
    free(transforms_3);
    free(transforms_4);
    free(transforms_5);
    free(forcefield);
    free(results);

    checkError(hipDeviceSynchronize());

    return sample;
  };
};
