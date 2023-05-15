#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include "../bude.h"
#include "CL/opencl.hpp"

#include <cstdint>
#include <cstdlib>
#include <string>

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS OclBude

#define STR_(s) #s
#define XSTR_(s) STR_(s)
#define MAKE_DEF_ARGS(name) " -D" #name "=" XSTR_(name)

constexpr static const char *constants =                                  //
    MAKE_DEF_ARGS(ZERO) MAKE_DEF_ARGS(QUARTER) MAKE_DEF_ARGS(HALF)        //
    MAKE_DEF_ARGS(ONE) MAKE_DEF_ARGS(TWO) MAKE_DEF_ARGS(FOUR)             //
    MAKE_DEF_ARGS(CNSTNT) MAKE_DEF_ARGS(HBTYPE_F) MAKE_DEF_ARGS(HBTYPE_E) //
    MAKE_DEF_ARGS(HARDNESS) MAKE_DEF_ARGS(NPNPDIST) MAKE_DEF_ARGS(NPPDIST);

#undef STR_
#undef xSTR_
#undef MAKE_DEF_ARGS

constexpr static const char *kernelSource{R"CLC(
#ifndef PPWI
  #error PPWI not defined
#endif

typedef struct {
  float x, y, z;
  int type;
} Atom;

typedef struct {
  int hbtype;
  float radius;
  float hphb;
  float elsc;
} FFParams;

__kernel void fasten_main(                                                                    //
    __local FFParams *restrict local_forcefields,                                             //
    const uint ntypes, const uint nposes, const uint natlig, const uint natpro,               //
    const __global Atom *restrict proteins,                                                   //
    const __global Atom *restrict ligands,                                                    //
    const __global FFParams *restrict forcefields,                                            //
    const __global float *restrict transforms_0, const __global float *restrict transforms_1, //
    const __global float *restrict transforms_2, const __global float *restrict transforms_3, //
    const __global float *restrict transforms_4, const __global float *restrict transforms_5, //
    __global float *restrict etotals) {
  // Get index of first TD
  int ix = get_group_id(0) * get_local_size(0) * PPWI + get_local_id(0);

  // Have extra threads do the last member instead of return.
  // A return would disable use of barriers, so not using return is better
  ix = ix < nposes ? ix : nposes - PPWI;

  // Copy forcefield parameter table to local memory
  event_t event = async_work_group_copy((__local float *)local_forcefields, (__global float *)forcefields,
                                        ntypes * sizeof(FFParams) / sizeof(float), 0);

  // Compute transformation matrix to private memory
  float etot[PPWI];
  float4 transform[PPWI][3];
  const int lsz = get_local_size(0);
  for (int i = 0; i < PPWI; i++) {
    int index = ix + i * lsz;

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

  // Wait for forcefield copy to finish
  wait_group_events(1, &event);

  // Loop over ligand atoms
  int il = 0;
  do {
    // Load ligand atom data
    const Atom l_atom = ligands[il];
    const FFParams l_params = local_forcefields[l_atom.type];
    const bool lhphb_ltz = l_params.hphb < ZERO;
    const bool lhphb_gtz = l_params.hphb > ZERO;

    float3 lpos[PPWI];
    const float4 linitpos = (float4)(l_atom.x, l_atom.y, l_atom.z, ONE);
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
      const Atom p_atom = proteins[ip];
      const FFParams p_params = local_forcefields[p_atom.type];

      const float radij = p_params.radius + l_params.radius;
      const float r_radij = native_recip(radij);

      const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
      const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
      const bool type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

      const bool phphb_ltz = p_params.hphb < ZERO;
      const bool phphb_gtz = p_params.hphb > ZERO;
      const bool phphb_nz = p_params.hphb != ZERO;
      const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
      const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
      const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FLT_MAX));
      const float r_distdslv = native_recip(distdslv);

      const float chrg_init = l_params.elsc * p_params.elsc;
      const float dslv_init = p_hphb + l_hphb;

      for (int i = 0; i < PPWI; i++) {
        // Calculate distance between atoms
        const float x = lpos[i].x - p_atom.x;
        const float y = lpos[i].y - p_atom.y;
        const float z = lpos[i].z - p_atom.z;
        const float distij = native_sqrt(x * x + y * y + z * z);

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
  const int td_base = get_group_id(0) * get_local_size(0) * PPWI + get_local_id(0);
  if (td_base < nposes) {
    for (int i = 0; i < PPWI; i++) {
      etotals[td_base + i * get_local_size(0)] = etot[i] * HALF;
    }
  }
}
)CLC"};

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

  std::vector<std::pair<cl::Platform, cl::Device>> devices;

public:
  IMPL_CLS() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto &platform : platforms) {
      std::vector<cl::Device> ds;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &ds);
      std::transform(ds.begin(), ds.end(), std::back_inserter(devices),
                     [&](auto &d) { return std::make_pair(platform, d); });
    }
  };

  [[nodiscard]] std::string name() override { return "ocl"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override {
    std::vector<Device> xs;
    for (size_t i = 0; i < devices.size(); i++) {
      xs.emplace_back(i, devices[i].second.template getInfo<CL_DEVICE_NAME>() + " (" +
                             devices[i].first.template getInfo<CL_PLATFORM_NAME>() + ")");
    }
    return xs;
  };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t device) const override {

    Sample sample(PPWI, wgsize, p.nposes());

    auto contextStart = now();

    auto context = cl::Context(devices[device].second);
    auto queue = cl::CommandQueue(context);

    auto p_ = p; // XXX cl::Buffer doesn't cast away const even if readOnly is true so we make a mutable copy here first
    cl::Buffer proteins(context, p_.protein.begin(), p_.protein.end(), true);
    cl::Buffer ligands(context, p_.ligand.begin(), p_.ligand.end(), true);
    cl::Buffer forcefields(context, p_.forcefield.begin(), p_.forcefield.end(), true);
    cl::Buffer transforms_0(context, p_.poses[0].begin(), p_.poses[0].end(), true);
    cl::Buffer transforms_1(context, p_.poses[1].begin(), p_.poses[1].end(), true);
    cl::Buffer transforms_2(context, p_.poses[2].begin(), p_.poses[2].end(), true);
    cl::Buffer transforms_3(context, p_.poses[3].begin(), p_.poses[3].end(), true);
    cl::Buffer transforms_4(context, p_.poses[4].begin(), p_.poses[4].end(), true);
    cl::Buffer transforms_5(context, p_.poses[5].begin(), p_.poses[5].end(), true);
    cl::Buffer energies(context, CL_MEM_READ_WRITE, sizeof(float) * p_.nposes());

    cl::Program program(context, kernelSource);

    const std::string options = " -cl-mad-enable"
                                " -cl-no-signed-zeros"
                                " -cl-unsafe-math-optimizations"
                                " -cl-finite-math-only" +
                                (" -DPPWI=" + std::to_string(PPWI)) + constants;

    try {
      program.build(options.c_str());
    } catch (...) {
      std::cerr << "Program failed to compile, flags=`" << options << "`, source:\n"
                << std::endl; // << kernelSource << std::endl;
      std::istringstream f(kernelSource);
      std::string line;
      int lineNum = 1;
      while (std::getline(f, line)) {
        std::cerr << std::setw(3) << lineNum++ << "｜" << line << std::endl;
      }
      auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
      std::cerr << "Compiler output(" << log.size() << " lines):\n" << std::endl;
      for (auto &[_, s] : log) {
        std::cerr << ">" << s << std::endl;
      }
    }

    auto kernel = cl::KernelFunctor<cl::LocalSpaceArg,                   //
                                    cl_uint, cl_uint, cl_uint, cl_uint,  //
                                    cl::Buffer, cl::Buffer, cl::Buffer,  //
                                    cl::Buffer, cl::Buffer,              //
                                    cl::Buffer, cl::Buffer,              //
                                    cl::Buffer, cl::Buffer,              //
                                    cl::Buffer>(program, "fasten_main"); //
    auto contextEnd = now();

    sample.contextTime = {contextStart, contextEnd};

    size_t global = std::ceil(double(p.nposes()) / PPWI);
    global = wgsize * size_t(std::ceil(double(global) / double(wgsize)));

    for (size_t i = 0; i < p.iterations + p.warmupIterations; ++i) {
      auto kernelStart = now();
      kernel(cl::EnqueueArgs(queue, cl::NDRange(global), cl::NDRange(wgsize)), //
             cl::Local(sizeof(FFParams) * p.ntypes()),                         //
             p.ntypes(), p.nposes(), p.natlig(), p.natpro(),                   //
             proteins, ligands, forcefields,                                   //
             transforms_0, transforms_1, transforms_2,                         //
             transforms_3, transforms_4, transforms_5,                         //
             energies                                                          //
      );
      queue.finish();
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }
    cl::copy(queue, energies, sample.energies.begin(), sample.energies.end());

    queue.finish();
    return sample;
  };
};
