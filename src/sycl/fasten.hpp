#pragma once

#include "../bude.h"
#include <CL/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <string>

using namespace cl;

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS SyclBude

template <size_t N> class bude_kernel_ndrange;

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

  static constexpr sycl::access::mode R = sycl::access::mode::read;
  static constexpr sycl::access::mode DW = sycl::access::mode::discard_write;
  static constexpr sycl::access::mode RW = sycl::access::mode::read_write;

  static constexpr sycl::access::target Global = sycl::access::target::global_buffer;
  static constexpr sycl::access::target Local = sycl::access::target::local;

  template <typename T, sycl::access::mode A = R> using accessor1 = sycl::accessor<T, 1, A, Global>;

  static void fasten_main(sycl::handler &h,                                                           //
                          size_t wgsize, size_t ntypes, size_t nposes,                                //
                          const accessor1<Atom> &proteins,                                            //
                          const accessor1<Atom> &ligands,                                             //
                          const accessor1<FFParams> &forcefields,                                     //
                          const accessor1<float> &transforms_0, const accessor1<float> &transforms_1, //
                          const accessor1<float> &transforms_2, const accessor1<float> &transforms_3, //
                          const accessor1<float> &transforms_4, const accessor1<float> &transforms_5, //
                          const accessor1<float, DW> &energies) {

    size_t global = std::ceil(double(nposes) / PPWI);
    global = wgsize * size_t(std::ceil(double(global) / double(wgsize)));

    sycl::accessor<FFParams, 1, RW, Local> local_forcefield(sycl::range<1>(ntypes), h);

    h.parallel_for<bude_kernel_ndrange<PPWI>>(sycl::nd_range<1>(global, wgsize), [=](sycl::nd_item<1> item) {
      const size_t lid = item.get_local_id(0);
      const size_t gid = item.get_group(0);
      const size_t lrange = item.get_local_range(0);

      float etot[PPWI];
      sycl::float4 transform[PPWI][3];

      size_t ix = gid * lrange * PPWI + lid;
      ix = ix < nposes ? ix : nposes - PPWI;

      // XXX async_work_group_copy takes only gentypes, so no FFParams,
      //  casting *_ptr<ElementType> parameter requires first converting to void and then to gentype
      //  although probably free, there must be a better way of doing this
      // sycl::device_event event = item.async_work_group_copy<float>(
      //     sycl::local_ptr<float>(sycl::local_ptr<void>(local_forcefield.get_pointer())),
      //     sycl::global_ptr<float>(sycl::global_ptr<void>(forcefields.get_pointer())),
      //     ntypes * sizeof(FFParams) / sizeof(float));

      for (int i = lid; i < ntypes; i += lrange)
        local_forcefield[i] = forcefields[i];

      // Compute transformation matrix to private memory
      const size_t lsz = lrange;
      for (size_t i = 0; i < PPWI; i++) {
        size_t index = ix + i * lsz;

        const float sx = sycl::sin(transforms_0[index]);
        const float cx = sycl::cos(transforms_0[index]);
        const float sy = sycl::sin(transforms_1[index]);
        const float cy = sycl::cos(transforms_1[index]);
        const float sz = sycl::sin(transforms_2[index]);
        const float cz = sycl::cos(transforms_2[index]);

        transform[i][0].x() = cy * cz;
        transform[i][0].y() = sx * sy * cz - cx * sz;
        transform[i][0].z() = cx * sy * cz + sx * sz;
        transform[i][0].w() = transforms_3[index];
        transform[i][1].x() = cy * sz;
        transform[i][1].y() = sx * sy * sz + cx * cz;
        transform[i][1].z() = cx * sy * sz - sx * cz;
        transform[i][1].w() = transforms_4[index];
        transform[i][2].x() = -sy;
        transform[i][2].y() = sx * cy;
        transform[i][2].z() = cx * cy;
        transform[i][2].w() = transforms_5[index];

        etot[i] = ZERO;
      }

      // item.wait_for(event);
      item.barrier(sycl::access::fence_space::local_space);

      // Loop over ligand atoms
      for (size_t il = 0; il < ligands.get_count(); il++) {
        // Load ligand atom data
        const Atom l_atom = ligands[il];
        const FFParams l_params = local_forcefield[l_atom.type];
        const bool lhphb_ltz = l_params.hphb < ZERO;
        const bool lhphb_gtz = l_params.hphb > ZERO;

        const sycl::float4 linitpos(l_atom.x, l_atom.y, l_atom.z, ONE);
        sycl::float3 lpos[PPWI];
        for (size_t i = 0; i < PPWI; i++) {
          // Transform ligand atom
          lpos[i].x() = transform[i][0].w() + linitpos.x() * transform[i][0].x() + linitpos.y() * transform[i][0].y() +
                        linitpos.z() * transform[i][0].z();
          lpos[i].y() = transform[i][1].w() + linitpos.x() * transform[i][1].x() + linitpos.y() * transform[i][1].y() +
                        linitpos.z() * transform[i][1].z();
          lpos[i].z() = transform[i][2].w() + linitpos.x() * transform[i][2].x() + linitpos.y() * transform[i][2].y() +
                        linitpos.z() * transform[i][2].z();
        }

        // Loop over protein atoms
        for (size_t ip = 0; ip < proteins.get_count(); ip++) {
          // Load protein atom data
          const Atom p_atom = proteins[ip];
          const FFParams p_params = local_forcefield[p_atom.type];

          const float radij = p_params.radius + l_params.radius;
          const float r_radij = ONE / (radij);

          const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
          const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
          const bool type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

          const bool phphb_ltz = p_params.hphb < ZERO;
          const bool phphb_gtz = p_params.hphb > ZERO;
          const bool phphb_nz = p_params.hphb != ZERO;
          const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
          const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
          const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
          const float r_distdslv = ONE / (distdslv);

          const float chrg_init = l_params.elsc * p_params.elsc;
          const float dslv_init = p_hphb + l_hphb;

          for (size_t i = 0; i < PPWI; i++) {
            // Calculate distance between atoms
            const float x = lpos[i].x() - p_atom.x;
            const float y = lpos[i].y() - p_atom.y;
            const float z = lpos[i].z() - p_atom.z;

            // XXX as of oneapi-2021.1-beta10, the sycl::native::sqrt variant is significantly slower for no apparent
            // reason
            const float distij = sycl::sqrt(x * x + y * y + z * z);

            // XXX as of oneapi-2021.1-beta10, the following variant is significantly slower for no apparent reason
            // const float distij = sycl::distance(lpos[i], sycl::float3(p_atom.x, p_atom.y, p_atom.z));

            // Calculate the sum of the sphere radii
            const float distbb = distij - radij;
            const bool zone1 = (distbb < ZERO);

            // Calculate steric energy
            etot[i] += (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : ZERO);

            // Calculate formal and dipole charge interactions
            float chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * (distbb < elcdst ? ONE : ZERO));
            const float neg_chrg_e = -sycl::fabs(chrg_e);
            chrg_e = type_E ? neg_chrg_e : chrg_e;
            etot[i] += chrg_e * CNSTNT;

            // Calculate the two cases for Nonpolar-Polar repulsive interactions
            const float coeff = (ONE - (distbb * r_distdslv));
            float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? ONE : ZERO);
            dslv_e *= (zone1 ? ONE : coeff);
            etot[i] += dslv_e;
          }
        };
      }

      // Write results
      const size_t td_base = gid * lrange * PPWI + lid;

      if (td_base < nposes) {
        for (size_t i = 0; i < PPWI; i++) {
          energies[td_base + i * lrange] = etot[i] * HALF;
        }
      }
    });
  }

  std::vector<cl::sycl::device> devices;

public:
  IMPL_CLS() : devices(sycl::device::get_devices()) {}

  [[nodiscard]] std::string name() override { return "sycl"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override {
    std::vector<Device> xs;
    for (size_t i = 0; i < devices.size(); i++)
      xs.emplace_back(i, devices[i].template get_info<sycl::info::device::name>());
    return xs;
  };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t deviceIdx) const override {
    auto device = devices[deviceIdx];

    Sample sample(PPWI, wgsize, p.nposes());

    auto contextStart = now();
    sycl::queue queue(device);
    sycl::buffer<Atom> proteins(p.protein.data(), p.protein.size());
    sycl::buffer<Atom> ligands(p.ligand.data(), p.ligand.size());
    sycl::buffer<FFParams> forcefields(p.forcefield.data(), p.forcefield.size());
    sycl::buffer<float> transforms_0(p.poses[0].data(), p.poses[0].size());
    sycl::buffer<float> transforms_1(p.poses[1].data(), p.poses[1].size());
    sycl::buffer<float> transforms_2(p.poses[2].data(), p.poses[2].size());
    sycl::buffer<float> transforms_3(p.poses[3].data(), p.poses[3].size());
    sycl::buffer<float> transforms_4(p.poses[4].data(), p.poses[4].size());
    sycl::buffer<float> transforms_5(p.poses[5].data(), p.poses[5].size());
    sycl::buffer<float> energies(sample.energies.size());
    queue.wait_and_throw();
    auto contextEnd = now();
    sample.contextTime = {contextStart, contextEnd};

    for (size_t i = 0; i < p.iterations + p.warmupIterations; ++i) {
      auto kernelStart = now();
      queue.submit([&](sycl::handler &h) {
        fasten_main(h, wgsize, p.ntypes(), p.nposes(),                                                 //
                    proteins.get_access<R>(h), ligands.get_access<R>(h), forcefields.get_access<R>(h), //
                    transforms_0.get_access<R>(h), transforms_1.get_access<R>(h), transforms_2.get_access<R>(h),
                    transforms_3.get_access<R>(h), transforms_4.get_access<R>(h), transforms_5.get_access<R>(h),
                    energies.get_access<DW>(h));
      });
      queue.wait_and_throw();
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    queue.submit([&](sycl::handler &h) { h.copy(energies.get_access<R>(h), sample.energies.data()); });
    queue.wait_and_throw();

    return sample;
  };
};